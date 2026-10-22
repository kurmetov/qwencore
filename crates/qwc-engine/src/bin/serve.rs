//! Text-only OpenAI-compatible server for the local Qwen checkpoint.
//!
//! The HTTP side is asynchronous; one dedicated worker owns CUDA, model
//! weights, the continuous-batching scheduler, and the shared paged KV pool.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use qwc_core::arch::{KV_ELEMS_PER_TOKEN, VOCAB_SIZE};
use qwc_cuda::delta_net::DeltaStateMode;
use qwc_cuda::paged_attention::{KvCacheDtype, PAGE_SIZE};
use qwc_engine::executor::MAX_SPECULATION_ROWS;
use qwc_engine::mtp::Speculator;
use qwc_engine::{DecodeLinearMode, Executor, ExecutorConfig, ModelWeights, PREFILL_CHUNK_SIZE};
use qwc_model::Checkpoint;
use qwc_runtime::{
    Batch, BatchLayout, CacheManager, Completion, FinishReason, Request, Scheduler, SchedulerConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::mpsc as async_mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

static NEXT_SEQUENCE: AtomicU32 = AtomicU32::new(1);

#[derive(Clone)]
struct AppState {
    command_tx: mpsc::Sender<Command>,
    tokenizer: Arc<Tokenizer>,
    model: Arc<str>,
    max_context: usize,
    /// Whether to reason if the request itself didn't set enable_thinking (--thinking).
    thinking: bool,
}

struct Args {
    model_path: PathBuf,
    model_name: String,
    bind: SocketAddr,
    max_context: usize,
    max_seqs: usize,
    prefill_chunk: usize,
    memory_limit: usize,
    kv_cache_bytes: usize,
    kv_cache_dtype: KvCacheDtype,
    delta_state: DeltaStateMode,
    prefix_snapshots: usize,
    speculation: SpeculationConfig,
    thinking: bool,
}

/// Speculation with the MTP head. Works when decode is running for a single
/// sequence: the head holds the KV of only one dialog.
#[derive(Clone, Copy)]
struct SpeculationConfig {
    /// Draft depth; 0 — speculation off.
    depth: usize,
    /// Vocab shortlist for draft logits; 0 — full projection.
    shortlist: usize,
    /// How many last prompt tokens the head runs during prefill.
    prime_window: usize,
}

struct WorkerConfig {
    model_path: PathBuf,
    max_context: usize,
    max_seqs: usize,
    prefill_chunk: usize,
    memory_limit: usize,
    kv_pool_blocks: usize,
    kv_cache_dtype: KvCacheDtype,
    delta_state: DeltaStateMode,
    prefix_snapshots: usize,
    speculation: SpeculationConfig,
    eos_ids: Vec<u32>,
}

enum Command {
    Generate(GenerateJob),
}

struct GenerateJob {
    id: u32,
    prompt: Vec<u32>,
    /// Length of the dialog history: everything before the generation prompt. The prefix cache
    /// is captured at this boundary — the next turn will repeat the history verbatim.
    history_tokens: usize,
    max_new_tokens: usize,
    stop: Vec<String>,
    buffer_output: bool,
    /// How many top tokens to return with log-probabilities at each step.
    logprobs: Option<usize>,
    /// Don't stop at EOS: matching against `eval` runs exactly `max_tokens`
    /// steps.
    ignore_eos: bool,
    event_tx: async_mpsc::UnboundedSender<EngineEvent>,
}

enum EngineEvent {
    Delta(String),
    Done {
        text: String,
        finish_reason: &'static str,
        completion_tokens: usize,
        buffered_output: bool,
        /// Bytes of `text` already sent as deltas before the first `<tool_call>`.
        streamed: usize,
        tokens: Vec<u32>,
        scores: Vec<StepScores>,
    },
    Error(String),
}

/// Log-probabilities of one emitted token: its own and those of the top
/// `top_k`, the way `eval` computes them.
struct StepScores {
    chosen: f32,
    top: Vec<(u32, f32)>,
}

struct ActiveJob {
    prompt: Vec<u32>,
    output: Vec<u32>,
    rendered: String,
    stop: Vec<String>,
    buffer_output: bool,
    /// How many bytes of `rendered` went out as deltas with `buffer_output`.
    streamed: usize,
    logprobs: Option<usize>,
    ignore_eos: bool,
    scores: Vec<StepScores>,
    /// Scores of the token the next `emit_token` will emit. Logits exist at
    /// sampling time, while emission happens a step later.
    next_scores: Option<StepScores>,
    event_tx: async_mpsc::UnboundedSender<EngineEvent>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    #[serde(rename = "temperature")]
    _temperature: Option<f32>,
    #[serde(default)]
    stop: Option<Stop>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    /// This is how reasoning is turned off in vLLM: `{"enable_thinking": false}`.
    #[serde(default)]
    chat_template_kwargs: Option<Value>,
    #[serde(default)]
    reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    tool_calls: Option<Vec<Value>>,
    /// The previous turn's reasoning, separate from the answer, as vLLM returns it.
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

/// The old OpenAI `/v1/completions`: the prompt bypasses the chat template, as text
/// or ready token ids. Needed for matching against `eval`, which needs the server's
/// logits on the same tokens.
#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: CompletionPrompt,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    #[serde(rename = "temperature")]
    _temperature: Option<f32>,
    #[serde(default)]
    stop: Option<Stop>,
    #[serde(default)]
    logprobs: Option<usize>,
    /// vLLM extension: generate exactly `max_tokens`, ignoring EOS.
    #[serde(default)]
    ignore_eos: bool,
    /// vLLM extension: tokens in `logprobs` as `token_id:N`, so the client
    /// doesn't have to recover ids from text.
    #[serde(default)]
    return_tokens_as_token_ids: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CompletionPrompt {
    Text(String),
    Tokens(Vec<u32>),
}

/// vLLM returns this many top tokens by default; a matching client doesn't
/// need more.
const MAX_LOGPROBS: usize = 20;

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    r#type: &'static str,
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorEnvelope {
                error: ErrorBody {
                    message: self.message,
                    r#type: "invalid_request_error",
                },
            }),
        )
            .into_response()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let tokenizer = Arc::new(
        Tokenizer::from_file(args.model_path.join("tokenizer.json"))
            .map_err(|error| format!("cannot load tokenizer.json: {error}"))?,
    );
    let eos_ids = load_eos_ids(&args.model_path)?;
    let bytes_per_block = PAGE_SIZE * KV_ELEMS_PER_TOKEN * args.kv_cache_dtype.bytes_per_element();
    let kv_pool_blocks = args.kv_cache_bytes / bytes_per_block;
    let max_blocks = args.max_context.div_ceil(PAGE_SIZE);
    if kv_pool_blocks < max_blocks {
        return Err(format!(
            "KV pool has {kv_pool_blocks} pages, but context {} needs {max_blocks}; increase --kv-cache-gb",
            args.max_context
        )
        .into());
    }

    let (command_tx, command_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let worker = WorkerConfig {
        model_path: args.model_path.clone(),
        max_context: args.max_context,
        max_seqs: args.max_seqs,
        prefill_chunk: args.prefill_chunk,
        memory_limit: args.memory_limit,
        kv_pool_blocks,
        kv_cache_dtype: args.kv_cache_dtype,
        delta_state: args.delta_state,
        prefix_snapshots: args.prefix_snapshots,
        speculation: args.speculation,
        eos_ids,
    };
    std::thread::Builder::new()
        .name("qwc-engine".into())
        .spawn(move || run_worker(worker, command_rx, ready_tx))?;
    let cache_bytes = ready_rx
        .recv()
        .map_err(|_| "engine exited during startup")?
        .map_err(|error| format!("engine startup failed: {error}"))?;

    let state = AppState {
        command_tx,
        tokenizer,
        model: Arc::from(args.model_name.as_str()),
        max_context: args.max_context,
        thinking: args.thinking,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    eprintln!(
        "qwc serve: http://{} model={} context={} KV={} pages ({:.2} GB cache+state), \
prefix cache {} snapshots ({:.2} GB)",
        args.bind,
        args.model_name,
        args.max_context,
        kv_pool_blocks,
        cache_bytes as f64 / 1e9,
        args.prefix_snapshots,
        (args.prefix_snapshots * Executor::prefix_snapshot_bytes()) as f64 / 1e9
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model.as_ref(),
            "object": "model",
            "owned_by": "local",
            "thinking": state.thinking
        }]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(mut request): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    if let Some(model) = request.model.as_deref() {
        if model != state.model.as_ref() {
            return Err(ApiError::bad_request(format!("unknown model: {model}")));
        }
    }

    // The client didn't say whether to reason — the server's launch mode decides.
    if request.enable_thinking.is_none()
        && request
            .chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("enable_thinking"))
            .is_none()
    {
        request.enable_thinking = Some(state.thinking);
    }
    let prompt_text = render_chat(&request)?;
    let encoding = state
        .tokenizer
        .encode(prompt_text, false)
        .map_err(|error| ApiError::bad_request(format!("tokenization failed: {error}")))?;
    let prompt = encoding.get_ids().to_vec();
    if prompt.is_empty() {
        return Err(ApiError::bad_request("rendered prompt is empty"));
    }
    let requested = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(256);
    let remaining = state.max_context.saturating_sub(prompt.len());
    if requested == 0 || requested > remaining {
        return Err(ApiError::bad_request(format!(
            "prompt has {} tokens; requested {} output tokens exceed context {}",
            prompt.len(),
            requested,
            state.max_context
        )));
    }
    if prompt.iter().any(|&token| token as usize >= VOCAB_SIZE) {
        return Err(ApiError::bad_request(
            "tokenizer produced an out-of-vocabulary ID",
        ));
    }

    let id = NEXT_SEQUENCE.fetch_add(1, Ordering::Relaxed).max(1);
    let request_id = format!("chatcmpl-qwc-{id}");
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let prompt_tokens = prompt.len();
    let history_tokens = history_boundary(&state.tokenizer, &prompt);
    let stop = match request.stop {
        None => Vec::new(),
        Some(Stop::One(stop)) => vec![stop],
        Some(Stop::Many(stops)) => stops,
    };
    // A tool call is emitted by this checkpoint as one XML object. Buffer it
    // so OpenAI clients receive a structured tool_calls delta instead of raw
    // model-specific markup that they cannot execute.
    let buffer_output = request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty());
    let (event_tx, event_rx) = async_mpsc::unbounded_channel();
    state
        .command_tx
        .send(Command::Generate(GenerateJob {
            id,
            prompt,
            history_tokens,
            max_new_tokens: requested,
            stop,
            buffer_output,
            logprobs: None,
            ignore_eos: false,
            event_tx,
        }))
        .map_err(|_| ApiError::unavailable("engine worker is not running"))?;

    if request.stream {
        let model = state.model.clone();
        let first = json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model.as_ref(),
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        });
        let first = tokio_stream::iter(vec![Ok::<Event, std::convert::Infallible>(
            Event::default().data(first.to_string()),
        )]);
        let stream_id = request_id.clone();
        let tools = request.tools.clone().unwrap_or_default();
        let event_stream = UnboundedReceiverStream::new(event_rx).flat_map(move |event| {
            let events = stream_events(&stream_id, created, model.as_ref(), &tools, event);
            tokio_stream::iter(
                events
                    .into_iter()
                    .map(Ok::<Event, std::convert::Infallible>),
            )
        });
        return Ok(Sse::new(first.chain(event_stream))
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response());
    }

    let mut events = UnboundedReceiverStream::new(event_rx);
    while let Some(event) = events.next().await {
        match event {
            EngineEvent::Delta(_) => {}
            EngineEvent::Done {
                text,
                finish_reason,
                completion_tokens,
                ..
            } => {
                let assistant = parse_assistant(&text, request.tools.as_deref().unwrap_or(&[]));
                let finish_reason = if assistant.tool_calls.is_empty() {
                    finish_reason
                } else {
                    "tool_calls"
                };
                return Ok(Json(json!({
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": state.model.as_ref(),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": assistant.content,
                            "tool_calls": assistant.tool_calls
                        },
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }))
                .into_response());
            }
            EngineEvent::Error(message) => return Err(ApiError::unavailable(message)),
        }
    }
    Err(ApiError::unavailable("engine closed the response channel"))
}

/// The generation prompt starts at the last `<|im_start|>`: the model's turn
/// will enter the next request's history already without `<think>` and with the answer. The prefix cache
/// is captured at this boundary.
fn history_boundary(tokenizer: &Tokenizer, prompt: &[u32]) -> usize {
    tokenizer
        .token_to_id("<|im_start|>")
        .and_then(|start| prompt.iter().rposition(|&token| token == start))
        .unwrap_or(prompt.len())
}

async fn completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    if let Some(model) = request.model.as_deref()
        && model != state.model.as_ref()
    {
        return Err(ApiError::bad_request(format!("unknown model: {model}")));
    }
    if request.stream {
        return Err(ApiError::bad_request(
            "stream is not supported on /v1/completions",
        ));
    }
    if request.logprobs.is_some_and(|top_k| top_k > MAX_LOGPROBS) {
        return Err(ApiError::bad_request(format!(
            "logprobs must be at most {MAX_LOGPROBS}"
        )));
    }
    let prompt = match request.prompt {
        CompletionPrompt::Tokens(tokens) => tokens,
        CompletionPrompt::Text(text) => state
            .tokenizer
            .encode(text, false)
            .map_err(|error| ApiError::bad_request(format!("tokenization failed: {error}")))?
            .get_ids()
            .to_vec(),
    };
    if prompt.is_empty() {
        return Err(ApiError::bad_request("prompt is empty"));
    }
    if prompt.iter().any(|&token| token as usize >= VOCAB_SIZE) {
        return Err(ApiError::bad_request("prompt has an out-of-vocabulary ID"));
    }
    let requested = request.max_tokens.unwrap_or(16);
    let remaining = state.max_context.saturating_sub(prompt.len());
    if requested == 0 || requested > remaining {
        return Err(ApiError::bad_request(format!(
            "prompt has {} tokens; requested {} output tokens exceed context {}",
            prompt.len(),
            requested,
            state.max_context
        )));
    }

    let id = NEXT_SEQUENCE.fetch_add(1, Ordering::Relaxed).max(1);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let prompt_tokens = prompt.len();
    let history_tokens = history_boundary(&state.tokenizer, &prompt);
    let stop = match request.stop {
        None => Vec::new(),
        Some(Stop::One(stop)) => vec![stop],
        Some(Stop::Many(stops)) => stops,
    };
    let (event_tx, event_rx) = async_mpsc::unbounded_channel();
    state
        .command_tx
        .send(Command::Generate(GenerateJob {
            id,
            prompt,
            history_tokens,
            max_new_tokens: requested,
            stop,
            buffer_output: true,
            logprobs: request.logprobs,
            ignore_eos: request.ignore_eos,
            event_tx,
        }))
        .map_err(|_| ApiError::unavailable("engine worker is not running"))?;

    let mut events = UnboundedReceiverStream::new(event_rx);
    while let Some(event) = events.next().await {
        match event {
            EngineEvent::Delta(_) => {}
            EngineEvent::Done {
                text,
                finish_reason,
                completion_tokens,
                tokens,
                scores,
                ..
            } => {
                let logprobs = request.logprobs.map(|_| {
                    completion_logprobs(
                        &state.tokenizer,
                        &tokens,
                        &scores,
                        request.return_tokens_as_token_ids,
                    )
                });
                return Ok(Json(json!({
                    "id": format!("cmpl-qwc-{id}"),
                    "object": "text_completion",
                    "created": created,
                    "model": state.model.as_ref(),
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "logprobs": logprobs,
                        "finish_reason": finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }))
                .into_response());
            }
            EngineEvent::Error(message) => return Err(ApiError::unavailable(message)),
        }
    }
    Err(ApiError::unavailable("engine closed the response channel"))
}

/// Old-format `logprobs`: tokens, their log-probabilities and top dictionaries.
fn completion_logprobs(
    tokenizer: &Tokenizer,
    tokens: &[u32],
    scores: &[StepScores],
    as_ids: bool,
) -> Value {
    let name = |token: u32| {
        if as_ids {
            format!("token_id:{token}")
        } else {
            tokenizer.decode(&[token], false).unwrap_or_default()
        }
    };
    let top: Vec<Value> = scores
        .iter()
        .map(|step| {
            let row: serde_json::Map<String, Value> = step
                .top
                .iter()
                .map(|&(token, logprob)| (name(token), json!(logprob)))
                .collect();
            Value::Object(row)
        })
        .collect();
    json!({
        "tokens": tokens.iter().map(|&token| name(token)).collect::<Vec<_>>(),
        "token_logprobs": scores.iter().map(|step| step.chosen).collect::<Vec<_>>(),
        "top_logprobs": top,
    })
}

fn stream_events(
    id: &str,
    created: u64,
    model: &str,
    tools: &[Value],
    event: EngineEvent,
) -> Vec<Event> {
    match event {
        EngineEvent::Delta(delta) => {
            vec![Event::default().data(
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": null}]
            })
            .to_string(),
        )]
        }
        EngineEvent::Done {
            text,
            finish_reason,
            buffered_output,
            streamed,
            ..
        } => {
            // What went out as deltas is a prefix without calls; send the rest.
            let assistant = parse_assistant(&text[streamed.min(text.len())..], tools);
            let finish_reason = if assistant.tool_calls.is_empty() {
                finish_reason
            } else {
                "tool_calls"
            };
            let mut events = Vec::new();
            if buffered_output {
                let mut delta = serde_json::Map::new();
                if let Some(content) = assistant.content {
                    delta.insert("content".into(), Value::String(content));
                }
                if !assistant.tool_calls.is_empty() {
                    // In a stream OpenAI merges call deltas by index;
                    // without it openai clients don't assemble the call.
                    let calls = assistant
                        .tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(index, mut call)| {
                            call["index"] = json!(index);
                            call
                        })
                        .collect();
                    delta.insert("tool_calls".into(), Value::Array(calls));
                }
                if !delta.is_empty() {
                    events.push(
                        Event::default().data(
                            json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
                            })
                            .to_string(),
                        ),
                    );
                }
            }
            events.push(
                Event::default().data(
                    json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
                    })
                    .to_string(),
                ),
            );
            events.push(Event::default().data("[DONE]"));
            events
        }
        EngineEvent::Error(message) => vec![
            Event::default().data(json!({"error": {"message": message}}).to_string()),
            Event::default().data("[DONE]"),
        ],
    }
}

fn run_worker(
    config: WorkerConfig,
    command_rx: mpsc::Receiver<Command>,
    ready_tx: mpsc::SyncSender<Result<usize, String>>,
) {
    let initialized = initialize_worker(&config);
    let (mut executor, weights, mut scheduler, mut speculator) = match initialized {
        Ok(worker) => worker,
        Err(error) => {
            let _ = ready_tx.send(Err(error));
            return;
        }
    };
    let _ = ready_tx.send(Ok(executor.cache_bytes()));
    let tokenizer = match Tokenizer::from_file(config.model_path.join("tokenizer.json")) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("worker tokenizer failed: {error}");
            return;
        }
    };
    let mut jobs = HashMap::<u32, ActiveJob>::new();
    let mut pending = HashMap::<u32, u32>::new();
    // For which sequence the speculator holds real hidden
    // states and for which tokens: after a regular step it's one pending
    // token, after verification — all accepted plus the corrected one.
    let mut hidden_for: Option<u32> = None;
    let mut head_tokens: Vec<u32> = Vec::new();

    loop {
        while let Ok(command) = command_rx.try_recv() {
            accept_command(command, &mut scheduler, &mut jobs);
        }

        let batch = match scheduler.next_batch() {
            Ok(Some(batch)) => batch,
            Ok(None) => {
                match command_rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(command) => accept_command(command, &mut scheduler, &mut jobs),
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) if jobs.is_empty() => return,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {}
                }
                continue;
            }
            Err(error) => {
                fail_all(&mut jobs, format!("scheduler failed: {error}"));
                return;
            }
        };

        let layout = match BatchLayout::build(&batch, scheduler.cache()) {
            Ok(layout) => layout,
            Err(error) => {
                let _ = scheduler.abort_batch();
                fail_all(&mut jobs, format!("batch layout failed: {error}"));
                return;
            }
        };
        // One decode sequence and fresh hidden states — the step
        // goes speculatively: a head draft, verification of k+1 rows in one
        // pass over the weights.
        let depth = match (speculator.as_ref(), batch.decode.as_slice()) {
            (Some(_), &[id]) if batch.prefill.is_empty() && hidden_for == Some(id) => {
                let remaining = scheduler.remaining_tokens(id).unwrap_or(0);
                config.speculation.depth.min(remaining.saturating_sub(1))
            }
            _ => 0,
        };
        if let (Some(spec), true) = (speculator.as_mut(), depth > 0) {
            let id = batch.decode[0];
            let token = pending[&id];
            // The head may have been serving another dialog: without binding the draft
            // would run on someone else's KV and shortlist.
            let bound = match jobs.get(&id) {
                Some(job) => spec.bind_prompt(id, &job.prompt).map(|_| ()),
                None => Ok(()),
            };
            if let Err(error) = bound {
                let _ = scheduler.abort_batch();
                fail_all(&mut jobs, format!("MTP bind failed: {error}"));
                return;
            }
            let step = speculative_step(
                &mut executor,
                &weights,
                &mut scheduler,
                spec,
                &layout,
                id,
                &head_tokens,
                depth,
            );
            let (truth, accepted) = match step {
                Ok(step) => step,
                Err(error) => {
                    let _ = scheduler.abort_batch();
                    fail_all(&mut jobs, format!("speculative step failed: {error}"));
                    return;
                }
            };

            // Logits of the verification rows are still with the executor: the commit doesn't
            // touch them.
            let mut verified = match jobs.get(&id).and_then(|job| job.logprobs) {
                Some(top_k) => {
                    let rows: Result<Vec<_>, String> = (0..=accepted)
                        .map(|row| score_row(&executor, row, truth[row], top_k).map(Some))
                        .collect();
                    match rows {
                        Ok(rows) => rows,
                        Err(error) => {
                            fail_all(&mut jobs, format!("logprobs failed: {error}"));
                            return;
                        }
                    }
                }
                None => Vec::new(),
            };

            // The step emitted the pending token and the accepted drafts; at the first
            // stop token or stop string emission stops.
            let mut produced = 0;
            let mut emitted_index = 0;
            let mut stopped = Vec::new();
            for &emitted in std::iter::once(&token).chain(&truth[..accepted]) {
                produced += 1;
                let stop = match jobs.get_mut(&id) {
                    Some(job) => {
                        // The first token was waiting from the previous step, its scores are already
                        // in the task; verification row i gave truth[i].
                        if emitted_index > 0
                            && let Some(scores) = verified.get_mut(emitted_index - 1)
                        {
                            job.next_scores = scores.take();
                        }
                        emitted_index += 1;
                        (!job.ignore_eos && config.eos_ids.contains(&emitted))
                            || emit_token(job, emitted, &tokenizer)
                    }
                    None => true,
                };
                if stop {
                    stopped.push(id);
                    break;
                }
            }
            pending.insert(id, truth[accepted]);
            if let Some(job) = jobs.get_mut(&id) {
                job.next_scores = verified.get_mut(accepted).and_then(Option::take);
            }
            let rows = accepted + 1;
            head_tokens.clear();
            head_tokens.extend_from_slice(&truth[..rows]);
            for &observed in &truth[..rows] {
                spec.observe(observed);
            }
            hidden_for = Some(id);

            let completed = match scheduler.complete_batch_multi(&stopped, &[(id, produced)]) {
                Ok(completed) => completed,
                Err(error) => {
                    fail_all(&mut jobs, format!("batch completion failed: {error}"));
                    return;
                }
            };
            for completion in completed {
                finish_job(&mut jobs, &mut pending, &tokenizer, completion);
                hidden_for = None;
            }
            continue;
        }

        let mut input = Vec::with_capacity(layout.num_tokens());
        for &id in &batch.decode {
            let Some(&token) = pending.get(&id) else {
                let _ = scheduler.abort_batch();
                fail_all(
                    &mut jobs,
                    format!("missing sampled token for sequence {id}"),
                );
                return;
            };
            input.push(token);
        }
        for chunk in &batch.prefill {
            let Some(job) = jobs.get(&chunk.id) else {
                let _ = scheduler.abort_batch();
                fail_all(
                    &mut jobs,
                    format!("missing prompt for sequence {}", chunk.id),
                );
                return;
            };
            input.extend_from_slice(&job.prompt[chunk.offset..chunk.offset + chunk.tokens]);
        }

        if let Err(error) = apply_prefix_ops(&mut executor, &batch) {
            let _ = scheduler.abort_batch();
            fail_all(&mut jobs, format!("prefix cache failed: {error}"));
            return;
        }

        let sampled = match executor.execute_layout(&weights, &layout, &input) {
            Ok(sampled) => sampled,
            Err(error) => {
                let _ = scheduler.abort_batch();
                fail_all(&mut jobs, format!("GPU execution failed: {error}"));
                return;
            }
        };
        let scored = match score_sampled(&executor, &jobs, &sampled) {
            Ok(scored) => scored,
            Err(error) => {
                let _ = scheduler.abort_batch();
                fail_all(&mut jobs, format!("logprobs failed: {error}"));
                return;
            }
        };

        let mut stopped = Vec::new();
        for &id in &batch.decode {
            let token = pending[&id];
            let should_stop = jobs.get_mut(&id).is_none_or(|job| {
                (!job.ignore_eos && config.eos_ids.contains(&token))
                    || emit_token(job, token, &tokenizer)
            });
            if should_stop {
                stopped.push(id);
            }
        }
        if let Some(spec) = speculator.as_mut() {
            let primed = prime_head(
                spec,
                &executor,
                &weights,
                &scheduler,
                &jobs,
                &batch,
                &layout,
                &sampled,
                config.speculation.prime_window,
            );
            if let Err(error) = primed {
                let _ = scheduler.abort_batch();
                fail_all(&mut jobs, format!("MTP priming failed: {error}"));
                return;
            }
        }
        for &(id, token) in &sampled {
            pending.insert(id, token);
        }
        for (id, scores) in scored {
            if let Some(job) = jobs.get_mut(&id) {
                job.next_scores = Some(scores);
            }
        }
        // The head's input is the hidden state of the step just taken.
        hidden_for = None;
        if let (Some(spec), &[id]) = (speculator.as_mut(), batch.decode.as_slice())
            && batch.prefill.is_empty()
        {
            if let Err(error) = executor.copy_decode_hidden_row(0, spec.hidden_mut()) {
                let _ = scheduler.abort_batch();
                fail_all(&mut jobs, format!("MTP hidden copy failed: {error}"));
                return;
            }
            hidden_for = Some(id);
            head_tokens.clear();
            head_tokens.push(pending[&id]);
        }

        let completed = match scheduler.complete_batch(&stopped) {
            Ok(completed) => completed,
            Err(error) => {
                fail_all(&mut jobs, format!("batch completion failed: {error}"));
                return;
            }
        };
        for completion in completed {
            if hidden_for == Some(completion.id) {
                hidden_for = None;
            }
            finish_job(&mut jobs, &mut pending, &tokenizer, completion);
        }

        let abandoned: Vec<u32> = jobs
            .iter()
            .filter_map(|(&id, job)| job.event_tx.is_closed().then_some(id))
            .collect();
        for id in abandoned {
            if scheduler.cancel(id).unwrap_or(false) {
                jobs.remove(&id);
                pending.remove(&id);
            }
        }
    }
}

/// Log-probabilities of row `row` of the last step, computed the same way
/// `eval` does. `chosen` was picked by argmax on the card: if its logit isn't the max,
/// it's the wrong row, and silently returning someone else's numbers is not allowed.
fn score_row(
    executor: &Executor,
    row: usize,
    chosen: u32,
    top_k: usize,
) -> Result<StepScores, String> {
    let logits = executor
        .logits_to_host(row)
        .map_err(|error| error.to_string())?;
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if logits[chosen as usize] != maximum {
        return Err(format!("row {row}: token {chosen} is not the argmax"));
    }
    let sum_exp: f64 = logits
        .iter()
        .map(|&value| f64::from(value - maximum).exp())
        .sum();
    let logsumexp = maximum + sum_exp.ln() as f32;
    let order = |left: &usize, right: &usize| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    };
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    let top_k = top_k.min(logits.len());
    if top_k > 0 {
        indices.select_nth_unstable_by(top_k - 1, order);
    }
    indices.truncate(top_k);
    indices.sort_unstable_by(order);
    Ok(StepScores {
        chosen: logits[chosen as usize] - logsumexp,
        top: indices
            .into_iter()
            .map(|token| (token as u32, logits[token] - logsumexp))
            .collect(),
    })
}

/// Scores of all rows of a regular step whose tasks asked for `logprobs`.
fn score_sampled(
    executor: &Executor,
    jobs: &HashMap<u32, ActiveJob>,
    sampled: &[(u32, u32)],
) -> Result<Vec<(u32, StepScores)>, String> {
    let mut scored = Vec::new();
    for (row, &(id, token)) in sampled.iter().enumerate() {
        if let Some(top_k) = jobs.get(&id).and_then(|job| job.logprobs) {
            scored.push((id, score_row(executor, row, token, top_k)?));
        }
    }
    Ok(scored)
}

fn finish_job(
    jobs: &mut HashMap<u32, ActiveJob>,
    pending: &mut HashMap<u32, u32>,
    tokenizer: &Tokenizer,
    completion: Completion,
) {
    if let Some(mut job) = jobs.remove(&completion.id) {
        finish_text(&mut job, tokenizer);
        let finish_reason = match completion.reason {
            FinishReason::Length => "length",
            FinishReason::Stopped => "stop",
        };
        let _ = job.event_tx.send(EngineEvent::Done {
            text: job.rendered,
            finish_reason,
            completion_tokens: job.output.len(),
            buffered_output: job.buffer_output,
            streamed: job.streamed,
            tokens: job.output,
            scores: job.scores,
        });
    }
    pending.remove(&completion.id);
}

/// A speculative step of one sequence: a head draft, verification by
/// the main model, commit of what was accepted. Returns the verification argmax per
/// row and the number of accepted drafts; `truth[accepted]` is the next
/// pending token.
#[allow(clippy::too_many_arguments)]
fn speculative_step(
    executor: &mut Executor,
    weights: &ModelWeights,
    scheduler: &mut Scheduler,
    spec: &mut Speculator,
    layout: &BatchLayout,
    id: u32,
    head_tokens: &[u32],
    depth: usize,
) -> Result<(Vec<u32>, usize), String> {
    let position = layout.position_starts[0] as usize;
    let state_slot = layout.state_slots[0] as usize;
    let token = *head_tokens.last().ok_or("no head tokens")?;
    let first = position + 1 - head_tokens.len();
    let drafts = spec
        .draft_chain(weights, head_tokens, first, depth)
        .map_err(|error| error.to_string())?;
    // Verification writes KV for all rows at once, pages are needed in advance. If the pool
    // didn't give them, the draft is shorter.
    let reserved = scheduler.reserve_extra(id, drafts.len());
    let drafts = &drafts[..reserved];
    let blocks = scheduler
        .cache()
        .sequence(id)
        .ok_or("sequence is gone")?
        .blocks
        .clone();
    let mut rows_in = Vec::with_capacity(reserved + 1);
    rows_in.push(token);
    rows_in.extend_from_slice(drafts);
    let truth = executor
        .verify_speculation(weights, &rows_in, position, state_slot, &blocks)
        .map_err(|error| error.to_string())?;
    let accepted = drafts
        .iter()
        .zip(&truth)
        .take_while(|(draft, truth)| draft == truth)
        .count();
    let rows = accepted + 1;
    executor
        .commit_speculation(weights, rows_in.len(), rows, state_slot)
        .map_err(|error| error.to_string())?;
    scheduler
        .release_extra(id, reserved - accepted)
        .map_err(|error| error.to_string())?;
    executor
        .copy_prefill_hidden_rows(0, rows, spec.hidden_mut())
        .map_err(|error| error.to_string())?;
    Ok((truth, accepted))
}

/// The head fills its KV from the prompt while it goes through prefill: without this
/// it has zeros on the prompt and acceptance drops from 0.92 to 0.88. Only when
/// the engine has one sequence: otherwise there will be no speculation, and priming
/// costs ~80 µs per token.
#[allow(clippy::too_many_arguments)]
fn prime_head(
    spec: &mut Speculator,
    executor: &Executor,
    weights: &ModelWeights,
    scheduler: &Scheduler,
    jobs: &HashMap<u32, ActiveJob>,
    batch: &Batch,
    layout: &BatchLayout,
    sampled: &[(u32, u32)],
    window: usize,
) -> qwc_cuda::Result<()> {
    let [chunk] = batch.prefill.as_slice() else {
        return Ok(());
    };
    if window == 0 || scheduler.active() != 1 {
        return Ok(());
    }
    let Some(job) = jobs.get(&chunk.id) else {
        return Ok(());
    };
    let prompt = &job.prompt;
    spec.bind_prompt(chunk.id, prompt)?;
    let Some(row) = layout.seq_ids.iter().position(|&id| id == chunk.id) else {
        return Ok(());
    };
    let row_begin = layout.token_offsets[row] as usize;
    let begin = chunk.offset.max(prompt.len().saturating_sub(window));
    let end = chunk.offset + chunk.tokens;
    if begin >= end {
        return Ok(());
    }
    // The pair for row t is (h_t, token t+1); for the last prompt position that's
    // the token the step has just emitted.
    let generated = sampled
        .iter()
        .find(|&&(id, _)| id == chunk.id)
        .map(|&(_, token)| token);
    let mut next = Vec::with_capacity(end - begin);
    for t in begin..end {
        match prompt.get(t + 1).copied().or(generated) {
            Some(token) => next.push(token),
            None => return Ok(()),
        }
    }
    spec.prime(weights, executor, row_begin + begin - chunk.offset, &next, begin + 1)
}

/// Prefix cache before a step: copy incomplete pages of continued
/// prefixes and tell the executor which slots to restore and which to capture.
fn apply_prefix_ops(executor: &mut Executor, batch: &Batch) -> qwc_cuda::Result<()> {
    for op in &batch.restores {
        if let Some((from, to)) = op.copy_block {
            executor.copy_kv_block(from as usize, to as usize)?;
        }
        if let Some(chunk) = batch.prefill.iter().find(|chunk| chunk.id == op.id) {
            eprintln!(
                "prefix cache: sequence {} continues {} cached tokens",
                op.id, chunk.offset
            );
        }
    }
    let restores: Vec<(usize, usize)> = batch
        .restores
        .iter()
        .map(|op| (op.state_slot as usize, op.snapshot as usize))
        .collect();
    let saves: Vec<(usize, usize)> = batch
        .saves
        .iter()
        .map(|op| (op.state_slot as usize, op.snapshot as usize))
        .collect();
    executor.plan_prefix(&restores, &saves);
    Ok(())
}

fn initialize_worker(
    config: &WorkerConfig,
) -> Result<(Executor, ModelWeights, Scheduler, Option<Speculator>), String> {
    qwc_cuda::Device::init(0).map_err(|error| error.to_string())?;
    qwc_cuda::set_memory_limit(config.memory_limit).map_err(str::to_owned)?;
    let checkpoint = Checkpoint::open(&config.model_path).map_err(|error| error.to_string())?;
    let weights = ModelWeights::load(&checkpoint).map_err(|error| error.to_string())?;
    let executor = Executor::new_with_pool_options(
        ExecutorConfig {
            max_batch: config.max_seqs,
            max_context: config.max_context,
        },
        config.kv_cache_dtype,
        DecodeLinearMode::Auto,
        config.delta_state,
        Some(config.kv_pool_blocks),
    )
    .map_err(|error| error.to_string())?;
    let mut executor = executor;
    executor
        .enable_prefix_snapshots(config.prefix_snapshots)
        .map_err(|error| format!("prefix snapshots: {error}"))?;
    let cache = CacheManager::new(config.max_seqs, config.kv_pool_blocks, PAGE_SIZE);
    let mut scheduler = Scheduler::new(
        SchedulerConfig::new(config.max_seqs, config.prefill_chunk),
        cache,
    );
    scheduler.enable_prefix_cache(config.prefix_snapshots);
    let speculator = match config.speculation.depth {
        0 => None,
        _ => {
            let mut speculator =
                Speculator::new(&checkpoint, config.max_context, config.kv_cache_dtype)
                    .map_err(|error| format!("MTP head: {error}"))?;
            if config.speculation.shortlist > 0 {
                speculator
                    .enable_shortlist(config.speculation.shortlist, config.max_context)
                    .map_err(|error| format!("MTP shortlist: {error}"))?;
            }
            Some(speculator)
        }
    };
    Ok((executor, weights, scheduler, speculator))
}

fn accept_command(command: Command, scheduler: &mut Scheduler, jobs: &mut HashMap<u32, ActiveJob>) {
    let Command::Generate(job) = command;
    let request = Request {
        id: job.id,
        prompt_tokens: job.prompt.len(),
        max_new_tokens: job.max_new_tokens,
    };
    if let Err(error) =
        scheduler.submit_prompt(request, Arc::from(job.prompt.as_slice()), job.history_tokens)
    {
        let _ = job.event_tx.send(EngineEvent::Error(error.to_string()));
        return;
    }
    jobs.insert(
        job.id,
        ActiveJob {
            prompt: job.prompt,
            output: Vec::with_capacity(job.max_new_tokens),
            rendered: String::new(),
            stop: job.stop,
            buffer_output: job.buffer_output,
            streamed: 0,
            logprobs: job.logprobs,
            ignore_eos: job.ignore_eos,
            scores: Vec::new(),
            next_scores: None,
            event_tx: job.event_tx,
        },
    );
}

/// Returns true when a stop string was reached or the client disconnected.
fn emit_token(job: &mut ActiveJob, token: u32, tokenizer: &Tokenizer) -> bool {
    job.output.push(token);
    if let Some(scores) = job.next_scores.take() {
        job.scores.push(scores);
    }
    let Ok(decoded) = tokenizer.decode(&job.output, true) else {
        let _ = job
            .event_tx
            .send(EngineEvent::Error("token decoding failed".into()));
        return true;
    };
    let stop_at = job.stop.iter().filter_map(|stop| decoded.find(stop)).min();
    let visible = stop_at.map_or(decoded.as_str(), |index| &decoded[..index]);
    // Do not stream a replacement marker produced by an incomplete UTF-8 byte
    // sequence. A later token will make the cumulative decode prefix valid.
    let stable = visible.trim_end_matches('\u{fffd}');
    if let Some(delta) = stable.strip_prefix(&job.rendered) {
        if !job.buffer_output
            && !delta.is_empty()
            && job.event_tx.send(EngineEvent::Delta(delta.into())).is_err()
        {
            return true;
        }
        job.rendered.push_str(delta);
    }
    // With tools the text before the call is still streamed: reasoning and answer
    // are visible right away, only the call's XML is buffered.
    if job.buffer_output {
        let safe = before_tool_call(&job.rendered);
        if safe > job.streamed {
            let delta = job.rendered[job.streamed..safe].to_owned();
            if job.event_tx.send(EngineEvent::Delta(delta)).is_err() {
                return true;
            }
            job.streamed = safe;
        }
    }
    stop_at.is_some()
}

/// Length of the prefix that definitely contains no start of `<tool_call>`: up to the tag itself
/// or without a tail that might turn out to be its start.
fn before_tool_call(text: &str) -> usize {
    const TAG: &str = "<tool_call>";
    if let Some(index) = text.find(TAG) {
        return index;
    }
    (1..TAG.len())
        .rev()
        .find(|&length| text.ends_with(&TAG[..length]))
        .map_or(text.len(), |length| text.len() - length)
}

fn finish_text(job: &mut ActiveJob, tokenizer: &Tokenizer) {
    let Ok(decoded) = tokenizer.decode(&job.output, true) else {
        return;
    };
    let stop_at = job.stop.iter().filter_map(|stop| decoded.find(stop)).min();
    let visible = stop_at.map_or(decoded.as_str(), |index| &decoded[..index]);
    if let Some(delta) = visible.strip_prefix(&job.rendered) {
        if !job.buffer_output && !delta.is_empty() {
            let _ = job.event_tx.send(EngineEvent::Delta(delta.into()));
        }
    }
    job.rendered = visible.into();
}

fn fail_all(jobs: &mut HashMap<u32, ActiveJob>, message: String) {
    for (_, job) in jobs.drain() {
        let _ = job.event_tx.send(EngineEvent::Error(message.clone()));
    }
}

struct ParsedAssistant {
    content: Option<String>,
    tool_calls: Vec<Value>,
}

/// Converts the checkpoint's documented XML function-call format into the
/// OpenAI `tool_calls` shape consumed by agent harnesses.
fn parse_assistant(text: &str, tools: &[Value]) -> ParsedAssistant {
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut rest = text;
    while let Some(begin) = rest.find("<tool_call>") {
        content.push_str(&rest[..begin]);
        let body = &rest[begin + "<tool_call>".len()..];
        let Some(end) = body.find("</tool_call>") else {
            content.push_str(&rest[begin..]);
            rest = "";
            break;
        };
        let call = &body[..end];
        if let Some((name, arguments)) = parse_xml_function(call, tools) {
            let index = tool_calls.len();
            tool_calls.push(json!({
                "id": format!("call_qwc_{index}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": Value::Object(arguments).to_string()
                }
            }));
        } else {
            content
                .push_str(&rest[begin..begin + "<tool_call>".len() + end + "</tool_call>".len()]);
        }
        rest = &body[end + "</tool_call>".len()..];
    }
    content.push_str(rest);
    let content = content.trim();
    ParsedAssistant {
        content: (!content.is_empty()).then(|| content.to_owned()),
        tool_calls,
    }
}

fn parse_xml_function(
    call: &str,
    tools: &[Value],
) -> Option<(String, serde_json::Map<String, Value>)> {
    let marker = "<function=";
    let begin = call.find(marker)? + marker.len();
    let name_end = call[begin..].find('>')? + begin;
    let name = call[begin..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let function_end = call[name_end + 1..].find("</function>")? + name_end + 1;
    let properties = tools
        .iter()
        .map(|tool| tool.get("function").unwrap_or(tool))
        .find(|function| function.get("name").and_then(Value::as_str) == Some(name))
        .and_then(|function| function.pointer("/parameters/properties"));
    let mut arguments = serde_json::Map::new();
    let mut body = &call[name_end + 1..function_end];
    while let Some(parameter) = body.find("<parameter=") {
        body = &body[parameter + "<parameter=".len()..];
        let key_end = body.find('>')?;
        let key = body[..key_end].trim();
        body = &body[key_end + 1..];
        let value_end = body.find("</parameter>")?;
        let raw = body[..value_end].trim();
        let value = match properties.and_then(|properties| properties.get(key)) {
            Some(schema) => coerce_to_schema(raw, &schema_types(schema)),
            None => serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned())),
        };
        arguments.insert(key.to_owned(), value);
        body = &body[value_end + "</parameter>".len()..];
    }
    Some((name.to_owned(), arguments))
}

/// Parameter types from a JSON Schema: `type`, `enum` value types, the
/// `anyOf`/`oneOf`/`allOf` branches. Like vLLM's `extract_types_from_schema`.
fn schema_types(schema: &Value) -> Vec<String> {
    let mut types = Vec::new();
    let mut add = |name: &str| {
        if !types.iter().any(|known| known == name) {
            types.push(name.to_owned());
        }
    };
    match schema.get("type") {
        Some(Value::String(name)) => add(name),
        Some(Value::Array(names)) => names.iter().filter_map(Value::as_str).for_each(&mut add),
        _ => {}
    }
    for value in schema.get("enum").and_then(Value::as_array).into_iter().flatten() {
        add(match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(number) if number.is_f64() => "number",
            Value::Number(_) => "integer",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        });
    }
    for field in ["anyOf", "oneOf", "allOf"] {
        for choice in schema.get(field).and_then(Value::as_array).into_iter().flatten() {
            for name in schema_types(choice) {
                add(&name);
            }
        }
    }
    if types.is_empty() {
        types.push("string".to_owned());
    }
    types
}

/// A parameter value from the call's XML by the schema type. The model writes `True`,
/// `500 000`-like things and bare strings; JSON parsing without a schema turned
/// `True` into a string and the string "123" into a number. The order of attempts is as in
/// `coerce_to_schema_type` vLLM: null, integer, number, boolean,
/// object/array, string.
fn coerce_to_schema(raw: &str, types: &[String]) -> Value {
    let has = |name: &str| {
        types.iter().any(|known| {
            let known = known.trim().to_lowercase();
            known == name
                || matches!(
                    (known.as_str(), name),
                    ("int" | "int32" | "int64" | "long", "integer")
                        | ("float" | "double", "number")
                        | ("bool", "boolean")
                        | ("str" | "text", "string")
                        | ("dict", "object")
                        | ("list", "array")
                )
        })
    };
    if has("null") && raw.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    if has("integer") {
        if let Ok(number) = raw.parse::<i64>() {
            return Value::from(number);
        }
    }
    if has("number") {
        if let Ok(number) = raw.parse::<f64>() {
            if number.is_finite() {
                return if number.fract() == 0.0 && number.abs() < 9.0e15 {
                    Value::from(number as i64)
                } else {
                    Value::from(number)
                };
            }
        }
    }
    if has("boolean") {
        match raw.trim().to_lowercase().as_str() {
            "true" | "1" => return Value::Bool(true),
            "false" | "0" => return Value::Bool(false),
            _ => {}
        }
    }
    if has("object") || has("array") {
        if let Ok(value) = serde_json::from_str::<Value>(raw) {
            return value;
        }
    }
    if has("string") {
        return Value::String(raw.to_owned());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

fn render_chat(request: &ChatRequest) -> Result<String, ApiError> {
    let mut output = String::new();
    let thinking = request
        .enable_thinking
        .or_else(|| {
            request
                .chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.get("enable_thinking"))
                .and_then(Value::as_bool)
        })
        .unwrap_or(true);
    let reasoning = match request.reasoning_effort.as_deref().unwrap_or("xhigh") {
        "high" | "xhigh" => Some(
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
        ),
        "medium" => None,
        "low" => Some(
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
        ),
        other => {
            return Err(ApiError::bad_request(format!(
                "unsupported reasoning_effort: {other}"
            )));
        }
    };

    let has_tools = request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty());
    let mut first_message = 0usize;
    if has_tools {
        output.push_str("<|im_start|>system\n");
        if thinking {
            if let Some(reasoning) = reasoning {
                output.push_str(reasoning);
                output.push_str("\n\n");
            }
        }
        output.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in request.tools.as_ref().unwrap() {
            output.push('\n');
            write_tojson(&mut output, tool);
        }
        // Verbatim from the checkpoint's chat_template.jinja: the model was trained
        // to call tools on this text, paraphrasing it made the calls worse.
        output.push_str("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
        if request
            .messages
            .first()
            .is_some_and(|message| message.role == "system")
        {
            let system = message_text(&request.messages[0].content)?;
            if !system.trim().is_empty() {
                output.push_str("\n\n");
                output.push_str(system.trim());
            }
            first_message = 1;
        }
        output.push_str("<|im_end|>\n");
    } else if request
        .messages
        .first()
        .is_some_and(|message| message.role == "system")
        && !message_text(&request.messages[0].content)?.trim().is_empty()
    {
        output.push_str("<|im_start|>system\n");
        if thinking {
            if let Some(reasoning) = reasoning {
                output.push_str(reasoning);
                output.push_str("\n\n");
            }
        }
        output.push_str(message_text(&request.messages[0].content)?.trim());
        output.push_str("<|im_end|>\n");
        first_message = 1;
    } else if thinking {
        if let Some(reasoning) = reasoning {
            output.push_str("<|im_start|>system\n");
            output.push_str(reasoning);
            output.push_str("<|im_end|>\n");
        }
    }

    // The template skips an empty system without tools entirely.
    if first_message == 0
        && request
            .messages
            .first()
            .is_some_and(|message| message.role == "system")
    {
        first_message = 1;
    }
    let mut in_tool_response = false;
    for message in &request.messages[first_message..] {
        let content = message_text(&message.content)?;
        match message.role.as_str() {
            "system" => return Err(ApiError::bad_request("system message must be first")),
            "user" => {
                if in_tool_response {
                    output.push_str("<|im_end|>\n");
                    in_tool_response = false;
                }
                output.push_str("<|im_start|>user\n");
                output.push_str(content.trim());
                output.push_str("<|im_end|>\n");
            }
            "assistant" => {
                if in_tool_response {
                    output.push_str("<|im_end|>\n");
                    in_tool_response = false;
                }
                // preserve_thinking isn't set in the template, so a <think> block
                // is present in every assistant turn, even an empty one.
                output.push_str("<|im_start|>assistant\n<think>\n");
                output.push_str(message.reasoning_content.as_deref().unwrap_or("").trim());
                output.push_str("\n</think>\n\n");
                output.push_str(content.trim());
                if let Some(tool_calls) = &message.tool_calls {
                    for (index, call) in tool_calls.iter().enumerate() {
                        output.push_str(match (index, content.trim().is_empty()) {
                            (0, true) => "",
                            (0, false) => "\n\n",
                            _ => "\n",
                        });
                        render_tool_call(&mut output, call)?;
                    }
                }
                output.push_str("<|im_end|>\n");
            }
            "tool" => {
                if !in_tool_response {
                    output.push_str("<|im_start|>user");
                    in_tool_response = true;
                }
                output.push_str("\n<tool_response>\n");
                output.push_str(content.trim());
                output.push_str("\n</tool_response>");
            }
            other => return Err(ApiError::bad_request(format!("unsupported role: {other}"))),
        }
    }
    if in_tool_response {
        output.push_str("<|im_end|>\n");
    }
    output.push_str("<|im_start|>assistant\n");
    if thinking {
        output.push_str("<think>\n");
    } else {
        output.push_str("<think>\n\n</think>\n\n");
    }
    Ok(output)
}

fn message_text(content: &Value) -> Result<String, ApiError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(ApiError::bad_request(
                        "only text message content is supported",
                    ));
                }
                text.push_str(
                    part.get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| ApiError::bad_request("text content part has no text"))?,
                );
            }
            Ok(text)
        }
        _ => Err(ApiError::bad_request("invalid message content")),
    }
}

fn render_tool_call(output: &mut String, call: &Value) -> Result<(), ApiError> {
    let function = call.get("function").unwrap_or(call);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("tool call has no function name"))?;
    output.push_str("<tool_call>\n<function=");
    output.push_str(name);
    output.push_str(">\n");
    let arguments = function.get("arguments").cloned().unwrap_or(Value::Null);
    let arguments = match arguments {
        Value::String(raw) => serde_json::from_str(&raw)
            .map_err(|error| ApiError::bad_request(format!("invalid tool arguments: {error}")))?,
        value => value,
    };
    if let Value::Object(arguments) = arguments {
        for (name, value) in arguments {
            output.push_str("<parameter=");
            output.push_str(&name);
            output.push_str(">\n");
            if let Some(value) = value.as_str() {
                output.push_str(value);
            } else {
                write_tojson(output, &value);
            }
            output.push_str("\n</parameter>\n");
        }
    }
    output.push_str("</function>\n</tool_call>");
    Ok(())
}

/// `tojson` from the template, i.e. Python `json.dumps(ensure_ascii=False)`:
/// separators ", " and ": ", keys in original order (serde_json is built with
/// preserve_order). The compact `to_string` gave the model a different schema text.
fn write_tojson(output: &mut String, value: &Value) {
    match value {
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                write_tojson(output, item);
            }
            output.push(']');
        }
        Value::Object(map) => {
            output.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                output.push_str(&Value::String(key.clone()).to_string());
                output.push_str(": ");
                write_tojson(output, item);
            }
            output.push('}');
        }
        scalar => output.push_str(&scalar.to_string()),
    }
}

fn load_eos_ids(model_path: &Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let data = std::fs::read(model_path.join("generation_config.json"))?;
    let value: Value = serde_json::from_slice(&data)?;
    let eos = value
        .get("eos_token_id")
        .ok_or("generation_config.json has no eos_token_id")?;
    let ids = match eos {
        Value::Number(number) => vec![number.as_u64().ok_or("invalid eos_token_id")? as u32],
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .ok_or("invalid eos_token_id")
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("invalid eos_token_id".into()),
    };
    Ok(ids)
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut model_path = home().join("models/Qwen3.8-27B-QUASAR-NVFP4");
    let mut model_name = None;
    let mut bind = "127.0.0.1:8000".parse()?;
    let mut max_context = 32 * 1024usize;
    let mut max_seqs = 32usize;
    // Token budget per step. More is faster prefill, but decode rows wait for
    // the whole chunk: on 4096-token prompts 512 -> 2048 gave +22% prefill and
    // ITL p50 66 -> 205 ms. The cap is the arena capacity.
    let mut prefill_chunk = PREFILL_CHUNK_SIZE;
    let mut memory_limit = 28_000_000_000usize;
    let mut kv_cache_bytes = 5_000_000_000usize;
    let mut kv_cache_dtype = KvCacheDtype::Fp8;
    let mut delta_state = DeltaStateMode::Wy;
    // State snapshots for the prefix cache, ~81 MB each. Eight means
    // eight dialogs whose next turn doesn't recompute the history.
    let mut prefix_snapshots = 8usize;
    // The best batch-1 measured point: depth 3 and a 32k shortlist (156.7 tok/s
    // vs 78 without speculation). The head costs 0.85 GB of VRAM.
    let mut speculation = SpeculationConfig {
        depth: 3,
        shortlist: 32768,
        prime_window: 2048,
    };
    // Reasons by default, as before; --thinking off answers right away.
    let mut thinking = true;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--thinking" => {
                thinking = match args.next().ok_or("--thinking needs on|off")?.as_str() {
                    "on" | "true" => true,
                    "off" | "false" => false,
                    other => return Err(format!("--thinking: {other}, expected on|off").into()),
                }
            }
            "--model" => model_path = PathBuf::from(args.next().ok_or("--model needs a path")?),
            "--served-model-name" => model_name = Some(args.next().ok_or("--served-model-name")?),
            "--bind" => bind = args.next().ok_or("--bind")?.parse()?,
            "--context" => max_context = args.next().ok_or("--context")?.parse()?,
            "--max-seqs" => max_seqs = args.next().ok_or("--max-seqs")?.parse()?,
            "--prefill-chunk" => {
                prefill_chunk = args.next().ok_or("--prefill-chunk")?.parse()?
            }
            "--memory-limit-gb" => {
                let gb: f64 = args.next().ok_or("--memory-limit-gb")?.parse()?;
                memory_limit = (gb * 1e9) as usize;
            }
            "--speculative" => {
                speculation.depth = args.next().ok_or("--speculative needs K (0 = off)")?.parse()?
            }
            "--shortlist" => {
                speculation.shortlist = args.next().ok_or("--shortlist needs N (0 = full)")?.parse()?
            }
            "--mtp-prime" => {
                speculation.prime_window = args.next().ok_or("--mtp-prime needs N")?.parse()?
            }
            "--prefix-cache" => {
                prefix_snapshots = args.next().ok_or("--prefix-cache needs N (0 = off)")?.parse()?
            }
            "--kv-cache-gb" => {
                let gb: f64 = args.next().ok_or("--kv-cache-gb")?.parse()?;
                kv_cache_bytes = (gb * 1e9) as usize;
            }
            "--delta-state" => {
                delta_state = match args.next().ok_or("--delta-state")?.as_str() {
                    "bf16" => DeltaStateMode::Bf16,
                    "fp32" => DeltaStateMode::Fp32,
                    "wy" => DeltaStateMode::Wy,
                    other => return Err(format!("unsupported delta state: {other}").into()),
                }
            }
            "--kv-cache" => {
                kv_cache_dtype = match args.next().ok_or("--kv-cache")?.as_str() {
                    "fp8" => KvCacheDtype::Fp8,
                    "bf16" => KvCacheDtype::Bf16,
                    other => return Err(format!("unsupported KV dtype: {other}").into()),
                }
            }
            "--help" | "-h" => {
                println!(
                    "qwc serve [--model PATH] [--bind 127.0.0.1:8000] \
[--context 32768] [--max-seqs 32] [--kv-cache fp8|bf16] [--kv-cache-gb 5] \
[--memory-limit-gb 28] [--prefill-chunk N] [--delta-state wy|bf16|fp32] \
[--served-model-name NAME] [--prefix-cache 8] [--speculative 3] \
[--shortlist 32768] [--mtp-prime 2048] [--thinking on|off]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    if max_context == 0 {
        return Err("--context must be positive".into());
    }
    if !(1..=qwc_engine::executor::MAX_BATCH).contains(&max_seqs) {
        return Err(format!("--max-seqs must be 1..={}", qwc_engine::executor::MAX_BATCH).into());
    }
    if !(1..=PREFILL_CHUNK_SIZE).contains(&prefill_chunk) {
        return Err(format!("--prefill-chunk must be 1..={PREFILL_CHUNK_SIZE}").into());
    }
    if speculation.depth + 1 > MAX_SPECULATION_ROWS {
        return Err(format!("--speculative must be 0..={}", MAX_SPECULATION_ROWS - 1).into());
    }
    if speculation.shortlist > VOCAB_SIZE {
        return Err(format!("--shortlist must be 0..={VOCAB_SIZE}").into());
    }
    if memory_limit == 0 || kv_cache_bytes == 0 {
        return Err("memory limits must be positive".into());
    }
    let model_name = model_name.unwrap_or_else(|| {
        model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("qwencore")
            .to_owned()
    });
    Ok(Args {
        model_path,
        model_name,
        bind,
        max_context,
        max_seqs,
        prefill_chunk,
        memory_limit,
        kv_cache_bytes,
        kv_cache_dtype,
        delta_state,
        prefix_snapshots,
        speculation,
        thinking,
    })
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_xml_tool_call_becomes_openai_tool_call() {
        let parsed = parse_assistant(
            "checking\n<tool_call>\n<function=read_file>\n<parameter=path>\n\"/tmp/a\"\n</parameter>\n<parameter=lines>\n[1,2]\n</parameter>\n</function>\n</tool_call>",
            &[],
        );
        assert_eq!(parsed.content.as_deref(), Some("checking"));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0]["function"]["name"], "read_file");
        let arguments: Value = serde_json::from_str(
            parsed.tool_calls[0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(arguments, json!({"path": "/tmp/a", "lines": [1, 2]}));
    }

    #[test]
    fn malformed_tool_call_stays_visible_content() {
        let source = "before <tool_call>broken</tool_call> after";
        let parsed = parse_assistant(source, &[]);
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content.as_deref(), Some(source));
    }

    #[test]
    fn tool_arguments_follow_schema_types() {
        // The model writes booleans Python-style; without a schema `True` went out as a string
        // and the remote filter silently didn't work.
        let tools = [json!({"type": "function", "function": {"name": "search", "parameters": {
            "type": "object", "properties": {
                "remote": {"type": "boolean"}, "salary_from": {"type": "integer"},
                "query": {"type": "string"}, "ratio": {"type": ["number", "null"]},
                "tags": {"type": "array"}, "level": {"enum": ["junior", "senior"]}}}}})];
        let parsed = parse_assistant(
            "<tool_call>\n<function=search>\n<parameter=remote>\nTrue\n</parameter>\n<parameter=salary_from>\n500000\n</parameter>\n<parameter=query>\n123\n</parameter>\n<parameter=ratio>\nnull\n</parameter>\n<parameter=tags>\n[\"a\"]\n</parameter>\n<parameter=level>\njunior\n</parameter>\n<parameter=extra>\n7\n</parameter>\n</function>\n</tool_call>",
            &tools,
        );
        let arguments: Value =
            serde_json::from_str(parsed.tool_calls[0]["function"]["arguments"].as_str().unwrap())
                .unwrap();
        assert_eq!(
            arguments,
            json!({"remote": true, "salary_from": 500000, "query": "123", "ratio": null,
                   "tags": ["a"], "level": "junior", "extra": 7})
        );
    }

    #[test]
    fn render_chat_matches_checkpoint_template() {
        // References are transformers.apply_chat_template over the checkpoint's
        // chat_template.jinja, see tests/chat_template/generate.py.
        let cases: Vec<Value> =
            serde_json::from_str(include_str!("../../tests/chat_template/cases.json")).unwrap();
        for case in cases {
            let request: ChatRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let Ok(rendered) = render_chat(&request) else {
                panic!("case {} rejected", case["name"]);
            };
            assert_eq!(rendered, case["expected"].as_str().unwrap(), "case {}", case["name"]);
        }
    }

    #[test]
    fn streaming_with_tools_holds_back_only_the_call() {
        assert_eq!(before_tool_call("ответ без вызова"), "ответ без вызова".len());
        assert_eq!(before_tool_call("ищу\n<tool_c"), "ищу\n".len());
        assert_eq!(before_tool_call("ищу\n<"), "ищу\n".len());
        assert_eq!(before_tool_call("ищу <tool_call>\n<function=x>"), "ищу ".len());
        assert_eq!(before_tool_call("a < b"), "a < b".len());
    }
}
