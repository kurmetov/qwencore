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
use qwc_engine::{DecodeLinearMode, Executor, ExecutorConfig, ModelWeights, PREFILL_CHUNK_SIZE};
use qwc_model::Checkpoint;
use qwc_runtime::{Batch, BatchLayout, CacheManager, FinishReason, Request, Scheduler, SchedulerConfig};
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
    eos_ids: Vec<u32>,
}

enum Command {
    Generate(GenerateJob),
}

struct GenerateJob {
    id: u32,
    prompt: Vec<u32>,
    /// Длина истории диалога: всё до промпта генерации. На этой границе
    /// снимается кэш префиксов — следующий ход повторит историю дословно.
    history_tokens: usize,
    max_new_tokens: usize,
    stop: Vec<String>,
    buffer_output: bool,
    event_tx: async_mpsc::UnboundedSender<EngineEvent>,
}

enum EngineEvent {
    Delta(String),
    Done {
        text: String,
        finish_reason: &'static str,
        completion_tokens: usize,
        buffered_output: bool,
    },
    Error(String),
}

struct ActiveJob {
    prompt: Vec<u32>,
    output: Vec<u32>,
    rendered: String,
    stop: Vec<String>,
    buffer_output: bool,
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
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

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
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
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
            "owned_by": "local"
        }]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    if let Some(model) = request.model.as_deref() {
        if model != state.model.as_ref() {
            return Err(ApiError::bad_request(format!("unknown model: {model}")));
        }
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
    // Промпт генерации начинается последним `<|im_start|>`: ход модели в
    // историю следующего запроса попадёт уже без `<think>` и с ответом.
    let history_tokens = state
        .tokenizer
        .token_to_id("<|im_start|>")
        .and_then(|start| prompt.iter().rposition(|&token| token == start))
        .unwrap_or(prompt_tokens);
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
        let event_stream = UnboundedReceiverStream::new(event_rx).flat_map(move |event| {
            let events = stream_events(&stream_id, created, model.as_ref(), event);
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
                let assistant = parse_assistant(&text);
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

fn stream_events(id: &str, created: u64, model: &str, event: EngineEvent) -> Vec<Event> {
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
            ..
        } => {
            let assistant = parse_assistant(&text);
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
                    delta.insert("tool_calls".into(), Value::Array(assistant.tool_calls));
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
    let (mut executor, weights, mut scheduler) = match initialized {
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

        let mut stopped = Vec::new();
        for &id in &batch.decode {
            let token = pending[&id];
            let should_stop = config.eos_ids.contains(&token)
                || jobs
                    .get_mut(&id)
                    .is_none_or(|job| emit_token(job, token, &tokenizer));
            if should_stop {
                stopped.push(id);
            }
        }
        for (id, token) in sampled {
            pending.insert(id, token);
        }

        let completed = match scheduler.complete_batch(&stopped) {
            Ok(completed) => completed,
            Err(error) => {
                fail_all(&mut jobs, format!("batch completion failed: {error}"));
                return;
            }
        };
        for completion in completed {
            if let Some(mut job) = jobs.remove(&completion.id) {
                finish_text(&mut job, &tokenizer);
                let finish_reason = match completion.reason {
                    FinishReason::Length => "length",
                    FinishReason::Stopped => "stop",
                };
                let _ = job.event_tx.send(EngineEvent::Done {
                    text: job.rendered,
                    finish_reason,
                    completion_tokens: job.output.len(),
                    buffered_output: job.buffer_output,
                });
            }
            pending.remove(&completion.id);
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

/// Кэш префиксов перед шагом: скопировать неполные страницы продолжаемых
/// префиксов и сказать исполнителю, какие слоты поднять и какие снять.
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

fn initialize_worker(config: &WorkerConfig) -> Result<(Executor, ModelWeights, Scheduler), String> {
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
    Ok((executor, weights, scheduler))
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
            event_tx: job.event_tx,
        },
    );
}

/// Returns true when a stop string was reached or the client disconnected.
fn emit_token(job: &mut ActiveJob, token: u32, tokenizer: &Tokenizer) -> bool {
    job.output.push(token);
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
    stop_at.is_some()
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
fn parse_assistant(text: &str) -> ParsedAssistant {
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
        if let Some((name, arguments)) = parse_xml_function(call) {
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

fn parse_xml_function(call: &str) -> Option<(String, serde_json::Map<String, Value>)> {
    let marker = "<function=";
    let begin = call.find(marker)? + marker.len();
    let name_end = call[begin..].find('>')? + begin;
    let name = call[begin..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let function_end = call[name_end + 1..].find("</function>")? + name_end + 1;
    let mut arguments = serde_json::Map::new();
    let mut body = &call[name_end + 1..function_end];
    while let Some(parameter) = body.find("<parameter=") {
        body = &body[parameter + "<parameter=".len()..];
        let key_end = body.find('>')?;
        let key = body[..key_end].trim();
        body = &body[key_end + 1..];
        let value_end = body.find("</parameter>")?;
        let raw = body[..value_end].trim();
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()));
        arguments.insert(key.to_owned(), value);
        body = &body[value_end + "</parameter>".len()..];
    }
    Some((name.to_owned(), arguments))
}

fn render_chat(request: &ChatRequest) -> Result<String, ApiError> {
    let mut output = String::new();
    let thinking = request.enable_thinking.unwrap_or(true);
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
            output.push_str(&tool.to_string());
        }
        output.push_str("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nFunction calls MUST use the XML format above. Required parameters MUST be specified. You may reason before a function call, but put nothing after it.\n</IMPORTANT>");
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
                output.push_str("<|im_start|>assistant\n");
                output.push_str(content.trim());
                if let Some(tool_calls) = &message.tool_calls {
                    for call in tool_calls {
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
    output.push_str("\n<tool_call>\n<function=");
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
                output.push_str(&value.to_string());
            }
            output.push_str("\n</parameter>\n");
        }
    }
    output.push_str("</function>\n</tool_call>");
    Ok(())
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
    // Бюджет токенов на шаг. Больше — быстрее префилл, но decode-строки ждут
    // весь чанк: на 4096-токенных промптах 512 -> 2048 дало +22% префилла и
    // ITL p50 66 -> 205 мс. Потолок — ёмкость арены.
    let mut prefill_chunk = PREFILL_CHUNK_SIZE;
    let mut memory_limit = 28_000_000_000usize;
    let mut kv_cache_bytes = 5_000_000_000usize;
    let mut kv_cache_dtype = KvCacheDtype::Fp8;
    let mut delta_state = DeltaStateMode::Wy;
    // Снимков состояния под кэш префиксов, ~81 МБ каждый. Восемь — это
    // восемь диалогов, чей следующий ход не пересчитывает историю.
    let mut prefix_snapshots = 8usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
[--served-model-name NAME] [--prefix-cache 8]"
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
        let parsed = parse_assistant(source);
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content.as_deref(), Some(source));
    }
}
