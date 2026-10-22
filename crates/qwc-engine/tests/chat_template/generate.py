"""Chat rendering references from the checkpoint's stock chat_template.jinja.

    ~/.venvs/baseline/bin/python crates/qwc-engine/tests/chat_template/generate.py

Writes cases.json: an OpenAI-format request and the text the template produces. The
`render_chat_matches_checkpoint_template` test in serve.rs checks render_chat against it.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(Path.home() / "models/Qwen3.8-27B-QUASAR-NVFP4")
TOOLS = [
    {"type": "function", "function": {
        "name": "search_vacancies", "description": "Поиск вакансий «по снимку».",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "слова"},
            "salary_from": {"type": "integer"},
            "remote": {"type": "boolean"},
            "category": {"type": "string", "enum": ["backend", "qa"]}},
            "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_vacancy", "description": "Карточка по id.",
                                      "parameters": {"type": "object", "properties": {"vacancy_id": {"type": "string"}},
                                                     "required": ["vacancy_id"]}}},
]
CALL = {"id": "c1", "type": "function",
        "function": {"name": "search_vacancies", "arguments": {"query": "python", "salary_from": 500000, "remote": True}}}
CALL2 = {"id": "c2", "type": "function", "function": {"name": "get_vacancy", "arguments": {"vacancy_id": "hh-1"}}}
CASES = {
    "plain": dict(messages=[{"role": "user", "content": "Привет"}]),
    "system_thinking": dict(messages=[{"role": "system", "content": " Будь краток. "}, {"role": "user", "content": "2+2?"}],
                            enable_thinking=True),
    "empty_system": dict(messages=[{"role": "system", "content": "  "}, {"role": "user", "content": "x"}]),
    "history": dict(messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                              {"role": "user", "content": "c"}]),
    "tools_round": dict(messages=[
        {"role": "system", "content": "Ты помощник."}, {"role": "user", "content": "Python на удалёнке?"},
        {"role": "assistant", "content": "", "tool_calls": [CALL, CALL2]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"total\": 3}"},
        {"role": "tool", "tool_call_id": "c2", "content": "{\"id\": \"hh-1\"}"}], tools=TOOLS),
    "tools_text_before_call": dict(messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Сейчас поищу.", "reasoning_content": "нужен поиск", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "[]"},
        {"role": "assistant", "content": "Ничего нет."}, {"role": "user", "content": "ещё"}],
        tools=TOOLS, enable_thinking=True),
}

cases = []
for name, case in CASES.items():
    thinking = case.get("enable_thinking", False)
    text = tokenizer.apply_chat_template(case["messages"], tools=case.get("tools"), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=thinking)
    request = {"messages": case["messages"], "enable_thinking": thinking}
    if "tools" in case:
        request["tools"] = case["tools"]
    cases.append({"name": name, "request": request, "expected": text})
(Path(__file__).parent / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))
print(f"{len(cases)} references")
