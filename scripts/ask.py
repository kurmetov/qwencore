#!/usr/bin/env python3
"""Console client for qwc serve using only the standard library.

    scripts/ask.py "What is 2+2?"               # one question, streamed answer
    echo "text" | scripts/ask.py "Summarize"    # stdin is appended to the question
    scripts/ask.py                               # dialogue; /reset, /exit

From another machine scripts/qwencore is handier: it also starts the server.

Flags: --think (reasoning, xhigh; printed in grey and kept out of history),
--max-tokens N (default: whatever is left of the context), --system TEXT,
--url (or QWC_URL, default http://127.0.0.1:8100).
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

DIM, RESET = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")
# The server answers 400 if max_tokens doesn't fit the context; from the error text
# we take the prompt length and the context size.
OVERFLOW = re.compile(r"prompt has (\d+) tokens; requested \d+ output tokens exceed context (\d+)")


def model_name(url):
    with urllib.request.urlopen(url + "/v1/models") as response:
        return json.load(response)["data"][0]["id"]


def post(url, body):
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return urllib.request.urlopen(request)


def open_stream(url, body):
    try:
        return post(url, body)
    except urllib.error.HTTPError as error:
        text = error.read().decode(errors="replace")
        match = OVERFLOW.search(text)
        if not match:
            raise RuntimeError(f"server error {error.code}: {text}") from None
        remaining = int(match[2]) - int(match[1])
        if remaining <= 0 or remaining >= body["max_tokens"]:
            raise RuntimeError("the dialogue doesn't fit the context, start over: /reset") from None
        return post(url, {**body, "max_tokens": remaining})


def stream(url, messages, args):
    body = {
        "model": args.model,
        "messages": messages,
        # Without a limit, ask for more than the context: the server replies with what fits.
        "max_tokens": args.max_tokens or 1 << 20,
        "stream": True,
        "enable_thinking": args.think,
        "reasoning_effort": "xhigh",
    }
    text = ""
    # Hold trailing newlines of thoughts until more text arrives.
    held = ""
    # The answer after </think> starts with newlines, sometimes as a separate chunk.
    opening = args.think
    # With reasoning the prompt ends with <think>: everything up to </think> is thoughts.
    thinking = args.think
    if thinking:
        sys.stdout.write(DIM)
    with open_stream(url, body) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content") or ""
                text += piece
                if thinking and "</think>" in text:
                    # The tag may arrive split across chunks: print
                    # the answer from the tag's end, not from its start in this chunk.
                    thinking = False
                    sys.stdout.write(RESET + "\n")
                    piece = text.split("</think>", 1)[1].lstrip("\n")
                    opening = not piece
                elif thinking:
                    if "</" in piece:
                        continue
                    pending = held + piece
                    piece = pending.rstrip("\n")
                    held = pending[len(piece):]
                elif opening:
                    piece = piece.lstrip("\n")
                    opening = not piece
                sys.stdout.write(piece)
                sys.stdout.flush()
    sys.stdout.write(RESET + "\n")
    return text.split("</think>", 1)[-1].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--url", default=os.environ.get("QWC_URL", "http://127.0.0.1:8100"))
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--system")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    try:
        args.model = model_name(url)
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as error:
        sys.exit(f"no qwc serve at {url} ({error}). Start it: scripts/llm-server.sh start")
    base = [{"role": "system", "content": args.system}] if args.system else []

    prompt = " ".join(args.prompt)
    if not sys.stdin.isatty():
        piped = sys.stdin.read()
        prompt = f"{prompt}\n\n{piped}" if prompt else piped

    try:
        if prompt:
            stream(url, base + [{"role": "user", "content": prompt}], args)
            return
        history = list(base)
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                print()
                return
            if line in ("/exit", "/quit"):
                return
            if line == "/reset":
                history = list(base)
                continue
            if not line:
                continue
            history.append({"role": "user", "content": line})
            try:
                history.append({"role": "assistant", "content": stream(url, history, args)})
            except RuntimeError as error:
                history.pop()
                print(f"{RESET}{error}")
    except RuntimeError as error:
        sys.exit(f"{RESET}{error}")
    except urllib.error.URLError as error:
        sys.exit(f"{RESET}no connection to {url}: {error}")
    except KeyboardInterrupt:
        print(RESET)


if __name__ == "__main__":
    main()
