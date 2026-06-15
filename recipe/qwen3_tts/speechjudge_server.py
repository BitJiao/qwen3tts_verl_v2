import argparse
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="HTTP server for SpeechJudge-GRM naturalness scoring.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model_path", default="/opt/data/private/jsj/Qwen3-TTS-main/pretrained/SpeechJudge-GRM")
    parser.add_argument("--speechjudge_repo", default="/opt/data/private/jsj/Qwen3-TTS-main/third_party/SpeechJudge")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args()


def ensure_import_path(repo: str) -> None:
    repo_path = Path(repo)
    for path in (repo_path, repo_path / "infer"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def extract_single_score(text: str) -> float:
    patterns = [
        r"(?:score|rating)\D{0,20}(\d+(?:\.\d+)?)",
        r"\b(\d+(?:\.\d+)?)\s*/\s*10\b",
        r"\b(10(?:\.0+)?|[1-9](?:\.\d+)?)\b",
        r"^\s*(\d+(?:\.\d+)?)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            score = float(matches[-1])
            return float(np.clip((score - 1.0) / 9.0, 0.0, 1.0))
    raise ValueError(f"Could not parse SpeechJudge score from: {text!r}")


class SpeechJudgeScorer:
    def __init__(self, args):
        ensure_import_path(args.speechjudge_repo)
        from utils import build_qwen_omni_inputs, build_rm_conversation

        self.build_qwen_omni_inputs = build_qwen_omni_inputs
        self.build_rm_conversation = build_rm_conversation
        self.max_new_tokens = args.max_new_tokens
        self.processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )

    @torch.no_grad()
    def score_one(self, target_text: str, wav_path: str) -> float:
        conversation = self.build_rm_conversation(wav_path, target_text)
        inputs = self.build_qwen_omni_inputs(self.processor, conversation)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        prompt_length = inputs["input_ids"].shape[1]
        text_ids = self.model.generate(
            **inputs,
            use_audio_in_video=False,
            do_sample=True,
            return_audio=False,
            max_new_tokens=self.max_new_tokens,
        )
        text = self.processor.batch_decode(
            text_ids[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return extract_single_score(text)


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    scorer = SpeechJudgeScorer(args)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            if self.path != "/score":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                target_text = request["target_text"]
                wav_paths = request["wav_paths"]
                scores = [scorer.score_one(target_text, wav_path) for wav_path in wav_paths]
                body = json.dumps({"scores": scores}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                print(f"[speechjudge_server] scoring error: {exc}", file=sys.stderr, flush=True)
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, fmt, *args):
            print(f"[speechjudge_server] {self.address_string()} - {fmt % args}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[speechjudge_server] listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
