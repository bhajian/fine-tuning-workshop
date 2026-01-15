#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Nemotron with vLLM (OpenAI API)")
    parser.add_argument(
        "--model_name",
        default="nvidia/Nemotron-Mini-4B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter_dir",
        default="",
        help="Optional LoRA adapter directory",
    )
    parser.add_argument(
        "--adapter_name",
        default="phishing",
        help="LoRA adapter name exposed to the OpenAI API",
    )
    parser.add_argument(
        "--served_model_name",
        default="",
        help="Override the model name exposed by the OpenAI server",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.served_model_name:
        cmd += ["--served-model-name", args.served_model_name]
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        if not adapter_dir.exists():
            raise SystemExit(f"Adapter directory not found: {adapter_dir}")
        cmd += ["--enable-lora", "--lora-modules", f"{args.adapter_name}={adapter_dir}"]
    return cmd


def main() -> None:
    args = parse_args()
    command = build_command(args)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
