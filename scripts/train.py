#!/usr/bin/env python3
import argparse
import inspect
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron with LoRA or SFT")
    parser.add_argument(
        "--model_name",
        default="nvidia/Nemotron-Mini-4B-Instruct",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--data_dir",
        default="data/processed",
        help="Directory containing train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Output directory for adapters and logs",
    )
    parser.add_argument(
        "--backend",
        choices=["trl", "nvidia", "nemo"],
        default="trl",
        help="Training backend: TRL/PEFT or an NVIDIA/NeMo command",
    )
    parser.add_argument(
        "--tuning_method",
        choices=["lora", "sft"],
        default="lora",
        help="Fine-tuning method: LoRA adapters or full SFT",
    )
    parser.add_argument(
        "--nvidia_library",
        choices=["nemo", "nemo-run", "custom"],
        default="nemo",
        help="NVIDIA library label when using the NVIDIA/NeMo backend",
    )
    parser.add_argument(
        "--nvidia_command",
        default="",
        help=(
            "Shell command to run when backend=nvidia/nemo. The command receives "
            "DATA_DIR, OUTPUT_DIR, MODEL_NAME, and TUNING_METHOD env vars."
        ),
    )
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules",
    )
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--bf16", action="store_true", help="Force bfloat16 compute")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16 compute")
    return parser.parse_args()


def resolve_bf16(force_bf16: bool, disable_bf16: bool) -> bool:
    import torch

    if disable_bf16:
        return False
    if force_bf16:
        return True
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def build_quant_config(use_4bit: bool, bf16: bool):
    if not use_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def run_external_backend(args: argparse.Namespace, data_dir: Path, output_dir: Path) -> None:
    script_dir = Path(__file__).resolve().parent
    default_nemo_cmd = f"bash {script_dir / 'nemo_launcher_finetune.sh'}"
    command = args.nvidia_command or os.environ.get("NVIDIA_TRAIN_CMD", "")
    if args.backend == "nemo":
        command = command or os.environ.get("NEMO_TRAIN_CMD", "") or default_nemo_cmd
    if not command:
        raise SystemExit(
            f"{args.backend} backend selected but no command provided. "
            "Use --nvidia_command or set NVIDIA_TRAIN_CMD/NEMO_TRAIN_CMD."
        )

    env = os.environ.copy()
    env.update(
        {
            "DATA_DIR": str(data_dir.resolve()),
            "OUTPUT_DIR": str(output_dir.resolve()),
            "MODEL_NAME": args.model_name,
            "TUNING_METHOD": args.tuning_method,
            "NVIDIA_LIBRARY": args.nvidia_library,
            "MAX_SEQ_LENGTH": str(args.max_seq_length),
            "TRAIN_BATCH_SIZE": str(args.per_device_train_batch_size),
            "EVAL_BATCH_SIZE": str(args.per_device_eval_batch_size),
            "GRADIENT_ACCUMULATION_STEPS": str(args.gradient_accumulation_steps),
            "NUM_TRAIN_EPOCHS": str(args.num_train_epochs),
            "LEARNING_RATE": str(args.learning_rate),
        }
    )

    subprocess.run(command, shell=True, check=True, env=env)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend in {"nvidia", "nemo"}:
        run_external_backend(args, data_dir, output_dir)
        return

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        from trl import SFTTrainer
        SFTConfig = None

    bf16 = resolve_bf16(args.bf16, args.no_bf16)
    use_4bit = args.tuning_method == "lora" and not args.no_4bit

    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    if not train_file.exists() or not val_file.exists():
        raise SystemExit("Missing train.jsonl or val.jsonl. Run prepare_jsonl.py first.")

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )

    quant_config = build_quant_config(use_4bit, bf16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = None
    if args.tuning_method == "lora":
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules,
        )

    base_args_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "report_to": "none",
        "bf16": bf16,
        "fp16": not bf16,
        "gradient_checkpointing": True,
    }
    training_args_kwargs = dict(base_args_kwargs)
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_args_kwargs["eval_strategy"] = "steps"
    else:
        training_args_kwargs["evaluation_strategy"] = "steps"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": lora_config,
    }
    if "dataset_text_field" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs.update(
            {
                "args": training_args,
                "dataset_text_field": "text",
                "tokenizer": tokenizer,
                "max_seq_length": args.max_seq_length,
                "packing": True,
            }
        )
    else:
        if SFTConfig is None:
            raise SystemExit("Installed trl version requires SFTConfig but it is missing.")
        sft_kwargs = dict(base_args_kwargs)
        if "eval_strategy" in inspect.signature(SFTConfig.__init__).parameters:
            sft_kwargs["eval_strategy"] = "steps"
        else:
            sft_kwargs["evaluation_strategy"] = "steps"
        sft_kwargs.update(
            {
                "max_length": args.max_seq_length,
                "packing": True,
                "dataset_text_field": "text",
            }
        )
        trainer_kwargs.update(
            {
                "args": SFTConfig(**sft_kwargs),
                "processing_class": tokenizer,
            }
        )

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    if args.tuning_method == "lora":
        adapter_dir = output_dir / "adapter"
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"Saved LoRA adapter to {adapter_dir}")
    else:
        sft_dir = output_dir / "sft_model"
        trainer.model.save_pretrained(sft_dir)
        tokenizer.save_pretrained(sft_dir)
        print(f"Saved SFT model to {sft_dir}")


if __name__ == "__main__":
    main()
