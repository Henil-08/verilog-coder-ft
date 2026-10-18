"""Fine-tune a causal LM on OpenRTLSet SFT splits with LoRA, QLoRA, or full fine-tuning.

Runs are defined in configs/experiments.yaml. GPU count comes from the launcher:

    uv run python -m training.train --experiment smoke --no-wandb          # local smoke test
    torchrun --nproc_per_node 1 -m training.train --experiment L2          # one GPU
    torchrun --nproc_per_node 2 -m training.train --experiment F2          # two GPUs, ZeRO shards across both
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from training.config import estimate_cost, load_run
from training.dataset import IGNORE_INDEX, load_sft_dataset


def build_model(cfg: dict):
    use_cuda = torch.cuda.is_available()
    kwargs = {"dtype": torch.bfloat16 if use_cuda else torch.float32, "attn_implementation": "sdpa"}

    if cfg["method"] == "qlora":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}

    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kwargs)
    model.config.use_cache = False

    if cfg["method"] == "full":
        return model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg["method"] == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if cfg["gradient_checkpointing"]:
        model.enable_input_require_grads()
    return model


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="run name from configs/experiments.yaml")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early (pilot runs)")
    parser.add_argument("--no-deepspeed", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_run(args.experiment)
    if args.no_deepspeed or not torch.cuda.is_available():
        cfg["deepspeed"] = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    use_wandb = not args.no_wandb
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])

    output_dir = Path(args.output_root) / cfg["run_name"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.padding_side = "right"

    train_ds = load_sft_dataset(cfg["train_file"], tokenizer, cfg["max_seq_len"])
    val_ds = load_sft_dataset(cfg["val_file"], tokenizer, cfg["max_seq_len"])

    model = build_model(cfg)
    trainable, total = count_params(model)
    if rank == 0:
        print(f"[{cfg['run_name']}] {cfg['method']}: {trainable:,} / {total:,} params trainable")

    micro = cfg["micro_batch_size"]
    grad_accum = max(1, cfg["effective_batch_size"] // (micro * world_size))

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        run_name=cfg["run_name"],
        num_train_epochs=cfg["epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler"],
        warmup_steps=cfg["warmup"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=cfg["gradient_checkpointing"] and cfg["method"] != "qlora",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=cfg["deepspeed"],
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        include_num_input_tokens_seen="non_padding",
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="no",
        report_to=["wandb"] if use_wandb else [],
        seed=cfg["seed"],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX, pad_to_multiple_of=8
        ),
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - start
    eval_metrics = trainer.evaluate()

    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))  # adapter only for LoRA/QLoRA, full weights otherwise

    if not trainer.is_world_process_zero():
        return
    tokenizer.save_pretrained(model_dir)

    tokens_seen = trainer.state.num_input_tokens_seen
    summary = {
        "run": cfg["run_name"],
        "method": cfg["method"],
        "base_model": cfg["base_model"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "train_examples": len(train_ds),
        "steps": trainer.state.global_step,
        "tokens_seen": int(tokens_seen),
        "train_seconds": round(train_seconds, 1),
        "tokens_per_second": round(tokens_seen / train_seconds, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "gpu": cfg["gpu"],
        "n_gpus": world_size,
        "train_cost_usd": estimate_cost(cfg["gpu"], world_size, train_seconds) if torch.cuda.is_available() else None,
        "final_train_loss": round(train_result.training_loss, 4),
        "final_eval_loss": round(eval_metrics["eval_loss"], 4),
        "artifact_mb": dir_size_mb(model_dir),
        "config": cfg,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))

    if use_wandb:
        import wandb

        if wandb.run is not None:
            wandb.run.summary.update({k: v for k, v in summary.items() if k != "config"})
            wandb.finish()


if __name__ == "__main__":
    main()
