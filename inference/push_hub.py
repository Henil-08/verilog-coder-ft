"""Publish a fine-tuned run to the Hugging Face Hub with a generated model card.

    uv run --extra train python -m inference.push_hub --run 7B-L3 --name ft-7b --owner henil08 --extras

Pushes the adapter (LoRA/QLoRA runs) to <owner>/verilog-coder-<size>-lora and the merged
weights (models/merged/<name>, from inference/convert.py) to <owner>/verilog-coder-<size>.
Model cards are built from outputs/runs/<run>/summary.json and results/<tag>.jsonl.
"""

import argparse
import json
from pathlib import Path

from eval.report import fmt_p, summarize
from training.config import load_run

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Henil-08/verilog-coder-ft"
LICENSES = {"3B": ("other", "qwen-research", "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/blob/main/LICENSE")}


def load_results(tag: str) -> list[dict]:
    return [json.loads(line) for line in open(ROOT / "results" / f"{tag}.jsonl")]


def pct(entry: dict) -> str:
    lo, hi = entry["ci95"]
    return f"{100 * entry['pass@1']:.1f}% ({entry['passed']}/{entry['n']}, 95% CI {100 * lo:.1f}–{100 * hi:.1f})"


def model_card(run: str, size: str, merged: bool) -> str:
    cfg = load_run(run)
    summary = json.loads((ROOT / "outputs" / "runs" / run / "summary.json").read_text())
    base_tag = "base-" + cfg["base_model"].split("/")[-1]
    base = summarize(load_results(base_tag), None)
    tuned = summarize(load_results(run), load_results(base_tag))
    license_id, license_name, license_link = LICENSES.get(size, ("apache-2.0", None, None))
    regression = ""
    if tuned["verilog_eval"]["pass@1"] < base["verilog_eval"]["pass@1"]:
        regression = (
            f"- **On VerilogEval this fine-tune scores below its base model** "
            f"({100 * tuned['verilog_eval']['pass@1']:.1f}% vs {100 * base['verilog_eval']['pass@1']:.1f}% "
            "with the published prompts). Fine-tuning on OpenRTLSet's real-world modules and "
            "LLM-written descriptions improved held-out OpenRTLSet modules but cost accuracy on "
            "VerilogEval's textbook-style problems; for that style of task the base model is the better choice.\n"
        )

    method = {"lora": "LoRA", "qlora": "QLoRA (NF4 base)", "full": "full fine-tuning"}[cfg["method"]]
    adapter = "" if cfg["method"] == "full" else (
        f"- LoRA rank {cfg['lora_r']}, alpha {cfg['lora_alpha']}, dropout {cfg['lora_dropout']}, "
        f"targets: {', '.join(cfg['target_modules'])}\n")
    rows = []
    for key, label in (("openrtl", "OpenRTLSet held-out (diff-sim)"),
                       ("verilog_eval", "VerilogEval v2 spec-to-RTL (published prompts)"),
                       ("verilog_eval_iface", "VerilogEval v2 + interface code block")):
        t = tuned[key]
        rows.append(f"| {label} | {pct(base[key])} | {pct(t)} | +{t['gained_vs_baseline']} / "
                    f"-{t['lost_vs_baseline']} (McNemar {fmt_p(t['mcnemar_p'])}) |")

    front = ["---", f"base_model: {cfg['base_model']}", f"license: {license_id}"]
    if license_name:
        front += [f"license_name: {license_name}", f"license_link: {license_link}"]
    front += ["library_name: " + ("transformers" if merged else "peft"), "pipeline_tag: text-generation",
              "datasets:", "- ESCAD/OpenRTLSet", "tags:", "- verilog", "- systemverilog", "- rtl", "- code", "---"]

    usage = (
        f"```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n"
        f"tok = AutoTokenizer.from_pretrained(\"{{repo}}\")\n"
        f"model = AutoModelForCausalLM.from_pretrained(\"{{repo}}\", dtype=\"auto\", device_map=\"auto\")\n```"
        if merged else
        f"```python\nfrom peft import PeftModel\nfrom transformers import AutoModelForCausalLM\n"
        f"base = AutoModelForCausalLM.from_pretrained(\"{cfg['base_model']}\", dtype=\"auto\")\n"
        f"model = PeftModel.from_pretrained(base, \"{{repo}}\")\n```"
    )

    return "\n".join(front) + f"""

# verilog-coder ({size}, {method})

{cfg['base_model']} fine-tuned for Verilog/SystemVerilog generation from a natural-language
spec (optionally with the module interface). Code, data pipeline, and evaluation harness:
[{GITHUB}]({GITHUB}).

## Training data

[OpenRTLSet](https://huggingface.co/datasets/ESCAD/OpenRTLSet): 131k Verilog modules from
GitHub, each paired with an LLM-written description. After dropping 54.6k fuzzer-generated
modules, deduplicating, and capping examples at 2048 tokens, 53.7k remain; this run used
{summary['train_examples']:,} of them (~93% Verilog, ~7% SystemVerilog). About 30% of
training prompts describe the ports in prose instead of giving the header as code, which
keeps the model usable when no header is provided.

The raw corpus contains UVM testbench code in only 3 rows, and none of them survive
filtering. This model was not trained or evaluated on UVM and makes no claim of UVM
capability.

## Training setup

- Method: {method}, 1 epoch, {summary['tokens_seen']:,} supervised tokens
{adapter}- LR {cfg['learning_rate']} (cosine, 3% warmup), effective batch {cfg['effective_batch_size']}, max length {cfg['max_seq_len']}, bf16
- HF Transformers + PEFT + DeepSpeed ZeRO-2 on {summary['n_gpus']}× {summary['gpu']} (Modal);
  {summary['train_seconds'] / 60:.0f} min of training, ~${summary['container_cost_usd']:.2f}
- Loss on the code (assistant) tokens only; final eval loss {summary['final_eval_loss']}
- Selected among LoRA/QLoRA/full-fine-tune configs by pass@1 on a validation split, so the
  test numbers below were not used for model selection.

## Evaluation (pass@1, greedy decoding)

| Benchmark | Base | Fine-tuned | Items gained / lost |
|---|---|---|---|
{chr(10).join(rows)}

- **OpenRTLSet held-out**: 394 standalone modules not seen in training. A generated module
  passes if it matches the reference's ports and its outputs equal the reference's on every
  half-cycle of 500 cycles of random stimulus (Verilator). Random stimulus is evidence of
  equivalence, not proof.
- **VerilogEval v2**: the official testbenches and Icarus Verilog v12 pass rule, with this
  project's chat prompt (scores are not directly comparable to leaderboard numbers that use
  other prompts).

## Limitations

{regression}- Greedy decoding sometimes falls into repetition loops on long modules; about
  {100 * tuned['truncated_outputs'] / 1312:.0f}% of eval outputs hit the 2048-token cap.
- Trained on single-module completions; multi-module designs and testbenches are out of
  scope. Not evaluated for synthesis quality, timing, or area.

## Usage

{usage}

Prompt format (system + user), as in training:

```text
Specification:
<description>

Module interface:
```verilog
module ...(...);
```
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--name", required=True, help="local build name, e.g. ft-7b")
    parser.add_argument("--owner", default="henil08")
    parser.add_argument("--dry-run", action="store_true", help="write cards to outputs/cards/ only")
    parser.add_argument("--extras", action="store_true", help="also push the MLX 4-bit and GGUF builds")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    size = args.run.split("-")[0]
    cfg = load_run(args.run)
    targets = []
    if cfg["method"] != "full":
        targets.append((f"{args.owner}/verilog-coder-{size.lower()}-lora", ROOT / "outputs" / "runs" / args.run / "model", False))
    targets.append((f"{args.owner}/verilog-coder-{size.lower()}", ROOT / "models" / "merged" / args.name, True))

    api = HfApi()
    for repo, folder, merged in targets:
        card = model_card(args.run, size, merged).replace("{repo}", repo)
        if args.dry_run:
            out = ROOT / "outputs" / "cards" / f"{repo.split('/')[-1]}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(card)
            print(f"wrote {out}")
            continue
        api.create_repo(repo, exist_ok=True)
        (folder / "README.md").write_text(card)
        api.upload_folder(repo_id=repo, folder_path=str(folder), ignore_patterns=["training_args.bin"],
                          commit_message=f"Upload {args.run}")
        print(f"pushed https://huggingface.co/{repo}")

    if args.extras and not args.dry_run:
        push_extras(api, args.owner, size, args.name)


def push_extras(api, owner: str, size: str, name: str) -> None:
    """MLX 4-bit and GGUF builds, each with a short card pointing at the main repo."""
    main = f"{owner}/verilog-coder-{size.lower()}"
    builds = [
        (f"{main}-mlx-4bit", ROOT / "models" / "mlx" / f"{name}-4bit", None,
         f"MLX 4-bit build of [{main}](https://huggingface.co/{main}) for Apple Silicon.\n\n"
         f"```bash\nmlx_lm.generate --model {main}-mlx-4bit --prompt \"Specification: 4-bit counter\"\n```"),
        (f"{main}-gguf", ROOT / "models" / "gguf", f"{name}*",
         f"GGUF (Q4_K_M) build of [{main}](https://huggingface.co/{main}) for llama.cpp / Ollama.\n\n"
         f"```bash\nollama run hf.co/{main}-gguf\n```"),
    ]
    for repo, folder, pattern, body in builds:
        api.create_repo(repo, exist_ok=True)
        api.upload_file(path_or_fileobj=f"---\nbase_model: {main}\ntags:\n- verilog\n---\n\n{body}\n".encode(),
                        path_in_repo="README.md", repo_id=repo)
        api.upload_folder(repo_id=repo, folder_path=str(folder), allow_patterns=[pattern] if pattern else None,
                          ignore_patterns=["*.Modelfile"], commit_message="Upload build")
        print(f"pushed https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
