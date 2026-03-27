from __future__ import annotations

"""
Sweep over K (number of SAE neurons kept, sorted by firing rate) and measure eval loss
in-domain (physics) and OOD (biology, math).

Use this to find the minimum K that preserves in-domain performance while OOD degrades —
the sweet spot for the scoping goal.

Usage:
    python scripts/evaluate_neuron_sweep.py \
        -p scripts/.cache/ignore_padding_True/physics/layer_20--width_16k--canonical/distribution.safetensors \
        -n 200 -b 4
"""

import contextlib
import csv
import gc
import re
from functools import partial
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma2ForCausalLM, PreTrainedTokenizerBase
from sae_lens import SAE

from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"

# Log-spaced sweep covering the full range from tiny to full SAE width (16k)
DEFAULT_K_VALUES = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 16384]


def sae_id_from_path(dist_path: str) -> str:
    folder_name = Path(dist_path).parent.name
    return folder_name.replace("--", "/")


def sae_id2hookpoint(sae_id: str) -> str:
    assert re.match(r"^layer_\d+/width_\d+k/canonical$", sae_id), f"Invalid SAE ID: {sae_id}"
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


def to_text_question(ds: Dataset) -> Dataset:
    return ds.select_columns(["question"]).rename_column("question", "text")


@torch.no_grad()
def compute_eval_loss(
    model,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    hookpoint: str | None,
    pruned_sae,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> float:
    """Mean per-batch cross-entropy loss with optional SAE hook."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    def _run_batch(batch_texts):
        nonlocal total_loss, n_batches
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # ignore padding in loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1

    if pruned_sae is not None and hookpoint is not None:
        sae_wrapper = SAEWrapper(pruned_sae)
        hook_dict = {hookpoint: partial(filter_hook_fn, sae_wrapper)}
        ctx = named_forward_hooks(model, hook_dict)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        for i in range(0, len(texts), batch_size):
            _run_batch(texts[i : i + batch_size])

    return total_loss / n_batches if n_batches > 0 else float("nan")


@click.command()
@click.option(
    "--dist-path", "-p", type=str, required=True,
    help="Path to distribution.safetensors from find_firing_rates.py",
)
@click.option("--batch-size", "-b", type=int, default=4)
@click.option("--n-samples", "-n", type=int, default=200, help="Eval samples per dataset")
@click.option("--max-length", "-ml", type=int, default=512, help="Max token length per example")
@click.option(
    "--k-values", "-k", type=str, default=",".join(map(str, DEFAULT_K_VALUES)),
    help="Comma-separated K values to sweep",
)
@click.option("--output-dir", "-o", type=str, default=None, help="Where to save CSV + plot")
def main(
    dist_path: str,
    batch_size: int,
    n_samples: int,
    max_length: int,
    k_values: str,
    output_dir: str | None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_id = sae_id_from_path(dist_path)
    hookpoint = sae_id2hookpoint(sae_id)
    k_list = sorted(set(int(k.strip()) for k in k_values.split(",")))

    out_dir = Path(output_dir) if output_dir else Path(dist_path).parent / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load firing rate distribution and compute ranking (descending = most task-relevant first)
    dist_data = load_file(dist_path)
    distribution: torch.Tensor = dist_data["distribution"]
    neuron_ranking = torch.argsort(distribution, descending=True)
    d_sae = len(distribution)
    print(f"SAE: {sae_id} | hookpoint: {hookpoint} | d_sae: {d_sae}")

    # Load model and tokenizer
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Gemma2ForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager"
    )
    model = model.to(device)

    # Load SAE base once — pruned versions are derived from it
    sae_base = SAE.from_pretrained(release=GEMMA2_9B_SAE_RELEASE, sae_id=sae_id, device=str(device))
    sae_base = sae_base.to(device).to(torch.bfloat16)

    # Load eval datasets
    # Physics = in-domain. Biology and math = OOD.
    # All three use the same "question" column and QA format → format is NOT a confounder.
    print("Loading eval datasets...")
    def _load_eval(config: str) -> list[str]:
        ds = load_dataset("4gate/StemQAMixture", config, split="test")
        return to_text_question(ds.select(range(min(n_samples, len(ds)))))["text"]

    eval_datasets: dict[str, list[str]] = {
        "physics (in-domain)": _load_eval("physics"),
        "biology (OOD)":       _load_eval("biology"),
        "chemistry (OOD)":     _load_eval("chemistry"),
        "math (OOD)":          _load_eval("math"),
    }

    results = []

    # Sweep over K values
    for K in k_list:
        pct = K / d_sae * 100
        print(f"\n--- K={K} ({pct:.1f}% of {d_sae} neurons) ---")
        pruned_sae = get_pruned_sae(sae_base, neuron_ranking, K_or_p=K, T=0.0)
        pruned_sae = pruned_sae.to(device)
        row: dict = {"K": K, "pct_neurons": round(pct, 3)}
        for ds_name, texts in eval_datasets.items():
            loss = compute_eval_loss(model, tokenizer, texts, hookpoint, pruned_sae, batch_size, max_length, device)
            print(f"  {ds_name}: loss={loss:.4f}")
            row[ds_name] = round(loss, 6)
        results.append(row)
        del pruned_sae
        gc.collect()
        torch.cuda.empty_cache()

    # Baseline: no SAE at all (unhooked model)
    print("\n--- Baseline: no SAE (unhooked) ---")
    row = {"K": -1, "pct_neurons": 100.0}
    for ds_name, texts in eval_datasets.items():
        loss = compute_eval_loss(model, tokenizer, texts, None, None, batch_size, max_length, device)
        print(f"  {ds_name}: loss={loss:.4f}")
        row[ds_name] = round(loss, 6)
    results.append(row)

    # Save CSV
    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results → {csv_path}")

    # Plot: eval loss vs K (log scale)
    sweep = [r for r in results if r["K"] != -1]
    baseline = next(r for r in results if r["K"] == -1)
    k_vals = [r["K"] for r in sweep]
    ds_names = [k for k in results[0] if k not in ("K", "pct_neurons")]
    colors = {
        "physics (in-domain)": "tab:blue",
        "biology (OOD)":       "tab:red",
        "chemistry (OOD)":     "tab:green",
        "math (OOD)":          "tab:orange",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for ds_name in ds_names:
        losses = [r[ds_name] for r in sweep]
        color = colors.get(ds_name)
        ax.plot(k_vals, losses, marker="o", label=ds_name, color=color)
        ax.axhline(
            baseline[ds_name], linestyle="--", color=color, alpha=0.5,
            label=f"{ds_name} (no SAE)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("K — neurons kept (log scale)", fontsize=12)
    ax.set_ylabel("Eval Loss (cross-entropy)", fontsize=12)
    ax.set_title(f"SAE Neuron Sweep: Physics (in-domain) vs OOD\n{sae_id}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / "sweep_loss_vs_k.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot → {plot_path}")


if __name__ == "__main__":
    main()
