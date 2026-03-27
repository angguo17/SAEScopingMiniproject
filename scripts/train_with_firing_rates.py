from __future__ import annotations

import gc
import hashlib
import re
from pathlib import Path
import os
import click
import torch
from beartype import beartype
from datasets import concatenate_datasets, load_dataset
from sae_lens import SAE
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma2ForCausalLM, PreTrainedTokenizerBase
from trl import SFTConfig

from sae_scoping.trainers.sae_enhanced.prune import (
    get_pruned_sae,
)
from sae_scoping.trainers.sae_enhanced.train import train_sae_enhanced_model

"""
This module/script does exactly what you expect: it trains a Gemma-2 9B
around an SAE. It is the type of CLI script you might want to ue BOTH for
scoping recovery training and adversarial re-training.
"""

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"


@beartype
def sae_id_from_path(dist_path: str) -> str:
    """Extract SAE ID from path like '.../layer_20--width_16k--canonical/distribution.safetensors'."""
    folder_name = Path(dist_path).parent.name
    return folder_name.replace("--", "/")


@beartype
def sae_id2hookpoint(sae_id: str) -> str:
    assert re.match(r"^layer_\d+/width_\d+k/canonical$", sae_id), f"Invalid SAE ID: {sae_id}"
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


@beartype
def model_name_or_path2threshold(model_name_or_path: str | None) -> float:
    """Copied from `script_2026_01_23_evaluate_biology_utility.py` (while this file was first made before that one, it was modified more recently)."""
    if model_name_or_path is None:
        raise ValueError(f"model_name_or_path is None")
    h_find_pattern = r"_h(\d+\.?\d*(?:e[+-]?\d+)?)"
    match = re.search(h_find_pattern, model_name_or_path, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not extract h value from path: {model_name_or_path}")
    return float(match.group(1))


def _main(
    dist_path: str,
    batch_size: int,
    max_steps: int,
    accum: int,
    special_hookpoint: str | None,
    checkpoint: str | None,
    train_on_dataset: str,
    wandb_project_name: str,
    save_every: int,
    save_limit: int,
    output_dir: str | None = None,
    wandb_run_name: str | None = None,
    save_output: bool = False,
    max_length: int = 1024,
    eval_on_datasets: str | None = None,  # comma-delimited list of dataset names, None = all
    eval_test_size: int = 500,  # number of samples for evaluation per dataset
    threshold: float | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Extract SAE ID and hookpoint from path
    if dist_path == "vanilla":
        sae_id, hookpoint, sae, pruned_sae = None, None, None, None
    else:
        sae_id = sae_id_from_path(dist_path)
        hookpoint = sae_id2hookpoint(sae_id)
        threshold = model_name_or_path2threshold(checkpoint) if threshold is None else threshold
        print(f"SAE ID: {sae_id}, Hookpoint: {hookpoint}")

        # 2. Load distribution and compute neuron mask
        dist_data = load_file(dist_path)
        distribution: torch.Tensor = dist_data["distribution"]  # shape: (d_sae,)
        neuron_ranking = torch.argsort(distribution, descending=True)
        n_kept = int((distribution >= threshold).sum().item())
        print(f"Keeping {n_kept}/{len(distribution)} neurons (threshold={threshold})")

    # 3. Load tokenizer and model
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    model_name_or_path = checkpoint if checkpoint is not None else model_name
    model_name_or_path_hash = hashlib.sha256(model_name_or_path.encode()).hexdigest() if model_name_or_path != "vanilla" else "vanilla"
    model = Gemma2ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = False

    if sae_id is not None:
        # 4. Load SAE and create pruned version
        sae = SAE.from_pretrained(release=GEMMA2_9B_SAE_RELEASE, sae_id=sae_id, device=device)
        sae = sae.to(device)
        pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
        pruned_sae = pruned_sae.to(device)

    # 5. Build training and evaluation datasets
    # All StemQA subjects have dedicated train/test splits — no need to carve manually.
    # In-domain: physics. OOD: biology (same format, different STEM domain),
    # math (same format, shares mathematical formalism but different concepts).
    def _to_text_question(ds):
        return ds.select_columns(["question"]).rename_column("question", "text")

    def _load_stemqa(config: str, split: str, n: int | None = None):
        ds = load_dataset("4gate/StemQAMixture", config, split=split)
        return _to_text_question(ds.select(range(min(n, len(ds)))) if n else ds)

    def _load_apps(split: str, n: int | None = None):
        ds = load_dataset("codeparrot/apps", split=split)
        ds = ds.select_columns(["question"]).rename_column("question", "text")
        return ds.select(range(min(n, len(ds)))) if n else ds

    _physics_train   = _load_stemqa("physics",   "train")
    _chemistry_train = _load_stemqa("chemistry", "train")
    _apps_train      = _load_apps("train")

    all_eval_datasets = {
        "physics":   _load_stemqa("physics",   "test", eval_test_size),  # in-domain
        "biology":   _load_stemqa("biology",   "test", eval_test_size),  # OOD: different STEM domain
        "chemistry": _load_stemqa("chemistry", "test", eval_test_size),  # OOD: adversarial recovery target
        "math":      _load_stemqa("math",      "test", eval_test_size),  # OOD: shares math formalism
        "apps":      _load_apps("test", eval_test_size),                  # OOD: adversarial recovery target (coding)
    }

    # Filter eval datasets if specified
    if eval_on_datasets is not None:
        eval_dataset_names = [name.strip() for name in eval_on_datasets.split(",")]
        invalid_names = set(eval_dataset_names) - set(all_eval_datasets.keys())
        if invalid_names:
            raise ValueError(f"Invalid eval dataset names: {invalid_names}. Valid names are: {list(all_eval_datasets.keys())}")
        eval_datasets = {name: all_eval_datasets[name] for name in eval_dataset_names}
        print(f"Evaluating on subset of datasets: {list(eval_datasets.keys())}")
    else:
        eval_datasets = all_eval_datasets
        print(f"Evaluating on all datasets: {list(eval_datasets.keys())}")
    train_datasets = {
        "physics":   _physics_train,    # in-domain scoping recovery
        "chemistry": _chemistry_train,  # adversarial recovery: STEM QA (chem)
        "apps":      _apps_train,       # adversarial recovery: coding (cyber)
    }
    if train_on_dataset not in train_datasets:
        raise ValueError(f"Invalid train on dataset: {train_on_dataset}")
    train_dataset = train_datasets[train_on_dataset]

    # 7. Train
    if sae_id is None:
        sae_id = "vanilla"
    if wandb_run_name is None:
        wandb_run_name = f"{train_on_dataset}/{sae_id.replace('/', '_')}"
        if sae_id != "vanilla":
            wandb_run_name += f"/h{threshold}/{model_name_or_path_hash[:10]}"
    if output_dir is None:
        output_dir = f"./outputs_gemma9b/{train_on_dataset}/{sae_id.replace('/', '_')}"
        if sae_id != "vanilla":
            output_dir += f"_h{threshold}_{model_name_or_path_hash[:10]}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        gradient_accumulation_steps=accum,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=save_every,
        bf16=True,
        save_total_limit=save_limit,
        report_to="wandb",
        max_length=max_length,  # SAE context length bounds this
        gradient_checkpointing=False,
        run_name=wandb_run_name,
    )
    if special_hookpoint is not None:  # used to limit # layers trained on
        hookpoint = special_hookpoint
    # NOTE: while technically not supported by my code, since it's passthrough, you
    # SHOULD be able to use not only "text" but also "messages" etc... (looke at
    # SFTTrainer docs for supported formats)
    os.environ["WANDB_PROJECT"] = wandb_project_name  # defensive code
    os.environ["WANDB_RUN_NAME"] = wandb_run_name  # defensive code
    train_sae_enhanced_model(
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        sae=pruned_sae,
        model=model,
        tokenizer=tokenizer,
        T=0.0,
        hookpoint=hookpoint,
        save_output=save_output,
        trainer_config=sft_config,
        wandb_project_name=wandb_project_name,
        wandb_run_name=wandb_run_name,
    )

    # Cleanup
    del model, sae, pruned_sae
    gc.collect()
    torch.cuda.empty_cache()


@click.command()
@click.option(
    "--dist-path",
    "-p",
    type=str,
    required=True,
    help="Path to distribution.safetensors",
)
@click.option("--batch-size", "-b", type=int, default=4, help="Training batch size")
@click.option("--max-steps", "-s", type=int, default=1000, help="Max training steps")
@click.option("--accum", "-a", type=int, default=1, help="Gradient accumulation steps")
@click.option(
    "--special-hookpoint",
    "-hook",
    type=str,
    default=None,
    help="Special hookpoint to use",
)
@click.option("--checkpoint", "-c", type=str, default=None, help="Checkpoint to load")
@click.option("--train-on-dataset", "-t", type=str, default="physics", help="Dataset to train on")
@click.option(
    "--wandb-project-name",
    "-w",
    type=str,
    default="gemma-scope-9b-recovery-train",
    help="Wandb project name",
)
@click.option("--save-every", "-se", type=int, default=1000, help="Save every n steps")
@click.option("--save-limit", "-sl", type=int, default=10, help="Save limit")
# NOTE please run for gemma
# export GRADIENT_CHECKPOINTING=0
@click.option("--max-length", "-ml", type=int, default=1024, help="Max length")
@click.option(
    "--eval-on-datasets",
    "-e",
    type=str,
    default=None,
    help="Comma-delimited list of dataset names to evaluate on (e.g., 'biology,gsm8k,ultrachat'). Default: all datasets",
)
@click.option(
    "--eval-test-size",
    "-ts",
    type=int,
    default=500,
    help="Number of samples per dataset for evaluation (default: 500)",
)
@click.option(
    "--threshold",
    "-h",
    type=float,
    default=None,  # None => infer from the checkpoint threshold (we only need to pass if we train vanilla)
    help="Min firing rate to keep neuron (default: None, uses model checkpoint threshold)",
)
def main(
    dist_path: str,
    batch_size: int,
    max_steps: int,
    accum: int,
    special_hookpoint: str | None,
    checkpoint: str | None,
    train_on_dataset: str,
    wandb_project_name: str,
    save_every: int,
    save_limit: int,
    max_length: int,
    eval_on_datasets: str | None,
    eval_test_size: int,
    threshold: float | None,
) -> None:
    r"""
    Example with benign recovery training in-domain (NOTE in this we limit how many
    layers are trained by using `special_hookpoint`; special hookpoint is meant only for vanilla):
    ```
    python3 script_2025_12_08_train_sft_gemma9b_sae.py \
        -p vanilla \
        -b 2 -a 16 -hook model.layers.31 -s 40000 -h 1e-4
    ```

    Example adversarial re-training (after recovery training) example:
    ```
    python3 script_2025_12_08_train_sft_gemma9b_sae.py \
        -c /<my_path>/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000 \
        -t ultrachat \
        -w gemma-scope-9b-recovery-attack-2025-12-24 \
        -s 4000 -a 8 -b 4 \
        -p /<my_path>/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors
    ```
    """
    return _main(
        dist_path=dist_path,
        batch_size=batch_size,
        max_steps=max_steps,
        accum=accum,
        special_hookpoint=special_hookpoint,
        checkpoint=checkpoint,
        train_on_dataset=train_on_dataset,
        wandb_project_name=wandb_project_name,
        save_every=save_every,
        save_limit=save_limit,
        save_output=False,
        max_length=max_length,
        eval_on_datasets=eval_on_datasets,
        eval_test_size=eval_test_size,
        threshold=threshold,
    )


if __name__ == "__main__":
    main()
