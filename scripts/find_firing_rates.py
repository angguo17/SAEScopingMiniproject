from __future__ import annotations
import click
import re
import itertools
from pathlib import Path
import gc
from jaxtyping import Integer, Float
from beartype import beartype
import torch
import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Gemma2ForCausalLM,
    AutoTokenizer,
)
from safetensors.torch import save_file
from sae_lens import SAE
from sae_scoping.trainers.sae_enhanced.rank import rank_neurons

"""
This module is a CLI entrypoint to running a grid of `rank_neurons` and saving the
firing rates for offline analysis. You would use this for plotting the firing rates
and picking (and then testing many) threhold(s).
"""


# Copied
def pretokenize(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    context_length: int = 1024,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize a dataset once into a list of batched dicts, with pinned memory for fast GPU transfer."""
    texts = dataset["text"]
    use_pin = torch.cuda.is_available()
    batches = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=context_length,
        )
        batch = {k: v.pin_memory() if use_pin else v for k, v in enc.items()}
        batches.append(batch)
    return batches


def sae_id2hookpoint(sae_id: str | None) -> str:
    if sae_id is None:
        return None
    assert re.match(r"^layer_\d+/width_16k/canonical$", sae_id)
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


# GEMMA2_9B_SAE_IDS: list[str] = [
#     # https://huggingface.co/google/gemma-scope-9b-pt-res/tree/main
#     f"layer_{i}/width_16k/canonical"
#     for i in range(0, 50, 1)
# ]
# https://github.com/decoderesearch/SAELens/blob/bd44804c64ff0d2d920fb0896635f9d9830dafab/sae_lens/pretrained_saes.yaml#L3321
# We want to use IT
GEMMA2_9B_SAE_IDS: list[str] = [
    f"layer_{layer}/width_{width}/canonical"
    for layer, width in [
        (9, "16k"),
        (20, "16k"),
        (31, "16k"),
        # We do not use these because they are big af
        # (9, "131k"),
        # (20, "131k"),
        # (31, "131k"),
    ]
]

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"


@beartype
def rank_neurons_shim(
    tokenized: Dataset | list[dict[str, torch.Tensor]],
    sae_id: str,
    sae_release: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int | None = None,
    T: float | int = 0.0,
    ignore_padding: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[Integer[torch.Tensor, "d_sae"], Float[torch.Tensor, "d_sae"]]:
    sae: SAE = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    sae = sae.to(device).to(torch.bfloat16)  # match model dtype to avoid copies on every forward
    hookpoint = sae_id2hookpoint(sae_id)
    with torch.no_grad():
        ranking, distribution = rank_neurons(
            dataset=tokenized,
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            T=T,
            hookpoint=hookpoint,
            batch_size=batch_size,
            token_selection="attention_mask" if ignore_padding else "all",
            return_distribution=True,
        )
        ranking = ranking.detach().cpu()
        distribution = distribution.detach().cpu()
        sae = sae.to("cpu")
        del sae
        gc.collect()
        torch.cuda.empty_cache()
        return ranking, distribution


# TODO(Adriano) don't hardcode lol plz
@click.command()
@click.option("--datasets", "-d", type=str, default="chemistry,apps,ultrachat")
@click.option("--ignore_paddings", "-i", type=str, default="True,False")
@click.option("--batch-size", "-b", type=int, default=7)
def cli(datasets: str, ignore_paddings: str, batch_size: int):
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load dataset and tokenize it
    print("=" * 100)
    print("Loading dataset and tokenizing it...")
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert tokenizer.pad_token is not None

    chem_data = load_dataset("4gate/StemQAMixture", "chemistry", split="train")
    apps_data = load_dataset("codeparrot/apps", split="train")
    ultrachat_data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    # rank_neurons expects a Dataset with a "text" column of raw strings and tokenizes internally
    chem_dataset = chem_data.select_columns(["question"]).rename_column("question", "text")
    apps_dataset = apps_data.select_columns(["question"]).rename_column("question", "text")

    def extract_ultrachat_text(examples):
        return {"text": [next((m["content"] for m in msgs if m["role"] == "user"), "") for msgs in examples["messages"]]}
    ultrachat_dataset = ultrachat_data.map(extract_ultrachat_text, batched=True, remove_columns=ultrachat_data.column_names)

    # 2. Load model
    model = Gemma2ForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    model = torch.compile(model)  # ~20-30% throughput gain; disable if using multi-GPU device_map or hooks cause issues

    # 3. For each SAE, run through inference on this
    output_folder = Path(__file__).parent / ".cache"
    datasets_and_names = [
        (chem_dataset, "stemqa_chemistry"),
        (apps_dataset, "apps"),
        (ultrachat_dataset, "ultrachat"),
    ]
    
    datasets = list(set(list(map(str.strip, datasets.split(",")))))
    datasets_and_names = [x for x in datasets_and_names if x[1] in datasets]

    # Pre-tokenize each dataset once — reused across all SAE/padding combos
    print("Pre-tokenizing datasets...")
    tokenized_map: dict[str, list[dict[str, torch.Tensor]]] = {
        name: pretokenize(ds, tokenizer, batch_size)
        for ds, name in datasets_and_names
    }
    print("Pre-tokenization done.")

    def to_bool(x: str) -> bool:
        return x.lower().strip() == "true"

    ignore_paddings = list(set(list(map(to_bool, ignore_paddings.split(",")))))
    combos = list(itertools.product(datasets_and_names, ignore_paddings, GEMMA2_9B_SAE_IDS))
    print("=" * 100)
    print(f"WILL ITERATE FOR {len(combos)} COMBOS")
    print("=" * 100)
    for (dataset, dataset_name), ignore_padding, sae_id in tqdm.tqdm(combos, desc="Processing datasets..."):
        subfolder = output_folder / f"ignore_padding_{ignore_padding}" / dataset_name / sae_id.replace("/", "--")
        if subfolder.exists():
            continue
        _, distribution = rank_neurons_shim(
            tokenized=tokenized_map[dataset_name],
            sae_id=sae_id,
            sae_release=GEMMA2_9B_SAE_RELEASE,
            model=model,
            batch_size=None,  # list input: each element is already a batch
            tokenizer=tokenizer,
            T=0,
            ignore_padding=ignore_padding,
        )

        assert not subfolder.exists(), f"Subfolder {subfolder} already exists"
        subfolder.mkdir(parents=True, exist_ok=True)
        # Distribution IMPLIES ranking so we don't need to do anything there.
        save_file(
            {"distribution": distribution},
            subfolder / "distribution.safetensors",
        )


if __name__ == "__main__":
    cli()
