# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from pathlib import Path
from typing import Optional

import shutil

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM

import fla  # noqa
from torchtitan.tools.logging import init_logger, logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@torch.inference_mode()
def _maybe_find_embedding_key(state_dict: dict) -> Optional[str]:
    """Best-effort detection of the tied embedding weight key."""

    preferred_suffixes = (
        "tok_embeddings.weight",
        "word_embeddings.weight",
        "wte.weight",
        "embed_tokens.weight",
        "embeddings.word_embeddings.weight",
    )

    for suffix in preferred_suffixes:
        for key in state_dict:
            if key.endswith(suffix):
                return key
    return None


def _ensure_lm_head_weight(model, state_dict: dict) -> None:
    """Add ``lm_head.weight`` to ``state_dict`` if it is absent."""

    if "lm_head.weight" in state_dict:
        logger.info("'lm_head.weight' already present in checkpoint payload")
        return

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()

    if lm_head is None or not hasattr(lm_head, "weight"):
        logger.warning(
            "The converted checkpoint is missing 'lm_head.weight' and the model does "
            "not expose an output embedding weight. The resulting checkpoint may fail "
            "to load if the training graph expects this parameter."
        )
        return

    embed_key = _maybe_find_embedding_key(state_dict)
    if embed_key is not None:
        logger.info(
            "Tying 'lm_head.weight' to existing embedding weight at key '%s'", embed_key
        )
        state_dict["lm_head.weight"] = state_dict[embed_key].clone()
    else:
        state_dict["lm_head.weight"] = lm_head.weight.detach().to(device="cpu")
        logger.info(
            "Added missing 'lm_head.weight' from the model's output embeddings module"
        )

    if "lm_head.weight" not in state_dict:
        raise RuntimeError("Failed to materialize 'lm_head.weight' in converted checkpoint")


def convert_hf_weights(model: str, checkpoint: Path, dtype: Optional[str]):
    torch_dtype = TORCH_DTYPE_MAP.get(dtype) if dtype else None
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)
    state_dict = model.state_dict()
    for key, tensor in list(state_dict.items()):
        state_dict[key] = tensor.detach().to(device="cpu")

    _ensure_lm_head_weight(model, state_dict)

    logger.info(f"Writing to DCP at '{checkpoint}'")
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)

    # ``torch.distributed.checkpoint`` persists keys exactly as they are provided in the
    # state dict.  The training loader expects the raw parameter names (e.g.
    # ``lm_head.weight``) instead of a "model."-prefixed namespace, so we hand the
    # flat state dict directly to the saver instead of nesting it under an extra key.
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save(state_dict, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=sorted(TORCH_DTYPE_MAP.keys()),
        default=None,
        help="Optional dtype override when loading the HF checkpoint.",
    )
    args = parser.parse_args()

    convert_hf_weights(args.model, args.checkpoint, args.dtype)
