# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from pathlib import Path
from typing import Optional

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
    state_dict = {
        key: tensor.detach().to(device="cpu")
        for key, tensor in model.state_dict().items()
    }

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)


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
