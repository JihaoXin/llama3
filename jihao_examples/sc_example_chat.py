# torchrun --nproc_per_node=4 sc_example_chat.py --ckpt_dir Meta-Llama-3-70B/ --tokenizer_path Meta-Llama-3-70B/tokenizer.model --max_seq_len 512 --max_batch_size 6
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import json
import os
import sys
from typing import Optional

import fire
import torch

from llama import Llama
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from pathlib import Path
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    # Set up parameters
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if torch.cuda.is_bf16_supported():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        print("Using BF16")
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        print("Using FP16")
    torch.manual_seed(42)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    # Initialize model parallel
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)
        print(f"Model parallel initialized with size {model_parallel_size}")

    # Load model and tokenizer
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model = Transformer(model_args).to(local_rank)
    generator = Llama(model, tokenizer)
    print("Llama loaded")
    # Check device allocation of model parameters
    for name, param in model.named_parameters():
        print(f"Layer: {name} - Shape: {param.shape} - Device: {param.device} - Dtype: {param.dtype} - Current Rank: {local_rank}")

    # Run chat completion
    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
