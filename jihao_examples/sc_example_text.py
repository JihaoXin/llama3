# torchrun --nproc_per_node=4 sc_example_text.py --ckpt_dir ~/llama3/Meta-Llama-3-70B/ --tokenizer_path  ~/llama3/Meta-Llama-3-70B/tokenizer.model --batch_size=4 --token_length=1024 --max_seq_len=1024 --max_batch_size=32 --max_gen_len=1
# ncu --set full --target-processes all --force-overwrite --export ../profile/llama3_1layer torchrun --nproc_per_node=4 sc_example_text.py --ckpt_dir ~/llama3/Meta-Llama-3-70B/ --tokenizer_path  ~/llama3/Meta-Llama-3-70B/tokenizer.model --max_gen_len=
# BUILD_CUDA_EXTENSIONS=1 TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation -e .
import json
import os
import sys
import time  # Import the time module
from typing import Optional, List
import random
import string

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
    batch_size = 1,
    token_length = 1024,
    max_gen_len: Optional[int] = None,
    warmup = 2,
    repeat = 5,
):
    # Set up parameters
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if torch.cuda.is_bf16_supported():
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
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

    with open("./params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    # Load model and tokenizer
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model = Transformer(model_args).to(local_rank)
    generator = Llama(model, tokenizer)
    print("\n==================================\n")
    print("Model loaded with config:")
    args_str = (f"ckpt_dir: {ckpt_dir}, "f"tokenizer_path: {tokenizer_path}, "f"temperature: {temperature}, "f"top_p: {top_p}, "f"max_seq_len: {max_seq_len}, "f"max_batch_size: {max_batch_size}, "f"batch_size: {batch_size}, "f"token_length: {token_length}, "f"max_gen_len: {max_gen_len}, "f"warmup: {warmup}, "f"repeat: {repeat}")
    print(args_str)
    print(model_args)
    print("\n==================================\n")
    # Auto generate toy prompts with batch size and sequence length
    prompts = [''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=token_length*10)) for _ in range(batch_size)]
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False)[:token_length] for x in prompts]
    assert len(prompt_tokens) == batch_size, "Batch size does not match"
    assert all(len(x) == token_length for x in prompt_tokens), "Token length does not match"


    # Run inference
    for _ in range(warmup):
        generator.text_completion(
            prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p)
    # Measure Time
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    for _ in range(repeat):
        generator.text_completion(
            prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p)
    end_time.record()
    torch.cuda.synchronize()

    print(f"Inference complete with duration: {start_time.elapsed_time(end_time)/repeat:.6f} ms")
    print("\n==================================\n")
    # for prompt, result in zip(prompts, results):
    #     print(prompt)
    #     print(f"> {result['generation']}")
    #     print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
