# launch command: python -m torch.distributed.launch --nproc_per_node=4 allreduce.py
import torch
import torch.distributed as dist
import os
import argparse

def setup(rank, world_size):
    # os.environ['NCCL_DEBUG'] = 'INFO'  # Set NCCL to report debug information
    dist.init_process_group("nccl",rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    print(f"Running Pytorch Allreduce example on rank {rank}.")
    setup(rank, world_size)
    print(f"Rank {rank} is Initialized.")
    warmup = 10
    repeat = 100
    # Ensure each GPU has a tensor to work with
    input = torch.randn([32, 1024, 8192], dtype=torch.bfloat16).cuda(rank)


    for i in range(warmup):
        dist.all_reduce(input)
    dist.barrier()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(repeat):
        dist.all_reduce(input)
    dist.barrier()
    end_event.record()
    end_event.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    if rank == 0:
        print(f"Elapsed time: {elapsed_time_ms/repeat} ms")

    cleanup()

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank, world_size)