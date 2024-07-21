#!/bin/bash

# Define the output file
output_file="sc_completion.log"

# Clear the output file if it already exists
> $output_file

export OMP_NUM_THREADS=4
docker exec -it distfuse nvidia-smi -pm 1
docker exec -it distfuse nvidia-smi -lgc 1410,1410

# Iterate over batch_size values
for batch_size in 1 4 32
do
  # Iterate over token_length values
  for token_length in 8 256 1024
  do
    # Run the torchrun command with the current batch_size and token_length
    torchrun --nproc_per_node=4 sc_example_text.py --ckpt_dir ~/llama3/Meta-Llama-3-70B/ --tokenizer_path ~/llama3/Meta-Llama-3-70B/tokenizer.model --batch_size=$batch_size --token_length=$token_length --max_seq_len=1024 --max_batch_size=32 --max_gen_len=1 >> $output_file 2>&1
  done
done

docker exec -it nvidia-smi -pm 0
docker exec -it nvidia-smi --reset-gpu-clocks