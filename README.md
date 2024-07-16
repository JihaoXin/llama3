# Llama 3
This repo is forked from Meta's LLAMA3.

We give examples to run the Llama3-70B model without loading the weights for fast prototyping.

The eventual target is to apply DistFuse to speedup the LLM inference.

For more information please refer to the original **META's README** below.

## Quick Start
#### Install
``` sh
git clone https://github.com/username/llama3.git
cd llama3
# Update Submodule - FairScale
git submodule update --init --recursive
git pull --recurse-submodules
git submodule update --recursive

# Install dependency
conda create --name <name> python=3.12.4
pip install -e .

# Install fairscale
cd fairscale
pip install -r requirements.txt
# -e signified dev mode since e stands for editable; "8.0" is for A100, replace the arch for different GPU.
BUILD_CUDA_EXTENSIONS=1 TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation -e .
cd ..
```
#### Execute
The main script is put at `jihao_examples/sc_example_text.py`, it will dry run the inference without loading the weights.

To change the model architecture such as the number of attention layers, please modify `jihao_examples/params.json`
```sh
cd jihao_examples

# Run with 4 GPUs
torchrun --nproc_per_node=4 sc_example_text.py --ckpt_dir ~/llama3/Meta-Llama-3-70B/ --tokenizer_path  ~/llama3/Meta-Llama-3-70B/tokenizer.model --batch_size=4 --token_length=1024 --max_seq_len=1024 --max_batch_size=32 --max_gen_len=1

# Profile with ncu
ncu --set full --target-processes all --force-overwrite --export ../profile/llama3_1layer torchrun --nproc_per_node=4 sc_example_text.py --ckpt_dir ~/llama3/Meta-Llama-3-70B/ --tokenizer_path  ~/llama3/Meta-Llama-3-70B/tokenizer.model --batch_size=4 --token_length=1024 --max_seq_len=1024 --max_batch_size=32 --max_gen_len=1
```
The `jihao_examples/allreduce.py` is designed to test the latency of the allreduce function from pytorch
```
python -m torch.distributed.launch --nproc_per_node=4 allreduce.py
```
The `jihao_examples/ds_example_chat.py` and `jihao_examples/hf_example_chat.py` are examples to run llama3 with deepspeed and huggingface. They are deprecated and cannot run directly.

---
---
# META's README
---
---

## Download
To download the model weights and tokenizer, please visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then, run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Ensure you have `wget` and `md5sum` installed. Then run the script: `./download.sh`.

Remember that the links expire after 24 hours and a certain amount of downloads. You can always re-request a link if you start seeing errors such as `403: Forbidden`.

### Access to Hugging Face

We also provide downloads on [Hugging Face](https://huggingface.co/meta-llama), in both transformers and native `llama3` formats. To download the weights from Hugging Face, please follow these steps:

- Visit one of the repos, for example [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- Read and accept the license. Once your request is approved, you'll be granted access to all the Llama 3 models. Note that requests used to take up to one hour to get processed.
- To download the original native weights to use with this repo, click on the "Files and versions" tab and download the contents of the `original` folder. You can also download them from the command line if you `pip install huggingface-hub`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct
```

- To use with transformers, the following [pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) snippet will download and cache the weights:

  ```python
  import transformers
  import torch

  model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

  pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
  )
  ```

## Quick Start

You can follow the steps below to get up and running with Llama 3 models quickly. These steps will let you run quick inference locally. For more examples, see the [Llama recipes repository](https://github.com/facebookresearch/llama-recipes).

1. Clone and download this repository in a conda env with PyTorch / CUDA.

2. In the top-level directory run:
    ```bash
    pip install -e .
    ```
3. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and register to download the model/s.

4. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

5. Once you get the email, navigate to your downloaded llama repository and run the download.sh script.
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email.
    - Do not use the “Copy Link” option; copy the link from the email manually.

6. Once the model/s you want have been downloaded, you can run the model locally using the command below:
```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```
**Note**
- Replace  `Meta-Llama-3-8B-Instruct/` with the path to your checkpoint directory and `Meta-Llama-3-8B-Instruct/tokenizer.model` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.
- This example runs the [example_chat_completion.py](example_chat_completion.py) found in this repository, but you can change that to a different .py file.

## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 8B     | 1  |
| 70B    | 8  |

All models support sequence length up to 8192 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware.

### Pretrained Models

These models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

See `example_text_completion.py` for some examples. To illustrate, see the command below to run it with the llama-3-8b model (`nproc_per_node` needs to be set to the `MP` value):

```
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir Meta-Llama-3-8B/ \
    --tokenizer_path Meta-Llama-3-8B/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

### Instruction-tuned Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, specific formatting defined in [`ChatFormat`](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202)
needs to be followed: The prompt begins with a `<|begin_of_text|>` special token, after which one or more messages follow. Each message starts with the `<|start_header_id|>` tag, the role `system`, `user` or `assistant`, and the `<|end_header_id|>` tag. After a double newline `\n\n`, the message's contents follow. The end of each message is marked by the `<|eot_id|>` token.

You can also deploy additional classifiers to filter out inputs and outputs that are deemed unsafe. See the llama-recipes repo for [an example](https://github.com/meta-llama/llama-recipes/blob/main/recipes/inference/local_inference/inference.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using llama-3-8b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

Llama 3 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
To help developers address these risks, we have created the [Responsible Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/).

## Issues

Please report any software “bug” or other problems with the models through one of the following means:
- Reporting issues with the model: [https://github.com/meta-llama/llama3/issues](https://github.com/meta-llama/llama3/issues)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md).

## License

Our model and weights are licensed for researchers and commercial entities, upholding the principles of openness. Our mission is to empower individuals and industry through this opportunity while fostering an environment of discovery and ethical AI advancements.

See the [LICENSE](LICENSE) file, as well as our accompanying [Acceptable Use Policy](USE_POLICY.md)

## Questions

For common questions, the FAQ can be found [here](https://llama.meta.com/faq), which will be updated over time as new questions arise.
