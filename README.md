# LLaMA-2 Setup Guide

A guide for setting up the LLaMA 2 LLM on a local ubuntu machine.

- [LLaMA-2 Setup Guide](#llama-2-setup-guide)
  - [Prerequisites](#prerequisites)
  - [Background](#background)
    - [Model Sizes](#model-sizes)
    - [Model Formats](#model-formats)
    - [CPU vs GPU Memory](#cpu-vs-gpu-memory)
  - [Obtaining a Model](#obtaining-a-model)
    - [Recommended GGUF Models](#recommended-gguf-models)
    - [Converting Transformers Models to GGUF Format](#converting-transformers-models-to-gguf-format)
  - [Setting Up GPU Support](#setting-up-gpu-support)
    - [CUDA Installation Troubleshooting](#cuda-installation-troubleshooting)
  - [Running Your Model](#running-your-model)
    - [Model Query Troubleshooting](#model-query-troubleshooting)
  - [Setting Up an API Server](#setting-up-an-api-server)
    - [Remote API Access](#remote-api-access)
  - [Integration with Microsoft Guidance](#integration-with-microsoft-guidance)
  - [Resources](#resources)

## Prerequisites

You'll need a capable machine that:

- Runs Ubuntu or some other Linux distribution
- Has enough RAM to fit any models you want to use in memory
  - GPU memory (VRAM) is highly preferred (see [CPU vs GPU Memory](#cpu-vs-gpu-memory))
- Has sufficient local storage to store models
- Has Python 3.8 or higher installed

## Background

### Model Sizes

LLaMA 2 is available in three sizes: 7B, 13B, and 70B. These numbers refer to the number of parameters in the model. Generally, smaller models are much faster but less accurate, while larger models are slower but more accurate.

In my experience, the smaller models are sufficient for question answering and text generation, but struggle with more complex tasks like logic puzzles and decision making based on evidence. I recommend trying out various models to see what works best for your use case.

### Model Formats

You might come across a few different formats of LLaMA2 models online, including transformers, GPTQ, GGML, GGUF, and others. Each of these has a different setup process and requirements (which I won't include here). I reccomend using **GGUF** models via **llama.cpp** because:

- They support splitting models between CPU and GPU memory (see [CPU vs GPU Memory](#cpu-vs-gpu-memory))
- They support quantization, which greatly reduces the memory footprint with minimal accuracy loss
- Easy setup and transportability (since they are just a single file)

### CPU vs GPU Memory

When running any LLM, the entire model must be loaded into the memory of the machine. Every model has different memory requirements, typically listed somewhere in the model's documentation. The larger the model, the more memory it requires.

There are two types of memory available on most machines: CPU (RAM) and GPU (VRAM). CPU memory is typically much larger than GPU memory, but is also significantly slower. I highly recommend running your model on a GPU if possible.

One benefit of using a GGUF model is that you can split the model between CPU and GPU memory. For instance, if you are running a 20GB model on a system with only 12GB of VRAM, you can split the model into roughly 12GB on the GPU and 8GB on the CPU. It will be slower than running the entire model on the GPU, but much faster than running the entire model on the CPU.

## Obtaining a Model

[HuggingFace](https://huggingface.co/) is a great resource for finding and downloading models. You can find the list of all Llama 2 models [here](https://huggingface.co/models?other=llama-2). Remember that there are multiple model formats, so look for GGUF if you want to use llama.cpp.

GGUF models often have multiple downloads for different quantization levels (`Q4_K_M`, `Q2_K`, etc). This affects the accuracy and memory footprint of the model. I recommend starting with `Q4_K_M` for balanced size and quality.

### Recommended GGUF Models

- [orca_mini_v3_7B](https://huggingface.co/TheBloke/orca_mini_v3_7B-GGUF) is a great starting point and one of the best 7B models I've found. It also comes in 13B and 70B sizes if you need better accuracy.
- [Upstage-Llama-2-70B-instruct-v2-GGUF](https://huggingface.co/TheBloke/Upstage-Llama-2-70B-instruct-v2-GGUF) is one of the top ranked 70B models on huggingface and has been effective for decision making and other complex tasks.

### Converting Transformers Models to GGUF Format

Many models on HuggingFace are only available in the transformers (hf) format. They can be converted to the GGUF format by following [this guide](https://www.substratus.ai/blog/converting-hf-model-gguf-model/).

## Setting Up GPU Support

This is optional if you only plan to run models on the CPU (not recommended). Make sure your GPU is CUDA compatible before following these steps. You can check your GPU's compatibility [here](https://developer.nvidia.com/cuda-gpus).

First, check if you have CUDA installed already:

```bash
nvcc --version
```

If you get a "command not found" error, you'll need to install the CUDA toolkit from [here](https://developer.nvidia.com/cuda-downloads). Once installed, `nvcc --version` should return a version number.

### CUDA Installation Troubleshooting

If you still get an error running `nvcc --version` after installation, you may need to manually add CUDA to your path. Check out [this](https://askubuntu.com/a/885627) post for instructions.

## Running Your Model

[!] I highly recommend using a virtual environment for this installation (either `conda` or `venv`). Documentation on this can be found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

We will use the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) package to run our model:

1. Install the `llama-cpp-python` package in your virtual environment:

```bash
# Compile with CUDA support
CMAKE_ARGS=-DLLAMA_CUBLAS=on FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

2. Create a python script to load and query your model. See this example (also in `sample_query.py`):

```python
from llama_cpp import Llama

llm = Llama(
    model_path="orca_mini_v3_7b.Q4_K_M.gguf",  # Path to model file
    n_ctx=4096,  # Context window size
    n_gpu_layers=-1,  # -1 to use all GPU layers, 0 to use only CPU
    verbose=False  # Whether to print debug info
)

output = llm(
    "Q: Name the planets in the solar system? A: ",  # Prompt
    max_tokens=50,  # Generate up to 32 tokens
    stop=["Q:", "\n"],  # Keywords to stop generation at
    echo=True  # Echo the prompt back in the output
)

print(output['choices'][0]['text'])  # Print the generated output
```

Remember to replace `PATH_TO_MODEL` with the path to your model file (from this script).

3. Run the script!

```bash
python sample_query.py
```

You should see an output similar to this:

```text
Q: Name the planets in the solar system? A: 8 Planets in our Solar System are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune
```

Note that this may take a while to run since the model must be loaded into memory. Once the model is loaded (by calling `Llama()`), subsequent calls to `llm()` will be much faster.

To ensure you are using GPU memory, you can run with `verbose=True` to verify that layers have been loaded onto the GPU's VRAM via CUDA. You can also run `nvidia-smi -l 1` in a separate terminal to monitor GPU memory usage (VRAM usage should go up when you run it).

### Model Query Troubleshooting

You may see `CUDA error: out of memory` if the model is too large to fit in GPU memory. There are a few ways to work around this:

- Split layers between the CPU and GPU by modifying the `n_gpu_layers` parameter in the code. The `verbose=True` output should show the total number of layers the model has. Try setting `n_gpu_layers` to a number less than this.
- Use a model with a smaller quantization level (e.g. `Q2_K` instead of `Q4_K_M`)
- Use a smaller model (e.g. 7B instead of 13B)

## Setting Up an API Server

While you could run and query your model from directly within your application, it is often more convenient to run the model on a separate server and query it via an API (similar to how OpenAI's API works). There are many possible ways to do this, but a simple solution is a basic Flask API. A sample of this can be found in `sample_api.py`.

To use this, simply `pip install flask`, run the script, and send a POST request to `http://127.0.0.1:5000` with the following JSON body:

```json
{
  "prompt": "Tell me a fun fact about dogs",
  "max_tokens": 50,
}
```

### Remote API Access

By default, your Flask API will only be accessible from your local machine. You can use a ervice like `ngrok` to expose your API to external requests. See [the docs](https://ngrok.com/docs/using-ngrok-with/flask/) for more information.

## Integration with Microsoft Guidance

TODO

## Resources

TODO
