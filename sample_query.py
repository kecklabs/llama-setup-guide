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
