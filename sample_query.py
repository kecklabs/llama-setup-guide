from llama_cpp import Llama

llm = Llama(
    model_path="PATH_TO_MODEL",  # Replace with path to your model file!
    n_ctx=4096,  # Context window size
    n_gpu_layers=-1  # Use all GPU layers
)

output = llm(
    "Q: Name the planets in the solar system? A: ",  # Prompt
    max_tokens=32,  # Generate up to 32 tokens
    stop=["Q:", "\n"],  # Keywords to stop generation at
    echo=True  # Echo the prompt back in the output
)

print(output)  # Print the generated output
