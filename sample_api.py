from llama_cpp import Llama
import flask

# Create the Flask app
app = flask.Flask(__name__)

# Load the model
llm = Llama(
    model_path="orca_mini_v3_7b.Q4_K_M.gguf",  # Path to model file
    n_ctx=4096,  # Context window size
    n_gpu_layers=-1,  # -1 to use all GPU layers, 0 to use only CPU
    verbose=False  # Whether to print debug info
)


@app.route('/chat', methods=['POST'])
def chat():
    """
    A simple endpoint for querying the model in a question-answer format.
    """
    data = flask.request.json

    # Read any parameters you'd like to include in the query
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 50)  # optional field with default

    # Run the query
    output = llm(
        f'Q: {prompt} A: ',  # format as question-answer
        max_tokens=max_tokens,
        stop=["Q:", "\n"]
    )

    response = output['choices'][0]['text']

    return flask.jsonify({'prompt': prompt, 'response': response})


if __name__ == '__main__':
    app.run()
