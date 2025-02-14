from smolagents import OpenAIServerModel, CodeAgent

# Define local model names
reasoning_model_id = "deepseek-r1:32b"  # Use DeepSeek for reasoning


def get_model(model_id):
    """Returns an Ollama model."""
    return OpenAIServerModel(
        model_id=model_id,
        api_base="http://ollama.service.prd.uit-auto-nemo.nvidia.com:8080/v1",  # Ollama API endpoint
        api_key="ollama"
    )


# Create reasoning model using DeepSeek
reasoning_model = get_model(reasoning_model_id)

# Create reasoner agent
reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)
