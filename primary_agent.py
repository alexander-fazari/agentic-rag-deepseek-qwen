# primary_agent.py
from smolagents import OpenAIServerModel, ToolCallingAgent
from rag_tool import rag_with_reasoner  # Import the tool function from rag_tool.py


def get_model(model_id):
    """Returns an Ollama model."""
    return OpenAIServerModel(
        model_id=model_id,
        api_base="http://localhost:11434/v1",  # Ollama API endpoint
        api_key="ollama"
    )


# Load tool model (Qwen-14B)
primary_model = get_model("qwen2.5:7b-instruct-8k")

# Create primary agent using Qwen for tool responses
primary_agent = ToolCallingAgent(tools=[rag_with_reasoner], model=primary_model, add_base_tools=False, max_steps=3)


# Export the agent to be used in the app
def get_primary_agent():
    return primary_agent
