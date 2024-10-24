from typing import Optional
from .core import Swarm as DefaultSwarm
from .core_zhipu import Swarm as ZhipuSwarm
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result
)

# Define available client types and their configurations
CLIENT_CONFIG = {
    "Zhipu": {
        "client_class": ZhipuSwarm,
        "models": {
            "options": [
                "glm-4", "glm-4-plus", "glm-4-0520", "glm-4-air",
                "glm-4-airx", "glm-4-long", "glm-4-flash", "glm-4-flashx"
            ],
            "default": "glm-4"
        }
    },
    "Default": {
        "client_class": DefaultSwarm,
        "models": {
            "options": ["default-model-1", "default-model-2", "default-model-3"],
            "default": "default-model-1"
        }
    }
}


def create_swarm(client_type: str = "Default", api_key: Optional[str] = None) -> DefaultSwarm | ZhipuSwarm:
    """
    Factory method to create a Swarm instance based on the specified client type.
    """
    if client_type not in CLIENT_CONFIG:
        raise ValueError(f"Unsupported client type: {client_type}. "
                         f"Supported types are: {list(CLIENT_CONFIG.keys())}")

    client_class = CLIENT_CONFIG[client_type]["client_class"]
    return client_class(api_key=api_key)


def get_model_config(client_type: str = "Default"):
    """
    Get the model configuration for a specific client type.
    """
    if client_type not in CLIENT_CONFIG:
        raise ValueError(f"Unsupported client type: {client_type}. "
                         f"Supported types are: {list(CLIENT_CONFIG.keys())}")

    return CLIENT_CONFIG[client_type]["models"]


# Re-export types for convenience
__all__ = [
    'create_swarm',
    'get_model_config',
    'Agent',
    'AgentFunction',
    'ChatCompletionMessage',
    'ChatCompletionMessageToolCall',
    'Function',
    'Response',
    'Result'
]