# __init__.py

from typing import Optional
from .core import Swarm as DefaultSwarm
from .core_zhipu import Swarm as ZhipuSwarm
from .types import *

# 修改配置以支持跨厂商对话
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
    "OpenAI": {  # 改名为更明确的 OpenAI
        "client_class": DefaultSwarm,
        "models": {
            "options": ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"],
            "default": "gpt-3.5-turbo"
        }
    }
}

# 添加存储不同厂商客户端的字典
_clients = {}


def create_swarm(client_type: str = "OpenAI", api_key: Optional[str] = None) -> DefaultSwarm | ZhipuSwarm:
    """
    Factory method to create a Swarm instance based on the specified client type.
    Stores the client instance for reuse.
    """
    if client_type not in CLIENT_CONFIG:
        raise ValueError(f"Unsupported client type: {client_type}. "
                         f"Supported types are: {list(CLIENT_CONFIG.keys())}")

    if api_key:
        client_class = CLIENT_CONFIG[client_type]["client_class"]
        _clients[client_type] = client_class(api_key=api_key)

    return _clients.get(client_type)


def get_client(client_type: str) -> Optional[DefaultSwarm | ZhipuSwarm]:
    """Get stored client instance for a specific type."""
    return _clients.get(client_type)


def get_model_config(client_type: str = "OpenAI"):
    """Get the model configuration for a specific client type."""
    if client_type not in CLIENT_CONFIG:
        raise ValueError(f"Unsupported client type: {client_type}. "
                         f"Supported types are: {list(CLIENT_CONFIG.keys())}")

    return CLIENT_CONFIG[client_type]["models"]