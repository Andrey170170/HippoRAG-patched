from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .Cohere import CohereEmbeddingModel
from .Transformers import TransformersEmbeddingModel
from .VLLM import VLLMEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(
    embedding_model_name: str = "nvidia/NV-Embed-v2", global_config=None
):
    if global_config is not None and hasattr(
        global_config, "resolved_embedding_provider"
    ):
        embedding_provider = global_config.resolved_embedding_provider()
    else:
        embedding_provider = getattr(global_config, "embedding_provider", "auto")

    if embedding_provider in {"openrouter", "openai", "azure", "openai_compatible"}:
        return OpenAIEmbeddingModel
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        return VLLMEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
