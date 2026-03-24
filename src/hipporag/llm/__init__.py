import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM
from .bedrock_llm import BedrockLLM
from .transformers_llm import TransformersLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    resolved_provider = config.resolved_llm_provider()

    if (
        resolved_provider == "openai_compatible"
        and config.llm_base_url is not None
        and "localhost" in config.llm_base_url
        and os.getenv("OPENAI_API_KEY") is None
    ):
        os.environ["OPENAI_API_KEY"] = "sk-"

    if resolved_provider == "bedrock":
        return BedrockLLM(config)

    if resolved_provider == "transformers":
        return TransformersLLM(config)

    return CacheOpenAI.from_experiment_config(config)
