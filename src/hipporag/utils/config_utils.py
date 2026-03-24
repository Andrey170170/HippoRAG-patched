import hashlib
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Literal, Mapping, Optional, Union

import yaml

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""

    # LLM specific attributes
    llm_provider: Literal[
        "auto",
        "openrouter",
        "openai",
        "azure",
        "openai_compatible",
        "bedrock",
        "transformers",
    ] = field(
        default="auto",
        metadata={
            "help": "Provider for the LLM backend. `auto` infers from the model name and endpoint configuration."
        },
    )
    llm_name: str = field(
        default="openai/gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."},
    )
    llm_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base URL for the LLM model, if none, means using OPENAI service."
        },
    )
    llm_api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Explicit API key for the LLM provider. Prefer environment variables for secrets when possible."
        },
    )
    llm_api_key_env: Optional[str] = field(
        default=None,
        metadata={
            "help": "Environment variable name from which to read the LLM API key."
        },
    )
    llm_headers: Dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": "Optional default HTTP headers passed to OpenAI-compatible LLM clients."
        },
    )
    llm_api_version: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional API version override for Azure OpenAI style endpoints."
        },
    )
    embedding_provider: Literal[
        "auto",
        "openrouter",
        "openai",
        "azure",
        "openai_compatible",
        "local",
        "cohere",
        "transformers",
        "vllm",
    ] = field(
        default="auto",
        metadata={
            "help": "Provider for the embedding backend. `auto` infers from the model name and endpoint configuration."
        },
    )
    embedding_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."
        },
    )
    embedding_api_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "Explicit API key for the embedding provider. Prefer environment variables for secrets when possible."
        },
    )
    embedding_api_key_env: Optional[str] = field(
        default=None,
        metadata={
            "help": "Environment variable name from which to read the embedding API key."
        },
    )
    embedding_api_version: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional API version override for Azure OpenAI style embedding endpoints."
        },
    )
    azure_endpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Azure Endpoint URI for the LLM model, if none, uses OPENAI service directly."
        },
    )
    azure_embedding_endpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Azure Endpoint URI for the OpenAI embedding model, if none, uses OPENAI service directly."
        },
    )
    max_new_tokens: Union[None, int] = field(
        default=2048, metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={
            "help": "How many chat completion choices to generate for each input message."
        },
    )
    seed: Union[None, int] = field(default=None, metadata={"help": "Random seed."})
    temperature: float = field(
        default=0, metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: {"type": "json_object"},
        metadata={"help": "Specifying the format that the model must output."},
    )

    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={
            "help": "Max number of retry attempts for an asynchronous API calling."
        },
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={
            "help": "If set to True, will ignore all existing openie files and rebuild them from scratch."
        },
    )

    # Storage specific attributes
    force_index_from_scratch: bool = field(
        default=False,
        metadata={
            "help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."
        },
    )
    rerank_dspy_file_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={
            "help": "Multiplicative factor that modified the passage node weights in PPR."
        },
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."},
    )

    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={
            "help": "Name of the text-based preprocessor to use in preprocessing."
        },
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={
            "help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."
        },
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."},
    )
    preprocess_chunk_max_token_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."
        },
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default="by_token")

    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt",] = field(
        default="openie_openai_gpt",
        metadata={
            "help": "Class name indicating which information extraction model to use."
        },
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online", metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."
        },
    )

    # Embedding specific attributes
    embedding_model_name: str = field(
        default="openai/text-embedding-3-small",
        metadata={"help": "Class name indicating which embedding model to use."},
    )
    embedding_batch_size: int = field(
        default=16, metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True, metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048, metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto", metadata={"help": "Data type for local embedding model."}
    )

    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."},
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={
            "help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."
        },
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={
            "help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."
        },
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."},
    )
    is_directed_graph: bool = field(
        default=False, metadata={"help": "Whether the graph is directed or not."}
    )

    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"},
    )
    retrieval_top_k: int = field(
        default=200, metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5, metadata={"help": "Damping factor for ppr algorithm."}
    )

    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={
            "help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."
        },
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."},
    )

    # Save dir (highest level directory)
    save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."
        },
    )

    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[
        Literal["hotpotqa", "hotpotqa_train", "musique", "2wikimultihopqa"]
    ] = field(
        default=None,
        metadata={
            "help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."
        },
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        "dpr_only",
        "entity",
        "passage_entity",
        "relation_aware_passage_entity",
        "passage_entity_relation",
        "facts_and_sim_passage_node_unidirectional",
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."},
    )
    corpus_len: Optional[int] = field(
        default=None, metadata={"help": "Length of the corpus to use."}
    )

    def __post_init__(self):
        self._normalize_provider_config()
        if self.save_dir is None:  # If save_dir not given
            if self.dataset is None:
                self.save_dir = "outputs"  # running freely
            else:
                self.save_dir = os.path.join(
                    "outputs", self.dataset
                )  # customize your dataset's output dir here
        logger.debug(
            f"Initializing the highest level of save_dir to be {self.save_dir}"
        )

    @classmethod
    def _field_names(cls) -> set[str]:
        return {field_info.name for field_info in fields(cls)}

    @classmethod
    def _normalize_config_dict(cls, raw_config: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = dict(raw_config)

        section_mappings = {
            "llm": {
                "provider": "llm_provider",
                "model": "llm_name",
                "name": "llm_name",
                "base_url": "llm_base_url",
                "api_key": "llm_api_key",
                "api_key_env": "llm_api_key_env",
                "headers": "llm_headers",
                "api_version": "llm_api_version",
                "azure_endpoint": "azure_endpoint",
            },
            "embedding": {
                "provider": "embedding_provider",
                "model": "embedding_model_name",
                "name": "embedding_model_name",
                "base_url": "embedding_base_url",
                "api_key": "embedding_api_key",
                "api_key_env": "embedding_api_key_env",
                "api_version": "embedding_api_version",
                "azure_endpoint": "azure_embedding_endpoint",
                "batch_size": "embedding_batch_size",
            },
            "runtime": None,
            "retrieval": None,
            "qa": None,
            "graph": None,
        }

        for section_name, mapping in section_mappings.items():
            section = normalized.pop(section_name, None)
            if section is None:
                continue
            if not isinstance(section, Mapping):
                raise ValueError(f"Config section '{section_name}' must be a mapping.")

            for key, value in section.items():
                target_key = mapping.get(key, key) if mapping is not None else key
                normalized[target_key] = value

        unknown_keys = sorted(set(normalized) - cls._field_names())
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

        return normalized

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> "BaseConfig":
        normalized = cls._normalize_config_dict(config_dict)
        return cls(**normalized)

    @classmethod
    def load_yaml_dict(cls, file_path: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}

        if not isinstance(loaded, Mapping):
            raise ValueError("YAML config must contain a mapping at the top level.")

        return cls._normalize_config_dict(loaded)

    @classmethod
    def from_yaml(cls, file_path: str) -> "BaseConfig":
        return cls.from_dict(cls.load_yaml_dict(file_path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def merged_with(self, overrides: Mapping[str, Any]) -> "BaseConfig":
        normalized = self._normalize_config_dict(overrides)
        base_dict = self.to_dict()
        base_dict.update(
            {key: value for key, value in normalized.items() if value is not None}
        )
        return self.__class__(**base_dict)

    def _normalize_provider_config(self) -> None:
        resolved_llm_provider = self.resolved_llm_provider()
        resolved_embedding_provider = self.resolved_embedding_provider()

        if self.llm_base_url is None:
            if resolved_llm_provider == "openrouter":
                self.llm_base_url = "https://openrouter.ai/api/v1"
            elif resolved_llm_provider == "openai":
                self.llm_base_url = "https://api.openai.com/v1"

        if self.embedding_base_url is None:
            if resolved_embedding_provider == "openai":
                self.embedding_base_url = "https://api.openai.com/v1"
            elif resolved_embedding_provider == "openrouter":
                self.embedding_base_url = "https://openrouter.ai/api/v1"

    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider

        if self.llm_name.startswith("bedrock"):
            return "bedrock"
        if self.llm_name.startswith("Transformers/"):
            return "transformers"
        if self.azure_endpoint is not None:
            return "azure"
        if self.llm_base_url:
            if "openrouter.ai" in self.llm_base_url:
                return "openrouter"
            if "api.openai.com" in self.llm_base_url:
                return "openai"
            return "openai_compatible"
        if "/" in self.llm_name:
            return "openrouter"
        return "openai"

    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider

        if self.azure_embedding_endpoint is not None:
            return "azure"
        if self.embedding_base_url:
            if "openrouter.ai" in self.embedding_base_url:
                return "openrouter"
            if "api.openai.com" in self.embedding_base_url:
                return "openai"
            return "openai_compatible"
        if self.embedding_model_name.startswith("Transformers/"):
            return "transformers"
        if self.embedding_model_name.startswith("VLLM/"):
            return "vllm"
        if "cohere" in self.embedding_model_name:
            return "cohere"
        if self.embedding_model_name.startswith("text-embedding"):
            return "openai"
        if "/" in self.embedding_model_name:
            return "openrouter"
        return "local"

    def resolve_llm_api_key(self) -> Optional[str]:
        if self.llm_api_key:
            return self.llm_api_key

        env_name = self.llm_api_key_env
        if env_name is None:
            env_name = {
                "openrouter": "OPENROUTER_API_KEY",
                "openai": "OPENAI_API_KEY",
                "azure": "AZURE_OPENAI_API_KEY",
                "openai_compatible": "OPENAI_API_KEY",
            }.get(self.resolved_llm_provider())

        return os.getenv(env_name) if env_name else None

    def resolve_embedding_api_key(self) -> Optional[str]:
        if self.embedding_api_key:
            return self.embedding_api_key

        env_name = self.embedding_api_key_env
        if env_name is None:
            env_name = {
                "openrouter": "OPENROUTER_API_KEY",
                "openai": "OPENAI_API_KEY",
                "azure": "AZURE_OPENAI_API_KEY",
                "openai_compatible": "OPENAI_API_KEY",
            }.get(self.resolved_embedding_provider())

        return os.getenv(env_name) if env_name else None

    @staticmethod
    def _sanitize_label(value: str) -> str:
        sanitized = []
        for char in value:
            sanitized.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
        return "".join(sanitized)

    @staticmethod
    def _short_hash(value: Optional[str]) -> str:
        if not value:
            return "default"
        return hashlib.md5(value.encode("utf-8")).hexdigest()[:10]

    def llm_runtime_label(self) -> str:
        endpoint_reference = (
            self.azure_endpoint
            if self.resolved_llm_provider() == "azure"
            else self.llm_base_url
        )
        identity = f"{endpoint_reference}|{self.llm_api_version}"
        return self._sanitize_label(
            f"{self.resolved_llm_provider()}_{self.llm_name}_{self._short_hash(identity)}"
        )

    def embedding_runtime_label(self) -> str:
        endpoint_reference = (
            self.azure_embedding_endpoint
            if self.resolved_embedding_provider() == "azure"
            else self.embedding_base_url
        )
        identity = f"{endpoint_reference}|{self.embedding_api_version}"
        return self._sanitize_label(
            f"{self.resolved_embedding_provider()}_{self.embedding_model_name}_{self._short_hash(identity)}"
        )
