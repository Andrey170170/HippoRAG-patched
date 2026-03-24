import functools
import hashlib
import json
import os
import sqlite3
from dataclasses import asdict
from copy import deepcopy
from typing import List, Tuple
from urllib.parse import parse_qs, urlparse

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        llm_config = getattr(self, "llm_config", None)
        gen_params = getattr(llm_config, "generate_params", {})
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
            "llm_provider": self.global_config.resolved_llm_provider()
            if getattr(self, "global_config", None)
            else None,
            "llm_base_url": getattr(self.global_config, "llm_base_url", None),
            "azure_endpoint": getattr(self.global_config, "azure_endpoint", None),
            "llm_api_version": getattr(self.global_config, "llm_api_version", None),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key_hash, message, metadata_str),
            )
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)

    return wrapper


class CacheOpenAI(BaseLLM):
    """OpenAI-compatible LLM implementation."""

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        cache_dir = os.path.join(global_config.save_dir or "outputs", "llm_cache")
        return cls(
            cache_dir=cache_dir,
            global_config=global_config,
            max_retries=global_config.max_retry_attempts,
        )

    def __init__(
        self,
        cache_dir,
        global_config,
        cache_filename: str | None = None,
        high_throughput: bool = True,
        **kwargs,
    ) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url
        self.llm_provider = global_config.resolved_llm_provider()

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.global_config.llm_runtime_label()}_cache.sqlite"
            legacy_cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
            legacy_cache_path = os.path.join(self.cache_dir, legacy_cache_filename)
            if not os.path.exists(
                os.path.join(self.cache_dir, cache_filename)
            ) and os.path.exists(legacy_cache_path):
                logger.info(f"Using legacy LLM cache file: {legacy_cache_path}")
                cache_filename = legacy_cache_filename
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(
                limits=limits, timeout=httpx.Timeout(5 * 60, read=5 * 60)
            )
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", 2)

        self.openai_client = self._build_client(client)

    def _extract_api_version(
        self, endpoint: str, configured_version: str | None = None
    ) -> str | None:
        if configured_version:
            return configured_version

        parsed = urlparse(endpoint)
        versions = parse_qs(parsed.query).get("api-version")
        return versions[0] if versions else None

    def _build_client(self, client: httpx.Client | None):
        api_key = self.global_config.resolve_llm_api_key()
        default_headers = self.global_config.llm_headers or None

        if self.global_config.resolved_llm_provider() == "azure":
            if self.global_config.azure_endpoint is None:
                raise ValueError(
                    "azure_endpoint must be configured when llm_provider='azure'."
                )
            api_version = self._extract_api_version(
                self.global_config.azure_endpoint,
                self.global_config.llm_api_version,
            )
            return AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=self.global_config.azure_endpoint,
                max_retries=self.max_retries,
            )

        if (
            api_key is None
            and self.global_config.resolved_llm_provider() == "openai_compatible"
            and self.llm_base_url
            and any(host in self.llm_base_url for host in ["localhost", "127.0.0.1"])
        ):
            api_key = "sk-"

        if api_key is None and self.global_config.resolved_llm_provider() in {
            "openrouter",
            "openai",
        }:
            env_name = self.global_config.llm_api_key_env or (
                "OPENROUTER_API_KEY"
                if self.global_config.resolved_llm_provider() == "openrouter"
                else "OPENAI_API_KEY"
            )
            raise ValueError(
                f"Missing API key for provider '{self.global_config.resolved_llm_provider()}'. "
                f"Set {env_name} or provide llm_api_key in config."
            )

        return OpenAI(
            api_key=api_key,
            base_url=self.llm_base_url,
            default_headers=default_headers,
            http_client=client,
            max_retries=self.max_retries,
        )

    def _init_llm_config(self) -> None:
        config_dict = asdict(self.global_config)

        config_dict["llm_name"] = self.global_config.llm_name
        config_dict["llm_base_url"] = self.global_config.llm_base_url
        config_dict["generate_params"] = {
            "model": self.global_config.llm_name,
            "max_completion_tokens": config_dict.get("max_new_tokens", 400),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
            "temperature": config_dict.get("temperature", 0.0),
        }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self, messages: List[TextChatMessage], **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI-compatible GPT API with:\n{params}")

        uses_native_openai_endpoint = (
            self.global_config.resolved_llm_provider() == "openai"
            and self.llm_base_url == "https://api.openai.com/v1"
        )
        if not uses_native_openai_endpoint or version.parse(
            openai.__version__
        ) < version.parse("1.45.0"):
            params["max_tokens"] = params.pop("max_completion_tokens")

        response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata
