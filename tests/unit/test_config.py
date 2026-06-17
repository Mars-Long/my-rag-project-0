"""Tests for the configuration system (src/rag0/config.py)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from rag0.config import RagConfig, get_config
from rag0.exceptions import ConfigurationError


class TestRagConfig:
    """Test the top-level configuration loading and validation."""

    def test_defaults_load_without_yaml(self) -> None:
        """Config should load with sensible defaults when no YAML file exists."""
        with patch.dict(os.environ, {}, clear=True):
            # Point to a non-existent path
            os.environ["RAG0_CONFIG_PATH"] = "/nonexistent/config.yaml"
            config = RagConfig()
            assert config.llm.model_name == "deepseek-chat"
            assert config.embedding.dimensions == 1024
            assert config.vector_store.host == "127.0.0.1"
            assert config.vector_store.port == 19530
            assert config.splitter.chunk_size == 500
            assert config.server.port == 7861
            assert config.telemetry.enabled is False

    def test_yaml_overrides_defaults(self) -> None:
        """YAML config values should override defaults."""
        yaml_content = {
            "llm": {"model_name": "custom-model"},
            "splitter": {"chunk_size": 1000},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(yaml_content, f)
            config_path = f.name

        try:
            with patch.dict(os.environ, {"RAG0_CONFIG_PATH": config_path}, clear=True):
                config = RagConfig()
                assert config.llm.model_name == "custom-model"
                assert config.splitter.chunk_size == 1000
                # Unchanged defaults
                assert config.vector_store.port == 19530
        finally:
            Path(config_path).unlink()

    def test_env_var_overrides_yaml(self) -> None:
        """Environment variables should take precedence over YAML values."""
        yaml_content = {"llm": {"model_name": "yaml-model"}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(yaml_content, f)
            config_path = f.name

        try:
            with patch.dict(
                os.environ,
                {
                    "RAG0_CONFIG_PATH": config_path,
                    "RAG0_LLM__MODEL_NAME": "env-model",
                },
                clear=True,
            ):
                config = RagConfig()
                # Env var should win
                assert config.llm.model_name == "env-model"
        finally:
            Path(config_path).unlink()

    def test_field_validation_rejects_invalid_port(self) -> None:
        """Field constraints (ge=1, le=65535) should reject invalid ports."""
        from pydantic import ValidationError as PydanticValidationError

        with patch.dict(
            os.environ,
            {"RAG0_SERVER__PORT": "99999"},
            clear=True,
        ):
            with pytest.raises(PydanticValidationError):
                RagConfig()

    def test_all_sub_configs_present(self) -> None:
        """All sub-config sections should be initialized."""
        config = RagConfig()
        assert config.llm is not None
        assert config.embedding is not None
        assert config.vector_store is not None
        assert config.database is not None
        assert config.splitter is not None
        assert config.reranker is not None
        assert config.server is not None
        assert config.telemetry is not None


class TestLLMConfig:
    """Test LLM-specific configuration."""

    def test_default_model_name(self) -> None:
        config = RagConfig()
        assert config.llm.model_name == "deepseek-chat"

    def test_temperature_bounds(self) -> None:
        config = RagConfig()
        # Valid range
        config.llm.temperature = 0.5
        assert config.llm.temperature == 0.5


class TestEmbeddingConfig:
    """Test embedding-specific configuration."""

    def test_default_dimensions(self) -> None:
        config = RagConfig()
        assert config.embedding.dimensions == 1024

    def test_batch_size_bounds(self) -> None:
        config = RagConfig()
        config.embedding.batch_size = 64
        assert config.embedding.batch_size == 64


class TestVectorStoreConfig:
    """Test vector store configuration."""

    def test_default_connection_params(self) -> None:
        config = RagConfig()
        assert config.vector_store.host == "127.0.0.1"
        assert config.vector_store.port == 19530
        assert config.vector_store.user == "root"

    def test_secure_disabled_by_default(self) -> None:
        config = RagConfig()
        assert config.vector_store.secure is False


class TestConfigSingleton:
    """Test the get_config() singleton accessor."""

    def test_get_config_returns_rag_config(self) -> None:
        # Clear the LRU cache
        get_config.cache_clear()
        config = get_config()
        assert isinstance(config, RagConfig)

    def test_get_config_is_cached(self) -> None:
        get_config.cache_clear()
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2  # Same instance
