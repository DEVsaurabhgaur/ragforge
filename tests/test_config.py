"""
tests/test_config.py — Smoke tests for RAGForge configuration defaults.
"""
import config


def test_chunk_size_positive():
    """Verify CHUNK_SIZE is configured as a positive integer."""
    assert config.CHUNK_SIZE > 0


def test_chunk_overlap_less_than_size():
    """Verify CHUNK_OVERLAP is strictly smaller than CHUNK_SIZE."""
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_top_k_positive():
    """Verify TOP_K_RESULTS is configured as a positive integer."""
    assert config.TOP_K_RESULTS > 0


def test_supported_extensions_includes_pdf():
    """Verify PDF extension is in the supported extension list."""
    assert ".pdf" in config.SUPPORTED_EXTENSIONS


def test_supported_extensions_includes_txt():
    """Verify TXT extension is in the supported extension list."""
    assert ".txt" in config.SUPPORTED_EXTENSIONS


def test_system_presets_not_empty():
    """Verify system presets dictionary contains configured modes."""
    assert len(config.SYSTEM_PRESETS) > 0


def test_default_system_prompt_not_empty():
    """Verify default system prompt is a non-empty string."""
    assert len(config.DEFAULT_SYSTEM_PROMPT) > 0


def test_retrieval_mode_valid():
    """Verify retrieval mode is set to hybrid or semantic."""
    assert config.RETRIEVAL_MODE in ("hybrid", "semantic")


def test_temperature_in_range():
    assert 0.0 <= config.DEFAULT_TEMPERATURE <= 1.0


def test_collection_name_set():
    assert config.COLLECTION_NAME != ""


def test_app_version_set():
    assert hasattr(config, "APP_VERSION")
    assert config.APP_VERSION != ""


def test_max_file_size_positive():
    assert hasattr(config, "MAX_FILE_SIZE_MB")
    assert config.MAX_FILE_SIZE_MB > 0


def test_max_documents_positive():
    assert hasattr(config, "MAX_DOCUMENTS")
    assert config.MAX_DOCUMENTS > 0


def test_technical_analyst_preset_exists():
    assert "Technical Analyst" in config.SYSTEM_PRESETS


def test_eli5_preset_exists():
    assert "ELI5 Explainer" in config.SYSTEM_PRESETS


def test_all_preset_values_nonempty():
    for key, value in config.SYSTEM_PRESETS.items():
        assert len(value) > 0, f"Preset '{key}' is empty"


def test_validate_config_invalid_parameters():
    import pytest
    # Save original values
    orig_provider = config.LLM_PROVIDER
    orig_chunk_size = config.CHUNK_SIZE
    orig_overlap = config.CHUNK_OVERLAP

    try:
        # Test invalid provider
        config.LLM_PROVIDER = "invalid-provider"
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            config.validate_config()
        config.LLM_PROVIDER = orig_provider

        # Test invalid chunk size
        config.CHUNK_SIZE = -10
        with pytest.raises(ValueError, match="CHUNK_SIZE"):
            config.validate_config()
        config.CHUNK_SIZE = orig_chunk_size

        # Test overlap >= size
        config.CHUNK_SIZE = 500
        config.CHUNK_OVERLAP = 600
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            config.validate_config()
    finally:
        # Ensure we restore original values
        config.LLM_PROVIDER = orig_provider
        config.CHUNK_SIZE = orig_chunk_size
        config.CHUNK_OVERLAP = orig_overlap
        config.validate_config()
