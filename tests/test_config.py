"""
tests/test_config.py — Smoke tests for RAGForge configuration defaults.
"""
import config


def test_chunk_size_positive():
    assert config.CHUNK_SIZE > 0


def test_chunk_overlap_less_than_size():
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_top_k_positive():
    assert config.TOP_K_RESULTS > 0


def test_supported_extensions_includes_pdf():
    assert ".pdf" in config.SUPPORTED_EXTENSIONS


def test_supported_extensions_includes_txt():
    assert ".txt" in config.SUPPORTED_EXTENSIONS


def test_system_presets_not_empty():
    assert len(config.SYSTEM_PRESETS) > 0


def test_default_system_prompt_not_empty():
    assert len(config.DEFAULT_SYSTEM_PROMPT) > 0


def test_retrieval_mode_valid():
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
