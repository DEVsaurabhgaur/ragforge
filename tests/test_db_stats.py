"""
tests/test_db_stats.py — Unit test for db_stats.py script.
"""
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from scripts.db_stats import main


def test_db_stats_no_db_plain():
    with patch("scripts.db_stats.vectorstore_exists", return_value=False), \
         patch("builtins.print") as mock_print:
        with patch.object(sys, "argv", ["db_stats.py"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0
            mock_print.assert_called_with("ChromaDB vector store does not exist on disk yet.")


def test_db_stats_no_db_json():
    with patch("scripts.db_stats.vectorstore_exists", return_value=False), \
         patch("builtins.print") as mock_print:
        with patch.object(sys, "argv", ["db_stats.py", "--json"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code == 0
            mock_print.assert_called_once()
            printed_arg = mock_print.call_args[0][0]
            parsed = json.loads(printed_arg)
            assert parsed["exists"] is False
            assert parsed["total_chunks"] == 0
            assert parsed["documents"] == []


def test_db_stats_with_db_json():
    mock_vs = MagicMock()
    mock_vs.get.return_value = {
        "ids": ["id1", "id2"],
        "metadatas": [{"source_file": "doc1.pdf"}, {"source_file": "doc1.pdf"}]
    }

    with patch("scripts.db_stats.vectorstore_exists", return_value=True), \
         patch("scripts.db_stats.load_existing_vectorstore", return_value=mock_vs), \
         patch("builtins.print") as mock_print:
        with patch.object(sys, "argv", ["db_stats.py", "--json"]):
            main()
            # print should display JSON output
            mock_print.assert_called_once()
            printed_arg = mock_print.call_args[0][0]
            parsed = json.loads(printed_arg)
            assert parsed["exists"] is True
            assert parsed["total_chunks"] == 2
            assert len(parsed["documents"]) == 1
            assert parsed["documents"][0]["document"] == "doc1.pdf"
            assert parsed["documents"][0]["chunks"] == 2


def test_db_stats_with_db_plain():
    mock_vs = MagicMock()
    mock_vs.get.return_value = {
        "ids": ["id1", "id2"],
        "metadatas": [{"source_file": "doc1.pdf"}, {"source_file": "doc1.pdf"}]
    }

    with patch("scripts.db_stats.vectorstore_exists", return_value=True), \
         patch("scripts.db_stats.load_existing_vectorstore", return_value=mock_vs), \
         patch("builtins.print") as mock_print:
        with patch.object(sys, "argv", ["db_stats.py"]):
            main()
            # Verify multiple print calls are made for headers and docs
            assert mock_print.call_count >= 5
            printed_calls = [call[0][0] for call in mock_print.call_args_list]
            assert any("RAGForge ChromaDB Stats" in line for line in printed_calls)
            assert any("doc1.pdf: 2 chunks" in line for line in printed_calls)
