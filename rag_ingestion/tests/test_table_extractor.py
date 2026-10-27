import pytest
from ..chunking.atomic_chunker import create_atomic_chunk

def test_atomic_chunk():
    """Verify that a table is wrapped as a single atomic chunk."""
    markdown_table = "| Col1 | Col2 |\n|---|---|\n| A | B |"
    chunk = create_atomic_chunk(markdown_table, "doc_1", 1, "table")
    
    assert chunk.atomic is True
    assert chunk.chunk_type == "table"
    assert chunk.content == markdown_table
