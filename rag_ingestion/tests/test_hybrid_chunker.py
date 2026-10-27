import pytest
from ..chunking.hybrid_chunker import chunk_text, count_tokens
from .. import config

def test_forced_cut():
    """Verify that chunks are cut when max token limit is reached."""
    # Create a long text
    long_sentence = "This is a sentence. " * 50 # ~250 tokens
    text = long_sentence * 2 # ~500 tokens
    
    chunks = chunk_text(text, "doc_1", 1)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk.content) <= config.MAX_CHUNK_TOKENS + 50 # allowing some margin for exact sentence boundaries
