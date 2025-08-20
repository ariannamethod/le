import torch
from memory import Memory
from le import _fallback_reply, ModelConfig, Transformer, CharDataset
import metrics


def test_fallback_reply_basic_functionality():
    """Test that _fallback_reply generates a valid response."""
    torch.manual_seed(42)
    words = ["hello", "world", "test", "fallback"]
    chars = sorted(set(''.join(words)))
    dataset = CharDataset(words, chars, max(len(w) for w in words))
    config = ModelConfig(block_size=32, vocab_size=dataset.get_vocab_size(), n_layer=1, n_embd=32, n_head=4)
    model = Transformer(config)
    
    with Memory(":memory:") as mem:
        mem.record_message("hello", "hi")
        result = _fallback_reply("world", model, dataset, mem, max_new_tokens=5, top_k=5, top_p=0.9)
    
    # Check basic formatting
    assert isinstance(result, str)
    assert len(result) > 0
    assert result[0].isupper()
    # Should end with period or ellipsis
    assert result.endswith('.') or result.endswith('...')


def test_fallback_reply_with_empty_prompt():
    """Test _fallback_reply handles empty prompts gracefully."""
    torch.manual_seed(42)
    words = ["hello", "world"]
    chars = sorted(set(''.join(words)))
    dataset = CharDataset(words, chars, max(len(w) for w in words))
    config = ModelConfig(block_size=16, vocab_size=dataset.get_vocab_size(), n_layer=1, n_embd=16, n_head=2)
    model = Transformer(config)
    
    with Memory(":memory:") as mem:
        result = _fallback_reply("", model, dataset, mem, max_new_tokens=3)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_fallback_reply_uses_charged_token():
    """Test that _fallback_reply attempts to use charged token logic."""
    torch.manual_seed(0)
    words = ["hello", "world", "test"]
    chars = sorted(set(''.join(words)))
    dataset = CharDataset(words, chars, max(len(w) for w in words))
    config = ModelConfig(block_size=16, vocab_size=dataset.get_vocab_size(), n_layer=1, n_embd=16, n_head=2)
    model = Transformer(config)
    
    with Memory(":memory:") as mem:
        # Test with a prompt that should have identifiable charged tokens
        result = _fallback_reply("hello", model, dataset, mem, max_new_tokens=3, temperature=0.8)
    
    assert isinstance(result, str)
    assert len(result) > 0
    # The result should not be the exact input
    assert result.lower().strip('.') != "hello"


def test_fallback_reply_quality_assessment():
    """Test that quality assessment doesn't break the generation."""
    torch.manual_seed(1)
    words = ["quality", "test", "assessment", "good", "bad"]
    chars = sorted(set(''.join(words)))
    dataset = CharDataset(words, chars, max(len(w) for w in words))
    config = ModelConfig(block_size=32, vocab_size=dataset.get_vocab_size(), n_layer=2, n_embd=32, n_head=4)
    model = Transformer(config)
    
    with Memory(":memory:") as mem:
        mem.record_message("quality", "good")
        result = _fallback_reply("assessment", model, dataset, mem, max_new_tokens=5)
    
    # Should still generate a valid response despite quality assessment
    assert isinstance(result, str) 
    assert len(result) > 0
    assert result[0].isupper()