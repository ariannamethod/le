import torch
from memory import Memory
from le import sample_prompt, ModelConfig, Transformer, CharDataset
import response_log

def test_sample_prompt_output_format():
    torch.manual_seed(0)
    words = ["hello", "world"]
    chars = sorted(set(''.join(words)))
    dataset = CharDataset(words, chars, max(len(w) for w in words))
    config = ModelConfig(block_size=32, vocab_size=dataset.get_vocab_size(), n_layer=1, n_embd=32, n_head=4)
    model = Transformer(config)
    with Memory(":memory:") as mem:
        mem.record_message("hello", "hi")
        result = sample_prompt("world", model, dataset, mem, max_new_tokens=5, top_k=5, top_p=0.9)
    assert result[0].isupper()
    assert result.endswith(".")
    assert not dataset.contains(result[:-1].lower())


def test_sample_prompt_uses_fallback_when_needed():
    """Test that sample_prompt uses the transformer fallback when response_log fails."""
    torch.manual_seed(42)
    words = ["test", "fallback", "needed"]
    chars = sorted(set(''.join(words)))
    dataset = CharDataset(words, chars, max(len(w) for w in words))
    config = ModelConfig(block_size=16, vocab_size=dataset.get_vocab_size(), n_layer=1, n_embd=16, n_head=2)
    model = Transformer(config)
    
    # Mock response_log to always return False (simulate repetition)
    original_check_and_log = response_log.check_and_log
    response_log.check_and_log = lambda text: False
    
    try:
        with Memory(":memory:") as mem:
            # Use smaller top_k to avoid index out of range
            vocab_size = dataset.get_vocab_size()
            result = sample_prompt("test", model, dataset, mem, max_new_tokens=3, top_k=min(5, vocab_size-1))
        
        # Should still get a valid response from fallback
        assert isinstance(result, str)
        assert len(result) > 0
        assert result[0].isupper()
        # Should not be the old hardcoded message
        assert result != "Повтор, попробуйте снова."
        
    finally:
        # Restore original function
        response_log.check_and_log = original_check_and_log
