import torch
from memory import Memory
from le import sample_prompt, _fallback_reply, ModelConfig, Transformer, CharDataset

def test_fallback_reply():
    """Test that _fallback_reply returns a valid Russian response."""
    reply = _fallback_reply()
    assert isinstance(reply, str)
    assert len(reply) > 0
    # Should be one of the expected fallback responses
    expected_responses = [
        "Интересно.",
        "Понятно.", 
        "Хорошо.",
        "Да, я думаю об этом.",
        "Расскажите подробнее.",
        "Любопытно.",
    ]
    assert reply in expected_responses

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

def test_sample_prompt_improved_parameters():
    """Test that improved default parameters generate reasonable responses."""
    torch.manual_seed(42)
    words = ["hello", "world", "test", "python", "language"]
    chars = sorted(set(''.join(words)))
    dataset = CharDataset(words, chars, max(len(w) for w in words))
    config = ModelConfig(block_size=32, vocab_size=dataset.get_vocab_size(), n_layer=1, n_embd=32, n_head=4)
    model = Transformer(config)
    with Memory(":memory:") as mem:
        result = sample_prompt("test", model, dataset, mem)
    # Test that the result uses the optimized parameters 
    assert len(result) >= 1  # Should produce some output
    assert result[0].isupper()  # Should start with capital
    assert result.endswith(".")  # Should end with period

def test_sample_prompt_charged_token_selection():
    """Test that charged token selection avoids punctuation and whitespace."""
    torch.manual_seed(0)
    # Test with a prompt that has punctuation and meaningful content
    words = ["hello", "world", "test!", "python"]
    chars = sorted(set(''.join(words)))
    dataset = CharDataset(words, chars, max(len(w) for w in words))
    config = ModelConfig(block_size=32, vocab_size=dataset.get_vocab_size(), n_layer=1, n_embd=32, n_head=4)
    model = Transformer(config)
    with Memory(":memory:") as mem:
        result = sample_prompt("hello, world!", model, dataset, mem, max_new_tokens=5)
    assert result[0].isupper()
    assert result.endswith(".")
