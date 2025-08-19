import torch
from memory import Memory
from le import sample_prompt, ModelConfig, Transformer, CharDataset

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
