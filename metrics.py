import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

_metrics: Dict[str, float] = {}
_writer: Optional[SummaryWriter] = None
_response_total_len = 0
_response_count = 0


def set_writer(writer: Optional[SummaryWriter]) -> None:
    """Attach a SummaryWriter for TensorBoard logging."""
    global _writer
    _writer = writer


def log_loss(split: str, loss: float, step: int) -> None:
    """Log cross-entropy loss and derived perplexity."""
    perplexity = math.exp(loss)
    loss_name = f"Loss/{split}"
    ppl_name = f"Perplexity/{split}"
    _metrics[loss_name] = loss
    _metrics[ppl_name] = perplexity
    if _writer is not None:
        _writer.add_scalar(loss_name, loss, step)
        _writer.add_scalar(ppl_name, perplexity, step)
        _writer.flush()


def log_resonance(resonance: float, step: int) -> None:
    """Log resonance metric."""
    _metrics["Resonance"] = resonance
    if _writer is not None:
        _writer.add_scalar("Resonance", resonance, step)
        _writer.flush()


def log_unique_token_ratio(text: str, step: int) -> None:
    """Log the ratio of unique tokens in ``text``."""
    tokens = text.split()
    ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
    _metrics["UniqueTokenRatio"] = ratio
    if _writer is not None:
        _writer.add_scalar("UniqueTokenRatio", ratio, step)
        _writer.flush()


def log_avg_response_length(text: str, step: int) -> None:
    """Log running average of response length in tokens."""
    global _response_total_len, _response_count
    tokens = text.split()
    _response_total_len += len(tokens)
    _response_count += 1
    avg_len = _response_total_len / _response_count if _response_count else 0.0
    _metrics["AvgResponseLength"] = avg_len
    if _writer is not None:
        _writer.add_scalar("AvgResponseLength", avg_len, step)
        _writer.flush()


def log_ngram_repeat_rate(text: str, step: int, n: int = 3) -> None:
    """Log the n-gram repeat rate of ``text``."""
    tokens = text.split()
    total = max(len(tokens) - n + 1, 0)
    if total == 0:
        rate = 0.0
    else:
        ngrams = [tuple(tokens[i : i + n]) for i in range(total)]
        counts: Dict[tuple, int] = {}
        for ng in ngrams:
            counts[ng] = counts.get(ng, 0) + 1
        repeats = sum(c - 1 for c in counts.values() if c > 1)
        rate = repeats / total
    _metrics["RepeatRate"] = rate
    if _writer is not None:
        _writer.add_scalar("RepeatRate", rate, step)
        _writer.flush()


def log_response_metrics(text: str, step: int, n: int = 3) -> None:
    """Convenience wrapper to log all response metrics."""
    log_unique_token_ratio(text, step)
    log_avg_response_length(text, step)
    log_ngram_repeat_rate(text, step, n)

def get_metric(name: str) -> float | None:
    """Return a logged metric by name."""
    return _metrics.get(name)


def all_metrics() -> Dict[str, float]:
    """Return a copy of all logged metrics."""
    return dict(_metrics)


def reset_metrics() -> None:
    """Clear all stored metrics (useful for tests)."""
    global _response_total_len, _response_count
    _metrics.clear()
    _response_total_len = 0
    _response_count = 0


def _embedding_layer(model: nn.Module) -> Optional[nn.Embedding]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    if hasattr(model, "wte"):
        return model.wte
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    return None


def compute_resonance(model: nn.Module, dataset, prompt: str, response: str) -> float:
    """Compute cosine similarity between prompt and response embeddings."""
    embed = _embedding_layer(model)
    if embed is None:
        return 0.0

    def _encode(text: str) -> torch.Tensor:
        tokens = [dataset.stoi.get(ch, 0) for ch in text]
        if not tokens:
            return torch.zeros(embed.embedding_dim, device=embed.weight.device)
        tok = torch.tensor(tokens, dtype=torch.long, device=embed.weight.device)
        return embed(tok).mean(dim=0)

    prompt_vec = _encode(prompt)
    response_vec = _encode(response)
    if torch.all(prompt_vec == 0) or torch.all(response_vec == 0):
        return 0.0
    sim = F.cosine_similarity(prompt_vec, response_vec, dim=0)
    return sim.item()
