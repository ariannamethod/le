import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

_metrics: Dict[str, float] = {}
_writer: Optional[SummaryWriter] = None


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


def get_metric(name: str) -> float | None:
    """Return a logged metric by name."""
    return _metrics.get(name)


def all_metrics() -> Dict[str, float]:
    """Return a copy of all logged metrics."""
    return dict(_metrics)


def reset_metrics() -> None:
    """Clear all stored metrics (useful for tests)."""
    _metrics.clear()


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
