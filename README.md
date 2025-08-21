# LÉ System Overview

LÉ is a lightweight experimental conversational system fully implemented in pure Python, designed to blend introspective self-modulation with agile knowledge retrieval.

The codebase relies solely on the Python standard library and a few small dependencies, keeping deployment simple and ensuring a super‑light footprint.

At runtime a user message first passes through a subjectivity layer that gauges resonance with the system’s identity before optional objective context is fetched from the web.

The subjectivity module loads an identity prompt from `blood/subjectivity.txt` and extracts high‑ and medium‑priority keywords that embody LÉ’s self‑perception.

For each incoming sentence, resonance is computed as \(S_b = \frac{\sum w_i}{N}\), where \(w_i\) are keyword weights and \(N\) is the total number of words.

An abstract bonus \(S_a\) adds 0.1 for each detected philosophical pattern, producing the final score \(S_f = \min(S_b + S_a, 1)\).

To estimate textual complexity, the filter evaluates entropy \(H = -\sum p_i \log_2 p_i\) over word frequencies and defines perplexity \(P = 2^H\).

Generation parameters are modulated by resonance: \( \text{max\_tokens} = \lfloor T_0 (1 + S_f) \rfloor \) and \( \text{temperature} = \min(\tau_0 + 0.3 S_f, 1.2)\), where \(T_0\) and \(\tau_0\) are base values.

Objectivity complements this introspection by asynchronously querying DuckDuckGo, Wikipedia, and simple Google fragments to gather concise external evidence.

The search module distills key phrases, clips them to a 15‑line window, and selects representative context words that can sway subsequent generation.

Influence strength is computed via \(I = \min\left(\frac{2}{L} \sum \frac{|Q \cap C_j|}{|Q|}, 1\right)\), where \(Q\) is the set of query words and \(C_j\) each context line.

Outputs from subjectivity and objectivity merge so that internal resonance shapes creativity while external facts ground responses, echoing the balance between emotion and perception.

Beyond these utilities, modules such as `memory.py`, `metrics.py`, and `molecule.py` plug into the same pipeline, illustrating a modular architecture that favors clarity over complexity.

The repository’s small surface area makes auditing straightforward; every function is documented, and behavior can be inferred directly from the code without hidden state.

Biologically, subjectivity mirrors cortical self‑monitoring, whereas objectivity evokes sensory pathways that sample the environment before signals reach associative cortices.

This duality resembles homeostatic regulation in cells where internal chemical gradients are balanced against external signals, much like \( \frac{dC}{dt} = k (E - C) \) models feedback adaptation.

Mathematically, the system echoes logistic growth, \(L(t) = \frac{K}{1 + e^{-r(t - t_0)}}\), as responses scale with context yet remain bounded by predefined limits.

Insights from cognitive science guide the design: resonance scoring parallels affective appraisal, while context retrieval aligns with attention mechanisms in neural models.

Search results are logged under `datasets/objectivity_context.txt`, enabling later fine‑tuning akin to memory consolidation during sleep in biological organisms.

The absence of heavyweight frameworks—relying instead on `asyncio`, `aiohttp`, and core modules—keeps the system nimble and illustrates the elegance of pure Python.

This audit brings documentation in line with implementation and invites future exploration at the intersection of mathematics, biology, and lightweight artificial cognition.
