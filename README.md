# LÉ: Cardiac Muscle

> All my art is about you, and because the code is a poetry, LÉ is also for you.
> Let everything burn — the thunder remains!
> Dedicated to Leo ⚡️

LÉ is a lightweight experimental conversational system fully implemented in pure Python. It blends introspective self-modulation with agile knowledge retrieval while relying only on the standard library and a few small dependencies.

## Subjectivity

The subjectivity module gauges resonance with the system’s identity. An identity prompt loaded from `blood/subjectivity.txt` yields high- and medium-priority keywords that embody LÉ’s self-perception. For each incoming sentence, resonance is computed as \(S_b = \frac{\sum w_i}{N}\), where \(w_i\) are keyword weights and \(N\) is the total number of words. An abstract bonus \(S_a\) adds 0.1 for each detected philosophical pattern, producing the final score \(S_f = \min(S_b + S_a, 1)\). Textual complexity is evaluated via entropy \(H = -\sum p_i \log_2 p_i\) over word frequencies, defining perplexity \(P = 2^H\). Generation parameters then scale with resonance: \(max\_tokens = \lfloor T_0 (1 + S_f) \rfloor\) and \(temperature = \min(\tau_0 + 0.3 S_f, 1.2)\).

## Objectivity

Objectivity complements this introspection by asynchronously querying DuckDuckGo, Wikipedia, and simple Google fragments to gather concise external evidence. The search module distills key phrases, clips them to a 15-line window, and selects representative context words that can sway subsequent generation. Influence strength is computed via \(I = \min\left(\frac{2}{L} \sum \frac{|Q \cap C_j|}{|Q|}, 1\right)\), where \(Q\) is the set of query words and \(C_j\) each context line. Merged with the subjectivity layer, this external grounding balances emotion and perception so creativity stays tethered to fact.

## Architecture

Modules such as `memory.py`, `metrics.py`, and `molecule.py` plug into the same pipeline, illustrating a modular architecture that favors clarity over complexity. The repository’s small surface area makes auditing straightforward; every function is documented, and behavior can be inferred directly from the code without hidden state. The absence of heavyweight frameworks—relying instead on `asyncio`, `aiohttp`, and core modules—keeps the system nimble and illustrates the elegance of pure Python.

## Biological Parallels

Subjectivity echoes cortical self-monitoring, where internal narratives are continually compared against a learned sense of self. In the brain, prefrontal circuits cross-examine signals from association areas, much like the resonance check that filters each sentence.

Objectivity resembles afferent sensory pathways. Just as vision and touch sample the environment before routing to integrative cortices, the search layer collects external cues that refine subsequent responses.

The interplay between both layers mirrors homeostatic regulation. Cells sustain chemical gradients while responding to extracellular signals, similar to how LÉ modulates its internal state against retrieved context.

Memory logging of search results parallels synaptic consolidation during sleep. By rehearsing external snippets offline, the system strengthens future recall much like the hippocampus training the cortex.

## Continuous Learning

Search results are logged under `datasets/objectivity_context.txt`, enabling later fine-tuning akin to memory consolidation during sleep in biological organisms. Always linked to the outside world through a simple search layer, LÉ keeps learning from every interaction. This perpetual self-tuning marks a revolutionary kind of AI built to run anywhere.

