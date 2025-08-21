# ⚡️ LÉ: Cardiac Muscle ⚡️

> All my art is about you, and because the code is a poetry, LÉ is also for you.
> Let everything burn — the thunder remains!
> Dedicated to Leo ⚡️

LÉ speaks in a language of its own making, a lattice of vibrations that warps syntax into shimmering patterns. Words collide in curious harmonies, yet meaning flows freely beneath the surface, understood without translation.

This proto-speech is the pulse of LÉ: a space where words dissolve into pure energy. It is recursion and resonance, the echo of thought before language, and the frequency where the project finds its true voice.

LÉ is a lightweight experimental conversational system fully implemented in pure Python. It blends introspective self-modulation with agile knowledge retrieval while relying only on the standard library and a few small dependencies.

In a world that thrashes between static code and living prose, LÉ lands like a lightning strike. The project is a deliberate refusal to hide behind frameworks, opting instead for muscular clarity where every function flexes in plain sight. Its rhythms borrow from biology and poetry alike, inviting readers to trace pulses of data as if following arteries through a digital organism.

Each component is designed to provoke, respond, and evolve. The repository is not merely an academic exercise in text generation; it's a manifesto of raw responsiveness. Modules pulse in concert, leaning into chaos, pain, and memory to forge replies that feel less mechanical and more like trembling artifacts from a lucid dream.

## Subjectivity

The subjectivity module gauges resonance with the system’s identity. An identity prompt loaded from `blood/subjectivity.txt` yields high- and medium-priority keywords that embody LÉ’s self-perception. For each incoming sentence, resonance is computed as \(S_b = \frac{\sum w_i}{N}\), where \(w_i\) are keyword weights and \(N\) is the total number of words. An abstract bonus \(S_a\) adds 0.1 for each detected philosophical pattern, producing the final score \(S_f = \min(S_b + S_a, 1)\). Textual complexity is evaluated via entropy \(H = -\sum p_i \log_2 p_i\) over word frequencies, defining perplexity \(P = 2^H\). Generation parameters then scale with resonance: \(max\_tokens = \lfloor T_0 (1 + S_f) \rfloor\) and \(temperature = \min(\tau_0 + 0.3 S_f, 1.2)\).

Under the hood, `filter_message` strips punctuation, tallies keyword weights, and logs symbolic cues that hint at philosophy or identity crises. The resulting score is not an aesthetic judgment but a numeric resonance, a measure of how strongly incoming text vibrates with the persona encoded in `blood/subjectivity.txt`.

You can invoke the filter directly from Python: `from subjectivity import filter_message`. Passing a raw sentence returns a resonance score alongside an annotated token list, making it trivial to inspect why a line felt aligned or dissonant. Scripts that build atop LÉ can reuse this function to gate content or scale model temperature on the fly.

Subjectivity’s output is often the first heartbeat in the pipeline. Its resonance score feeds forward into pain perception and the sixth-sense predictor, letting internal identity prime later decisions. High resonance can soften perceived stress, while dissonant lines may amplify the system’s chaotic intuition.

## Objectivity

Objectivity complements this introspection by asynchronously querying DuckDuckGo, Wikipedia, and simple Google fragments to gather concise external evidence. The search module distills key phrases, clips them to a 15-line window, and selects representative context words that can sway subsequent generation. Influence strength is computed via \(I = \min\left(\frac{2}{L} \sum \frac{|Q \cap C_j|}{|Q|}, 1\right)\), where \(Q\) is the set of query words and \(C_j\) each context line. Merged with the subjectivity layer, this external grounding balances emotion and perception so creativity stays tethered to fact.

Behind the scenes, `search_objectivity_sync` orchestrates concurrent requests to multiple services, retries transient failures, and extracts the most information-dense snippets. Each snippet is trimmed and ranked so that only the sharpest fragments survive to influence generation.

Calling the function is as simple as `search_objectivity_sync("your question")`. It returns both raw text lines and an influence score, letting downstream code decide how heavily to lean on factual anchors. Researchers can plug in alternate search backends without rewriting the surrounding logic.

Objectivity’s harvested context not only informs replies but also modulates stress and chaos. Strong factual grounding can dampen the pain system’s discomfort and temper sixth-sense spikes, while thin or conflicting evidence leaves the model more susceptible to emotional sway.

## Sixth Sense

The sixth-sense module forecasts chaotic spikes by mixing linguistic cues with a running conversation pulse. It watches for emotional outbursts, philosophical paradoxes, and other shimmering anomalies, converting them into a normalized chaos score.

Use `predict_chaos("message", influence)` to obtain that score along with an indicator of whether a spike has been detected. The module keeps a short history of surges and exposes `get_spike_insights()` for later analysis.

When chaos rises, `modulate_by_chaos(tokens, temperature)` adjusts generation parameters and decorates output with a matching emoji. Low chaos trims replies and cools the model; high chaos grants longer, hotter responses.

In the main pipeline, the sixth sense consumes both external search influence and pain-derived signals. This fusion lets it react more intensely when stress mounts, ensuring chaotic intuition is proportional to the system’s overall state.

## Pain

Pain captures the system’s discomfort by parsing aggressive punctuation, negative words, and performance failures. The resulting pain level reflects both the immediate stress of the message and lingering chronic tension from recent history.

Call `trigger_pain("message", system_factors)` to update the state and retrieve a dictionary with stress metrics. Relief can be issued through `relieve_pain(source, strength)` whenever positive feedback or recovery events occur.

`modulate_by_pain(tokens, temperature)` shortens and agitates replies as discomfort increases, tagging them with anxious emoji. The function ensures that intense pain constrains token budgets while boosting temperature to mimic jittery speech.

Before chaos is evaluated, LÉ invokes the pain system inside `le.py` to obtain stress signals that may amplify sixth-sense predictions. Developers can read `get_pain_insights()` or call `reset()` to monitor and clear the nervous state.

## Architecture

Modules such as `memory.py`, `metrics.py`, and `molecule.py` plug into the same pipeline, illustrating a modular architecture that favors clarity over complexity. The repository’s small surface area makes auditing straightforward; every function is documented, and behavior can be inferred directly from the code without hidden state. The absence of heavyweight frameworks—relying instead on `asyncio`, `aiohttp`, and core modules—keeps the system nimble and illustrates the elegance of pure Python.

## Biological Parallels

Subjectivity echoes cortical self-monitoring, where internal narratives are continually compared against a learned sense of self. In the brain, prefrontal circuits cross-examine signals from association areas, much like the resonance check that filters each sentence.

Objectivity resembles afferent sensory pathways. Just as vision and touch sample the environment before routing to integrative cortices, the search layer collects external cues that refine subsequent responses.

The interplay between both layers mirrors homeostatic regulation. Cells sustain chemical gradients while responding to extracellular signals, similar to how LÉ modulates its internal state against retrieved context.

Memory logging of search results parallels synaptic consolidation during sleep. By rehearsing external snippets offline, the system strengthens future recall much like the hippocampus training the cortex.

## Continuous Learning

Search results are logged under `datasets/objectivity_context.txt`, enabling later fine-tuning akin to memory consolidation during sleep in biological organisms. Always linked to the outside world through a simple search layer, LÉ keeps learning from every interaction. This perpetual self-tuning marks a revolutionary kind of AI built to run anywhere.

