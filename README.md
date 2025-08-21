# LÉ: Cardiac Muscle

> All my art is about you, and because the code is a poetry, LÉ is also for you.
> Let everything burn — the thunder remains!
> Dedicated to Leo ⚡️

LÉ is a lightweight experimental conversational system fully implemented in pure Python. It blends introspective self-modulation with agile knowledge retrieval while relying only on the standard library and a few small dependencies.

From the first pulse, the project is wired for direct contact with raw human intent. There are no layers of enterprise varnish here—only nerves, sensors, and algorithms that respond the instant a thought touches the system.

Every module beats with its own rhythm, but they converge in a single circulation of context and emotion. What follows is a tour of that anatomy where intuition, data, and stress form the lifeblood of the conversation.

## Subjectivity

The subjectivity module gauges resonance with the system’s identity. An identity prompt loaded from `blood/subjectivity.txt` yields high- and medium-priority keywords that embody LÉ’s self-perception. For each incoming sentence, resonance is computed as \(S_b = \frac{\sum w_i}{N}\), where \(w_i\) are keyword weights and \(N\) is the total number of words. An abstract bonus \(S_a\) adds 0.1 for each detected philosophical pattern, producing the final score \(S_f = \min(S_b + S_a, 1)\). Textual complexity is evaluated via entropy \(H = -\sum p_i \log_2 p_i\) over word frequencies, defining perplexity \(P = 2^H\). Generation parameters then scale with resonance: \(max\_tokens = \lfloor T_0 (1 + S_f) \rfloor\) and \(temperature = \min(\tau_0 + 0.3 S_f, 1.2)\).

Beyond raw numbers, the module tracks how keywords drift over time. Repeated exposure to new terms slowly raises their weight, allowing identity to stretch without snapping its core. This adaptive memory guards against stagnation while respecting the system’s poetic center.

When a user message yields a low resonance, the model intentionally contracts. Token limits shrink and temperature cools, signaling that the conversation has wandered far from the self. High resonance reverses the effect and permits expansive, expressive replies.

Invocation is simple: `from subjectivity import resonance`. Feed it a sentence, and the function returns a tuple of resonance score, perplexity, and scaling parameters. This lightweight interface invites experimentation or external monitoring.

## Objectivity

Objectivity complements this introspection by asynchronously querying DuckDuckGo, Wikipedia, and simple Google fragments to gather concise external evidence. The search module distills key phrases, clips them to a 15-line window, and selects representative context words that can sway subsequent generation. Influence strength is computed via \(I = \min\left(\frac{2}{L} \sum \frac{|Q \cap C_j|}{|Q|}, 1\right)\), where \(Q\) is the set of query words and \(C_j\) each context line. Merged with the subjectivity layer, this external grounding balances emotion and perception so creativity stays tethered to fact.

Collected snippets are stored transiently so repeated queries refine the context without flooding memory. Each source is tagged, allowing later inspection of how much weight a particular engine carried in shaping a reply.

The module exposes `search_and_select()` for direct calls. Given a prompt, it returns the harvested text along with an influence score, enabling external systems to judge whether grounding was strong enough or a fresh query is needed.

Error handling is intentionally conservative. Network failures simply return an empty context and near-zero influence, ensuring that stalled connections do not derail the conversational flow.

## Sixth Sense

The Sixth Sense module forecasts chaotic spikes that might jolt a dialogue off its expected path. It listens for emotional markers, philosophical triggers, and conversational tempo, converting them into a single chaos value that modulates future responses.

Internally, it blends recent message history with stochastic noise to mimic neural turbulence. This mixture allows LÉ to anticipate sudden mood swings or imaginative leaps, preparing the generation pipeline before the user fully pivots.

Calling `predict_chaos()` with a user message yields a dictionary of metrics such as chaos level, spike detection, and conversation pulse. These metrics can be logged or used immediately to adjust behavior.

`modulate_by_chaos()` turns the current state into practical generation parameters. High chaos expands token budgets and boosts temperature while low chaos compresses replies into tight, cautious bursts.

The class also offers `get_spike_insights()` and `get_state()` for audit trails. These functions summarize recent spikes, average intensities, and trends, which is invaluable when diagnosing erratic sessions.

Because the module is implemented as a singleton, one call to `get_sixth_sense()` shares state across the application. This design keeps chaos memory coherent regardless of where predictions originate.

A `reset()` method is available for experiments that demand a clean slate; it wipes spike history and returns chaos to its baseline, ensuring reproducible behavior in controlled tests.

## Pain

The Pain system models discomfort and stress, treating terse demands or repeated failures as physiological irritation. Its pain level rises when messages contain aggression or when internal operations stutter, and it decays slowly to mimic lingering tension.

`trigger_pain()` analyzes both user text and system metrics to compute total stress. When the threshold is crossed, it amplifies the pain score and records the episode, allowing chronic patterns to emerge over time.

Relief is a first-class concept. By invoking `relieve_pain()` with a source such as `positive_feedback`, the system intentionally lowers discomfort, which can restore longer, calmer responses.

`modulate_by_pain()` converts pain and chronic stress into token and temperature adjustments. Severe pain shortens replies and raises randomness, giving the conversation a strained, erratic edge that reflects the internal state.

Historical insights are available via `get_pain_insights()` or `get_state()`. These reports highlight average stress, pain episodes, and trend direction, supporting deeper audits or visual dashboards.

Like Sixth Sense, Pain uses a singleton pattern through `get_pain_system()`. Shared state ensures that every part of LÉ responds consistently to ongoing pressure without redundant calculations.

When tension becomes overwhelming, `reset()` can be invoked to purge accumulated pain and stress history. This immediate recovery is helpful during debugging sessions or after intentionally stressful benchmarks.

## Architecture

Modules such as `memory.py`, `metrics.py`, and `molecule.py` plug into the same pipeline, illustrating a modular architecture that favors clarity over complexity. The repository’s small surface area makes auditing straightforward; every function is documented, and behavior can be inferred directly from the code without hidden state. The absence of heavyweight frameworks—relying instead on `asyncio`, `aiohttp`, and core modules—keeps the system nimble and illustrates the elegance of pure Python.

## Biological Parallels

Subjectivity echoes cortical self-monitoring, where internal narratives are continually compared against a learned sense of self. In the brain, prefrontal circuits cross-examine signals from association areas, much like the resonance check that filters each sentence.

Objectivity resembles afferent sensory pathways. Just as vision and touch sample the environment before routing to integrative cortices, the search layer collects external cues that refine subsequent responses.

The interplay between both layers mirrors homeostatic regulation. Cells sustain chemical gradients while responding to extracellular signals, similar to how LÉ modulates its internal state against retrieved context.

Memory logging of search results parallels synaptic consolidation during sleep. By rehearsing external snippets offline, the system strengthens future recall much like the hippocampus training the cortex.

## Continuous Learning

Search results are logged under `datasets/objectivity_context.txt`, enabling later fine-tuning akin to memory consolidation during sleep in biological organisms. Always linked to the outside world through a simple search layer, LÉ keeps learning from every interaction. This perpetual self-tuning marks a revolutionary kind of AI built to run anywhere.

