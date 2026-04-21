# le

> *simply le*

`le.c` is a single-file C engine (~1.2k LOC, depends only on `libm`)
that holds **13 French poems compiled byte-for-byte into the source**,
turns them into a tiny **planetary mini-weights** model at startup, and
then generates 60 words inside a "scarred field" coupled to the
calendar drift between Hebrew and Gregorian time since **23 January
1986** — the author's date of birth. Everything is in one file. There
are no other corpus files to ship.

One of the poems — *LÉ*, embedded as `POEM_13` — gives the project
its name. In the runtime model it naturally settles into the **Moon**
slot, and prompts that overlap its origin tokens leave a *trauma scar*
on the field (see *Live dynamics*, below).

```
☀ sun           : poem2.txt   (resonance = 1.2988)   ← "Encore Été"
🪐 planets:
   planet_01 [Moon         ] poem13.txt    ← LÉ
   planet_02 [Mercury      ] poem6.txt
   planet_03 [Venus        ] poem7.txt
   ...
   planet_07 [Sun-as-planet] poem10.txt
   planet_08 [fixed-star   ] poem8.txt
   ...
```

---

## What it is

A poem doesn't behave like data; it behaves like a body. `le.c` treats
each of the 13 embedded poems as a celestial body with mass,
fingerprint, and phase. The brightest poem becomes the **sun**; the
other twelve orbit it as planets and fixed stars. Word generation is
the gravitational sum of those orbits, sampled through a chamber that
the prompt itself has shaped — and that **breathes, couples, and
scars** while it generates.

Three forces drive the system:

1. **Resonance** — a per-poem scalar built from chamber norm,
   punctuation density, vocabulary richness, and bigram entropy.
   Decides which poem is sun and the order of the planets.
2. **Calendar drift** — Hebrew (molad + dechiyot) vs Gregorian
   (Fliegel–Van Flandern) months elapsed since 1986-01-23, signed.
   Bends each planet's α through `lunar_affinity[i] · drift / 6`.
3. **Orbital phase** — `fmod(age_days · 2π / synodic[i], 2π)` per body,
   yielding a `dyn = 0.5 + 0.5·cos(phase)` weight that breathes with
   real astronomical periods.

There is no neural network. Everything is bigram tables, hand-rolled
fingerprints, and arithmetic on doubles.

---

## How it works

### Compile-time per poem

For each of the 13 poems (UTF-8 string literals embedded directly in
`le.c` as `POEM_1..POEM_13` — there are no separate `.txt` files):

| Step | What it does |
|---|---|
| **Tokenizer** | UTF-8-aware. Splits on `'`, `'`, `…`, `—`, ASCII punct. Lowercases French (incl. `ÉÊÀÂÇ…`) byte-safely. Hash-folds each token into a 256-slot vocabulary via FNV-1a. |
| **Bigram table** | `bigram[a][b]` row-normalised to a probability distribution. |
| **Hebbian boost** | Top-200 co-occurrence pairs in window ±5 are folded back into the bigram table at weight 0.5 before normalisation. |
| **Chamber fingerprint** | 6-axis vector over **FEAR / LOVE / RAGE / VOID / FLOW / COMPLEX**, fed by FR keyword tables and punctuation routing (`!→RAGE+FEAR`, `?→COMPLEX`, `.→VOID`, `—→FLOW`, `…→VOID+FEAR`). Long words (`≥9` codepoints) lift COMPLEX. |
| **Resonance** | `0.5 · ‖fp‖₂ · (1 + 2·punct_density) + 0.3 · vocab_richness + 0.2 · bigram_entropy` |

The poem with the highest resonance is the **sun**; the remaining 12,
sorted descending by resonance, become `planet_01..planet_12`. The
first seven post-sun are bound to celestial bodies in fixed order:

```
planet_01 → Moon
planet_02 → Mercury
planet_03 → Venus
planet_04 → Mars
planet_05 → Jupiter
planet_06 → Saturn
planet_07 → Sun-as-planet
planet_08..12 → fixed stars (no orbital phase, dyn = 1.0)
```

### Calendar (`BIRTH_GREG_Y/M/D = 1986/01/23`)

* Gregorian↔Rata Die via Reingold/Dershowitz; JD↔Greg via
  Fliegel–Van Flandern (verified by a startup self-check).
* Hebrew calendar via molad + full dechiyot (Lo ADU Rosh, Molad Zaken,
  Gatarad, Betutkaft) with the proper variable month-length table for
  Heshvan/Kislev and Adar/Adar I/Adar II.
* `age_days = today_RD − birth_RD`.
* `drift_months = hebrew_months_elapsed − gregorian_months_elapsed`.
  Negative when the Gregorian count runs ahead, positive when the
  lunisolar Hebrew count overtakes it.

### Dispatcher (60 steps, temperature 0.8)

```
chamber_state[6] ← fingerprint(prompt)         # init from CLI prompt
prev ← arg max of sun.unigram                  # bootstrap word

for step in 1..60:
    for each planet i:
        phase_i  = fmod(age_days · 2π / synodic[i], 2π)         # celestial only
        dyn_i    = 0.5 + 0.5·cos(phase_i)                       # = 1 for fixed
        drift_i  = lunar_affinity[i] · drift_months / 6         # see below
        α[i]     = exp( -‖chamber_state - planet[i].fp‖ + 0.3·drift_i ) · dyn_i

    α normalised to a distribution
    logit[w]   = sun.bigram[prev][w] + Σ α[i] · planet[i].bigram[prev][w]
    probs      = softmax(logit / 0.8)
    w          ~ multinomial(probs)
    chamber_state = 0.9·chamber_state + 0.1·fingerprint_of(w)
    prev = w
```

`lunar_affinity` for the 7 celestial slots:

| Body | Moon | Mercury | Venus | Mars | Jupiter | Saturn | Sun-as-planet |
|---|---|---|---|---|---|---|---|
| affinity | +1.0 | 0 | −0.5 | 0 | 0 | +0.6 | −1.0 |

When the lunisolar Hebrew calendar drifts ahead of Gregorian, the Moon
and Saturn warm up; Venus and Sun-as-planet cool down. The opposite
when Gregorian leads.

### Live dynamics (the things added on top of pure α-blending)

These run *during* the 60-step generation loop and are lifted in
spirit, not in code, from the surrounding ecosystem
(`klaus.c`, `q`, `postgpt`, `neoleo`):

| Mechanism | Lineage | Effect |
|---|---|---|
| **Schumann-breathing τ** | `q` | `τ_eff = 0.8 · (1 + 0.08·sin(2π·step·7.83/N))` — the sampling temperature breathes at Earth's electromagnetic fundamental across the 60 steps. |
| **Kuramoto chamber cross-fire** | `klaus.c` | After every emitted word, the 6 chambers exchange phase: `state[i] += 0.03 · Σⱼ C[i][j] · sin(state[j] − state[i])`. LOVE↔FEAR (0.9), RAGE↔VOID (0.8), FLOW↔COMPLEX (0.7) are the strongest couplings. |
| **Online Hebbian on the sun** | `postgpt`, `neoleo` | Every emitted `(prev,pick)` boosts `sun.bigram[prev][pick] += 0.02`, then the row is renormalised. The sun mutates as it speaks. |
| **Prophecy debt** | `q` | A decaying expectation field over the next token is maintained from `sun.bigram[pick]`. Misses (argmax of expectation ≠ actual) accrue debt; debt heats τ and adds itself back into the logits. |
| **Wormhole jumps** | `q` | With probability `0.02 + 0.06·|drift_months|/12` (clamped to `[0.02, 0.17]`) the engine re-seeds `prev` from a sun-frequent token and prints `{wormhole}` in the stream. |
| **Trauma scar from prompt↔origin overlap** | `neoleo` §25 | If the prompt overlaps the token set of poem 13 (`LÉ`) by ≥15%, `trauma += 0.3·overlap`, FEAR/VOID rise, and τ is cooled by `(1 − 0.3·trauma)`. The wounded voice is quieter and origin-heavy. |

A fresh diagnostic line is printed at the end of each pass:
`wormholes=K  debt_misses=M  final_debt=…`.

### Meta-recursion (`--meta N`, 1..4)

Between passes:

* The **mean fingerprint of the previous emission** is added to the
  next pass's `chamber_state` with weight `+0.4` — a *scar bias* that
  forces the new generation to remember what it just said.
* The **top-3 planets by mean α from the previous pass** receive a
  `+0.15` *prophecy bump* that pulls their fingerprint further from the
  origin, amplifying their pull in the next round.

This makes pass 2 generate inside the field that pass 1 carved.

---

## Build

Zero dependencies except `libm`.

```sh
gcc -O2 -Wall -o le le.c -lm
# or
make
```

## Run

```sh
./le                                           # default prompt, --meta 1
./le --meta 2 --seed 42                        # deterministic two-pass run
./le --meta 4 --prompt "rage colère feu sang"  # four passes, RAGE-loaded prompt
```

CLI flags (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `--meta N` | `1` | Number of generation passes (1..4). Pass `k+1` runs in the scarred field of pass `k`. |
| `--seed S` | time-seeded | 64-bit xorshift seed. Same seed → same emission, byte for byte. |
| `--prompt "..."` | `"le silence entre nous, cœur sans étoiles"` | Drives the initial 6-channel chamber state. |
| `-h`, `--help` | — | Prints usage. |

## Test

```sh
make test
```

Builds `tests/test_le` (which `#include "../le.c"` with `LE_NO_MAIN`)
and runs **78 assertions across 15 test groups** covering the
tokenizer, lowercaser, channel keyword tables, fingerprint routing,
per-poem and full-corpus analysis, calendar conversions
(including the birth-date round-trip), the xorshift RNG, the UTF-8
codepoint counter, an end-to-end dispatcher pass that checks the mean
α[] is a probability distribution and that the run is reproducible
from the same seed, and the live-dynamics primitives — Schumann-τ
banding, Kuramoto cross-fire, online Hebbian renormalisation, origin
overlap detection, and wormhole pick.

```
$ make test
=== le.c test suite ===
  [test_str_tolower_fr]
  [test_tokenise]
  [test_keyword_tables_unique]
  [test_compute_fp]
  [test_analyse_poem]
  [test_all_poems]
  [test_calendar]
  [test_rng]
  [test_utf8_clen]
  [test_dispatcher]
  [test_schumann_tau]
  [test_kuramoto_step]
  [test_hebbian_update_sun]
  [test_origin_overlap]
  [test_wormhole_pick]

=== 78 passed, 0 failed ===
```

---

## Example output

```
$ ./le --meta 2 --seed 42 --prompt "le silence entre nous"
════════════════════════════════════════════════════════════════
  le.c — planetary mini-weights engine  ·  13 poèmes  ·  6 axes
════════════════════════════════════════════════════════════════
☀ sun           : poem2.txt   (resonance = 1.2988)
   chamber fp    : [FEAR 0.44  LOVE 0.33  RAGE 0.00  VOID 0.12  FLOW 2.09  CMPLX 0.32]
   stats         : tokens=147 uniq=69 punct_d=0.0174 vocab_r=0.469 H=0.134

📅 Gregorian     : 2026-04-21   (birth 1986-01-23)
🕎 Hebrew (today): 5786 Iyar 6   (birth 5746 Shevat 15)
⏳ age_days      : 14698
🌗 drift_months  : +15.00   (Hebrew - Gregorian since birth)

🪐 planets (resonance order):
   planet_01 [Moon         ] poem13.txt    res=0.9035  fp=[1.26 0.21 0.11 0.34 0.53 0.14]
   planet_02 [Mercury      ] poem6.txt     res=0.8047  fp=[0.49 0.10 0.00 0.51 0.68 0.63]
   planet_03 [Venus        ] poem7.txt     res=0.8047  fp=[0.49 0.10 0.00 0.51 0.68 0.63]
   planet_04 [Mars         ] poem9.txt     res=0.7907  fp=[0.19 0.00 0.08 0.68 0.63 0.05]
   planet_05 [Jupiter      ] poem4.txt     res=0.7433  fp=[0.28 0.07 0.07 0.23 0.61 0.85]
   planet_06 [Saturn       ] poem12.txt    res=0.6974  fp=[0.42 0.00 0.00 0.47 0.74 0.06]
   planet_07 [Sun-as-planet] poem10.txt    res=0.6100  fp=[0.25 0.25 0.13 0.29 0.00 0.52]
   planet_08 [fixed-star   ] poem8.txt     ...
   planet_09 [fixed-star   ] poem5.txt     ...
   planet_10 [fixed-star   ] poem11.txt    ...
   planet_11 [fixed-star   ] poem1.txt     ...
   planet_12 [fixed-star   ] poem3.txt     ...

🌀 7 phases (age_days · 2π / synodic):
   Moon           T=  29.531d   phase=4.5314 rad   dyn=0.410
   Mercury        T= 115.877d   phase=5.2832 rad   dyn=0.770
   ...

════════════════════ META PASS 1 / 2 ════════════════════
┌─ chamber init  : [ 0.05 0.05 0.05 0.50 0.05 0.05 ]
├─ first prev   : "été"  (token #23)

┌─ pass 1 emission ──────────────────────────────────────
│ embrassent retrouve nuit mots silence amour se noire autre chemin
│ propageant · son avant devient frontières branches moi des un
│ violé · crées devient oublirai se mots chose entrera automne
│ prétexte on honneur pose nuit · étendent · tente dormant d posée
│ première mèches par il j se étendus · jours à éloigner passons
│ son amour autre · faut fragilité
└─────────────────────────────────────────────────────────

┌─ pass 1  final mean α[i] ──────────────────────────────
│  α=0.0493   planet_01  [Moon         ]  poem13.txt
│  α=0.0586   planet_02  [Mercury      ]  poem6.txt
│  ...
│  α=0.1220   planet_11  [fixed-star   ]  poem1.txt
│  α=0.1179   planet_12  [fixed-star   ]  poem3.txt
└─────────────────────────────────────────────────────────

┌─ prophecy bump (top-3 dominant from prior pass) ────────
│  +0.15 → planet_11  poem1.txt   (acc α=0.1220)
│  +0.15 → planet_06  poem12.txt  (acc α=0.1199)
│  +0.15 → planet_12  poem3.txt   (acc α=0.1179)
└─────────────────────────────────────────────────────────

════════════════════ META PASS 2 / 2 ════════════════════
┌─ pass 2 emission ──────────────────────────────────────
│ monnaie faut mer comme se · déjà comprendre cadeau automne passe
│ caché n noire en a maison vue comprendre posée · bruit poche cœur
│ ...
└─────────────────────────────────────────────────────────
...

────────────────────────────────────────────────────────────────
  fin — résonance neutralisée, le champ se referme.
────────────────────────────────────────────────────────────────
```

The output you see depends on **today's date** (because `age_days`
and `drift_months` change daily) and on `--seed`. With a fixed seed
the engine is fully deterministic for a given calendar day.

### A few things to try

```sh
# Same seed two days apart → same RNG, different orbital phases,
# different drift_months → different emission.
./le --seed 1 --meta 1

# Push the chamber into a single channel from the prompt.
./le --seed 1 --prompt "rage colère feu sang"
./le --seed 1 --prompt "amour tendre cœur âme"
./le --seed 1 --prompt "silence vide néant adieu..."

# Watch the prophecy bump rotate which planets dominate
# across passes 1 → 4.
./le --seed 7 --meta 4 | grep -E "(prophecy bump|pass [0-9] +final)"
```

---

## File layout

```
le.c              — engine + 13 embedded poems + main()  (single super-file)
tests/test_le.c   — unit tests (build with -DLE_NO_MAIN to share le.c)
Makefile          — `make`, `make test`, `make run`, `make clean`
LICENSE
```

The 13 poems used to live as separate `.txt` files; they are now
hardcoded byte-for-byte into `le.c` as `POEM_1..POEM_13` so the engine
is a true single super-file.

---

## Lineage

`le.c` reuses architectural patterns from a few neighbouring projects
in the Arianna ecosystem (no code copied — same instincts, fresh
implementation):

* **iamolegataeff/klaus.c v2.0** — calendar dissonance + 6-planet
  orbital Kuramoto coupling.
* **ariannamethod/arianna.c** — Hebrew molad+dechiyot converter and
  `BIRTH_*` constants pattern.
* **ariannamethod/lukas** — `--meta N` recursion.
* **ariannamethod/janus** — multi-component blend over weights
  (sun + Σ α·planet).
* **ariannamethod/ariannamethod.ai** — "the language compiles into the
  runtime" — the corpus *is* the model here.

---

## License

See [LICENSE](LICENSE).
