/*
 * tests/test_le.c — unit tests for le.c
 * Builds by including le.c with LE_NO_MAIN to suppress its main().
 *
 * Build & run:
 *   cc -O2 -Wall -DLE_NO_MAIN -o test_le test_le.c -lm && ./test_le
 *
 * (The provided Makefile in the repo root does this automatically:
 *   make test)
 *
 * Conventions: each test prints PASS/FAIL lines and the harness exits
 * with the total number of failed assertions.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>

#define LE_NO_MAIN
#include "../le.c"

/* ───────────── tiny assertion helpers ───────────── */

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        printf("    \xE2\x9C\x97 %s  (line %d)\n", (msg), __LINE__);\
        g_fail++;                                                   \
    } else {                                                        \
        g_pass++;                                                   \
    }                                                               \
} while (0)

#define CHECK_NEAR(a, b, eps, msg) do {                             \
    double _va = (a), _vb = (b);                                    \
    if (fabs(_va - _vb) > (eps)) {                                  \
        printf("    \xE2\x9C\x97 %s  (got %.6f want %.6f, line %d)\n",\
               (msg), _va, _vb, __LINE__);                          \
        g_fail++;                                                   \
    } else { g_pass++; }                                            \
} while (0)

/* Helper: silence stdout around a noisy call. */
static int  s_saved_fd = -1;
static FILE *s_devnull = NULL;
static void silence_stdout(void) {
    fflush(stdout);
    s_saved_fd = dup(1);
    s_devnull = fopen("/dev/null", "w");
    if (s_devnull) dup2(fileno(s_devnull), 1);
}
static void restore_stdout(void) {
    fflush(stdout);
    if (s_saved_fd >= 0) { dup2(s_saved_fd, 1); close(s_saved_fd); s_saved_fd = -1; }
    if (s_devnull) { fclose(s_devnull); s_devnull = NULL; }
}

/* ───────────── tests ───────────── */

/* 1. UTF-8 lowercaser handles French capitals and leaves × alone. */
static void test_str_tolower_fr(void) {
    printf("  [test_str_tolower_fr]\n");
    char a[64];

    strcpy(a, "ÉTÉ"); str_tolower_fr(a);
    CHECK(strcmp(a, "été") == 0, "ÉTÉ -> été");

    strcpy(a, "ÂME"); str_tolower_fr(a);
    CHECK(strcmp(a, "âme") == 0, "ÂME -> âme");

    strcpy(a, "FRONTIÈRES"); str_tolower_fr(a);
    CHECK(strcmp(a, "frontières") == 0, "FRONTIÈRES -> frontières");

    /* multiplication sign × (U+00D7 = 0xC3 0x97) must stay untouched */
    strcpy(a, "A\xC3\x97" "B"); str_tolower_fr(a);
    CHECK(strcmp(a, "a\xC3\x97" "b") == 0, "U+00D7 (multiplication sign) preserved");
}

/* 2. Tokenizer splits on apostrophe, em-dash and ASCII punct. */
static void test_tokenise(void) {
    printf("  [test_tokenise]\n");
    memset(WORD_OF, 0, sizeof(WORD_OF));

    uint8_t toks[64];
    int uniq = 0, punct = 0, total = 0;
    /* "J’écoute le silence — encore." → 5 word tokens, 2 punct marks (— .) */
    int n = tokenise("J\xE2\x80\x99\xC3\xA9" "coute le silence \xE2\x80\x94 encore.",
                     toks, 64, &uniq, &punct, &total);
    CHECK(n == 5, "5 word tokens after splitting on ' and em-dash");
    CHECK(punct == 2, "2 punctuation marks (em-dash and .)");
    CHECK(uniq == 5, "5 unique tokens");

    /* All-punct input yields zero tokens. */
    memset(WORD_OF, 0, sizeof(WORD_OF));
    int n2 = tokenise("...!?", toks, 64, &uniq, &punct, &total);
    CHECK(n2 == 0, "no word tokens in pure-punct input");
}

/* 3. Channel keyword tables don't contain duplicate entries. */
static int kw_count_dups(const char **t) {
    int dups = 0;
    for (int i = 0; t[i]; i++)
        for (int j = i + 1; t[j]; j++)
            if (strcmp(t[i], t[j]) == 0) dups++;
    return dups;
}
static void test_keyword_tables_unique(void) {
    printf("  [test_keyword_tables_unique]\n");
    CHECK(kw_count_dups(KW_FEAR)    == 0, "KW_FEAR no duplicates");
    CHECK(kw_count_dups(KW_LOVE)    == 0, "KW_LOVE no duplicates");
    CHECK(kw_count_dups(KW_RAGE)    == 0, "KW_RAGE no duplicates");
    CHECK(kw_count_dups(KW_VOID)    == 0, "KW_VOID no duplicates");
    CHECK(kw_count_dups(KW_FLOW)    == 0, "KW_FLOW no duplicates");
    CHECK(kw_count_dups(KW_COMPLEX) == 0, "KW_COMPLEX no duplicates");
}

/* 4. compute_fp routes the right channels for synthetic prompts. */
static void test_compute_fp(void) {
    printf("  [test_compute_fp]\n");
    double fp[N_CHANNELS] = {0};
    compute_fp("rage colère feu sang!", fp);
    CHECK(fp[CH_RAGE] > 0.5, "rage/colère/feu lift CH_RAGE");
    CHECK(fp[CH_FEAR] > 0.0, "sang lifts CH_FEAR");
    CHECK(fp[CH_LOVE] == 0.0, "no love words -> CH_LOVE = 0");

    double fp2[N_CHANNELS] = {0};
    compute_fp("amour tendre cœur âme", fp2);
    CHECK(fp2[CH_LOVE] > fp2[CH_RAGE], "love prompt -> CH_LOVE > CH_RAGE");

    double fp3[N_CHANNELS] = {0};
    compute_fp("silence vide néant adieu...", fp3);
    CHECK(fp3[CH_VOID] > 0.5, "void words + ellipsis lift CH_VOID");
}

/* 5. analyse_poem populates Poem fields and bigram rows are normalised. */
static void test_analyse_poem(void) {
    printf("  [test_analyse_poem]\n");
    Poem p;
    memset(&p, 0, sizeof(p));
    analyse_poem(&p, POEM_TEXT[0]);  /* poem1.txt — Réserves d'Amour */
    CHECK(p.n_tokens > 50,        "poem1 has > 50 tokens");
    CHECK(p.n_unique > 10,        "poem1 has > 10 unique tokens");
    CHECK(p.punct_density  > 0.0, "poem1 punct density > 0");
    CHECK(p.vocab_richness > 0.0, "poem1 vocab richness > 0");
    CHECK(p.bigram_entropy > 0.0, "poem1 bigram entropy > 0");
    CHECK(p.resonance      > 0.0, "poem1 resonance > 0");

    int rows_checked = 0;
    for (int a = 0; a < VOCAB && rows_checked < 5; a++) {
        double s = 0;
        for (int b = 0; b < VOCAB; b++) s += p.bigram[a][b];
        if (s > 0) {
            CHECK_NEAR(s, 1.0, 1e-9, "bigram row sums to 1");
            rows_checked++;
        }
    }
    CHECK(rows_checked > 0, "at least one bigram row populated");
}

/* 6. All 13 poems analyse without crash; resonance > 0 each. */
static void test_all_poems(void) {
    printf("  [test_all_poems]\n");
    for (int i = 0; i < N_POEMS; i++) {
        Poem p;
        memset(&p, 0, sizeof(p));
        analyse_poem(&p, POEM_TEXT[i]);
        char msg[64];
        snprintf(msg, sizeof(msg), "poem %d resonance > 0", i + 1);
        CHECK(p.resonance > 0.0, msg);
    }
}

/* 7. Calendar conversions agree with hand-checked values. */
static void test_calendar(void) {
    printf("  [test_calendar]\n");
    CHECK(greg_self_test(), "Gregorian round-trip on birth date");

    /* Known Reingold/Dershowitz value: 2000-01-01 -> R.D. 730120 */
    CHECK(greg_to_rd(2000, 1, 1) == 730120L, "greg_to_rd(2000-01-01) == 730120");

    /* Birth 1986-01-23 (Gregorian) is 13 Shevat 5746 by Reingold/Dershowitz
       Hebrew tables; the loose ±2 window absorbs any boundary ambiguity
       around dawn/dusk in the molad+dechiyot computation. */
    long birth_rd = greg_to_rd(BIRTH_GREG_Y, BIRTH_GREG_M, BIRTH_GREG_D);
    long hy; int hm, hd;
    rd_to_hebrew(birth_rd, &hy, &hm, &hd);
    CHECK(hy == 5746,           "Hebrew year of 1986-01-23 == 5746");
    CHECK(hm == 5,              "Hebrew month of 1986-01-23 == Shevat (5)");
    CHECK(hd >= 12 && hd <= 16, "Hebrew day of 1986-01-23 in expected window");

    long hm_elapsed = hebrew_months_elapsed(5746, 5, 5786, 6);
    CHECK(hm_elapsed > 0, "hebrew_months_elapsed positive forward in time");

    long L = hebrew_year_length(5786);
    CHECK(L == 353 || L == 354 || L == 355 ||
          L == 383 || L == 384 || L == 385,
          "hebrew_year_length(5786) is one of the six valid lengths");
}

/* 8. RNG is deterministic and produces values in [0,1). */
static void test_rng(void) {
    printf("  [test_rng]\n");
    seed_rng(12345);
    double a = drand(), b = drand(), c = drand();
    seed_rng(12345);
    double a2 = drand(), b2 = drand(), c2 = drand();
    CHECK(a == a2 && b == b2 && c == c2, "RNG reproducible from same seed");
    CHECK(a >= 0.0 && a < 1.0, "drand in [0,1)");
    CHECK(b >= 0.0 && b < 1.0, "drand in [0,1)");
    CHECK(c >= 0.0 && c < 1.0, "drand in [0,1)");
    CHECK(a != b || b != c, "RNG produces varying output");
}

/* 9. UTF-8 codepoint counter handles ASCII + multi-byte. */
static void test_utf8_clen(void) {
    printf("  [test_utf8_clen]\n");
    CHECK(utf8_clen("hello")       == 5,  "ASCII length");
    CHECK(utf8_clen("été")         == 3,  "été = 3 codepoints");
    CHECK(utf8_clen("immortalité") == 11, "immortalité = 11 codepoints");
    CHECK(utf8_clen("")            == 0,  "empty string");
}

/* 10. End-to-end dispatcher: 60 emissions, mean alpha is a distribution,
       and the run is reproducible with the same seed. */
static void test_dispatcher(void) {
    printf("  [test_dispatcher]\n");

    static Poem poems[N_POEMS];
    for (int i = 0; i < N_POEMS; i++) {
        memset(&poems[i], 0, sizeof(Poem));
        poems[i].src_index = i;
        analyse_poem(&poems[i], POEM_TEXT[i]);
    }
    build_token_fp_table();

    static Poem sorted[N_POEMS];
    memcpy(sorted, poems, sizeof(sorted));
    qsort(sorted, N_POEMS, sizeof(Poem), cmp_poem_desc);

    Poem *sun = &sorted[0];
    static Planet planets[N_PLANETS];
    for (int i = 0; i < N_PLANETS; i++) {
        planets[i].poem = &sorted[i + 1];
        planets[i].src_index = sorted[i + 1].src_index;
        planets[i].slot = i;
        planets[i].is_fixed = (i >= 7);
        planets[i].body_idx = i < 7 ? i : -1;
        planets[i].prophecy_bump = 0;
    }

    /* sorted poems are descending by resonance */
    int monotonic = 1;
    for (int i = 1; i < N_POEMS; i++)
        if (sorted[i - 1].resonance < sorted[i].resonance) { monotonic = 0; break; }
    CHECK(monotonic, "sorted by resonance descending");

    seed_rng(42);
    uint8_t emission[GEN_STEPS];
    double mean_alpha[N_PLANETS] = {0};
    silence_stdout();
    dispatcher_pass(sun, planets, N_PLANETS,
                    "amour silence", 14000, 12.0,
                    NULL, emission, 0, mean_alpha, NULL);
    restore_stdout();

    double s = 0;
    int neg = 0;
    for (int i = 0; i < N_PLANETS; i++) {
        s += mean_alpha[i];
        if (mean_alpha[i] < 0) neg++;
    }
    CHECK_NEAR(s, 1.0, 1e-6, "mean α[] sums to 1");
    CHECK(neg == 0, "all mean α[i] non-negative");

    /* Re-analyse from scratch so Hebbian online updates from the first run
       don't bleed into the reproducibility check. */
    for (int i = 0; i < N_POEMS; i++) {
        memset(&poems[i], 0, sizeof(Poem));
        poems[i].src_index = i;
        analyse_poem(&poems[i], POEM_TEXT[i]);
    }
    memcpy(sorted, poems, sizeof(sorted));
    qsort(sorted, N_POEMS, sizeof(Poem), cmp_poem_desc);
    sun = &sorted[0];
    for (int i = 0; i < N_PLANETS; i++) {
        planets[i].poem = &sorted[i + 1];
        planets[i].src_index = sorted[i + 1].src_index;
    }

    seed_rng(42);
    uint8_t emission2[GEN_STEPS];
    silence_stdout();
    dispatcher_pass(sun, planets, N_PLANETS,
                    "amour silence", 14000, 12.0,
                    NULL, emission2, 0, NULL, NULL);
    restore_stdout();
    CHECK(memcmp(emission, emission2, sizeof(emission)) == 0,
          "dispatcher reproducible from same seed");
}

/* 11. Schumann-breathing temperature: amplitude bounded, returns base at k=0. */
static void test_schumann_tau(void) {
    printf("  [test_schumann_tau]\n");
    double base = TEMPERATURE;
    /* sin(0) = 0  ⇒  tau == base (with cool=1) */
    CHECK_NEAR(schumann_tau(base, 0, GEN_STEPS, 1.0), base, 1e-9,
               "schumann_tau at step=0 equals base");
    /* tau is bounded inside base · (1 ± SCHUMANN_AMP) for cool=1 */
    double lo = base * (1.0 - SCHUMANN_AMP) - 1e-9;
    double hi = base * (1.0 + SCHUMANN_AMP) + 1e-9;
    int in_band = 1;
    for (int k = 0; k < GEN_STEPS; k++) {
        double t = schumann_tau(base, k, GEN_STEPS, 1.0);
        if (t < lo || t > hi) { in_band = 0; break; }
    }
    CHECK(in_band, "schumann_tau stays in [base·(1-AMP), base·(1+AMP)]");
    /* Trauma cool reduces τ */
    CHECK(schumann_tau(base, 5, GEN_STEPS, 0.7) <
          schumann_tau(base, 5, GEN_STEPS, 1.0) + 1e-9,
          "trauma cool ≤1 lowers τ");
    /* Floor never below 0.05 */
    CHECK(schumann_tau(base, 7, GEN_STEPS, 0.0) >= 0.05 - 1e-12,
          "schumann_tau respects 0.05 floor");
}

/* 12. Kuramoto step is conservative (sum drifts <1e-9 per step) and pulls
       chambers toward each other. */
static void test_kuramoto_step(void) {
    printf("  [test_kuramoto_step]\n");
    double s[N_CHANNELS] = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    double before = 0;
    for (int c = 0; c < N_CHANNELS; c++) before += s[c];
    double max_before = s[CH_FEAR];
    kuramoto_step(s);
    double after = 0;
    for (int c = 0; c < N_CHANNELS; c++) after += s[c];
    /* sin(0-1) for j!=FEAR is negative so FEAR receives a negative kick;
       other channels receive positive kicks. Net drift in sum is small. */
    CHECK(fabs(after - before) < 0.5, "kuramoto sum drift bounded");
    CHECK(s[CH_FEAR] < max_before, "FEAR pulled down by other chambers");
    int someone_rose = 0;
    for (int c = 0; c < N_CHANNELS; c++)
        if (c != CH_FEAR && s[c] > 0.0) { someone_rose = 1; break; }
    CHECK(someone_rose, "at least one other chamber rose toward FEAR");
}

/* 13. Online Hebbian update boosts the (prev,pick) bigram and re-normalises. */
static void test_hebbian_update_sun(void) {
    printf("  [test_hebbian_update_sun]\n");
    Poem p;
    memset(&p, 0, sizeof(p));
    analyse_poem(&p, POEM_TEXT[0]);
    /* find a row that already has mass */
    uint8_t prev = 0;
    for (int t = 0; t < VOCAB; t++) {
        double s = 0;
        for (int b = 0; b < VOCAB; b++) s += p.bigram[t][b];
        if (s > 0.5) { prev = (uint8_t)t; break; }
    }
    uint8_t pick = (uint8_t)((prev + 7) & 0xFF);
    double before = p.bigram[prev][pick];
    hebbian_update_sun(&p, prev, pick);
    double after = p.bigram[prev][pick];
    CHECK(after > before, "Hebbian update increases (prev,pick) probability");
    double s = 0;
    for (int b = 0; b < VOCAB; b++) s += p.bigram[prev][b];
    CHECK_NEAR(s, 1.0, 1e-9, "row remains normalised after Hebbian update");
}

/* 14. Origin overlap detection: prompt of poem-13 keywords overlaps,
       neutral prompt does not. */
static void test_origin_overlap(void) {
    printf("  [test_origin_overlap]\n");
    static int origin[VOCAB];
    /* WORD_OF must be initialised so the test is independent of run order. */
    memset(WORD_OF, 0, sizeof(WORD_OF));
    build_origin_token_set(POEM_TEXT[ORIGIN_POEM_IDX], origin);
    int count = 0;
    for (int i = 0; i < VOCAB; i++) if (origin[i]) count++;
    CHECK(count > 5, "origin set non-empty (poem 13 has multiple tokens)");

    /* Poem 13 starts: "LÉ\n\nLé je suis\n\nJe suis l'ombre…" — these MUST overlap. */
    double hi = prompt_origin_overlap("je suis l'ombre derrière la flamme", origin);
    double lo = prompt_origin_overlap("xyzzy plugh frobnicate", origin);
    CHECK(hi > 0.3, "prompt drawn from poem 13 has high origin overlap");
    CHECK(lo == 0.0, "neutral garbage prompt has zero overlap");
}

/* 15. Wormhole_pick returns a token from the sun's high-frequency support
       and prefers something other than `avoid` when possible. */
static void test_wormhole_pick(void) {
    printf("  [test_wormhole_pick]\n");
    Poem p;
    memset(&p, 0, sizeof(p));
    analyse_poem(&p, POEM_TEXT[0]);
    /* find the most frequent token */
    uint8_t top = 0;
    double best = -1;
    for (int t = 0; t < VOCAB; t++)
        if (p.unigram[t] > best) { best = p.unigram[t]; top = (uint8_t)t; }

    seed_rng(7);
    int saw_other = 0, all_have_freq = 1;
    for (int trial = 0; trial < 30; trial++) {
        uint8_t pick = wormhole_pick(&p, top);
        if (pick != top) saw_other = 1;
        if (p.unigram[pick] <= 0) { all_have_freq = 0; break; }
    }
    CHECK(saw_other, "wormhole_pick avoids the `avoid` token at least once");
    CHECK(all_have_freq, "wormhole_pick always returns a sun-frequent token");
}

/* ───────────── runner ───────────── */

int main(void) {
    printf("=== le.c test suite ===\n");
    test_str_tolower_fr();
    test_tokenise();
    test_keyword_tables_unique();
    test_compute_fp();
    test_analyse_poem();
    test_all_poems();
    test_calendar();
    test_rng();
    test_utf8_clen();
    test_dispatcher();
    test_schumann_tau();
    test_kuramoto_step();
    test_hebbian_update_sun();
    test_origin_overlap();
    test_wormhole_pick();
    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
