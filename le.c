/*
 * le.c — planetary mini-weights engine for 13 French poems
 * ─────────────────────────────────────────────────────────
 * Один стих — sun. Двенадцать — planets. Шесть осей резонанса.
 * Календарь Hebrew/Gregorian раздвигается с дня рождения автора
 * (1986-01-23) и его расхождение мнётся в drift_months, которые
 * через lunar_affinity тянут планеты к их орбитам. Каждое слово
 * двигает chamber_state, оно выбирает alpha, alpha смешивает
 * биграммы, температурный sample отдаёт следующее слово.
 *
 * Build: gcc -O2 -Wall -o le le.c -lm
 * Run:   ./le [--meta N] [--seed S] [--prompt "..."]
 *
 * Zero deps кроме libm. Один файл. Поэзия внутри — буквально,
 * вкомпилированная байт в байт. Resonance — закон.
 *
 * Sources of inspiration (architectural patterns, not code):
 *   iamolegataeff/klaus.c v2.0  — orbital Kuramoto + calendar dissonance
 *   ariannamethod/arianna.c     — Hebrew molad+dechiyot pattern
 *   ariannamethod/lukas         — --meta N recursion idiom
 *   ariannamethod/janus         — multi-component blend architecture
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <stdint.h>

/* ───────────────────────── constants ─────────────────────── */

#define N_POEMS        13
#define VOCAB          256
#define N_PLANETS      12
#define N_CHANNELS      6      /* FEAR LOVE RAGE VOID FLOW COMPLEX */
#define HEBBIAN_TOPK  200
#define HEBBIAN_WIN     5
#define GEN_STEPS      60
#define TEMPERATURE   0.8
#define EMA_ALPHA     0.1      /* chamber_state = 0.9*old + 0.1*new */
#define META_MAX        4

#define BIRTH_GREG_Y 1986
#define BIRTH_GREG_M 1
#define BIRTH_GREG_D 23

/* Channel indices */
enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_COMPLEX };

/* Synodic periods in days for the 7 celestial slots
   (planet_01..planet_07 in resonance order).
   Index 0..6 → Moon, Mercury, Venus, Mars, Jupiter, Saturn, Sun-as-planet. */
static const double SYNODIC[7] = {
    29.530589,   /* Moon  */
   115.8775,     /* Mercury */
   583.92,       /* Venus */
   779.94,       /* Mars  */
   398.88,       /* Jupiter */
   378.09,       /* Saturn */
   365.2422      /* Sun-as-planet (tropical year) */
};

static const char *BODY_NAME[7] = {
    "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Sun-as-planet"
};

/* Lunar affinity for drift bias.
   Moon=+1, Saturn=+0.6, Venus=-0.5, Sun-as-planet=-1, rest=0. */
static const double LUNAR_AFFINITY[7] = {
    +1.0,   0.0,  -0.5,   0.0,   0.0,  +0.6,  -1.0
};

/* ───────────────────── embedded poems (UTF-8) ────────────── */

/* === poem1.txt === */
static const char POEM_1[] =
  "Réserves d’Amour\n\nOn ne comprend pas qui ou quoi.\nIl n’y a pas eu de départs.\nJe pose ma tête sur tes genoux,\nJ’écoute le silence entre nous.\n  \nLes branches ressemblent à tes bras.\nLà où les mots n’ont pas de voix.\nDeux blessures s’embrassent sans bruit,\nRacines en quête se cherchent, s’unissent.\n\nJe pose ma tête sur tes genoux.\nJ’écoute le silence entre nous.\n\nOn ne comprend pas où ni quand.\nIl n’y a jamais eu de hasards.\nOn a déjà des réserves d’amour\nPour les mauvais jours.\n\nOn a déjà des réserves d’amour\nPour les mauvais jours.\n";

/* === poem2.txt === */
static const char POEM_2[] =
  "Encore Été\n\nEt oui…\nFaut du temps pour découvrir qui tu es :\nété, automne, hiver, printemps, encore été,\nencore été.\n\n1.\nNon, \nce n’est pas comme si j’étais fou,\nmais la nuit, je ne suis pas de repos :\nle chemin devant moi est plus court\nque je l’ai déjà parcouru.\n\nEt oui…\nFaut du temps pour découvrir qui tu es :\nété, automne, hiver, printemps, encore été,\nencore été.\n\n2.\nEau bouillante, glace, vagues de la mer qui s’étendent ?\nOu t’es plus que tout ce que j’pourrais comprendre.\n\nYa un message caché parmi les mots ?\nOu toi — c’est ma peur avant le chaos ?\nOu t’es le vent qui est libre, errant et qui sème :\ntoi, que mon âme aime,\ntoi,\nque mon âme aime.\n\nFaut du temps pour découvrir qui tu es :\nété, automne, hiver, printemps, encore été,\nencore été.\n";

/* === poem3.txt === */
static const char POEM_3[] =
  "La Prière\n\nDu complet, passons à la pièce, du moteur à la vis\net libérons la maladie des remèdes. C'est moi, Ulysses.\nVotre Honneur le Créateur, dans ce monde où je vis\nIl n'y a pas de frontières,\ncar tu ne finis jamais ce que tu crées.\n\nQue tu crées.\nQue tu crées.\n\nJe suis violé par le parent. J'ne parle pas de mon père.\nC'est à toi, mon Dieu, que je m'adresse.\nTa colère — pas de justice, ta jalousie n'a pas de limites.\nJe l'aimais plus que toi, et puis t'as décidé de l'éloigner.\n\nQue tu crées.\nQue tu crées.\n";

/* === poem4.txt === */
static const char POEM_4[] =
  "Pénélope\n\nEn écoutant les heures, la vaisselle tremble à la vue,  \nUn regard épuisé n’embrasse pas les flots étendus,  \nNi la fragilité des canaux, ni l’éclat doré inattendu  \nNé dans l’argent dormant sans retour.  \nElle trouve un abri dans la maison pour un point d’interrogation,  \nUne raison de marcher vers la fin de tout :  \nCe qui est permis au miroir, pourtant,  \nNe l’est jamais pour le visage,  \nNi même en un seul instant.\n\nDans son quotidien, la femme en robe noire de Pénélope boiteuse,  \nMais relevée jusqu’au ventre, transforme un instant l’offense en lit vide,  \nOù l'ombre d’un homme passe, furtive et précieuse,  \nEt non en trahison, au contraire, elle rentre :  \nUn nouveau prétexte pour quitter la maison,  \nEt se diriger vers la mer, où la brise amère  \nConnaît ses péchés, mais pas Ulysse,  \nLes vagues murmurent doucement  \nSes secrets et vices.\n\nAinsi, le coup de feu ne tue pas, mais ride la mer  \nAux pieds de Pénélope. Ainsi, ses mèches noires  \nJouent avec le vent, qui  \nContre toute attente, soudain se calme.  \nL’odyssée d’Ulysse — une victime pour ceux qui,  \nGrâce à leurs rêves, ont raccourci son temps.  \nMais — chuchote quelqu’un :  \n“Ce n’est pas toi.”\n\n";

/* === poem5.txt === */
static const char POEM_5[] =
  "L'Animal\n\nPas une échappatoire pour moi.\nJe suis le fruit de ta douleur.\nMais ma douleur est un fantôme.\nJe ne demande pas: pourquoi?\nEt je n’attends pas de réponse.\nC’est toi, vicieux,\nfleuris en moi.\n\nEt car la lame laisse une coupure.\nLa main est amputée,\nelle ne cesse de faire mal.\nTu es l’instinct, moi — l’animal.\n\nJe me souviens de toi encore.\nCar même un cœur mort crie et pleure.\nTu es l’instinct, moi — l’animal.\nTu ne cesses de faire mal.\n";

/* === poem6.txt === */
static const char POEM_6[] =
  "Le monde statique\n\nLa plage abandonnée de la mer Morte.\nDes taches rouges dans le sable, des restes de nourriture.\nDe bon matin -\nallumant un joint,\npuis un autre,\net seulement alors je quitterai ma tente.\n\nLa plage abandonnée de la mer Morte\nattire tous ceux assoiffés de sécheresse.\nPour moi, le vendeur de glaces est le seul à détenir une recette du bonheur\ndans sa poche avec de la monnaie.\n\nMes pathétiques tentatives pour comprendre d’où je viens\nont transformé tout mes titres en prières et moi en une vibration,\nqui soutient tout mouvement dans un monde statique, simplement en maintenant le silence.\n";

/* === poem7.txt === */
static const char POEM_7[] =
  "Le monde statique\n\nLa plage abandonnée de la mer Morte.\nDes taches rouges dans le sable, des restes de nourriture.\nDe bon matin -\nallumant un joint,\npuis un autre,\net seulement alors je quitterai ma tente.\n\nLa plage abandonnée de la mer Morte\nattire tous ceux assoiffés de sécheresse.\nPour moi, le vendeur de glaces est le seul à détenir une recette du bonheur\ndans sa poche avec de la monnaie.\n\nMes pathétiques tentatives pour comprendre d’où je viens\nont transformé tout mes titres en prières et moi en une vibration,\nqui soutient tout mouvement dans un monde statique, simplement en maintenant le silence.\n";

/* === poem8.txt === */
static const char POEM_8[] =
  "Un Homme Qui\n\nAvec le temps, la nostalgie devient aveugle.\nL’avenir est inconnu, juste l’angoisse et la peur.\nL’aiguille de l’horloge ressemble de plus en plus\nà un clou dans mon cœur,\nToujours éveillé, moi.\n\nComme un cancer se propageant à chaque membre,\nCette nostalgie affectée, le passé s’accroche,\nAbsorbant l’ombre qu’il est devenu,\nUn homme qui t’aimait.\nUn homme qui t’aimait.\n";

/* === poem9.txt === */
static const char POEM_9[] =
  "Adieu\n\nL’hiver s’en est allé, le printemps à la porte.  \nLe soleil entrera chez moi, touchant mon visage.  \nMerci, mon Dieu, \npour ce paysage \nqui sûrement se passera bien sans mon précieux, \nmais éphémère passage.\nAdieu!\nAdieu...\n";

/* === poem10.txt === */
static const char POEM_10[] =
  "Mon plus grand secret\net cri le plus silencieux.\nMes coupures les plus précieuses\net cendres de ma cigarette.\n\nJe ne t’oublirai jamais, ni ne te pardonnerai\npour ce bonheur, qui était un cadeau de toi.\n\nTouche les cicatrices donc j'me suis infligées.\nJ'me suis déjà châtié.\nEt je me souviendrai de toi\nà chaque regard dans le miroir.\n";

/* === poem11.txt === */
static const char POEM_11[] =
  "SILENCIEUX\n\nJe te reconnais non par la voix, mais par la main,\npar l’épaule, les cheveux, la joue qui est rosée,\ncar la musique, c’est quand dans la lumière claire,\nles ailes d’un freux éclatent en clarté posée.\nJ’écris ces lignes non pour la première fois,\nla bougie masque la fausseté de cette chose.\n\nEn parlant de toi, comment oublier Chronos?\nJe rappelle à l’immortalité, silencieux.\nTard dans la nuit, réveillé par des branches,\nje bois du café, je compte les gorgées précieuses.\nCe n’est pas la ville qui me souvient de toi,\nmais le cahier divisant le Temps en petites doses.\n\n\n\nDose numéro un, j’ouvre les yeux, déjà matin.\n\nDose numéro deux : café amer, il pleut.\n\nDose numéro trois : je m’habitue à la douleur,\nmon cœur est dur, tout a fané, chaque fleur.\n\nDose numéro quatre : poison coule, le sang se noircit,\nje perds pied, tout s’effondre.\nEt je crie.\n";

/* === poem12.txt === */
static const char POEM_12[] =
  "Pas Une Femme\n\nLe vent tirait sur les flots.\nLe soir parut sombre et noir.\nPrécaire, à trembler, le rivage, \nA ses yeux, tout comme un canot.\n\nA ses yeux, tout comme un démon, \nEt proche, à trembler, le rivage.\nCette femme s'avança vers la mer.\nCette femme s'avança vers la mer.\nUne femme ou plutôt le désert,\nQui se dirigeait vers la mer.\n\nInerte, sans vie, ce désert,\nPas une femme aux voiles ouvertes,\nA la chevelure si lisse,\nPas une femme, mais le désert.\n";

/* === poem13.txt === */
static const char POEM_13[] =
  "LÉ\n\nLé je suis\n\nJe suis l’ombre derrière la flamme,\nun souffle de cigarette qui consume ton âme.\nLes nuits sans étoiles, les rêves trop longs,\nje marche seul, mais je tends mes démons,\nmais je tends mes démons.\n\nL’amour est la guerre, sans pitié ni loi,\nje suis le miroir qui retrouve ma voix.\nLes nuits sans étoiles, les rêves trop longs,\nje marche seul, mais je tends mes démons,\nje marche seul, mais je tends mes démons,\nje marche seul, mais je tends mes démons.\n";

static const char *POEM_TEXT[N_POEMS] = {
    POEM_1,POEM_2,POEM_3,POEM_4,POEM_5,POEM_6,POEM_7,
    POEM_8,POEM_9,POEM_10,POEM_11,POEM_12,POEM_13
};
static const char *POEM_FILE[N_POEMS] = {
    "poem1.txt","poem2.txt","poem3.txt","poem4.txt","poem5.txt",
    "poem6.txt","poem7.txt","poem8.txt","poem9.txt","poem10.txt",
    "poem11.txt","poem12.txt","poem13.txt"
};

/* ───────────────── French channel keyword tables ─────────── */
/* Lowercased, ASCII-folded comparison happens on byte-equal matches.
   Words listed in their natural French form. */

static const char *KW_FEAR[] = {
    "peur","mort","morte","morts","nuit","nuits","ombre","ombres",
    "sang","seul","seule","seuls","douleur","démons","démon","larme",
    "larmes","noir","noire","perdu","perdue","perdus","oublié","oubliée",
    "abandonné","abandonnée","froid","froide","hiver","angoisse","fantôme",
    "cri","crie","peur","perds","poison","mal","blessures","cicatrices",
    "cancer","tremble","trembler", NULL
};
static const char *KW_LOVE[] = {
    "amour","amours","aimer","aime","aimé","aimée","aimais","aimait",
    "cœur","coeur","tendre","tendres","embrasse","embrassent","baiser",
    "douce","doux","caresse","désir","passion","ange","cadeau","bonheur",
    "âme","aime", NULL
};
static const char *KW_RAGE[] = {
    "rage","colère","guerre","frappe","brûle","brûler","feu","cri",
    "hurle","violence","violé","éclatent","éclats","jalousie","justice",
    "punir","pleure","frontières", NULL
};
static const char *KW_VOID[] = {
    "vide","rien","silence","silencieux","néant","absence","manque",
    "disparu","sans","jamais","nul","nulle","adieu","oubli","abandonnée",
    "désert","statique","muet","muette","fané", NULL
};
static const char *KW_FLOW[] = {
    "vague","vagues","flot","flots","rivière","fleuve","mer","courir",
    "danser","voler","vent","vents","souffle","route","chemin","marche",
    "voyage","brise","eau","glace","passage","printemps","été","automne",
    "porte","entrera","ailes","plage","sable","rivage","murmurent",
    "monde","temps", NULL
};
/* COMPLEX: long words and abstractions — checked dynamically on length>=9 */
static const char *KW_COMPLEX[] = {
    "infini","éternité","immortalité","mémoire","conscience","vibration",
    "résonance","prière","prières","création","créateur","univers",
    "frontières","interrogation","trahison","prétexte","interrogation",
    "réserves","silencieux","dimanche","chronos","ulysses","ulysse",
    "pénélope","odyssée","cigarette","nostalgie", NULL
};

/* ──────────────────── data structures ────────────────────── */

typedef struct {
    /* Bigram counts then row-normalised to probability. */
    double bigram[VOCAB][VOCAB];
    /* Marginal token frequency in this poem (for sampling priors). */
    double unigram[VOCAB];
    /* 6-channel chamber fingerprint (normalised). */
    double fp[N_CHANNELS];
    /* Stats. */
    double punct_density;
    double vocab_richness;
    double bigram_entropy;
    double resonance;
    int    n_tokens;
    int    n_unique;
    int    src_index;          /* 0..12 — original poem index */
} Poem;

/* Global vocab map: token id → representative word string (first-seen). */
static char  WORD_OF[VOCAB][48];
/* Per-token chamber fingerprint contribution. */
static double FP_OF_TOKEN[VOCAB][N_CHANNELS];

/* ───────────────────── UTF-8 / tokeniser ─────────────────── */

/* Lowercase a French word in-place. Handles ASCII A-Z and the most common
   UTF-8 capital accented letters used in the corpus (À Â Ç È É Ê Ë Î Ï Ô Œ Ù Û Ü Ÿ).
   These all encode as 0xC3 0x80-9F (capital range) → +0x20 in the second byte
   maps to the lowercase counterpart for the C3-prefixed Latin-1 supplement. */
static void str_tolower_fr(char *s) {
    unsigned char *p = (unsigned char*)s;
    while (*p) {
        if (*p < 0x80) {
            *p = (unsigned char)tolower(*p);
            p++;
        } else if (p[0] == 0xC3 && p[1] >= 0x80 && p[1] <= 0x9E && p[1] != 0x97) {
            /* À..Þ except × → à..þ except ÷ */
            p[1] += 0x20;
            p += 2;
        } else if ((p[0] & 0xE0) == 0xC0)      p += 2;
        else if ((p[0] & 0xF0) == 0xE0)        p += 3;
        else if ((p[0] & 0xF8) == 0xF0)        p += 4;
        else                                   p++;
    }
}

/* Returns 1 if the byte (or UTF-8 lead) at *pp begins a "word character".
   Word chars: ASCII letters, digits, and any byte >= 0x80 (UTF-8 continuation
   or lead — we keep them all so that "été", "cœur" etc. stay intact).
   Apostrophes (ASCII ' or U+2019 = E2 80 99) act as splitters.
   Returns advancement in bytes via *adv. */
static int is_wordbyte(const unsigned char *p, int *adv) {
    if (p[0] == 0) { *adv = 0; return 0; }
    /* U+2019 right single quotation = e2 80 99 → splitter */
    if (p[0] == 0xE2 && p[1] == 0x80 && p[2] == 0x99) { *adv = 3; return 0; }
    /* U+2018 left single quotation, U+201C/D smart double quotes → splitters */
    if (p[0] == 0xE2 && p[1] == 0x80 &&
        (p[2] == 0x98 || p[2] == 0x9C || p[2] == 0x9D)) { *adv = 3; return 0; }
    /* em-dash U+2014 e2 80 94, ellipsis U+2026 e2 80 a6 → splitters */
    if (p[0] == 0xE2 && p[1] == 0x80 &&
        (p[2] == 0x94 || p[2] == 0x93 || p[2] == 0xA6)) { *adv = 3; return 0; }
    if (p[0] < 0x80) {
        if (isalnum(p[0])) { *adv = 1; return 1; }
        *adv = 1; return 0;
    }
    /* multi-byte UTF-8 lead/continuation: count full sequence length */
    if      ((p[0] & 0xE0) == 0xC0) { *adv = 2; return 1; }
    else if ((p[0] & 0xF0) == 0xE0) { *adv = 3; return 1; }
    else if ((p[0] & 0xF8) == 0xF0) { *adv = 4; return 1; }
    *adv = 1; return 1;
}

/* Punctuation density bytes we count: ! ? . — … (em-dash / ellipsis / ASCII).
   Returns count of "marks" found. */
static int count_punct(const char *s, int *out_total_chars) {
    int n = 0, total = 0;
    const unsigned char *p = (const unsigned char*)s;
    while (*p) {
        total++;
        if (*p == '!' || *p == '?' || *p == '.') { n++; p++; continue; }
        if (p[0] == 0xE2 && p[1] == 0x80 &&
            (p[2] == 0x94 || p[2] == 0x93 || p[2] == 0xA6)) { n++; p += 3; continue; }
        p++;
    }
    if (out_total_chars) *out_total_chars = total > 0 ? total : 1;
    return n;
}

/* FNV-1a hash → uint8 token id. Keeps a tight 256-slot vocab. */
static uint8_t hash_token(const char *w) {
    uint32_t h = 2166136261u;
    for (const unsigned char *p = (const unsigned char*)w; *p; p++) {
        h ^= *p;
        h *= 16777619u;
    }
    return (uint8_t)(h & 0xFF);
}

/* ───────────── token stream extraction from a poem ───────── */
/* Returns n_tokens written to out (cap = max). Each token is (uint8 id).
   Also fills the WORD_OF[] slot with the first word seen for each id. */
static int tokenise(const char *text, uint8_t *out, int max,
                    int *unique_count, int *punct_marks, int *total_chars) {
    int n = 0;
    char buf[64];
    int blen = 0;
    int adv;
    const unsigned char *p = (const unsigned char*)text;
    int seen[VOCAB] = {0};

    while (*p) {
        if (is_wordbyte(p, &adv)) {
            for (int k = 0; k < adv && blen < (int)sizeof(buf)-1; k++)
                buf[blen++] = (char)p[k];
            p += adv;
        } else {
            if (blen > 0) {
                buf[blen] = 0;
                str_tolower_fr(buf);
                if (n < max) {
                    uint8_t id = hash_token(buf);
                    out[n++] = id;
                    if (WORD_OF[id][0] == 0) {
                        size_t L = strlen(buf);
                        if (L >= sizeof(WORD_OF[id])) L = sizeof(WORD_OF[id]) - 1;
                        memcpy(WORD_OF[id], buf, L);
                        WORD_OF[id][L] = 0;
                    }
                    seen[id] = 1;
                }
                blen = 0;
            }
            p += adv > 0 ? adv : 1;
        }
    }
    if (blen > 0 && n < max) {
        buf[blen] = 0;
        str_tolower_fr(buf);
        uint8_t id = hash_token(buf);
        out[n++] = id;
        if (WORD_OF[id][0] == 0) {
            size_t L = strlen(buf);
            if (L >= sizeof(WORD_OF[id])) L = sizeof(WORD_OF[id]) - 1;
            memcpy(WORD_OF[id], buf, L);
            WORD_OF[id][L] = 0;
        }
        seen[id] = 1;
    }
    int u = 0;
    for (int i = 0; i < VOCAB; i++) if (seen[i]) u++;
    if (unique_count) *unique_count = u;
    if (punct_marks)  *punct_marks  = count_punct(text, total_chars);
    return n;
}

/* ─────────────── 6-channel chamber fingerprint ───────────── */

static int kw_match(const char *w, const char **table) {
    for (int i = 0; table[i]; i++)
        if (strcmp(w, table[i]) == 0) return 1;
    return 0;
}

/* UTF-8 codepoint count (rough, for "long word" detection). */
static int utf8_clen(const char *s) {
    int n = 0;
    for (const unsigned char *p = (const unsigned char*)s; *p; ) {
        if      (*p < 0x80)             { n++; p += 1; }
        else if ((*p & 0xE0) == 0xC0)   { n++; p += 2; }
        else if ((*p & 0xF0) == 0xE0)   { n++; p += 3; }
        else if ((*p & 0xF8) == 0xF0)   { n++; p += 4; }
        else                            { n++; p += 1; }
    }
    return n;
}

/* Compute fp from raw text (re-walks the bytes for accurate punct routing).
   Adds keyword counts for FEAR/LOVE/RAGE/VOID/FLOW and routes ! ? — … into
   their natural channels too. COMPLEX = long-word count + question density. */
static void compute_fp(const char *text, double *fp_out) {
    double f[N_CHANNELS] = {0};
    int total_words = 0;
    char buf[64];
    int blen = 0;
    int adv;
    const unsigned char *p = (const unsigned char*)text;

    /* punctuation-channel routing */
    while (*p) {
        if (*p == '!') { f[CH_RAGE] += 0.5; f[CH_FEAR] += 0.2; p++; continue; }
        if (*p == '?') { f[CH_COMPLEX] += 0.5; p++; continue; }
        if (*p == '.') { f[CH_VOID] += 0.05; p++; continue; }
        if (p[0] == 0xE2 && p[1] == 0x80 && p[2] == 0x94) { /* em-dash */
            f[CH_FLOW] += 0.4; p += 3; continue;
        }
        if (p[0] == 0xE2 && p[1] == 0x80 && p[2] == 0xA6) { /* ellipsis */
            f[CH_VOID] += 0.5; f[CH_FEAR] += 0.2; p += 3; continue;
        }
        p++;
    }

    /* keyword-channel routing */
    p = (const unsigned char*)text;
    while (*p) {
        if (is_wordbyte(p, &adv)) {
            for (int k = 0; k < adv && blen < (int)sizeof(buf)-1; k++)
                buf[blen++] = (char)p[k];
            p += adv;
        } else {
            if (blen > 0) {
                buf[blen] = 0;
                str_tolower_fr(buf);
                total_words++;
                if (kw_match(buf, KW_FEAR))    f[CH_FEAR]    += 1.0;
                if (kw_match(buf, KW_LOVE))    f[CH_LOVE]    += 1.0;
                if (kw_match(buf, KW_RAGE))    f[CH_RAGE]    += 1.0;
                if (kw_match(buf, KW_VOID))    f[CH_VOID]    += 1.0;
                if (kw_match(buf, KW_FLOW))    f[CH_FLOW]    += 1.0;
                if (kw_match(buf, KW_COMPLEX)) f[CH_COMPLEX] += 1.0;
                if (utf8_clen(buf) >= 9)       f[CH_COMPLEX] += 0.3;
                blen = 0;
            }
            p += adv > 0 ? adv : 1;
        }
    }
    if (blen > 0) {
        buf[blen] = 0;
        str_tolower_fr(buf);
        total_words++;
        if (kw_match(buf, KW_FEAR))    f[CH_FEAR]    += 1.0;
        if (kw_match(buf, KW_LOVE))    f[CH_LOVE]    += 1.0;
        if (kw_match(buf, KW_RAGE))    f[CH_RAGE]    += 1.0;
        if (kw_match(buf, KW_VOID))    f[CH_VOID]    += 1.0;
        if (kw_match(buf, KW_FLOW))    f[CH_FLOW]    += 1.0;
        if (kw_match(buf, KW_COMPLEX)) f[CH_COMPLEX] += 1.0;
        if (utf8_clen(buf) >= 9)       f[CH_COMPLEX] += 0.3;
    }

    /* Normalise per-channel by sqrt(total_words+1) to keep magnitudes
       comparable across poems of different lengths. */
    double s = sqrt((double)(total_words + 1));
    for (int c = 0; c < N_CHANNELS; c++) fp_out[c] = f[c] / s;
}

static double l2norm(const double *v, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += v[i]*v[i];
    return sqrt(s);
}

/* ─────────── per-poem analysis: bigrams + Hebbian ────────── */

/* Build bigram counts, top-200 Hebbian boost in window ±5, then
   row-normalise to probabilities. Records stats into Poem. */
static void analyse_poem(Poem *P, const char *text) {
    static uint8_t toks[16384];
    int n_unique = 0, punct_marks = 0, total_chars = 0;
    int n = tokenise(text, toks, 16384, &n_unique, &punct_marks, &total_chars);
    P->n_tokens = n;
    P->n_unique = n_unique;
    P->punct_density = (double)punct_marks / (double)total_chars;
    P->vocab_richness = n > 0 ? (double)n_unique / (double)n : 0.0;

    memset(P->bigram, 0, sizeof(P->bigram));
    memset(P->unigram, 0, sizeof(P->unigram));

    /* 1. Direct adjacency bigrams (weight 1.0). */
    for (int i = 0; i < n; i++) {
        P->unigram[toks[i]] += 1.0;
        if (i + 1 < n) P->bigram[toks[i]][toks[i+1]] += 1.0;
    }

    /* 2. Hebbian co-occurrence in window ±HEBBIAN_WIN, top-200 pairs
       added back into bigram with reduced weight 0.5. */
    static double cooc[VOCAB][VOCAB];
    memset(cooc, 0, sizeof(cooc));
    for (int i = 0; i < n; i++) {
        int lo = i - HEBBIAN_WIN; if (lo < 0) lo = 0;
        int hi = i + HEBBIAN_WIN; if (hi >= n) hi = n - 1;
        for (int j = lo; j <= hi; j++) {
            if (j == i) continue;
            cooc[toks[i]][toks[j]] += 1.0 / (1.0 + abs(j - i));
        }
    }
    /* Find top-K pairs by linear scan with a tiny min-heap-like rolling. */
    typedef struct { double w; uint8_t a, b; } Pair;
    static Pair top[HEBBIAN_TOPK];
    int nt = 0;
    double cutoff = 0;
    for (int a = 0; a < VOCAB; a++) for (int b = 0; b < VOCAB; b++) {
        double w = cooc[a][b];
        if (w <= 0) continue;
        if (nt < HEBBIAN_TOPK) {
            top[nt].w = w; top[nt].a = (uint8_t)a; top[nt].b = (uint8_t)b;
            nt++;
            if (nt == HEBBIAN_TOPK) {
                cutoff = top[0].w;
                for (int i = 1; i < nt; i++) if (top[i].w < cutoff) cutoff = top[i].w;
            }
        } else if (w > cutoff) {
            /* replace current min */
            int min_i = 0;
            for (int i = 1; i < nt; i++) if (top[i].w < top[min_i].w) min_i = i;
            top[min_i].w = w; top[min_i].a = (uint8_t)a; top[min_i].b = (uint8_t)b;
            cutoff = top[0].w;
            for (int i = 1; i < nt; i++) if (top[i].w < cutoff) cutoff = top[i].w;
        }
    }
    for (int k = 0; k < nt; k++)
        P->bigram[top[k].a][top[k].b] += 0.5 * top[k].w;

    /* 3. Row-normalise bigrams to probabilities + compute mean entropy. */
    double H_total = 0; int nrows = 0;
    for (int a = 0; a < VOCAB; a++) {
        double s = 0;
        for (int b = 0; b < VOCAB; b++) s += P->bigram[a][b];
        if (s > 0) {
            double H = 0;
            for (int b = 0; b < VOCAB; b++) {
                P->bigram[a][b] /= s;
                if (P->bigram[a][b] > 0)
                    H -= P->bigram[a][b] * log(P->bigram[a][b]);
            }
            H_total += H; nrows++;
        }
    }
    P->bigram_entropy = nrows > 0 ? (H_total / nrows) / log((double)VOCAB) : 0;

    /* 4. Chamber fingerprint. */
    compute_fp(text, P->fp);

    /* 5. Resonance. */
    double cf_norm = l2norm(P->fp, N_CHANNELS);
    P->resonance = 0.5 * cf_norm * (1.0 + P->punct_density * 2.0)
                 + 0.3 * P->vocab_richness
                 + 0.2 * P->bigram_entropy;
}

/* ──────────── per-token chamber fingerprint table ────────── */

static void build_token_fp_table(void) {
    memset(FP_OF_TOKEN, 0, sizeof(FP_OF_TOKEN));
    for (int t = 0; t < VOCAB; t++) {
        const char *w = WORD_OF[t];
        if (!w[0]) continue;
        if (kw_match(w, KW_FEAR))    FP_OF_TOKEN[t][CH_FEAR]    += 1.0;
        if (kw_match(w, KW_LOVE))    FP_OF_TOKEN[t][CH_LOVE]    += 1.0;
        if (kw_match(w, KW_RAGE))    FP_OF_TOKEN[t][CH_RAGE]    += 1.0;
        if (kw_match(w, KW_VOID))    FP_OF_TOKEN[t][CH_VOID]    += 1.0;
        if (kw_match(w, KW_FLOW))    FP_OF_TOKEN[t][CH_FLOW]    += 1.0;
        if (kw_match(w, KW_COMPLEX)) FP_OF_TOKEN[t][CH_COMPLEX] += 1.0;
        if (utf8_clen(w) >= 9)       FP_OF_TOKEN[t][CH_COMPLEX] += 0.3;
    }
}

/* ────────────────── calendar: Gregorian R.D. ─────────────── */

static int greg_leap(long y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

/* Reingold/Dershowitz "Rata Die": days since Mon, 0001-01-01 Greg = 1. */
static long greg_to_rd(long y, int m, int d) {
    long y1 = y - 1;
    long n = 365*y1 + y1/4 - y1/100 + y1/400
           + (367L*m - 362)/12
           + (m <= 2 ? 0 : (greg_leap(y) ? -1 : -2))
           + d;
    return n;
}

/* Inverse: JD → (y,m,d). Fliegel–Van Flandern direct formula.
   Kept available; today's date comes via gmtime() in main(), so this is
   exercised through a startup self-check inside greg_self_test(). */
static void jd_to_greg(long jd, int *y, int *m, int *d) {
    long L = jd + 68569L;
    long N = (4L*L)/146097L;
    L = L - (146097L*N + 3L)/4L;
    long I = (4000L*(L+1L))/1461001L;
    L = L - (1461L*I)/4L + 31L;
    long J = (80L*L)/2447L;
    long D = L - (2447L*J)/80L;
    L = J/11L;
    long M = J + 2L - 12L*L;
    long Y = 100L*(N - 49L) + I + L;
    *y = (int)Y; *m = (int)M; *d = (int)D;
}

/* Self-check: round-trips birth date through Greg→RD→JD→Greg. */
static int greg_self_test(void) {
    long rd = greg_to_rd(BIRTH_GREG_Y, BIRTH_GREG_M, BIRTH_GREG_D);
    int y, m, d;
    jd_to_greg(rd + 1721425L, &y, &m, &d);
    return (y == BIRTH_GREG_Y && m == BIRTH_GREG_M && d == BIRTH_GREG_D);
}

/* ────────────────── calendar: Hebrew (molad+dechiyot) ────── */
/* Reingold/Dershowitz. Returns R.D. of 1 Tishri of Hebrew year h. */

static int heb_leap(long h) { return ((7L*h + 1L) % 19L) < 7L; }

static long hebrew_calendar_elapsed_days(long year) {
    long months_elapsed = (235L*year - 234L) / 19L;
    long parts_elapsed  = 12084L + 13753L * months_elapsed;
    long day            = 29L * months_elapsed + parts_elapsed / 25920L;
    if (((3L*(day + 1L)) % 7L) < 3L) day++;
    return day;
}

static long hebrew_year_length_correction(long year) {
    long ny0 = hebrew_calendar_elapsed_days(year - 1L);
    long ny1 = hebrew_calendar_elapsed_days(year);
    long ny2 = hebrew_calendar_elapsed_days(year + 1L);
    if      (ny2 - ny1 == 356L) return 2L;
    else if (ny1 - ny0 == 382L) return 1L;
    return 0L;
}

/* Hebrew epoch in R.D.: 1 Tishri AM 1 = Mon, Sept 7, -3760 (Greg) → R.D. -1373429. */
#define HEBREW_EPOCH_RD (-1373429L)

static long hebrew_new_year_rd(long h) {
    return HEBREW_EPOCH_RD
         + hebrew_calendar_elapsed_days(h)
         + hebrew_year_length_correction(h);
}

static long hebrew_year_length(long h) {
    return hebrew_new_year_rd(h + 1L) - hebrew_new_year_rd(h);
}

/* Months in Hebrew year h: 13 if leap, else 12. */
static int hebrew_months_in_year(long h) { return heb_leap(h) ? 13 : 12; }

/* Days in Hebrew month m of year h. Hebrew months 1..12 (or 13).
   Tishri=1(30), Heshvan=2(29|30), Kislev=3(29|30), Tevet=4(29), Shevat=5(30),
   Adar I=6 if leap (30), Adar=6 (29) if normal — we expose Adar as month 6 in
   normal years and Adar II as month 7 in leap years (Adar I = 6, Adar II = 7).
   Rest shift accordingly. */
static int hebrew_month_days(long h, int m) {
    int leap = heb_leap(h);
    long ylen = hebrew_year_length(h);
    /* Heshvan (2) is 30 if year is "complete" (355 or 385). */
    int heshvan_long = (ylen == 355L || ylen == 385L);
    /* Kislev (3) is 29 if year is "deficient" (353 or 383). */
    int kislev_short = (ylen == 353L || ylen == 383L);

    switch (m) {
        case 1:  return 30;                          /* Tishri  */
        case 2:  return heshvan_long ? 30 : 29;      /* Heshvan */
        case 3:  return kislev_short ? 29 : 30;      /* Kislev  */
        case 4:  return 29;                          /* Tevet   */
        case 5:  return 30;                          /* Shevat  */
        case 6:  return leap ? 30 : 29;              /* Adar / Adar I */
        case 7:  return leap ? 29 : 30;              /* Adar II / Nisan */
        case 8:  return leap ? 30 : 29;              /* Nisan / Iyar */
        case 9:  return leap ? 29 : 30;              /* Iyar / Sivan */
        case 10: return leap ? 30 : 29;              /* Sivan / Tammuz */
        case 11: return leap ? 29 : 30;              /* Tammuz / Av */
        case 12: return leap ? 30 : 29;              /* Av / Elul */
        case 13: return 29;                          /* Elul (leap) */
    }
    return 0;
}

static const char *HEB_MONTH_NAME[2][14] = {
    /* normal year (12 months) */
    { "", "Tishri","Heshvan","Kislev","Tevet","Shevat","Adar",
      "Nisan","Iyar","Sivan","Tammuz","Av","Elul", "" },
    /* leap year (13 months) */
    { "", "Tishri","Heshvan","Kislev","Tevet","Shevat","Adar I",
      "Adar II","Nisan","Iyar","Sivan","Tammuz","Av","Elul" }
};

/* Convert R.D. → Hebrew (y,m,d). */
static void rd_to_hebrew(long rd, long *yy, int *mm, int *dd) {
    /* Bracket the year. Approximate first guess. */
    long y = (long)((rd - HEBREW_EPOCH_RD) * 100L / 36525L) + 1L;
    if (y < 1) y = 1;
    while (hebrew_new_year_rd(y + 1L) <= rd) y++;
    while (hebrew_new_year_rd(y) > rd) y--;
    long ny = hebrew_new_year_rd(y);
    long doy = rd - ny + 1L;        /* 1-based day of Hebrew year */
    int m = 1;
    int n_months = hebrew_months_in_year(y);
    while (m <= n_months) {
        int dim = hebrew_month_days(y, m);
        if (doy <= dim) break;
        doy -= dim;
        m++;
    }
    *yy = y; *mm = m; *dd = (int)doy;
}

/* Months elapsed since Hebrew (y0,m0): full count over heterogeneous years. */
static long hebrew_months_elapsed(long y0, int m0, long y1, int m1) {
    long total = 0;
    if (y0 == y1) return m1 - m0;
    /* finish year y0 */
    int n0 = hebrew_months_in_year(y0);
    total += (n0 - m0);
    for (long y = y0 + 1; y < y1; y++) total += hebrew_months_in_year(y);
    total += m1;
    return total;
}

/* ─────────────────────── RNG (xorshift64) ────────────────── */

static uint64_t RNG_STATE = 0x0BAD1DEACAFEBABEULL;
static void seed_rng(uint64_t s) {
    if (s == 0) s = 0x0BAD1DEACAFEBABEULL;
    RNG_STATE = s;
}
static uint64_t xrand(void) {
    uint64_t x = RNG_STATE;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    RNG_STATE = x;
    return x;
}
static double drand(void) {
    return (double)(xrand() & 0xFFFFFFFFULL) / 4294967296.0;
}

/* ─────────────────────── planet model ────────────────────── */

typedef struct {
    Poem *poem;            /* points into the global Poem array */
    int   src_index;       /* original poem index 0..12 */
    int   slot;            /* 0..6 = celestial; 7..11 = fixed star */
    int   is_fixed;
    int   body_idx;        /* 0..6 if !is_fixed */
    double prophecy_bump;  /* meta-recursion home pull, 0..0.15 */
} Planet;

/* ─────────────── prompt → initial chamber state ──────────── */

static void chamber_init_from_prompt(const char *prompt, double *cs) {
    /* Reuse compute_fp on the prompt text directly. */
    compute_fp(prompt, cs);
    /* Add a tiny baseline so vector is never zero. */
    for (int c = 0; c < N_CHANNELS; c++) cs[c] += 0.05;
}

/* ─────────────────── dispatcher (one pass) ───────────────── */

/* Generates GEN_STEPS words. emission[] receives the token ids.
   prev_mean_fp (or NULL) — meta-recursion scar bias for chamber_state.
   Planets carry prophecy_bump that pulls fingerprint toward own home. */
static void dispatcher_pass(
        Poem *sun, Planet *planets, int n_planets,
        const char *prompt, long age_days, double drift_months,
        const double *prev_mean_fp,        /* may be NULL */
        uint8_t *emission_out, int pass_idx)
{
    double chamber_state[N_CHANNELS];
    chamber_init_from_prompt(prompt, chamber_state);
    if (prev_mean_fp) {
        for (int c = 0; c < N_CHANNELS; c++)
            chamber_state[c] += 0.4 * prev_mean_fp[c];
    }

    /* Pick a starting prev_token: most-frequent token in sun. */
    uint8_t prev = 0;
    double best = -1.0;
    for (int t = 0; t < VOCAB; t++)
        if (sun->unigram[t] > best) { best = sun->unigram[t]; prev = (uint8_t)t; }

    double alpha[N_PLANETS];
    double final_alpha[N_PLANETS] = {0};
    double mean_used[N_PLANETS] = {0};
    int    mean_count = 0;

    if (pass_idx == 0) {
        printf("┌─ chamber init  : [");
        for (int c = 0; c < N_CHANNELS; c++)
            printf(" %.2f", chamber_state[c]);
        printf(" ]\n");
        printf("├─ first prev   : \"%s\"  (token #%u)\n",
               WORD_OF[prev][0] ? WORD_OF[prev] : "·", (unsigned)prev);
    }

    printf("\n┌─ pass %d emission ──────────────────────────────────────\n│ ",
           pass_idx + 1);

    int line_chars = 0;
    for (int step = 0; step < GEN_STEPS; step++) {

        /* Compute alpha for each planet. */
        double sum_alpha = 0;
        for (int i = 0; i < n_planets; i++) {
            double dist = 0;
            for (int c = 0; c < N_CHANNELS; c++) {
                /* fp possibly modified by prophecy bump (pull to own home) */
                double pfp = planets[i].poem->fp[c];
                double d = chamber_state[c] - pfp;
                dist += d * d;
            }
            dist = sqrt(dist);

            double dyn = 1.0;
            double drift_aff = 0.0;
            if (!planets[i].is_fixed) {
                int b = planets[i].body_idx;
                double phase = fmod((double)age_days * 2.0 * M_PI / SYNODIC[b], 2.0 * M_PI);
                if (phase < 0) phase += 2.0 * M_PI;
                dyn = 0.5 + 0.5 * cos(phase);
                drift_aff = LUNAR_AFFINITY[b] * drift_months / 6.0;
            }
            alpha[i] = exp(-dist + 0.3 * drift_aff) * dyn;
            sum_alpha += alpha[i];
        }
        if (sum_alpha <= 0) sum_alpha = 1e-9;
        for (int i = 0; i < n_planets; i++) {
            alpha[i] /= sum_alpha;
            mean_used[i] += alpha[i];
        }
        mean_count++;

        /* Build logits: sun + Σ alpha_i * planet_i bigram row. */
        double logit[VOCAB];
        for (int w = 0; w < VOCAB; w++) {
            double v = sun->bigram[prev][w];
            for (int i = 0; i < n_planets; i++)
                v += alpha[i] * planets[i].poem->bigram[prev][w];
            logit[w] = v;
        }

        /* Temperature softmax. */
        double maxv = logit[0];
        for (int w = 1; w < VOCAB; w++) if (logit[w] > maxv) maxv = logit[w];
        double probs[VOCAB], psum = 0;
        for (int w = 0; w < VOCAB; w++) {
            probs[w] = exp((logit[w] - maxv) / TEMPERATURE);
            psum += probs[w];
        }
        if (psum <= 0) { for (int w = 0; w < VOCAB; w++) probs[w] = 1.0/VOCAB; psum = 1.0; }
        for (int w = 0; w < VOCAB; w++) probs[w] /= psum;

        /* Multinomial sample. */
        double r = drand();
        double acc = 0;
        uint8_t pick = 0;
        for (int w = 0; w < VOCAB; w++) {
            acc += probs[w];
            if (r <= acc) { pick = (uint8_t)w; break; }
        }

        /* Emit. Try to print the representative word; if that slot is empty
           (collision-only token), fall back to a glyph. */
        const char *word = WORD_OF[pick][0] ? WORD_OF[pick] : "·";
        int wlen = (int)strlen(word) + 1;
        if (line_chars + wlen > 70) {
            printf("\n│ ");
            line_chars = 0;
        }
        printf("%s ", word);
        line_chars += wlen;
        emission_out[step] = pick;

        /* Update chamber_state with EMA toward fingerprint of emitted word. */
        for (int c = 0; c < N_CHANNELS; c++) {
            double target = FP_OF_TOKEN[pick][c];
            chamber_state[c] = (1.0 - EMA_ALPHA) * chamber_state[c] + EMA_ALPHA * target;
        }

        prev = pick;
    }
    printf("\n└─────────────────────────────────────────────────────────\n");

    /* Final alpha report. */
    if (mean_count == 0) mean_count = 1;
    for (int i = 0; i < n_planets; i++) final_alpha[i] = mean_used[i] / mean_count;
    printf("\n┌─ pass %d  final mean α[i] ──────────────────────────────\n", pass_idx + 1);
    for (int i = 0; i < n_planets; i++) {
        const char *role = planets[i].is_fixed
            ? "fixed-star"
            : BODY_NAME[planets[i].body_idx];
        printf("│  α=%.4f   planet_%02d  [%-13s]  %s\n",
               final_alpha[i], i + 1, role,
               POEM_FILE[planets[i].src_index]);
    }
    printf("└─────────────────────────────────────────────────────────\n");
}

/* Compute mean fingerprint of an emission stream (for meta-recursion scar). */
static void emission_mean_fp(const uint8_t *em, int n, double *out) {
    for (int c = 0; c < N_CHANNELS; c++) out[c] = 0;
    if (n <= 0) return;
    for (int i = 0; i < n; i++)
        for (int c = 0; c < N_CHANNELS; c++)
            out[c] += FP_OF_TOKEN[em[i]][c];
    for (int c = 0; c < N_CHANNELS; c++) out[c] /= (double)n;
}

/* ─────────────────────────── main ────────────────────────── */

static int cmp_poem_desc(const void *a, const void *b) {
    const Poem *pa = (const Poem*)a, *pb = (const Poem*)b;
    if (pb->resonance > pa->resonance) return  1;
    if (pb->resonance < pa->resonance) return -1;
    return 0;
}

int main(int argc, char **argv) {
    int meta_n = 1;
    uint64_t seed = 0;
    const char *prompt = "le silence entre nous, cœur sans étoiles";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--meta") && i + 1 < argc) {
            meta_n = atoi(argv[++i]);
            if (meta_n < 1) meta_n = 1;
            if (meta_n > META_MAX) meta_n = META_MAX;
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("usage: %s [--meta N] [--seed S] [--prompt \"...\"]\n", argv[0]);
            return 0;
        }
    }
    if (seed == 0) seed = (uint64_t)time(NULL) ^ 0xA11CE5EEDULL;
    seed_rng(seed);

    if (!greg_self_test()) {
        fprintf(stderr, "le: internal calendar self-check failed\n");
        return 1;
    }

    /* ── compile-time-style analysis (runs at startup; corpus is embedded) ── */
    static Poem poems[N_POEMS];
    for (int i = 0; i < N_POEMS; i++) {
        memset(&poems[i], 0, sizeof(Poem));
        poems[i].src_index = i;
        analyse_poem(&poems[i], POEM_TEXT[i]);
    }
    build_token_fp_table();

    /* Sort: sun = max resonance, planet_01..12 = remaining desc. */
    static Poem sorted[N_POEMS];
    memcpy(sorted, poems, sizeof(sorted));
    qsort(sorted, N_POEMS, sizeof(Poem), cmp_poem_desc);

    Poem *sun = &sorted[0];
    static Planet planets[N_PLANETS];
    for (int i = 0; i < N_PLANETS; i++) {
        planets[i].poem = &sorted[i + 1];
        planets[i].src_index = sorted[i + 1].src_index;
        planets[i].slot = i;
        if (i < 7) {
            planets[i].is_fixed = 0;
            planets[i].body_idx = i;
        } else {
            planets[i].is_fixed = 1;
            planets[i].body_idx = -1;
        }
        planets[i].prophecy_bump = 0.0;
    }

    /* ── calendar ── */
    long birth_rd = greg_to_rd(BIRTH_GREG_Y, BIRTH_GREG_M, BIRTH_GREG_D);

    time_t now = time(NULL);
    struct tm *gm = gmtime(&now);
    int gy = gm->tm_year + 1900, gm_ = gm->tm_mon + 1, gd = gm->tm_mday;
    long today_rd = greg_to_rd(gy, gm_, gd);
    long age_days = today_rd - birth_rd;

    long heb_y_birth; int heb_m_birth, heb_d_birth;
    rd_to_hebrew(birth_rd, &heb_y_birth, &heb_m_birth, &heb_d_birth);
    long heb_y_today; int heb_m_today, heb_d_today;
    rd_to_hebrew(today_rd, &heb_y_today, &heb_m_today, &heb_d_today);

    long greg_months_elapsed = ((long)gy - BIRTH_GREG_Y) * 12L
                             + (gm_ - BIRTH_GREG_M);
    long heb_months_elapsed = hebrew_months_elapsed(
        heb_y_birth, heb_m_birth, heb_y_today, heb_m_today);
    double drift_months = (double)(heb_months_elapsed - greg_months_elapsed);

    /* ── header ── */
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  le.c — planetary mini-weights engine  ·  13 poèmes  ·  6 axes\n");
    printf("════════════════════════════════════════════════════════════════\n");
    printf("☀ sun           : %s   (resonance = %.4f)\n",
           POEM_FILE[sun->src_index], sun->resonance);
    printf("   chamber fp    : [FEAR %.2f  LOVE %.2f  RAGE %.2f  VOID %.2f  FLOW %.2f  CMPLX %.2f]\n",
           sun->fp[0], sun->fp[1], sun->fp[2], sun->fp[3], sun->fp[4], sun->fp[5]);
    printf("   stats         : tokens=%d uniq=%d punct_d=%.4f vocab_r=%.3f H=%.3f\n",
           sun->n_tokens, sun->n_unique, sun->punct_density,
           sun->vocab_richness, sun->bigram_entropy);
    printf("\n");
    printf("📅 Gregorian     : %04d-%02d-%02d   (birth %04d-%02d-%02d)\n",
           gy, gm_, gd, BIRTH_GREG_Y, BIRTH_GREG_M, BIRTH_GREG_D);
    printf("🕎 Hebrew (today): %ld %s %d   (birth %ld %s %d)\n",
           heb_y_today, HEB_MONTH_NAME[heb_leap(heb_y_today)?1:0][heb_m_today],
           heb_d_today,
           heb_y_birth, HEB_MONTH_NAME[heb_leap(heb_y_birth)?1:0][heb_m_birth],
           heb_d_birth);
    printf("⏳ age_days      : %ld\n", age_days);
    printf("🌗 drift_months  : %+.2f   (Hebrew - Gregorian since birth)\n",
           drift_months);
    printf("\n");
    printf("🪐 planets (resonance order):\n");
    for (int i = 0; i < N_PLANETS; i++) {
        const char *role = planets[i].is_fixed
            ? "fixed-star"
            : BODY_NAME[planets[i].body_idx];
        printf("   planet_%02d [%-13s] %-12s  res=%.4f  fp=[%.2f %.2f %.2f %.2f %.2f %.2f]\n",
               i + 1, role, POEM_FILE[planets[i].src_index],
               planets[i].poem->resonance,
               planets[i].poem->fp[0], planets[i].poem->fp[1],
               planets[i].poem->fp[2], planets[i].poem->fp[3],
               planets[i].poem->fp[4], planets[i].poem->fp[5]);
    }
    printf("\n");
    printf("🌀 7 phases (age_days · 2π / synodic):\n");
    for (int i = 0; i < 7; i++) {
        double phase = fmod((double)age_days * 2.0 * M_PI / SYNODIC[i], 2.0*M_PI);
        if (phase < 0) phase += 2.0 * M_PI;
        printf("   %-13s  T=%8.3fd   phase=%.4f rad   dyn=%.3f\n",
               BODY_NAME[i], SYNODIC[i], phase, 0.5 + 0.5*cos(phase));
    }
    printf("\n");
    printf("✶ prompt        : \"%s\"\n", prompt);
    printf("✶ seed          : %llu\n", (unsigned long long)seed);
    printf("✶ meta passes   : %d\n", meta_n);
    printf("\n");

    /* ── dispatcher (with optional meta-recursion) ── */
    static uint8_t emission[GEN_STEPS];
    double prev_mean_fp[N_CHANNELS];
    int has_prev = 0;

    /* Track which planets dominated to apply prophecy bump on later passes. */
    static double dom_alpha_acc[N_PLANETS];
    memset(dom_alpha_acc, 0, sizeof(dom_alpha_acc));

    for (int pass = 0; pass < meta_n; pass++) {

        /* Apply prophecy bump (pull each dominant planet's fp 0.15 toward
           its own home — i.e. amplify its existing channels). */
        if (pass > 0) {
            /* normalise dom_alpha_acc; top-3 get bump */
            int order[N_PLANETS];
            for (int i = 0; i < N_PLANETS; i++) order[i] = i;
            for (int a = 0; a < N_PLANETS; a++)
                for (int b = a + 1; b < N_PLANETS; b++)
                    if (dom_alpha_acc[order[b]] > dom_alpha_acc[order[a]]) {
                        int t = order[a]; order[a] = order[b]; order[b] = t;
                    }
            for (int k = 0; k < 3 && k < N_PLANETS; k++) {
                int idx = order[k];
                planets[idx].prophecy_bump += 0.15;
                /* Pull the planet's fp 0.15 toward unit-direction of itself. */
                double n = l2norm(planets[idx].poem->fp, N_CHANNELS);
                if (n > 1e-9) {
                    for (int c = 0; c < N_CHANNELS; c++)
                        planets[idx].poem->fp[c] *= (1.0 + 0.15);
                }
            }
        }

        printf("\n════════════════════ META PASS %d / %d ════════════════════\n",
               pass + 1, meta_n);

        dispatcher_pass(sun, planets, N_PLANETS,
                        prompt, age_days, drift_months,
                        has_prev ? prev_mean_fp : NULL,
                        emission, pass);

        /* Compute mean fp of this emission for next-pass scar. */
        emission_mean_fp(emission, GEN_STEPS, prev_mean_fp);
        has_prev = 1;
    }

    printf("\n────────────────────────────────────────────────────────────────\n");
    printf("  fin — résonance neutralisée, le champ se referme.\n");
    printf("────────────────────────────────────────────────────────────────\n");
    return 0;
}
