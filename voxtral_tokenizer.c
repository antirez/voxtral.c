/*
 * voxtral_tokenizer.c - Tekken tokenizer (decode only)
 *
 * Tekken tokenizer format (tekken.json):
 *   - vocab: array of {rank, token_bytes (base64), token_str}
 *   - special_tokens: array of {rank, token_str, is_control}
 *   - config: {default_vocab_size: 131072, default_num_special_tokens: 1000}
 *
 * Token ID mapping:
 *   - IDs 0..999: special tokens (special_tokens[rank])  (BOS=1, EOS=2, [STREAMING_PAD]=32, ...)
 *   - IDs 1000..131071: regular vocabulary tokens, where:
 *       token_id = 1000 + vocab_rank
 *       bytes = vocab[vocab_rank].token_bytes
 */

#include "voxtral_tokenizer.h"
#include "voxtral.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int vox_verbose;

#define MAX_VOCAB     130072
#define MAX_SPECIAL   1000
#define MAX_TOKEN_LEN 256

#define TEKKEN_NUM_SPECIAL 1000

struct vox_tokenizer {
    char **vocab;           /* Regular vocabulary strings [MAX_VOCAB] */
    char **special;         /* Special token strings [MAX_SPECIAL] */
    int n_vocab;
    int n_special;
    int bos_id;
    int eos_id;
};

/* ========================================================================
 * Minimal Base64 Decoder
 * ======================================================================== */

static const int b64_table[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
};

static int b64_decode(const char *in, char *out, int max_out) {
    int len = 0;
    int val = 0, bits = 0;

    for (; *in; in++) {
        int c = b64_table[(unsigned char)*in];
        if (c == -1) {
            if (*in == '=') break;
            continue;
        }
        val = (val << 6) | c;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (len < max_out - 1) {
                out[len++] = (char)((val >> bits) & 0xFF);
            }
        }
    }
    out[len] = '\0';
    return len;
}

/* ========================================================================
 * Minimal JSON Parser (for tekken.json)
 * ======================================================================== */

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

static int parse_str(const char **p, char *out, int max_len) {
    skip_ws(p);
    if (**p != '"') return -1;
    (*p)++;
    int i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') out[i++] = '\n';
            else if (**p == 't') out[i++] = '\t';
            else if (**p == 'r') out[i++] = '\r';
            else if (**p == '"') out[i++] = '"';
            else if (**p == '\\') out[i++] = '\\';
            else if (**p == 'u') {
                /* Unicode escape \uXXXX */
                (*p)++;
                unsigned int cp = 0;
                for (int j = 0; j < 4 && **p; j++, (*p)++) {
                    cp <<= 4;
                    if (**p >= '0' && **p <= '9') cp |= **p - '0';
                    else if (**p >= 'a' && **p <= 'f') cp |= **p - 'a' + 10;
                    else if (**p >= 'A' && **p <= 'F') cp |= **p - 'A' + 10;
                }
                /* Encode as UTF-8 */
                if (cp < 0x80 && i < max_len - 1) {
                    out[i++] = cp;
                } else if (cp < 0x800 && i < max_len - 2) {
                    out[i++] = 0xC0 | (cp >> 6);
                    out[i++] = 0x80 | (cp & 0x3F);
                } else if (i < max_len - 3) {
                    out[i++] = 0xE0 | (cp >> 12);
                    out[i++] = 0x80 | ((cp >> 6) & 0x3F);
                    out[i++] = 0x80 | (cp & 0x3F);
                }
                continue; /* Already advanced past digits */
            } else {
                out[i++] = **p;
            }
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';
    if (**p == '"') (*p)++;
    return 0;
}

static long parse_long(const char **p) {
    skip_ws(p);
    long val = 0;
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

static void skip_value(const char **p) {
    skip_ws(p);
    if (**p == '"') {
        (*p)++;
        while (**p && **p != '"') {
            if (**p == '\\') (*p)++;
            if (**p) (*p)++;
        }
        if (**p == '"') (*p)++;
    } else if (**p == '{') {
        int d = 1; (*p)++;
        while (**p && d > 0) {
            if (**p == '"') { (*p)++; while (**p && **p != '"') { if (**p == '\\') (*p)++; (*p)++; } if (**p) (*p)++; }
            else if (**p == '{') { d++; (*p)++; }
            else if (**p == '}') { d--; (*p)++; }
            else (*p)++;
        }
    } else if (**p == '[') {
        int d = 1; (*p)++;
        while (**p && d > 0) {
            if (**p == '"') { (*p)++; while (**p && **p != '"') { if (**p == '\\') (*p)++; (*p)++; } if (**p) (*p)++; }
            else if (**p == '[') { d++; (*p)++; }
            else if (**p == ']') { d--; (*p)++; }
            else (*p)++;
        }
    } else {
        while (**p && **p != ',' && **p != '}' && **p != ']') (*p)++;
    }
}

/* ========================================================================
 * Tokenizer Loading
 * ======================================================================== */

vox_tokenizer_t *vox_tokenizer_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "vox_tokenizer_load: cannot open %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc(size + 1);
    if (!json || fread(json, 1, size, f) != (size_t)size) {
        fclose(f);
        free(json);
        return NULL;
    }
    fclose(f);
    json[size] = '\0';

    vox_tokenizer_t *tok = (vox_tokenizer_t *)calloc(1, sizeof(vox_tokenizer_t));
    tok->vocab = (char **)calloc(MAX_VOCAB, sizeof(char *));
    tok->special = (char **)calloc(MAX_SPECIAL, sizeof(char *));
    tok->bos_id = 1;  /* <s> */
    tok->eos_id = 2;  /* </s> */

    const char *p = json;
    skip_ws(&p);
    if (*p != '{') goto fail;
    p++;

    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }

        char key[64];
        if (parse_str(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        skip_ws(&p);

        if (strcmp(key, "vocab") == 0) {
            /* Parse vocab array */
            if (*p != '[') break;
            p++;

            while (*p && *p != ']') {
                skip_ws(&p);
                if (*p == ',') { p++; continue; }
                if (*p != '{') break;
                p++;

                int rank = -1;
                char token_bytes[512] = {0};

                while (*p && *p != '}') {
                    skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    char k[32];
                    if (parse_str(&p, k, sizeof(k)) != 0) break;
                    skip_ws(&p);
                    if (*p != ':') break;
                    p++;
                    skip_ws(&p);

                    if (strcmp(k, "rank") == 0) {
                        rank = (int)parse_long(&p);
                    } else if (strcmp(k, "token_bytes") == 0) {
                        parse_str(&p, token_bytes, sizeof(token_bytes));
                    } else {
                        skip_value(&p);
                    }
                }
                if (*p == '}') p++;

                if (rank >= 0 && rank < MAX_VOCAB && token_bytes[0]) {
                    char decoded[MAX_TOKEN_LEN];
                    int len = b64_decode(token_bytes, decoded, sizeof(decoded));
                    tok->vocab[rank] = (char *)malloc(len + 1);
                    memcpy(tok->vocab[rank], decoded, len);
                    tok->vocab[rank][len] = '\0';
                    if (rank >= tok->n_vocab) tok->n_vocab = rank + 1;
                }
            }
            if (*p == ']') p++;
        } else if (strcmp(key, "special_tokens") == 0) {
            /* Parse special tokens array */
            if (*p != '[') break;
            p++;

            while (*p && *p != ']') {
                skip_ws(&p);
                if (*p == ',') { p++; continue; }
                if (*p != '{') break;
                p++;

                int rank = -1;
                char token_str[256] = {0};

                while (*p && *p != '}') {
                    skip_ws(&p);
                    if (*p == ',') { p++; continue; }
                    char k[32];
                    if (parse_str(&p, k, sizeof(k)) != 0) break;
                    skip_ws(&p);
                    if (*p != ':') break;
                    p++;
                    skip_ws(&p);

                    if (strcmp(k, "rank") == 0) {
                        rank = (int)parse_long(&p);
                    } else if (strcmp(k, "token_str") == 0) {
                        parse_str(&p, token_str, sizeof(token_str));
                    } else {
                        skip_value(&p);
                    }
                }
                if (*p == '}') p++;

                if (rank >= 0 && rank < MAX_SPECIAL && token_str[0]) {
                    tok->special[rank] = strdup(token_str);
                    if (rank >= tok->n_special) tok->n_special = rank + 1;
                }
            }
            if (*p == ']') p++;
        } else {
            skip_value(&p);
        }
    }

    free(json);

    if (vox_verbose >= 2)
        fprintf(stderr, "Tokenizer: %d vocab + %d special tokens\n",
                tok->n_vocab, tok->n_special);
    return tok;

fail:
    free(json);
    vox_tokenizer_free(tok);
    return NULL;
}

void vox_tokenizer_free(vox_tokenizer_t *tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (int i = 0; i < MAX_VOCAB; i++) free(tok->vocab[i]);
        free(tok->vocab);
    }
    if (tok->special) {
        for (int i = 0; i < MAX_SPECIAL; i++) free(tok->special[i]);
        free(tok->special);
    }
    free(tok);
}

const char *vox_tokenizer_decode(vox_tokenizer_t *tok, int token_id) {
    if (token_id >= TEKKEN_NUM_SPECIAL && token_id < TEKKEN_NUM_SPECIAL + tok->n_vocab) {
        return tok->vocab[token_id - TEKKEN_NUM_SPECIAL];
    }
    if (token_id >= 0 && token_id < tok->n_special) {
        return tok->special[token_id];
    }
    return NULL;
}

char *vox_tokenizer_decode_seq(vox_tokenizer_t *tok, const int *tokens, int n_tokens) {
    /* First pass: compute total length */
    int total = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < TEKKEN_NUM_SPECIAL) continue; /* ignore special/control */
        const char *s = vox_tokenizer_decode(tok, tokens[i]);
        if (s) total += strlen(s);
    }

    char *result = (char *)malloc(total + 1);
    result[0] = '\0';
    int pos = 0;

    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < TEKKEN_NUM_SPECIAL) continue; /* ignore special/control */
        const char *s = vox_tokenizer_decode(tok, tokens[i]);
        if (s) {
            int len = strlen(s);
            memcpy(result + pos, s, len);
            pos += len;
        }
    }
    result[pos] = '\0';

    return result;
}

int vox_tokenizer_bos(vox_tokenizer_t *tok) {
    return tok->bos_id;
}

int vox_tokenizer_eos(vox_tokenizer_t *tok) {
    return tok->eos_id;
}

int vox_tokenizer_vocab_size(vox_tokenizer_t *tok) {
    (void)tok;
    return TEKKEN_NUM_SPECIAL + MAX_VOCAB;
}
