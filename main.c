/*
 * main.c - CLI entry point for voxtral.c
 *
 * Usage: voxtral -d <model_dir> -i <input.wav> [options]
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_audio.h"
#include "voxtral_mic.h"
#ifdef USE_METAL
#include "voxtral_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#define DEFAULT_FEED_CHUNK 16000 /* 1 second at 16kHz */

static void usage(const char *prog) {
    fprintf(stderr, "voxtral.c — Voxtral Realtime 4B speech-to-text\n\n");
    fprintf(stderr, "Usage: %s -d <model_dir> (-i <input.wav> | --stdin | --mic) [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>    Model directory (with consolidated.safetensors, tekken.json)\n");
    fprintf(stderr, "  -i <file>   Input WAV file (16-bit PCM, any sample rate)\n");
    fprintf(stderr, "  --stdin     Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)\n");
    fprintf(stderr, "  --mic       Capture audio from default microphone (macOS)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -I <secs>   Encoder processing interval in seconds (default: 2.0)\n");
    fprintf(stderr, "  --mic-secs <s>  Stop mic capture after <s> seconds (default: until Ctrl+C)\n");
    fprintf(stderr, "  --alt <c>   Show alternative tokens within cutoff distance (0.0-1.0)\n");
    fprintf(stderr, "  --debug     Debug output (per-layer, per-chunk details)\n");
    fprintf(stderr, "  --silent    No status output (only transcription on stdout)\n");
    fprintf(stderr, "  -h          Show this help\n");
}

/* Drain pending tokens from stream and print to stdout */
static int first_token = 1;
static float alt_cutoff = -1; /* <0 means disabled */

static void drain_tokens(vox_stream_t *s) {
    if (alt_cutoff < 0) {
        /* Fast path: no alternatives */
        const char *tokens[64];
        int n;
        while ((n = vox_stream_get(s, tokens, 64)) > 0) {
            for (int i = 0; i < n; i++) {
                const char *t = tokens[i];
                if (first_token) {
                    while (*t == ' ') t++;
                    first_token = 0;
                }
                fputs(t, stdout);
            }
            fflush(stdout);
        }
    } else {
        /* Alternatives mode */
        const int n_alt = 3;
        const char *tokens[64 * 3];
        int n;
        while ((n = vox_stream_get_alt(s, tokens, 64, n_alt)) > 0) {
            for (int i = 0; i < n; i++) {
                const char *best = tokens[i * n_alt];
                if (!best) continue;
                /* Check for alternatives */
                int has_alt = 0;
                for (int a = 1; a < n_alt; a++) {
                    if (tokens[i * n_alt + a]) { has_alt = 1; break; }
                }
                if (has_alt) {
                    fputc('[', stdout);
                    for (int a = 0; a < n_alt; a++) {
                        const char *alt = tokens[i * n_alt + a];
                        if (!alt) break;
                        if (a > 0) fputc('|', stdout);
                        const char *t = alt;
                        if (a == 0 && first_token) {
                            while (*t == ' ') t++;
                            first_token = 0;
                        }
                        fputs(t, stdout);
                    }
                    fputc(']', stdout);
                } else {
                    const char *t = best;
                    if (first_token) {
                        while (*t == ' ') t++;
                        first_token = 0;
                    }
                    fputs(t, stdout);
                }
            }
            fflush(stdout);
        }
    }
}

/* Feed audio in chunks, printing tokens as they become available.
 * feed_chunk controls granularity: smaller = more responsive token output. */
static int feed_chunk = DEFAULT_FEED_CHUNK;
static void feed_and_drain(vox_stream_t *s, const float *samples, int n_samples) {
    int off = 0;
    while (off < n_samples) {
        int chunk = n_samples - off;
        if (chunk > feed_chunk) chunk = feed_chunk;
        vox_stream_feed(s, samples + off, chunk);
        off += chunk;
        drain_tokens(s);
    }
}

static volatile sig_atomic_t g_stop_mic = 0;

typedef struct {
    vox_stream_t *stream;
} mic_cb_ctx_t;

static void handle_sigint(int signo) {
    (void)signo;
    g_stop_mic = 1;
}

static int mic_chunk_cb(const float *samples, int n_samples, void *user) {
    mic_cb_ctx_t *cb = (mic_cb_ctx_t *)user;
    if (!cb || !cb->stream) return 1;
    if (g_stop_mic) return 1;
    feed_and_drain(cb->stream, samples, n_samples);
    return g_stop_mic ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *input_wav = NULL;
    int verbosity = 1; /* 0=silent, 1=normal, 2=debug */
    int use_stdin = 0;
    int use_mic = 0;
    float interval = -1.0f; /* <0 means use default */
    float mic_seconds = -1.0f; /* <=0 means unlimited */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_wav = argv[++i];
        } else if (strcmp(argv[i], "-I") == 0 && i + 1 < argc) {
            interval = (float)atof(argv[++i]);
            if (interval <= 0) {
                fprintf(stderr, "Error: -I requires a positive number of seconds\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--alt") == 0 && i + 1 < argc) {
            alt_cutoff = (float)atof(argv[++i]);
            if (alt_cutoff < 0 || alt_cutoff > 1) {
                fprintf(stderr, "Error: --alt requires a value between 0.0 and 1.0\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = 1;
        } else if (strcmp(argv[i], "--mic") == 0) {
            use_mic = 1;
        } else if (strcmp(argv[i], "--mic-secs") == 0 && i + 1 < argc) {
            mic_seconds = (float)atof(argv[++i]);
            if (mic_seconds <= 0) {
                fprintf(stderr, "Error: --mic-secs requires a positive number of seconds\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--debug") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--silent") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || (!input_wav && !use_stdin && !use_mic)) {
        usage(argv[0]);
        return 1;
    }
    if ((input_wav ? 1 : 0) + use_stdin + use_mic > 1) {
        fprintf(stderr, "Error: -i, --stdin, and --mic are mutually exclusive\n");
        return 1;
    }
    if (mic_seconds > 0 && !use_mic) {
        fprintf(stderr, "Error: --mic-secs requires --mic\n");
        return 1;
    }

    vox_verbose = verbosity;
    vox_verbose_audio = (verbosity >= 2) ? 1 : 0;

#ifdef USE_METAL
    vox_metal_init();
#endif

    /* Load model */
    vox_ctx_t *ctx = vox_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    vox_stream_t *s = vox_stream_init(ctx);
    if (!s) {
        fprintf(stderr, "Failed to init stream\n");
        vox_free(ctx);
        return 1;
    }
    if (alt_cutoff >= 0)
        vox_stream_set_alt(s, 3, alt_cutoff);
    if (interval > 0) {
        vox_set_processing_interval(s, interval);
        feed_chunk = (int)(interval * VOX_SAMPLE_RATE);
        if (feed_chunk < 160) feed_chunk = 160;
        if (feed_chunk > DEFAULT_FEED_CHUNK) feed_chunk = DEFAULT_FEED_CHUNK;
    }

    if (use_mic) {
#ifndef __APPLE__
        fprintf(stderr, "Error: --mic is only supported on macOS\n");
        vox_stream_free(s);
        vox_free(ctx);
        return 1;
#else
        mic_cb_ctx_t cb = { .stream = s };
        int mic_total_samples = 0;
        g_stop_mic = 0;
        signal(SIGINT, handle_sigint);
        if (vox_verbose >= 1) {
            if (mic_seconds > 0)
                fprintf(stderr, "Capturing microphone for %.2f seconds...\n", mic_seconds);
            else
                fprintf(stderr, "Capturing microphone... press Ctrl+C to stop.\n");
        }

        if (vox_capture_mic_macos(mic_seconds, feed_chunk, mic_chunk_cb, &cb,
                                  &g_stop_mic, &mic_total_samples) != 0) {
            fprintf(stderr, "Microphone capture failed\n");
            vox_stream_free(s);
            vox_free(ctx);
            return 1;
        }
        if (mic_total_samples == 0 && vox_verbose >= 1) {
            fprintf(stderr, "No microphone audio was captured.\n");
            fprintf(stderr, "Check macOS microphone permission and System Settings input device.\n");
        }
#endif
    } else if (use_stdin) {
        /* Peek at first 4 bytes to detect WAV vs raw */
        uint8_t hdr[4];
        size_t hdr_read = fread(hdr, 1, 4, stdin);
        if (hdr_read < 4) {
            fprintf(stderr, "Not enough data on stdin\n");
            vox_stream_free(s);
            vox_free(ctx);
            return 1;
        }

        if (memcmp(hdr, "RIFF", 4) == 0) {
            /* WAV on stdin: buffer all, parse, feed in chunks */
            size_t capacity = 1024 * 1024;
            size_t size = 4;
            uint8_t *buf = (uint8_t *)malloc(capacity);
            if (!buf) { vox_stream_free(s); vox_free(ctx); return 1; }
            memcpy(buf, hdr, 4);

            while (1) {
                if (size == capacity) {
                    capacity *= 2;
                    uint8_t *tmp = (uint8_t *)realloc(buf, capacity);
                    if (!tmp) { free(buf); vox_stream_free(s); vox_free(ctx); return 1; }
                    buf = tmp;
                }
                size_t n = fread(buf + size, 1, capacity - size, stdin);
                if (n == 0) break;
                size += n;
            }

            int n_samples = 0;
            float *samples = vox_parse_wav_buffer(buf, size, &n_samples);
            free(buf);
            if (!samples) {
                fprintf(stderr, "Invalid WAV data on stdin\n");
                vox_stream_free(s);
                vox_free(ctx);
                return 1;
            }
            if (vox_verbose >= 1)
                fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                        n_samples, (float)n_samples / VOX_SAMPLE_RATE);

            feed_and_drain(s, samples, n_samples);
            free(samples);
        } else {
            /* Raw s16le 16kHz mono: stream incrementally */
            if (vox_verbose >= 2)
                fprintf(stderr, "Streaming raw s16le 16kHz mono from stdin\n");

            /* Feed the 4 peeked header bytes as 2 s16le samples */
            int16_t sv[2];
            memcpy(sv, hdr, 4);
            float f[2] = { sv[0] / 32768.0f, sv[1] / 32768.0f };
            vox_stream_feed(s, f, 2);
            drain_tokens(s);

            /* Read loop */
            int16_t raw_buf[4096];
            float fbuf[4096];
            while (1) {
                size_t nread = fread(raw_buf, sizeof(int16_t), 4096, stdin);
                if (nread == 0) break;
                for (size_t i = 0; i < nread; i++)
                    fbuf[i] = raw_buf[i] / 32768.0f;
                vox_stream_feed(s, fbuf, (int)nread);
                drain_tokens(s);
            }
        }
    } else {
        /* File input: load WAV, feed in chunks */
        int n_samples = 0;
        float *samples = vox_load_wav(input_wav, &n_samples);
        if (!samples) {
            fprintf(stderr, "Failed to load %s\n", input_wav);
            vox_stream_free(s);
            vox_free(ctx);
            return 1;
        }
        if (vox_verbose >= 1)
            fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                    n_samples, (float)n_samples / VOX_SAMPLE_RATE);

        feed_and_drain(s, samples, n_samples);
        free(samples);
    }

    vox_stream_finish(s);
    drain_tokens(s);
    fputs("\n", stdout);
    fflush(stdout);

    vox_stream_free(s);
    vox_free(ctx);
#ifdef USE_METAL
    vox_metal_shutdown();
#endif
    return 0;
}
