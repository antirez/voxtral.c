/*
 * voxtral_audio.c - WAV loading and mel spectrogram computation
 *
 * Mel spectrogram parameters (from params.json):
 *   Sample rate: 16000 Hz
 *   Mel bins: 128
 *   Hop length: 160 (10ms)
 *   Window size: 400 (25ms)
 *   global_log_mel_max: 1.5
 *
 * This implementation matches vLLM:
 *   vllm/vllm/model_executor/models/voxtral.py::VoxtralEncoderModel.compute_whisper_melspec()
 * which uses:
 *   - torch.hann_window(window_size=400) (periodic)
 *   - torch.stft(n_fft=400, hop_length=160, center=True, pad_mode="reflect")
 *   - magnitudes = stft[..., :-1].abs() ** 2 (drop last frame)
 *   - mel_filter_bank from mistral_common (Slaney-style)
 *   - log10 clamp to [global_log_mel_max-8, global_log_mel_max], then (val+4)/4
 */

#include "voxtral_audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SAMPLE_RATE  16000
#define N_MEL        128
#define HOP_LENGTH   160
#define WIN_LENGTH   400
#define N_FFT        400
#define LOG_MEL_MAX  1.5f
#define N_FREQ       (N_FFT / 2 + 1)    /* 201 bins (matches training) */

/* ========================================================================
 * WAV File Loading
 * ======================================================================== */

/* Read little-endian values from buffer */
static uint16_t read_u16(const uint8_t *p) { return p[0] | (p[1] << 8); }
static uint32_t read_u32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24); }

/* Parse WAV data from a buffer. Returns mono float32 samples at 16kHz.
 * The caller keeps ownership of data. Returns NULL on error. */
float *vox_parse_wav_buffer(const uint8_t *data, size_t file_size, int *out_n_samples) {
    if (file_size < 44 || memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "parse_wav_buffer: not a valid WAV file\n");
        return NULL;
    }

    int channels = 0, sample_rate = 0, bits_per_sample = 0;
    int audio_format = 0;
    const uint8_t *pcm_data = NULL;
    int pcm_size = 0;

    const uint8_t *p = data + 12;
    const uint8_t *end = data + file_size;

    while (p + 8 <= end) {
        uint32_t chunk_size = read_u32(p + 4);
        if (memcmp(p, "fmt ", 4) == 0 && chunk_size >= 16 && p + 8 + chunk_size <= end) {
            audio_format = read_u16(p + 8);
            channels = read_u16(p + 10);
            sample_rate = read_u32(p + 12);
            bits_per_sample = read_u16(p + 22);
        } else if (memcmp(p, "data", 4) == 0) {
            pcm_data = p + 8;
            pcm_size = chunk_size;
            /* Piped ffmpeg writes 0xFFFFFFFF as data size (int -1); use rest of file */
            if (pcm_size <= 0 || pcm_data + pcm_size > end)
                pcm_size = (int)(end - pcm_data);
            break; /* data is always last meaningful chunk */
        }
        if (p + 8 + chunk_size > end) break;
        p += 8 + chunk_size;
        if (chunk_size & 1) p++;
    }

    if (audio_format != 1 || bits_per_sample != 16 || pcm_data == NULL || channels < 1) {
        fprintf(stderr, "parse_wav_buffer: unsupported format (need 16-bit PCM, got fmt=%d bits=%d)\n",
                audio_format, bits_per_sample);
        return NULL;
    }

    int n_frames = pcm_size / (channels * 2);

    float *samples = (float *)malloc(n_frames * sizeof(float));
    if (!samples) return NULL;

    const int16_t *src = (const int16_t *)pcm_data;
    for (int i = 0; i < n_frames; i++) {
        if (channels == 1) {
            samples[i] = src[i] / 32768.0f;
        } else {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                int16_t val;
                memcpy(&val, &src[i * channels + c], sizeof(int16_t));
                sum += val;
            }
            samples[i] = (sum / channels) / 32768.0f;
        }
    }

    /* Resample to 16kHz if needed */
    if (sample_rate != SAMPLE_RATE) {
        int new_n = (int)((long long)n_frames * SAMPLE_RATE / sample_rate);
        float *resampled = (float *)malloc(new_n * sizeof(float));
        if (!resampled) {
            free(samples);
            return NULL;
        }

        for (int i = 0; i < new_n; i++) {
            float src_pos = (float)i * sample_rate / SAMPLE_RATE;
            int idx = (int)src_pos;
            float frac = src_pos - idx;
            if (idx + 1 < n_frames) {
                resampled[i] = samples[idx] * (1.0f - frac) + samples[idx + 1] * frac;
            } else {
                resampled[i] = (idx < n_frames) ? samples[idx] : 0.0f;
            }
        }

        free(samples);
        samples = resampled;
        n_frames = new_n;

        if (vox_verbose_audio) {
            fprintf(stderr, "  Resampled %d -> %d Hz (%d samples)\n",
                    sample_rate, SAMPLE_RATE, n_frames);
        }
    }

    *out_n_samples = n_frames;
    return samples;
}

float *vox_load_wav(const char *path, int *out_n_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "vox_load_wav: cannot open %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    if (file_size <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    uint8_t *data = (uint8_t *)malloc(file_size);
    if (!data || fread(data, 1, file_size, f) != (size_t)file_size) {
        fclose(f);
        free(data);
        return NULL;
    }
    fclose(f);

    float *samples = vox_parse_wav_buffer(data, (size_t)file_size, out_n_samples);
    free(data);
    return samples;
}

float *vox_read_pcm_stdin(int *out_n_samples) {
    /* Read all of stdin into a growing buffer */
    size_t capacity = 1024 * 1024; /* 1 MB initial */
    size_t size = 0;
    uint8_t *buf = (uint8_t *)malloc(capacity);
    if (!buf) return NULL;

    while (1) {
        if (size == capacity) {
            capacity *= 2;
            uint8_t *tmp = (uint8_t *)realloc(buf, capacity);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
        size_t n = fread(buf + size, 1, capacity - size, stdin);
        if (n == 0) break;
        size += n;
    }

    if (size < 4) {
        fprintf(stderr, "vox_read_pcm_stdin: no data on stdin\n");
        free(buf);
        return NULL;
    }

    fprintf(stderr, "Read %zu bytes from stdin\n", size);

    /* Auto-detect: WAV (RIFF header) or raw s16le */
    if (memcmp(buf, "RIFF", 4) == 0) {
        fprintf(stderr, "Detected WAV format on stdin\n");
        float *samples = vox_parse_wav_buffer(buf, size, out_n_samples);
        free(buf);
        return samples;
    }

    /* Treat as raw s16le 16kHz mono */
    fprintf(stderr, "Treating stdin as raw s16le 16kHz mono\n");
    int n_frames = (int)(size / 2);
    float *samples = (float *)malloc(n_frames * sizeof(float));
    if (!samples) { free(buf); return NULL; }

    const int16_t *src = (const int16_t *)buf;
    for (int i = 0; i < n_frames; i++) {
        samples[i] = src[i] / 32768.0f;
    }

    free(buf);
    *out_n_samples = n_frames;
    return samples;
}

/* ========================================================================
 * Mel Filter Bank (Slaney-style, from mistral_common)
 * ======================================================================== */

static float hertz_to_mel(float freq) {
    const float min_log_hertz = 1000.0f;
    const float min_log_mel = 15.0f;
    const float logstep = 27.0f / logf(6.4f);

    float mels = 3.0f * freq / 200.0f;
    if (freq >= min_log_hertz) {
        mels = min_log_mel + logf(freq / min_log_hertz) * logstep;
    }
    return mels;
}

static float mel_to_hertz(float mels) {
    const float min_log_hertz = 1000.0f;
    const float min_log_mel = 15.0f;
    const float logstep = logf(6.4f) / 27.0f;

    float freq = 200.0f * mels / 3.0f;
    if (mels >= min_log_mel) {
        freq = min_log_hertz * expf(logstep * (mels - min_log_mel));
    }
    return freq;
}

/* Build mel filter bank: [N_MEL, N_FREQ] */
static float *build_mel_filters(void) {
    float *filters = (float *)calloc((size_t)N_MEL * N_FREQ, sizeof(float));
    if (!filters) return NULL;

    /* FFT bin center frequencies (0..8000 inclusive) */
    float fft_freqs[N_FREQ];
    for (int i = 0; i < N_FREQ; i++) {
        fft_freqs[i] = (float)i * ((float)SAMPLE_RATE / 2.0f) / (float)(N_FREQ - 1);
    }

    float mel_min = hertz_to_mel(0.0f);
    float mel_max = hertz_to_mel((float)SAMPLE_RATE / 2.0f);

    float filter_freqs[N_MEL + 2];
    float filter_diff[N_MEL + 1];

    for (int i = 0; i < N_MEL + 2; i++) {
        float mel = mel_min + (mel_max - mel_min) * (float)i / (float)(N_MEL + 1);
        filter_freqs[i] = mel_to_hertz(mel);
    }
    for (int i = 0; i < N_MEL + 1; i++) {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
        if (filter_diff[i] == 0.0f) filter_diff[i] = 1e-6f;
    }

    for (int m = 0; m < N_MEL; m++) {
        float enorm = 2.0f / (filter_freqs[m + 2] - filter_freqs[m]);
        for (int f = 0; f < N_FREQ; f++) {
            float down = (fft_freqs[f] - filter_freqs[m]) / filter_diff[m];
            float up = (filter_freqs[m + 2] - fft_freqs[f]) / filter_diff[m + 1];
            float val = fminf(down, up);
            if (val < 0.0f) val = 0.0f;
            filters[(size_t)m * N_FREQ + f] = val * enorm;
        }
    }

    return filters;
}

/* ========================================================================
 * Mel Spectrogram
 * ======================================================================== */

/* Global verbose flag for audio module (declared extern in header, defined here) */
int vox_verbose_audio = 0;

float *vox_mel_spectrogram(const float *samples, int n_samples, int *out_frames) {
    int n_fft = N_FFT;
    int n_freqs = N_FREQ;
    int pad_len = n_fft / 2; /* center=True padding (reflect) */

    /* Reflect-pad the signal */
    int padded_len = n_samples + 2 * pad_len;
    float *padded = (float *)malloc(padded_len * sizeof(float));

    /* Left reflect pad: samples[pad_len..1] (reversed, excluding samples[0]) */
    for (int i = 0; i < pad_len; i++) {
        int src = pad_len - i;
        padded[i] = (src < n_samples) ? samples[src] : 0.0f;
    }
    /* Copy original signal */
    memcpy(padded + pad_len, samples, n_samples * sizeof(float));
    /* Right reflect pad: samples[n-2..n-pad_len-1] (reversed) */
    for (int i = 0; i < pad_len; i++) {
        int src = n_samples - 2 - i;
        padded[pad_len + n_samples + i] = (src >= 0) ? samples[src] : 0.0f;
    }

    int n_frames_total = (padded_len - n_fft) / HOP_LENGTH + 1;
    /* vLLM drops the last STFT frame: magnitudes = stft[..., :-1] */
    int n_frames = n_frames_total - 1;
    if (n_frames <= 0) {
        fprintf(stderr, "vox_mel_spectrogram: audio too short (%d samples)\n", n_samples);
        free(padded);
        return NULL;
    }

    /* Build mel filter bank */
    float *mel_filters = build_mel_filters();
    if (!mel_filters) {
        free(padded);
        return NULL;
    }

    /* Periodic Hann window (length 400, periodic=True) */
    float window[WIN_LENGTH];
    for (int i = 0; i < WIN_LENGTH; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)WIN_LENGTH));
    }

    /* Precompute DFT cos/sin tables: [k * N_FFT + n] = cos/sin(2*pi*k*n/N_FFT) */
    float *dft_cos = (float *)malloc((size_t)N_FREQ * N_FFT * sizeof(float));
    float *dft_sin = (float *)malloc((size_t)N_FREQ * N_FFT * sizeof(float));
    for (int k = 0; k < N_FREQ; k++) {
        for (int n = 0; n < N_FFT; n++) {
            float angle = 2.0f * (float)M_PI * (float)k * (float)n / (float)N_FFT;
            dft_cos[k * N_FFT + n] = cosf(angle);
            dft_sin[k * N_FFT + n] = sinf(angle);
        }
    }

    /* Allocate output: [n_frames, N_MEL] */
    float *mel = (float *)calloc(n_frames * N_MEL, sizeof(float));

    /* Working buffers */
    float windowed[N_FFT];
    float power[N_FREQ];

    /* Process each frame */
    for (int t = 0; t < n_frames; t++) {
        int start = t * HOP_LENGTH;

        /* Windowed frame (N_FFT == WIN_LENGTH, no zero-padding needed) */
        for (int i = 0; i < N_FFT; i++)
            windowed[i] = padded[start + i] * window[i];

        /* Direct DFT -> power spectrum (exact, no interpolation) */
        for (int k = 0; k < n_freqs; k++) {
            float re = 0, im = 0;
            const float *cos_row = dft_cos + k * N_FFT;
            const float *sin_row = dft_sin + k * N_FFT;
            for (int n = 0; n < N_FFT; n++) {
                re += windowed[n] * cos_row[n];
                im += windowed[n] * sin_row[n];
            }
            power[k] = re * re + im * im;
        }

        /* Apply mel filters: mel[t] = mel_filters @ power */
        for (int m = 0; m < N_MEL; m++) {
            float sum = 0.0f;
            const float *filt = mel_filters + (size_t)m * n_freqs;
            for (int k = 0; k < n_freqs; k++) {
                sum += filt[k] * power[k];
            }
            /* Log10 clamp */
            if (sum < 1e-10f) sum = 1e-10f;
            float val = log10f(sum);
            float min_val = LOG_MEL_MAX - 8.0f;
            if (val < min_val) val = min_val;
            mel[t * N_MEL + m] = (val + 4.0f) / 4.0f;
        }
    }

    free(dft_cos);
    free(dft_sin);
    free(padded);
    free(mel_filters);

    *out_frames = n_frames;
    return mel;
}

/* ========================================================================
 * Incremental Mel Spectrogram
 * ======================================================================== */

struct vox_mel_ctx {
    /* Precomputed tables (built once in init) */
    float *mel_filters;     /* [N_MEL * N_FREQ] */
    float *dft_cos;         /* [N_FREQ * N_FFT] DFT cos table */
    float *dft_sin;         /* [N_FREQ * N_FFT] DFT sin table */
    float window[WIN_LENGTH];

    /* Padded audio sample buffer (growing).
     * Starts with left_pad zeros, then real samples appended via feed(). */
    float *samples;
    int n_samples;          /* current length of samples buffer */
    int samples_cap;
    int64_t sample_offset;  /* global sample index of samples[0] */

    /* Mel frame buffer (growing) */
    float *mel;
    int n_mel_frames;       /* currently available frames in mel[] */
    int mel_cap;            /* allocated frame capacity */
    int mel_frame_offset;   /* global frame index of mel[0] */

    int left_pad;           /* total left padding = 200 + left_pad_samples */
    int finished;           /* set by vox_mel_finish */
};

#define MEL_SAMPLE_COMPACT_MIN 16000 /* compact every ~1s of audio progress */

/* Drop old audio samples that can no longer contribute to future mel frames. */
static void mel_compact_samples(vox_mel_ctx_t *ctx) {
    if (!ctx || ctx->n_samples <= WIN_LENGTH) return;

    int64_t next_frame_global = (int64_t)ctx->mel_frame_offset + ctx->n_mel_frames;
    int64_t needed_from_global = next_frame_global * HOP_LENGTH;
    int64_t discard64 = needed_from_global - ctx->sample_offset;
    if (discard64 <= 0) return;

    int discard = (discard64 > ctx->n_samples) ? ctx->n_samples : (int)discard64;
    if (discard < MEL_SAMPLE_COMPACT_MIN &&
        ctx->n_samples < (ctx->samples_cap * 3) / 4) return;

    int remain = ctx->n_samples - discard;
    if (remain > 0) {
        memmove(ctx->samples, ctx->samples + discard, (size_t)remain * sizeof(float));
    }
    ctx->n_samples = remain;
    ctx->sample_offset += discard;
}

/* Compute mel frames for all windows that fit in available data.
 * Each frame t needs samples[t*HOP_LENGTH .. t*HOP_LENGTH + WIN_LENGTH - 1]. */
static int mel_compute_available(vox_mel_ctx_t *ctx) {
    int new_frames = 0;
    float windowed[N_FFT];
    float power[N_FREQ];

    while (1) {
        int t = ctx->n_mel_frames;
        int64_t t_global = (int64_t)ctx->mel_frame_offset + t;
        int64_t start_global = t_global * HOP_LENGTH;
        int64_t start64 = start_global - ctx->sample_offset;
        int64_t end64 = start64 + WIN_LENGTH;
        if (start64 < 0 || end64 > ctx->n_samples) break;
        int start = (int)start64;

        /* Ensure mel buffer capacity */
        if (t >= ctx->mel_cap) {
            int new_cap = ctx->mel_cap ? ctx->mel_cap * 2 : 1024;
            float *tmp = (float *)realloc(ctx->mel,
                (size_t)new_cap * N_MEL * sizeof(float));
            if (!tmp) break;
            ctx->mel = tmp;
            ctx->mel_cap = new_cap;
        }

        /* Windowed frame (N_FFT == WIN_LENGTH, no zero-padding needed) */
        for (int i = 0; i < N_FFT; i++)
            windowed[i] = ctx->samples[start + i] * ctx->window[i];

        /* Direct DFT -> power spectrum (exact, no interpolation) */
        for (int k = 0; k < N_FREQ; k++) {
            float re = 0, im = 0;
            const float *cos_row = ctx->dft_cos + k * N_FFT;
            const float *sin_row = ctx->dft_sin + k * N_FFT;
            for (int n = 0; n < N_FFT; n++) {
                re += windowed[n] * cos_row[n];
                im += windowed[n] * sin_row[n];
            }
            power[k] = re * re + im * im;
        }

        /* Apply mel filters + log10 clamp */
        float *mel_row = ctx->mel + (size_t)t * N_MEL;
        for (int m = 0; m < N_MEL; m++) {
            float sum = 0.0f;
            const float *filt = ctx->mel_filters + (size_t)m * N_FREQ;
            for (int k = 0; k < N_FREQ; k++) {
                sum += filt[k] * power[k];
            }
            if (sum < 1e-10f) sum = 1e-10f;
            float val = log10f(sum);
            float min_val = LOG_MEL_MAX - 8.0f;
            if (val < min_val) val = min_val;
            mel_row[m] = (val + 4.0f) / 4.0f;
        }

        ctx->n_mel_frames++;
        new_frames++;
    }
    return new_frames;
}

vox_mel_ctx_t *vox_mel_ctx_init(int left_pad_samples) {
    vox_mel_ctx_t *ctx = (vox_mel_ctx_t *)calloc(1, sizeof(vox_mel_ctx_t));
    if (!ctx) return NULL;

    /* Build mel filters */
    ctx->mel_filters = build_mel_filters();
    if (!ctx->mel_filters) { free(ctx); return NULL; }

    /* Precompute DFT cos/sin tables */
    ctx->dft_cos = (float *)malloc((size_t)N_FREQ * N_FFT * sizeof(float));
    ctx->dft_sin = (float *)malloc((size_t)N_FREQ * N_FFT * sizeof(float));
    if (!ctx->dft_cos || !ctx->dft_sin) {
        free(ctx->dft_cos); free(ctx->dft_sin);
        free(ctx->mel_filters); free(ctx);
        return NULL;
    }
    for (int k = 0; k < N_FREQ; k++) {
        for (int n = 0; n < N_FFT; n++) {
            float angle = 2.0f * (float)M_PI * (float)k * (float)n / (float)N_FFT;
            ctx->dft_cos[k * N_FFT + n] = cosf(angle);
            ctx->dft_sin[k * N_FFT + n] = sinf(angle);
        }
    }

    /* Periodic Hann window */
    for (int i = 0; i < WIN_LENGTH; i++) {
        ctx->window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)WIN_LENGTH));
    }

    /* Left padding: 200 (center=True reflect over silence = zeros) + left_pad_samples */
    ctx->left_pad = 200 + left_pad_samples;

    /* Allocate initial sample buffer with left padding (all zeros) */
    ctx->samples_cap = ctx->left_pad + 16000; /* room for ~1s of audio */
    ctx->samples = (float *)calloc((size_t)ctx->samples_cap, sizeof(float));
    if (!ctx->samples) {
        free(ctx->mel_filters);
        free(ctx);
        return NULL;
    }
    ctx->n_samples = ctx->left_pad; /* starts with zeros */

    return ctx;
}

int vox_mel_feed(vox_mel_ctx_t *ctx, const float *samples, int n_samples) {
    if (!ctx || n_samples <= 0) return 0;

    /* Grow sample buffer if needed */
    int needed = ctx->n_samples + n_samples;
    if (needed > ctx->samples_cap) {
        int new_cap = ctx->samples_cap;
        while (new_cap < needed) new_cap *= 2;
        float *tmp = (float *)realloc(ctx->samples, (size_t)new_cap * sizeof(float));
        if (!tmp) return 0;
        ctx->samples = tmp;
        ctx->samples_cap = new_cap;
    }

    /* Append new samples */
    memcpy(ctx->samples + ctx->n_samples, samples, (size_t)n_samples * sizeof(float));
    ctx->n_samples += n_samples;

    /* Compute all new mel frames that fit */
    int n_new = mel_compute_available(ctx);
    mel_compact_samples(ctx);
    return n_new;
}

int vox_mel_finish(vox_mel_ctx_t *ctx, int right_pad_samples) {
    if (!ctx || ctx->finished) return ctx ? ctx->n_mel_frames : 0;

    /* Append right_pad_samples zeros */
    if (right_pad_samples > 0) {
        int needed = ctx->n_samples + right_pad_samples;
        if (needed > ctx->samples_cap) {
            int new_cap = ctx->samples_cap;
            while (new_cap < needed) new_cap *= 2;
            float *tmp = (float *)realloc(ctx->samples, (size_t)new_cap * sizeof(float));
            if (!tmp) return ctx->n_mel_frames;
            ctx->samples = tmp;
            ctx->samples_cap = new_cap;
        }
        memset(ctx->samples + ctx->n_samples, 0, (size_t)right_pad_samples * sizeof(float));
        ctx->n_samples += right_pad_samples;
    }

    /* Right reflect padding (200 samples from end of real audio).
     * Find the last non-padding real sample and reflect from there. */
    int reflect_len = 200;
    int needed2 = ctx->n_samples + reflect_len;
    if (needed2 > ctx->samples_cap) {
        int new_cap = ctx->samples_cap;
        while (new_cap < needed2) new_cap *= 2;
        float *tmp = (float *)realloc(ctx->samples, (size_t)new_cap * sizeof(float));
        if (!tmp) return ctx->n_mel_frames;
        ctx->samples = tmp;
        ctx->samples_cap = new_cap;
    }

    /* The end of real content is at n_samples (after right_pad zeros).
     * Reflect from the last real sample before the right-pad zeros:
     * samples[n_samples - right_pad - 2 - i] for i=0..199 */
    int real_end = ctx->n_samples - right_pad_samples;
    for (int i = 0; i < reflect_len; i++) {
        int src = real_end - 2 - i;
        ctx->samples[ctx->n_samples + i] = (src >= 0) ? ctx->samples[src] : 0.0f;
    }
    ctx->n_samples += reflect_len;

    /* Compute remaining frames */
    mel_compute_available(ctx);

    /* Drop last frame (vLLM convention: magnitudes = stft[..., :-1]) */
    if (ctx->n_mel_frames > 0) ctx->n_mel_frames--;

    ctx->finished = 1;
    return ctx->n_mel_frames;
}

float *vox_mel_data(vox_mel_ctx_t *ctx, int *out_n_frames) {
    if (!ctx) { if (out_n_frames) *out_n_frames = 0; return NULL; }
    if (out_n_frames) *out_n_frames = ctx->n_mel_frames;
    return ctx->mel;
}

int vox_mel_frame_offset(vox_mel_ctx_t *ctx) {
    return ctx ? ctx->mel_frame_offset : 0;
}

void vox_mel_discard_before(vox_mel_ctx_t *ctx, int keep_from_frame) {
    if (!ctx) return;
    if (keep_from_frame <= ctx->mel_frame_offset) return;

    int discard = keep_from_frame - ctx->mel_frame_offset;
    if (discard > ctx->n_mel_frames) discard = ctx->n_mel_frames;
    if (discard <= 0) return;

    int remain = ctx->n_mel_frames - discard;
    if (remain > 0) {
        memmove(ctx->mel, ctx->mel + (size_t)discard * N_MEL,
                (size_t)remain * N_MEL * sizeof(float));
    }
    ctx->n_mel_frames = remain;
    ctx->mel_frame_offset += discard;

    mel_compact_samples(ctx);
}

void vox_mel_free(vox_mel_ctx_t *ctx) {
    if (!ctx) return;
    free(ctx->mel_filters);
    free(ctx->dft_cos);
    free(ctx->dft_sin);
    free(ctx->samples);
    free(ctx->mel);
    free(ctx);
}
