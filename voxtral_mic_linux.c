/*
 * voxtral_mic_linux.c - Microphone capture using ALSA (GNU/Linux)
 * (need libasound2-dev)
 *
 * Captures audio from the default microphone at 16 kHz, mono S16LE,
 * converts samples to float [-1, 1], and writes them into a
 * mutex‑protected ring buffer.
 * The main thread polls vox_mic_read() to drain samples.
 */

#ifdef __linux__

#include "voxtral_mic.h"
#include <alsa/asoundlib.h>
#include <pthread.h>
#include <string.h>
#include <stdio.h>

#define MIC_SAMPLE_RATE   16000
#define MIC_BUF_FRAMES    1600    // 1600 frames = 100 ms at 16 kHz (mono)
#define RING_CAPACITY     160000  // 10 seconds at 16kHz

static snd_pcm_t          *pcm_handle = NULL;   // ALSA PCM capture handle
static pthread_t           capture_thread;   // background thread for audio capture
static pthread_mutex_t     ring_mutex = PTHREAD_MUTEX_INITIALIZER;  // protects ring buffer state
static float               ring[RING_CAPACITY];   // circular buffer
static int                 ring_head = 0;   // next write position
static int                 ring_count = 0;  // n of valid samples in ring
static volatile int        running = 0;    // capture running/not running



/* The audio capture thread:
 * reads from ALSA pcm_handle, converts s16 to float, writes into ring buffer */
static void *mic_capture_thread(void *arg) {
    (void)arg;

    int16_t buf[MIC_BUF_FRAMES];

    while (running) {
        snd_pcm_sframes_t n = snd_pcm_readi(pcm_handle, buf, MIC_BUF_FRAMES);
        if (n == -EPIPE) {
            // XRUN: buffer overrun/underrun, reset pcm_handle with prepare and try again
            snd_pcm_prepare(pcm_handle);
            continue;
        } else if (n == -EAGAIN || n == -EINTR) {
            // Alsa is occupied or call interrupted, try again
            continue;
        } else if (n < 0) {
            // Unrecoverable ALSA error
            fprintf(stderr, "ALSA read error: %s\n", snd_strerror((int)n));
            continue;
        }

        int frames = (int)n;
        pthread_mutex_lock(&ring_mutex);
        for (int i = 0; i < frames; i++) {
            float sample = buf[i] / 32768.0f;
            ring[ring_head] = sample;
            ring_head = (ring_head + 1) % RING_CAPACITY;
            if (ring_count < RING_CAPACITY) {
                ring_count++;
            }
            /* If ring buffer is full, new samples overwrite the oldest ones.
             * ring_head advances, ring_tail advances implicitly,
             * ring_count stays at RING_CAPACITY. */
        }
        pthread_mutex_unlock(&ring_mutex);
    }

    return NULL;
}



/* Errors checking helper func */
static int snd_check(int err, const char *msg) {
    if(err < 0) {
        fprintf(stderr, "%s: %s\n", msg, snd_strerror(err));
        return -1;
    }
    return 0;
}



int vox_mic_start(void) {
    /* pcm_handle already running, skip start process */
    if (running) return 0;

    int err;
    snd_pcm_hw_params_t *hw_params = NULL;

    /* Open ALSA pcm_handle for capture */
    err = snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_CAPTURE, 0);
    if (snd_check(err, "snd_pcm_open failed")) goto fail;

    /* Allocate and initialize hw_params with ALSA default configuration */
    snd_pcm_hw_params_alloca(&hw_params);
    err = snd_pcm_hw_params_any(pcm_handle, hw_params);
    if (snd_check(err, "snd_pcm_hw_params_any failed")) goto fail;

    /* Set ACCESS param to INTERLEAVED */
    err = snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (snd_check(err, "snd_pcm_hw_params_set_access failed")) goto fail;

    /* Set CHANNELS param to MONO */
    err = snd_pcm_hw_params_set_channels(pcm_handle, hw_params, 1);
    if (snd_check(err, "snd_pcm_hw_params_set_channels failed")) goto fail;

    /* Set FORMAT param to S16LE */
    err = snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_S16_LE);
    if (snd_check(err, "snd_pcm_hw_params_set_format failed")) goto fail;

    /* Set PERIOD SIZE (number of frames per read) to MIC_BUF_FRAMES */
    snd_pcm_uframes_t period_size = MIC_BUF_FRAMES;
    err = snd_pcm_hw_params_set_period_size_near(pcm_handle, hw_params, &period_size, NULL);
    if (snd_check(err, "snd_pcm_hw_params_set_period_size_near failed")) goto fail;

    /* Set RATE param to 16kHz (mandatory for Voxtral) */
    unsigned int rate = MIC_SAMPLE_RATE;
    err = snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &rate, NULL);
    if (snd_check(err, "snd_pcm_hw_params_set_rate_near failed")) goto fail;
    if (rate != MIC_SAMPLE_RATE) {
        fprintf(stderr, "Your audio pcm_handle does not support 16000 Hz\n");
        goto fail;
    }

    /* Finally commit all previous params settings to the ALSA pcm_handle */
    err = snd_pcm_hw_params(pcm_handle, hw_params);
    if (snd_check(err, "snd_pcm_hw_params failed")) goto fail;

    /* Set the ALSA pcm_handle as ready */
    err = snd_pcm_prepare(pcm_handle);
    if (snd_check(err, "snd_pcm_prepare failed")) goto fail;

    /* Init ring buffer */
    pthread_mutex_lock(&ring_mutex);
    ring_head = 0;
    ring_count = 0;
    pthread_mutex_unlock(&ring_mutex);

    /* Start capture thread */
    running = 1;
    err = pthread_create(&capture_thread, NULL, mic_capture_thread, NULL);
    if (err != 0) {
        fprintf(stderr, "pthread_create failed: %d\n", err);
        running = 0;
        goto fail;
    }

    return 0;

fail:
     if (pcm_handle) {
         snd_pcm_close(pcm_handle);
         pcm_handle = NULL;
     }
     return -1;
}



int vox_mic_read(float *out, int max_samples) {
    if (!out || max_samples <= 0) return 0;

    pthread_mutex_lock(&ring_mutex);
    int n = ring_count < max_samples ? ring_count : max_samples;
    if (n > 0) {
        /* ring_tail = posizione del campione più vecchio */
        int ring_tail = (ring_head - ring_count + RING_CAPACITY) % RING_CAPACITY;
        for (int i = 0; i < n; i++) {
            out[i] = ring[(ring_tail + i) % RING_CAPACITY];
        }
        ring_count -= n;
    }
    pthread_mutex_unlock(&ring_mutex);

    return n;
}



int vox_mic_read_available(void) {
    pthread_mutex_lock(&ring_mutex);
    int n = ring_count;
    pthread_mutex_unlock(&ring_mutex);
    return n;
}



void vox_mic_stop(void) {
    if (!running) return;
    running = 0;
    // Force the blocking snd_pcm_readi() call to wake up by dropping the PCM stream
    if (pcm_handle) snd_pcm_drop(pcm_handle);
    pthread_join(capture_thread, NULL);
    if (pcm_handle) {
        snd_pcm_close(pcm_handle);
        pcm_handle = NULL;
    }
}



#else  /* !__linux__ */
#include "voxtral_mic.h"
#include <stdio.h>

int vox_mic_start(void) {
    fprintf(stderr, "Microphone capture with ALSA is not supported on this platform\n");
    return -1;
}

int vox_mic_read(float *out, int max_samples) {
    (void)out; (void)max_samples;
    return 0;
}

int vox_mic_read_available(void) { return 0; }

void vox_mic_stop(void) {}

#endif


