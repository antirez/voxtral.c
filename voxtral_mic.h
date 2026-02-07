/*
 * voxtral_mic.h - macOS microphone capture helpers
 */

#ifndef VOXTRAL_MIC_H
#define VOXTRAL_MIC_H

#include <signal.h>

/* Callback receives mono float32 samples in [-1,1] at 16kHz.
 * Return 0 to continue capture, non-zero to stop. */
typedef int (*vox_mic_chunk_cb)(const float *samples, int n_samples, void *user);

/* Capture audio from the default system microphone (macOS).
 * max_seconds <= 0 means run until callback/stop_flag requests stop.
 * chunk_samples controls callback granularity at 16kHz.
 * out_total_samples (optional) reports captured sample count.
 * Returns 0 on success, non-zero on error. */
int vox_capture_mic_macos(float max_seconds, int chunk_samples,
                          vox_mic_chunk_cb callback, void *user,
                          const volatile sig_atomic_t *stop_flag,
                          int *out_total_samples);

#endif /* VOXTRAL_MIC_H */
