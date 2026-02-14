/*
 * voxtral_audio.h - WAV loading and mel spectrogram computation
 */

#ifndef VOXTRAL_AUDIO_H
#define VOXTRAL_AUDIO_H

#include <stddef.h>
#include <stdint.h>

/* Verbose flag for audio module */
extern int vox_verbose_audio;

/* Load a WAV file, returns mono float32 samples in [-1,1] at 16kHz.
 * Handles: 16-bit PCM, mono or stereo (mixed to mono).
 * Resamples to 16kHz if needed.
 * Returns NULL on error. Caller must free returned buffer. */
float *vox_load_wav(const char *path, int *out_n_samples);

/* Parse a WAV file from a memory buffer. Same behavior as vox_load_wav
 * but operates on data already in memory. Caller must free returned buffer. */
float *vox_parse_wav_buffer(const uint8_t *data, size_t size, int *out_n_samples);

/* Read audio from stdin, returns mono float32 samples in [-1,1] at 16kHz.
 * Auto-detects format: WAV (RIFF header) or raw s16le 16kHz mono.
 * Returns NULL on error. Caller must free returned buffer. */
float *vox_read_pcm_stdin(int *out_n_samples);

/* Compute log-mel spectrogram from audio samples.
 * samples: mono float32 at 16kHz
 * n_samples: number of samples
 * out_frames: set to number of mel frames produced
 * Returns: [n_frames, 128] mel spectrogram (caller must free) */
float *vox_mel_spectrogram(const float *samples, int n_samples, int *out_frames);

/* ========================================================================
 * Incremental Mel Spectrogram (for real-time streaming)
 * ======================================================================== */

/* Opaque context for incremental mel computation */
typedef struct vox_mel_ctx vox_mel_ctx_t;

/* Create incremental mel context. left_pad_samples zeros are prepended
 * (e.g. 40960 for 32 left-pad tokens). An additional 200 samples of
 * center=True padding are added automatically. */
vox_mel_ctx_t *vox_mel_ctx_init(int left_pad_samples);

/* Feed new audio samples. Computes all mel frames whose 400-sample window
 * fits within available data. Returns number of NEW frames computed. */
int vox_mel_feed(vox_mel_ctx_t *ctx, const float *samples, int n_samples);

/* Finalize: append right_pad_samples zeros, then 200-sample right reflect
 * padding, compute remaining frames, drop last frame (vLLM convention).
 * Returns number of available mel frames. */
int vox_mel_finish(vox_mel_ctx_t *ctx, int right_pad_samples);

/* Get pointer to mel buffer and available frame count. */
float *vox_mel_data(vox_mel_ctx_t *ctx, int *out_n_frames);

/* Global mel frame index corresponding to mel_data()[0].
 * Needed when the context compacts old frames in long-running streams. */
int vox_mel_frame_offset(vox_mel_ctx_t *ctx);

/* Discard mel frames with global index < keep_from_frame.
 * Keeps memory bounded in non-stop streams. */
void vox_mel_discard_before(vox_mel_ctx_t *ctx, int keep_from_frame);

/* Free incremental mel context. */
void vox_mel_free(vox_mel_ctx_t *ctx);

#endif /* VOXTRAL_AUDIO_H */
