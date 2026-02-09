/*
 * voxtral_mic.h - Microphone capture API
 *
 * macOS implementation uses AudioQueue Services (AudioToolbox).
 * Windows implementation uses WASAPI.
 * Other platforms: stubs that return errors.
 */

#ifndef VOXTRAL_MIC_H
#define VOXTRAL_MIC_H

/* Open default microphone and start capturing (16kHz s16le mono, converted
 * to float internally). Returns 0 on success, -1 on error. */
int vox_mic_start(void);

/* Non-blocking read of available samples as float [-1,1].
 * Copies up to max_samples into out. Returns number of samples copied. */
int vox_mic_read(float *out, int max_samples);

/* Return number of samples currently buffered (for over-buffer detection). */
int vox_mic_read_available(void);

/* Stop capture and free resources. */
void vox_mic_stop(void);

#endif /* VOXTRAL_MIC_H */
