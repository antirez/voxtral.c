/*
 * voxtral_mic.c - macOS microphone capture helpers
 */

#include "voxtral_mic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>

#define VOX_MIC_SAMPLE_RATE 16000

typedef struct {
    AudioQueueRef queue;
    int max_samples;
    int total_samples;
    int should_stop;
    int failed;
    vox_mic_chunk_cb callback;
    void *user;
} vox_mic_state_t;

static void vox_mic_input_cb(void *inUserData,
                             AudioQueueRef inAQ,
                             AudioQueueBufferRef inBuffer,
                             const AudioTimeStamp *inStartTime,
                             UInt32 inNumPackets,
                             const AudioStreamPacketDescription *inPacketDesc) {
    (void)inStartTime;
    (void)inNumPackets;
    (void)inPacketDesc;

    vox_mic_state_t *st = (vox_mic_state_t *)inUserData;
    if (!st || st->should_stop || st->failed) return;

    int n_samples = (int)(inBuffer->mAudioDataByteSize / sizeof(int16_t));
    if (n_samples > 0) {
        const int16_t *src = (const int16_t *)inBuffer->mAudioData;
        float *tmp = (float *)malloc((size_t)n_samples * sizeof(float));
        if (!tmp) {
            st->failed = 1;
            st->should_stop = 1;
        } else {
            for (int i = 0; i < n_samples; i++)
                tmp[i] = src[i] / 32768.0f;
            if (st->callback && st->callback(tmp, n_samples, st->user) != 0)
                st->should_stop = 1;
            free(tmp);
            st->total_samples += n_samples;
            if (st->max_samples > 0 && st->total_samples >= st->max_samples)
                st->should_stop = 1;
        }
    }

    if (!st->should_stop && !st->failed) {
        if (AudioQueueEnqueueBuffer(inAQ, inBuffer, 0, NULL) != noErr) {
            st->failed = 1;
            st->should_stop = 1;
        }
    }
}

int vox_capture_mic_macos(float max_seconds, int chunk_samples,
                          vox_mic_chunk_cb callback, void *user,
                          const volatile sig_atomic_t *stop_flag,
                          int *out_total_samples) {
    if (!callback) return -1;
    if (chunk_samples <= 0) chunk_samples = 1600; /* 100ms */

    vox_mic_state_t st;
    memset(&st, 0, sizeof(st));
    st.callback = callback;
    st.user = user;
    if (max_seconds > 0.0f)
        st.max_samples = (int)(max_seconds * VOX_MIC_SAMPLE_RATE);

    AudioStreamBasicDescription fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.mSampleRate = VOX_MIC_SAMPLE_RATE;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    fmt.mBitsPerChannel = 16;
    fmt.mChannelsPerFrame = 1;
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerFrame = 2;
    fmt.mBytesPerPacket = 2;

    OSStatus err = AudioQueueNewInput(&fmt, vox_mic_input_cb, &st, NULL, NULL, 0, &st.queue);
    if (err != noErr) {
        fprintf(stderr, "vox_capture_mic_macos: AudioQueueNewInput failed (%d)\n", (int)err);
        return -1;
    }

    /* Best-effort device info for debugging routing issues */
    CFStringRef dev = NULL;
    UInt32 dev_size = (UInt32)sizeof(dev);
    err = AudioQueueGetProperty(st.queue, kAudioQueueProperty_CurrentDevice, &dev, &dev_size);
    if (err == noErr && dev) {
        char name[256];
        if (CFStringGetCString(dev, name, sizeof(name), kCFStringEncodingUTF8)) {
            fprintf(stderr, "Using input device: %s\n", name);
        }
        CFRelease(dev);
    } else {
        fprintf(stderr, "Using input device: (default system input)\n");
    }

    const int nbuf = 3;
    UInt32 bytes = (UInt32)((size_t)chunk_samples * sizeof(int16_t));
    for (int i = 0; i < nbuf; i++) {
        AudioQueueBufferRef buf = NULL;
        err = AudioQueueAllocateBuffer(st.queue, bytes, &buf);
        if (err != noErr || !buf) {
            fprintf(stderr, "vox_capture_mic_macos: AudioQueueAllocateBuffer failed (%d)\n", (int)err);
            AudioQueueDispose(st.queue, true);
            return -1;
        }
        err = AudioQueueEnqueueBuffer(st.queue, buf, 0, NULL);
        if (err != noErr) {
            fprintf(stderr, "vox_capture_mic_macos: AudioQueueEnqueueBuffer failed (%d)\n", (int)err);
            AudioQueueDispose(st.queue, true);
            return -1;
        }
    }

    err = AudioQueueStart(st.queue, NULL);
    if (err != noErr) {
        fprintf(stderr, "vox_capture_mic_macos: AudioQueueStart failed (%d)\n", (int)err);
        AudioQueueDispose(st.queue, true);
        return -1;
    }

    while (!st.should_stop && !st.failed) {
        if (stop_flag && *stop_flag) {
            st.should_stop = 1;
            break;
        }
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, false);
    }

    AudioQueueStop(st.queue, true);
    AudioQueueDispose(st.queue, true);

    if (out_total_samples) *out_total_samples = st.total_samples;
    return st.failed ? -1 : 0;
}

#else

int vox_capture_mic_macos(float max_seconds, int chunk_samples,
                          vox_mic_chunk_cb callback, void *user,
                          const volatile sig_atomic_t *stop_flag,
                          int *out_total_samples) {
    (void)max_seconds;
    (void)chunk_samples;
    (void)callback;
    (void)user;
    (void)stop_flag;
    if (out_total_samples) *out_total_samples = 0;
    fprintf(stderr, "vox_capture_mic_macos: microphone input is only available on macOS\n");
    return -1;
}

#endif
