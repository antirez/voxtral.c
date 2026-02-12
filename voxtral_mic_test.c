/*
 * voxtral_mic_test.c - Microphone backends test (macOS and GNU/Linux)
 *
 * This test is usefull to verify if compiled backend is receiving microphone input.
 * Since it uses the voxtral_mic.h interface, it is OS-agnostic
 * and works with both AudioQueue (macOS) and ALSA (GNU/Linux) backends.
 * */

#include "voxtral_mic.h"
#include <unistd.h>
#include <math.h>
#include <stdio.h>

static float rms(const float *buf, int n) {
    if (n <= 0) return 0.0f;
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += buf[i] * buf[i];
    return sqrt(acc / n);
}

int main() {
    printf("=== Mic backend test ===\n");

    printf("Starting microphone...\n");
    if (vox_mic_start() != 0) {
        fprintf(stderr, "ERROR: unable to start microphone\n");
        return 1;
    }

    float buf[1024];

    for (int iter = 0; iter < 100; iter++) {
        int available = vox_mic_read_available();
        int n = vox_mic_read(buf, 1024);
        float r = rms(buf, n);

        printf("iter=%d  available=%d  read=%d | RMS=%.6f\n",
               iter, available, n, r);

        usleep(100000);
    }

    printf("Stopping microphone...\n");
    vox_mic_stop();
    printf("Stopped.\n");

    printf("=== End test ===\n");
    return 0;
}


