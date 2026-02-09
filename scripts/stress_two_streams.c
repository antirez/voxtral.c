#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "voxtral.h"

extern int vox_verbose;

typedef struct {
    int tid;
    vox_ctx_t *ctx;
    const char *wav_path;
    char *out;
    int ok;
} task_t;

static void *run_transcribe(void *arg) {
    task_t *t = (task_t *)arg;
    t->out = vox_transcribe(t->ctx, t->wav_path);
    if (!t->out) {
        fprintf(stderr, "[t%d] vox_transcribe failed\n", t->tid);
        t->ok = 0;
        return NULL;
    }
    size_t n = strlen(t->out);
    fprintf(stderr, "[t%d] transcript bytes=%zu\n", t->tid, n);
    t->ok = (n > 0);
    return NULL;
}

int main(int argc, char **argv) {
    const char *model_dir = (argc > 1) ? argv[1] : "voxtral-model";
    const char *wav_path = (argc > 2) ? argv[2] : "samples/test_speech.wav";

    /* Keep noise low; this is meant as a concurrency smoke test. */
    vox_verbose = 0;

    vox_ctx_t *ctx1 = vox_load(model_dir);
    if (!ctx1) {
        fprintf(stderr, "[err] vox_load failed for ctx1 (model_dir=%s)\n", model_dir);
        return 1;
    }
    vox_ctx_t *ctx2 = vox_load(model_dir);
    if (!ctx2) {
        fprintf(stderr, "[err] vox_load failed for ctx2 (model_dir=%s)\n", model_dir);
        vox_free(ctx1);
        return 1;
    }

    task_t t1 = { .tid = 1, .ctx = ctx1, .wav_path = wav_path, .out = NULL, .ok = 0 };
    task_t t2 = { .tid = 2, .ctx = ctx2, .wav_path = wav_path, .out = NULL, .ok = 0 };

    pthread_t th1, th2;
    if (pthread_create(&th1, NULL, run_transcribe, &t1) != 0) {
        fprintf(stderr, "[err] pthread_create(th1) failed\n");
        vox_free(ctx1);
        vox_free(ctx2);
        return 1;
    }
    if (pthread_create(&th2, NULL, run_transcribe, &t2) != 0) {
        fprintf(stderr, "[err] pthread_create(th2) failed\n");
        pthread_join(th1, NULL);
        free(t1.out);
        vox_free(ctx1);
        vox_free(ctx2);
        return 1;
    }

    pthread_join(th1, NULL);
    pthread_join(th2, NULL);

    int ok = (t1.ok && t2.ok);
    if (ok && t1.out && t2.out && strcmp(t1.out, t2.out) != 0) {
        fprintf(stderr, "[warn] transcripts differ between threads\n");
    }

    free(t1.out);
    free(t2.out);
    vox_free(ctx1);
    vox_free(ctx2);

    return ok ? 0 : 1;
}

