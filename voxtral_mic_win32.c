/*
 * voxtral_mic_win32.c - Microphone capture using WASAPI (Windows)
 */

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <stdio.h>
#include <math.h>
#include "voxtral_mic.h"

/* Link against these libraries (handled in build.ps1) */
/* #pragma comment(lib, "ole32.lib") */
/* #pragma comment(lib, "uuid.lib") */
/* #pragma comment(lib, "mmdevapi.lib") */

extern int vox_verbose;

/* 
 * Local GUIDs (unique names -> never collide with header externs).
 * These bytes are verified canonical values for WASAPI.
 */
static const GUID V_CLSID_MMDeviceEnumerator = 
    {0xbcde0395, 0xe52f, 0x467c, {0x8e,0x3d,0xc4,0x57,0x92,0x91,0x69,0x2e}};

static const GUID V_IID_IMMDeviceEnumerator = 
    {0xa95664d2, 0x9614, 0x4f35, {0xa7,0x46,0xde,0x8d,0xb6,0x36,0x17,0xe6}};

static const GUID V_IID_IAudioClient = 
    {0x1cb9ad4c, 0xdbfa, 0x4c32, {0xb1,0x78,0xc2,0xf5,0x68,0xa7,0x03,0xb2}};

static const GUID V_IID_IAudioCaptureClient = 
    {0xc8adbd64, 0xe71e, 0x48a0, {0xa4,0xde,0x18,0x5c,0x39,0x5c,0xd3,0x17}};

/* Constants */
#define MIC_SAMPLE_RATE   16000
#define RING_CAPACITY     160000 
#define REFTIMES_PER_SEC  10000000

/* Ring Buffer */
static float                    ring[RING_CAPACITY];
static int                      ring_head;  
static int                      ring_count; 
static CRITICAL_SECTION         ring_lock;

/* WASAPI State */
static HANDLE                   capture_thread = NULL;
static HANDLE                   shutdown_event = NULL;
static int                      running = 0;
static int                      thread_ready = 0;
static HRESULT                  thread_hr = S_OK;

static DWORD WINAPI mic_thread_proc(LPVOID lpParam) {
    (void)lpParam;
    HRESULT hr;
    IMMDeviceEnumerator *enumerator = NULL;
    IMMDevice *device = NULL;
    IAudioClient *audio_client = NULL;
    IAudioCaptureClient *capture_client = NULL;

    /* 1. Initialize COM (STA is safest for WASAPI) */
    hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        thread_hr = hr;
        thread_ready = -1;
        return 1;
    }

    /* 2. Create Enumerator */
    hr = CoCreateInstance(&V_CLSID_MMDeviceEnumerator, NULL, CLSCTX_INPROC_SERVER, 
                          &V_IID_IMMDeviceEnumerator, (void**)&enumerator);
    if (FAILED(hr)) goto cleanup;

    /* 3. Get Default Endpoint */
    hr = enumerator->lpVtbl->GetDefaultAudioEndpoint(enumerator, eCapture, eCommunications, &device);
    if (FAILED(hr)) hr = enumerator->lpVtbl->GetDefaultAudioEndpoint(enumerator, eCapture, eConsole, &device);
    if (FAILED(hr)) goto cleanup;

    /* 4. Activate Audio Client */
    hr = device->lpVtbl->Activate(device, &V_IID_IAudioClient, CLSCTX_INPROC_SERVER, NULL, (void**)&audio_client);
    if (FAILED(hr)) goto cleanup;

    /* 5. Initialize Stream (16kHz Mono Float) */
    WAVEFORMATEX wfx = {0};
    wfx.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    wfx.nChannels = 1;
    wfx.nSamplesPerSec = MIC_SAMPLE_RATE;
    wfx.wBitsPerSample = 32;
    wfx.nBlockAlign = (wfx.nChannels * wfx.wBitsPerSample) / 8;
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;

    DWORD stream_flags = 0x80000000 | 0x08000000; /* AUTOCONVERTPCM | SRC_DEFAULT_QUALITY */
    hr = audio_client->lpVtbl->Initialize(audio_client, AUDCLNT_SHAREMODE_SHARED, 
                                          stream_flags, REFTIMES_PER_SEC, 0, &wfx, NULL);
    if (FAILED(hr)) goto cleanup;

    /* 6. Get Capture Client */
    hr = audio_client->lpVtbl->GetService(audio_client, &V_IID_IAudioCaptureClient, (void**)&capture_client);
    if (FAILED(hr)) goto cleanup;

    /* 7. Start Capturing */
    hr = audio_client->lpVtbl->Start(audio_client);
    if (FAILED(hr)) goto cleanup;

    /* Signal success to the main thread */
    thread_hr = S_OK;
    thread_ready = 1;

    while (running) {
        if (WaitForSingleObject(shutdown_event, 10) == WAIT_OBJECT_0) break;

        UINT32 packet_length = 0;
        hr = capture_client->lpVtbl->GetNextPacketSize(capture_client, &packet_length);
        
        while (SUCCEEDED(hr) && packet_length != 0) {
            BYTE *data;
            UINT32 frames_available;
            DWORD flags;

            hr = capture_client->lpVtbl->GetBuffer(capture_client, &data, &frames_available, &flags, NULL, NULL);
            if (SUCCEEDED(hr)) {
                EnterCriticalSection(&ring_lock);
                if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                    for (UINT32 i = 0; i < frames_available; i++) {
                        ring[ring_head] = 0.0f;
                        ring_head = (ring_head + 1) % RING_CAPACITY;
                        if (ring_count < RING_CAPACITY) ring_count++;
                    }
                } else {
                    float *samples = (float *)data;
                    for (UINT32 i = 0; i < frames_available; i++) {
                        ring[ring_head] = samples[i];
                        ring_head = (ring_head + 1) % RING_CAPACITY;
                        if (ring_count < RING_CAPACITY) ring_count++;
                    }
                }
                LeaveCriticalSection(&ring_lock);
                capture_client->lpVtbl->ReleaseBuffer(capture_client, frames_available);
            }
            hr = capture_client->lpVtbl->GetNextPacketSize(capture_client, &packet_length);
        }
    }

    audio_client->lpVtbl->Stop(audio_client);

cleanup:
    if (FAILED(hr)) {
        thread_hr = hr;
        thread_ready = -1;
    }
    if (capture_client) capture_client->lpVtbl->Release(capture_client);
    if (audio_client) audio_client->lpVtbl->Release(audio_client);
    if (device) device->lpVtbl->Release(device);
    if (enumerator) enumerator->lpVtbl->Release(enumerator);
    CoUninitialize();
    return 0;
}

int vox_mic_start(void) {
    if (running) return 0;

    InitializeCriticalSection(&ring_lock);
    ring_head = 0;
    ring_count = 0;
    thread_ready = 0;
    thread_hr = S_OK;
    running = 1;

    shutdown_event = CreateEvent(NULL, TRUE, FALSE, NULL);
    capture_thread = CreateThread(NULL, 0, mic_thread_proc, NULL, 0, NULL);
    if (!capture_thread) {
        running = 0;
        DeleteCriticalSection(&ring_lock);
        return -1;
    }

    /* Wait for thread to initialize WASAPI (max 2 seconds) */
    int timeout = 200;
    while (thread_ready == 0 && timeout-- > 0) Sleep(10);

    if (thread_ready != 1) {
        fprintf(stderr, "WASAPI: Thread initialization failed (0x%lx)\n", thread_hr);
        vox_mic_stop();
        return -1;
    }

    if (vox_verbose >= 1) fprintf(stderr, "WASAPI: Capture started successfully\n");
    return 0;
}

int vox_mic_read(float *out, int max_samples) {
    if (!running) return 0;
    EnterCriticalSection(&ring_lock);
    int n = ring_count < max_samples ? ring_count : max_samples;
    if (n > 0) {
        int tail = (ring_head - ring_count + RING_CAPACITY) % RING_CAPACITY;
        for (int i = 0; i < n; i++) {
            out[i] = ring[(tail + i) % RING_CAPACITY];
        }
        ring_count -= n;
    }
    LeaveCriticalSection(&ring_lock);
    return n;
}

int vox_mic_read_available(void) {
    if (!running) return 0;
    EnterCriticalSection(&ring_lock);
    int n = ring_count;
    LeaveCriticalSection(&ring_lock);
    return n;
}

void vox_mic_stop(void) {
    if (!running) return;
    running = 0;
    if (shutdown_event) SetEvent(shutdown_event);
    if (capture_thread) {
        WaitForSingleObject(capture_thread, INFINITE);
        CloseHandle(capture_thread);
        capture_thread = NULL;
    }
    if (shutdown_event) {
        CloseHandle(shutdown_event);
        shutdown_event = NULL;
    }
    DeleteCriticalSection(&ring_lock);
}
#endif
