# Voxtral: Windows CUDA & Real-time Guide

This guide explains how to build and use Voxtral on Windows, with optional NVIDIA CUDA acceleration and native microphone support via WASAPI.

## Prerequisites

1.  **Visual Studio 2022**: Install "Desktop development with C++".
2.  **CUDA Toolkit**: Version 12.x or 13.x (13.0 recommended).
3.  **FFmpeg**: Required for running regression tests or processing non-WAV files (e.g., `.ogg`).
4.  **PowerShell**: Used for building and testing.

## Building

The project uses a unified `build.ps1` script that automatically detects your Visual Studio environment.

## Downloading The Model

Download the Voxtral model into `voxtral-model/`:
```powershell
.\download_model.ps1
```

### 1. Build with CUDA (Recommended)
This enables GPU acceleration for the encoder and decoder, providing near-instant transcription.
```powershell
.\build.ps1 -Cuda
```
*Note: This will automatically generate CUDA kernel headers targeting your specific GPU architecture.*

### 2. Build for CPU only
```powershell
.\build.ps1
```

### 3. Clean Build
If you encounter issues, perform a clean build:
```powershell
.\build.ps1 -Clean -Cuda
```

## Using Voxtral

### 1. File Transcription (WAV)
Voxtral expects 16-bit PCM WAV files at any sample rate (it resamples internally).
```powershell
.\voxtral.exe -d voxtral-model -i samples/jfk.wav
```

### 2. Real-time Microphone Transcription
The `--from-mic` flag uses the Windows Audio Session API (WASAPI) to capture from your default system microphone.
```powershell
.\voxtral.exe -d voxtral-model --from-mic
```

### 3. Tuning for Responsiveness
For real-time usage, you can adjust the processing interval. A smaller interval (e.g., 0.5s or 1.0s) reduces latency but increases overhead.
```powershell
# Update every 0.5 seconds (default for mic)
.\voxtral.exe -d voxtral-model --from-mic -I 0.5

# Show alternative tokens (beam search visualization)
.\voxtral.exe -d voxtral-model --from-mic --alt 0.4
```

## Advanced Configuration

### Environment Variables
-   **`VOX_PRINT_TIMINGS=1`**: Prints detailed latency breakdowns for the Encoder and Decoder after transcription.
-   **`CUDA_PATH`**: If the build script cannot find CUDA, ensure this variable points to your toolkit directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`).

### Performance Metrics
In `--debug` mode, you can monitor how the system is keeping up:
-   **`Encoder`**: Processes audio chunks into model features.
-   **`Decoder`**: Generates text tokens. On an RTX 5080, you should see ~20ms/step or faster.
-   **`Warning: can't keep up`**: If you see this, the system is falling behind real-time. The buffer limit is 10 seconds.

## Regression Testing
To verify that your build is working correctly and producing accurate text:
```powershell
powershell -ExecutionPolicy Bypass -File .\runtest.ps1
```
This script checks batch-cpu, batch-cuda, and streaming modes against known phrases.
