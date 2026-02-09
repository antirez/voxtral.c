# runtest.ps1 - Voxtral regression test for Windows
# Requires: ffmpeg, voxtral.exe, voxtral-model directory.

$MODEL_DIR = "voxtral-model"
$INPUT = "samples/jfk.wav"
$VOXTRAL = ".\voxtral.exe"
$global:PASS = 0
$global:FAIL = 0

$PHRASES = @(
    "And so my fellow Americans",
    "ask not what your country can do for you",
    "ask what you can do for your country"
)

function Cleanup {
    # No cleanup needed for jfk.wav as it's already a WAV
}

# Check prerequisites
if (-not (Test-Path $VOXTRAL)) {
    Write-Host "FAIL: $VOXTRAL not found. Run .\build.ps1 first."
    exit 1
}
if (-not (Test-Path $MODEL_DIR)) {
    Write-Host "FAIL: $MODEL_DIR not found. Run .\\download_model.ps1 first."
    exit 1
}
if (-not (Test-Path $INPUT)) {
    Write-Host "FAIL: $INPUT not found."
    exit 1
}

$global:SAW_CUDA = $false

function Check-Output($name, $got, $requireCuda = $false) {
    $ok = $true
    # Robust check: remove punctuation from both strings
    $clean_got = $got -replace "[,.!?]", ""
    foreach ($phrase in $PHRASES) {
        $clean_phrase = $phrase -replace "[,.!?]", ""
        if ($clean_got -notlike "*$clean_phrase*") {
            Write-Host "  MISSING: ""$phrase"""
            $ok = $false
        }
    }

    if ($requireCuda -and -not $global:SAW_CUDA) {
        Write-Host "  FAIL: CUDA initialization message not found in output!"
        $ok = $false
    }

    if ($ok) {
        Write-Host "PASS: $name (all $($PHRASES.Count) phrases found)"
        $global:PASS++
    } else {
        Write-Host "FAIL: $name"
        $global:FAIL++
    }
    Write-Host ""
}

function Run-Test($name, $cmd, $cmdArgs, $requireCuda = $false) {
    Write-Host "=== Test: $name ==="
    $got = ""
    $global:SAW_CUDA = $false
    # Run and stream output
    & $cmd $cmdArgs 2>&1 | ForEach-Object {
        $line = $_.ToString()
        if ($line -match "\[kernels\] backend=CUDA") {
            $global:SAW_CUDA = $true
            Write-Host "  [status] $line" -ForegroundColor Green
        } elseif ($line -match "Loading|Metal|Model|Audio:|Encoder:|Decoder:|\\\[DEBUG\\\]|\\\[kernels\\\]|\\\[cuda\\\]") {
            Write-Host "  [status] $line" -ForegroundColor Cyan
        } else {
            Write-Host $line -NoNewline
            $got += $line
        }
    }
    Write-Host "`n"
    Check-Output $name $got $requireCuda
}

# Test 1: Batch mode (Default/CPU)
Run-Test "batch-cpu" $VOXTRAL @("-d", $MODEL_DIR, "-i", $INPUT)

# Test 2: Batch mode (CUDA)
Run-Test "batch-cuda" $VOXTRAL @("-d", $MODEL_DIR, "-i", $INPUT) -requireCuda $true

# Test 3: Streaming mode with small chunks
Write-Host "=== Test: streaming -I 0.1 ==="
$got = ""
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    # Use powershell to handle the pipe more reliably
    $ffmpeg_cmd = "ffmpeg -i ""$INPUT"" -f s16le -ar 16000 -ac 1 - 2>NUL"
    $voxtral_cmd = "$VOXTRAL -d ""$MODEL_DIR"" --stdin -I 0.1"
    
    cmd /c "$ffmpeg_cmd | $voxtral_cmd" 2>&1 | ForEach-Object {
        $line = $_.ToString()
        if ($line -match "\[kernels\] backend=CUDA") {
            $global:SAW_CUDA = $true
            Write-Host "  [status] $line" -ForegroundColor Green
        } elseif ($line -match "Loading|Metal|Model|Audio:|Encoder:|Decoder:|\\\[DEBUG\\\]|\\\[kernels\\\]|\\\[cuda\\\]") {
            Write-Host "  [status] $line" -ForegroundColor Cyan
        } else {
            Write-Host $line -NoNewline
            $got += $line
        }
    }
    Write-Host "`n"
    Check-Output "streaming -I 0.1" $got
} else {
    Write-Host "  [skip] ffmpeg not found, skipping streaming test."
    $global:PASS++
}

Cleanup
Write-Host "=== Results: $global:PASS passed, $global:FAIL failed ==="
if ($global:FAIL -eq 0) { exit 0 } else { exit 1 }
