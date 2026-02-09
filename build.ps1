# build.ps1 - Build script for voxtral.c on Windows
param(
    [switch]$Clean,
    [switch]$Blas,
    [switch]$Debug,
    [switch]$Avx512,
    [switch]$Cuda
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

function Test-Command ([string]$Name) {
    (Get-Command $Name -ErrorAction SilentlyContinue) -ne $null
}

function Import-VSEnv {
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        Write-Warning "vswhere.exe not found at $vswhere"
        return $false
    }
    
    $vsroot  = & $vswhere -latest -products * `
               -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
               -property installationPath 2>$null
    if (-not $vsroot) {
        Write-Warning "VS Build Tools not found."
        return $false
    }

    $vcvars = Join-Path $vsroot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) {
        Write-Warning "vcvars64.bat not found at $vcvars"
        return $false
    }

    Write-Host "Importing MSVC environment from $vcvars"
    $envDump = cmd /s /c "`"$vcvars`" && set"
    foreach ($line in $envDump -split "`r?`n") {
        if ($line -match '^(.*?)=(.*)$') {
            $name,$value = $Matches[1],$Matches[2]
            Set-Item -Path "Env:$name" -Value $value
        }
    }
    return $true
}

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

$SRCS = "voxtral.c", "voxtral_kernels.c", "voxtral_audio.c", "voxtral_encoder.c", "voxtral_decoder.c", "voxtral_tokenizer.c", "voxtral_safetensors.c", "main.c"
if ($true) { # Since build.ps1 is only for Windows
    $SRCS += "voxtral_mic_win32.c"
} else {
    $SRCS += "voxtral_mic_macos.c"
}
$TARGET = "voxtral.exe"

if ($Clean) {
    Write-Host "Cleaning..."
    if (Test-Path $TARGET) { Remove-Item $TARGET }
    Get-ChildItem -Filter *.o | Remove-Item
    Get-ChildItem -Filter *.obj | Remove-Item
    if (Test-Path "voxtral_cuda_kernels_cubin.h") { Remove-Item "voxtral_cuda_kernels_cubin.h" }
}

# Try to find a compiler
$CC = ""
if (Test-Command gcc) {
    $CC = "gcc"
} elseif (Test-Command cl) {
    $CC = "cl"
} else {
    Write-Host "Compiler not found in PATH. Trying to import VS environment..."
    if (Import-VSEnv) {
        if (Test-Command cl) {
            $CC = "cl"
        }
    }
}

if (-not $CC) {
    Write-Error "No compiler (gcc or cl) found. Please install MinGW-w64 or Visual Studio Build Tools."
    exit 1
}

Write-Host "Using compiler: $CC"

# Helper for CUDA detection
$NVCC = ""
$CUDA_LIB_PATH = ""
if ($Cuda) {
    if (Test-Command nvcc) {
        $NVCC = "nvcc"
    } elseif ($env:CUDA_PATH) {
        $nvccPath = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        if (Test-Path $nvccPath) {
            $NVCC = $nvccPath
        }
    }
    
    if (-not $NVCC) {
        Write-Error "CUDA requested but nvcc not found. Please install CUDA Toolkit."
        exit 1
    }
    
    if ($env:CUDA_PATH) {
        $CUDA_LIB_PATH = Join-Path $env:CUDA_PATH "lib\x64"
    } else {
        # Try default location
        $CUDA_LIB_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64" # Adjust version if needed logic
        if (-not (Test-Path $CUDA_LIB_PATH)) {
             Write-Warning "Could not guess CUDA lib path. Linking might fail."
        }
    }
    Write-Host "Using NVCC: $NVCC"
}

if ($CC -eq "gcc") {
    $CFLAGS = "-Wall", "-Wextra", "-O3", "-march=native", "-ffast-math", "-mavx2", "-mfma"
    $LDFLAGS = "-lm"
    if ($Debug) {
        $CFLAGS = "-Wall", "-Wextra", "-g", "-O0", "-DDEBUG"
    }
    if ($Avx512) {
        $CFLAGS += "-mavx512f", "-mavx512bf16", "-DUSE_AVX512BF16"
    }
    if ($Blas) {
        $CFLAGS += "-DUSE_BLAS", "-DUSE_OPENBLAS"
        $LDFLAGS += "-lopenblas"
    }
    if ($Cuda) {
        Write-Host "Generating CUDA kernel header..."
        powershell.exe -ExecutionPolicy Bypass -File scripts\gen_cuda_header.ps1
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        $CFLAGS += "-DUSE_CUDA"
        $SRCS += "voxtral_cuda.c"
        if ($env:CUDA_PATH) {
            $CUDA_INC_PATH = Join-Path $env:CUDA_PATH "include"
            $CFLAGS += "-I`"$CUDA_INC_PATH`""
        }
        $LDFLAGS += "-L`"$CUDA_LIB_PATH`"", "-lcuda", "-lcublas", "-lcublasLt"
    } else {
        $SRCS += "voxtral_cuda_stub.c"
    }
    $cmd = "$CC $CFLAGS -o $TARGET $SRCS $LDFLAGS"
} else {
    # MSVC (cl.exe)
    $CFLAGS = "/O2", "/W3", "/MT", "/D_CRT_SECURE_NO_WARNINGS", "/openmp", "/arch:AVX2"
    $LINK_FLAGS = "ole32.lib uuid.lib mmdevapi.lib"
    
    if ($Debug) {
        $CFLAGS = "/Zi", "/Od", "/DDEBUG", "/D_CRT_SECURE_NO_WARNINGS", "/openmp", "/arch:AVX2"
        $LINK_FLAGS += "/DEBUG"
    }
    
    if ($Avx512) {
        $CFLAGS = "/O2", "/W3", "/MT", "/D_CRT_SECURE_NO_WARNINGS", "/openmp", "/arch:AVX512", "/DUSE_AVX512BF16"
    }
    
    if ($Blas) {
        $CFLAGS += "/DUSE_BLAS", "/DUSE_OPENBLAS"
        $LINK_FLAGS += " openblas.lib"
    }
    
    if ($Cuda) {
        Write-Host "Generating CUDA kernel header..."
        # Run in current session so MSVC environment is preserved
        & (Join-Path $PSScriptRoot "scripts\gen_cuda_header.ps1")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        
        $CFLAGS += "/DUSE_CUDA"
        if ($env:CUDA_PATH) {
            $CUDA_INC_PATH = Join-Path $env:CUDA_PATH "include"
            $CFLAGS += "/I`"$CUDA_INC_PATH`""
        }
        $SRCS += "voxtral_cuda.c"
        $LINK_FLAGS += " /LIBPATH:`"$CUDA_LIB_PATH`" cuda.lib cudart.lib cublas.lib cublaslt.lib"
    } else {
        $SRCS += "voxtral_cuda_stub.c"
    }

    $cmd = "$CC $CFLAGS $SRCS /Fe$TARGET /link $LINK_FLAGS"
}

Write-Host "Building $TARGET..."
Write-Host "Running: $cmd"
Invoke-Expression $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful: $TARGET"
} else {
    Write-Host "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
