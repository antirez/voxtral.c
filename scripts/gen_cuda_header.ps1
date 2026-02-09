param(
    [string]$CuFile = "voxtral_cuda_kernels.cu",
    [string]$HFile = "voxtral_cuda_kernels_cubin.h",
    [string]$Arch = "native"
)

# Use current directory if not specified
if (-not (Test-Path $CuFile)) {
    $CuFile = Join-Path $PSScriptRoot "..\$CuFile"
}

$NVCC = "nvcc"
if ($env:CUDA_PATH) {
    $nvccPath = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path $nvccPath) { $NVCC = $nvccPath }
}

$CubinFile = "voxtral_cuda_kernels.cubin"

Write-Host "Compiling $CuFile to $CubinFile for $Arch..."
# Use -O3 and -arch
$nvccArgs = "-cubin", $CuFile, "-o", $CubinFile, "-arch=$Arch", "-O3"
& $NVCC @nvccArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "nvcc failed to compile kernels."
    exit 1
}

Write-Host "Converting $CubinFile to $HFile..."
$bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $CubinFile))
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("// Auto-generated from $CubinFile")
[void]$sb.AppendLine("#include <stddef.h>")
[void]$sb.AppendLine("static const unsigned char voxtral_cuda_kernels_cubin[] = {")

for ($i = 0; $i -lt $bytes.Length; $i++) {
    [void]$sb.Append("0x{0:X2}" -f $bytes[$i])
    if ($i -lt $bytes.Length - 1) {
        [void]$sb.Append(", ")
    }
    if (($i + 1) % 12 -eq 0) {
        [void]$sb.AppendLine()
    }
}

[void]$sb.AppendLine("};")
[void]$sb.AppendLine("static const size_t voxtral_cuda_kernels_cubin_len = $($bytes.Length);")

$outputPath = Join-Path (Get-Item .).FullName $HFile
[System.IO.File]::WriteAllText($outputPath, $sb.ToString())
Write-Host "Generated $outputPath"
Remove-Item $CubinFile