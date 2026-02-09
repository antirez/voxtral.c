# download_model.ps1 - Download Voxtral Realtime 4B model from HuggingFace
#
# Usage:
#   .\download_model.ps1 [-Dir voxtral-model]
#
# Notes:
# - Uses curl.exe for resumable downloads (avoid PowerShell's curl alias).

param(
    [string]$Dir = "voxtral-model"
)

$ModelId = "mistralai/Voxtral-Mini-4B-Realtime-2602"
$Files = @(
    "consolidated.safetensors",
    "params.json",
    "tekken.json"
)

$BaseUrl = "https://huggingface.co/$ModelId/resolve/main"

Write-Host "Downloading Voxtral Realtime 4B to $Dir/"
Write-Host "Model: $ModelId"
Write-Host ""

New-Item -ItemType Directory -Force -Path $Dir | Out-Null

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    Write-Error "curl.exe not found. Install a recent Windows build (Windows 10/11 usually includes it)."
    exit 1
}

foreach ($file in $Files) {
    $dest = Join-Path $Dir $file
    $url = "$BaseUrl/$file"

    $localSize = 0
    if (Test-Path $dest) {
        $localSize = (Get-Item $dest).Length
    }

    $remoteSize = $null
    try {
        $headers = curl.exe -sIL $url
        foreach ($line in $headers) {
            if ($line -match '^[Cc]ontent-[Ll]ength:\s*(\d+)\s*$') {
                $remoteSize = [int64]$Matches[1]
            }
        }
    } catch {
        $remoteSize = $null
    }

    if ($localSize -gt 0 -and $remoteSize -ne $null -and $localSize -eq $remoteSize) {
        Write-Host "  [skip] $file (already complete)"
        continue
    }

    if ($localSize -gt 0) {
        Write-Host "  [resume] $file ($localSize bytes present)..."
    } else {
        Write-Host "  [download] $file..."
    }

    curl.exe -L --fail --retry 5 --retry-delay 2 --continue-at - -o "$dest" "$url"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "curl.exe failed downloading $file"
        exit $LASTEXITCODE
    }

    Write-Host "  [done] $file"
}

Write-Host ""
Write-Host "Download complete. Model files in $Dir/"
Get-ChildItem -Path $Dir | Format-Table -AutoSize

