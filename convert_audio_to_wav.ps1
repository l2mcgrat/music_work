[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [Parameter(Mandatory = $false)]
    [string]$OutDir = "",

    [Parameter(Mandatory = $false)]
    [string]$Suffix = "_converted",

    [Parameter(Mandatory = $false)]
    [ValidateSet("copy", "convert", "skip")]
    [string]$OnValidWav = "copy",

    [Parameter(Mandatory = $false)]
    [switch]$Mono,

    [Parameter(Mandatory = $false)]
    [int]$SampleRate = 44100,

    [Parameter(Mandatory = $false)]
    [switch]$Force,

    [Parameter(Mandatory = $false)]
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-IsRiffWave {
    param([Parameter(Mandatory = $true)][string]$Path)

    try {
        $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try {
            $buf = New-Object byte[] 12
            $read = $fs.Read($buf, 0, 12)
            if ($read -lt 12) { return $false }

            $riff = [System.Text.Encoding]::ASCII.GetString($buf, 0, 4)
            $wave = [System.Text.Encoding]::ASCII.GetString($buf, 8, 4)
            return ($riff -eq "RIFF" -and $wave -eq "WAVE")
        } finally {
            $fs.Dispose()
        }
    } catch {
        return $false
    }
}

$src = (Resolve-Path -LiteralPath $SourceDir).Path
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $src "converted_wav"
}
$dst = $OutDir

if (-not (Test-Path -LiteralPath $dst)) {
    if ($DryRun) {
        Write-Host "DRY RUN: mkdir '$dst'"
    } else {
        New-Item -ItemType Directory -Path $dst | Out-Null
    }
}

$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue)
if (-not $ffmpeg) {
    throw "ffmpeg not found on PATH. Install it or run 'winget install Gyan.FFmpeg'."
}

$audioExts = @(".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")
$files = Get-ChildItem -Path $src -Recurse -File | Where-Object { $audioExts -contains $_.Extension.ToLowerInvariant() }

if (-not $files) {
    Write-Host "No audio files found under: $src"
    exit 0
}

$converted = 0
$copied = 0
$skipped = 0
$failed = 0

foreach ($f in $files) {
    $base = $f.BaseName
    if ($dst -eq $f.DirectoryName) {
        if ([string]::IsNullOrWhiteSpace($Suffix)) {
            throw "OutDir is the same as the source file folder, but -Suffix is empty. Refusing to overwrite in-place. Provide -Suffix (e.g. _converted) or choose a different -OutDir."
        }
        $outPath = Join-Path $dst ($base + $Suffix + ".wav")
    } else {
        $outPath = Join-Path $dst ($base + ".wav")
    }

    if ((Test-Path -LiteralPath $outPath) -and (-not $Force)) {
        Write-Host "Skip (exists): $outPath"
        $skipped++
        continue
    }

    $isValidWav = $false
    if ($f.Extension.ToLowerInvariant() -eq ".wav") {
        $isValidWav = Test-IsRiffWave -Path $f.FullName
    }

    if ($isValidWav -and $OnValidWav -eq "skip") {
        Write-Host "Skip (valid WAV): $($f.FullName)"
        $skipped++
        continue
    }

    if ($isValidWav -and $OnValidWav -eq "copy") {
        Write-Host "Copy: $($f.FullName) -> $outPath"
        if (-not $DryRun) {
            Copy-Item -LiteralPath $f.FullName -Destination $outPath -Force
        }
        $copied++
        continue
    }

    $args = @(
        "-hide_banner",
        "-y",
        "-i", $f.FullName
    )
    if ($Mono) {
        $args += @("-ac", "1")
    }
    if ($SampleRate -gt 0) {
        $args += @("-ar", "$SampleRate")
    }
    $args += @($outPath)

    Write-Host "Convert: $($f.FullName) -> $outPath"
    if ($DryRun) {
        Write-Host "DRY RUN: ffmpeg $($args -join ' ')"
        continue
    }

    try {
        & $ffmpeg.Source @args | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "ffmpeg failed with exit code $LASTEXITCODE"
        }
        $converted++
    } catch {
        Write-Warning "FAILED: $($f.FullName) ($($_.Exception.Message))"
        $failed++
    }
}

Write-Host ""
Write-Host "Done. Copied=$copied Converted=$converted Skipped=$skipped Failed=$failed"
if ($failed -gt 0) { exit 1 } else { exit 0 }
