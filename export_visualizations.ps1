[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [Parameter(Mandatory = $false)]
    [ValidateSet('png','mp4','gif')]
    [string]$Format = 'mp4',

    [Parameter(Mandatory = $false)]
    [ValidateSet('scroll','full')]
    [string]$WaveformView = 'scroll',

    [Parameter(Mandatory = $false)]
    [int]$MaxWaveformPoints = 20000,

    [Parameter(Mandatory = $false)]
    [int]$Fps = 30,

    [Parameter(Mandatory = $false)]
    [switch]$UseDb,

    [Parameter(Mandatory = $false)]
    [switch]$Mono,

    [Parameter(Mandatory = $false)]
    [int]$Seconds = 10,

    [Parameter(Mandatory = $false)]
    [switch]$Full,

    [Parameter(Mandatory = $false)]
    [switch]$Overwrite,

    [Parameter(Mandatory = $false)]
    [switch]$NoAudio,

    [Parameter(Mandatory = $false)]
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
if (-not $repoRoot) {
    $repoRoot = (Get-Location).Path
}

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
            return ($riff -eq 'RIFF' -and $wave -eq 'WAVE')
        } finally {
            $fs.Dispose()
        }
    } catch {
        return $false
    }
}

$src = (Resolve-Path -LiteralPath $SourceDir).Path
$out = $OutDir
if (-not (Test-Path -LiteralPath $out)) {
    if ($DryRun) {
        Write-Host "DRY RUN: mkdir '$out'"
    } else {
        New-Item -ItemType Directory -Path $out -Force | Out-Null
    }
}

$py = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    throw "Could not find venv python at: $py (expected at repoRoot/.venv/Scripts/python.exe)"
}

$viz = Join-Path $repoRoot 'music_visualizer.py'
if (-not (Test-Path -LiteralPath $viz)) {
    throw "Could not find visualizer script at: $viz"
}

$allWavs = Get-ChildItem -Path $src -File -Filter "*.wav"
if (-not $allWavs) {
    throw "No .wav files found in: $src"
}

# Prefer *_converted.wav when present for the same stem.
$selected = @()
$groups = $allWavs | Group-Object {
    if ($_.BaseName -match '^(.*)_converted$') { $Matches[1] } else { $_.BaseName }
}
foreach ($g in $groups) {
    $converted = $g.Group | Where-Object { $_.BaseName -match '_converted$' } | Select-Object -First 1
    $candidate = if ($converted) { $converted } else { $g.Group | Select-Object -First 1 }
    if (-not (Test-IsRiffWave -Path $candidate.FullName)) {
        Write-Warning "Skip (not RIFF/WAVE): $($candidate.FullName)"
        continue
    }
    $selected += $candidate
}

$wavs = $selected | Sort-Object FullName

foreach ($w in $wavs) {
    $args = @(
        $viz,
        $w.FullName,
        '--mode','both',
        '--no-gui',
        '--waveform-view', $WaveformView,
        '--max-waveform-points', "$MaxWaveformPoints",
        '--export-auto',
        '--out-dir', $out,
        '--export-format', $Format
    )

    if ($Fps -gt 0) {
        $args += @('--fps', "$Fps")
    }

    if ($Full) {
        $args += '--export-full'
    } else {
        $args += @('--export-seconds', "$Seconds")
    }

    if ($UseDb) { $args += '--db' }
    if ($Mono) { $args += '--mono' }
    if ($Overwrite) { $args += '--overwrite-export' }
    if ($NoAudio) { $args += '--no-export-audio' }

    $cmdLine = "$py " + ($args | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    Write-Host "Export: $($w.Name) -> *.$Format"

    if ($DryRun) {
        Write-Host "DRY RUN: $cmdLine"
        continue
    }

    & $py @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "FAILED: $($w.FullName)"
    }
}
