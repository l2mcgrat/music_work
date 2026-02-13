[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [Parameter(Mandatory = $false)]
    [string]$OutDir = "",

    [Parameter(Mandatory = $false)]
    [string]$Suffix = "_converted",

    [Parameter(Mandatory = $false)]
    [ValidateSet("flat", "mirror", "tags")]
    [string]$Layout = "flat",

    [Parameter(Mandatory = $false)]
    [switch]$NameFromTags,

    [Parameter(Mandatory = $false)]
    [switch]$FoldersFromTags,

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

function Get-SafeName {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [int]$MaxLen = 120
    )

    $n = $Name.Trim()
    if ([string]::IsNullOrWhiteSpace($n)) { return "unknown" }

    # Replace invalid Windows filename characters
    $n = $n -replace '[<>:"/\\|?*]', '_'
    # Remove control chars
    $n = -join ($n.ToCharArray() | Where-Object { [int]$_ -ge 32 })
    # Collapse whitespace
    $n = $n -replace '\s+', ' '
    $n = $n.Trim(' .')
    if ($n.Length -gt $MaxLen) { $n = $n.Substring(0, $MaxLen).Trim() }
    if ([string]::IsNullOrWhiteSpace($n)) { return "unknown" }
    return $n
}

function Get-UniquePath {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) { return $Path }

    $dir = Split-Path -Parent $Path
    $leaf = Split-Path -Leaf $Path
    $base = [System.IO.Path]::GetFileNameWithoutExtension($leaf)
    $ext = [System.IO.Path]::GetExtension($leaf)

    for ($i = 2; $i -lt 1000; $i++) {
        $candidate = Join-Path $dir ("$base ($i)$ext")
        if (-not (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    throw "Could not find a unique filename for: $Path"
}

function Get-TagValue {
    param(
        [Parameter(Mandatory = $false)]$Tags,
        [Parameter(Mandatory = $true)][string]$Key
    )

    if (-not $Tags) { return $null }

    # Tags can be emitted with varying case. Try a few common variations.
    foreach ($k in @($Key, $Key.ToLowerInvariant(), $Key.ToUpperInvariant())) {
        try {
            $v = $Tags.$k
            if ($v) { return [string]$v }
        } catch {
            # ignore
        }
    }
    return $null
}

function Get-AudioTags {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$FfprobePath
    )

    try {
        $json = & $FfprobePath -v error -print_format json -show_format -show_entries format_tags=title,artist,album,track,disc $Path
        if ($LASTEXITCODE -ne 0) { return $null }
        $obj = $json | ConvertFrom-Json
        return $obj.format.tags
    } catch {
        return $null
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

$dstResolved = $dst
try {
    $dstResolved = (Resolve-Path -LiteralPath $dst).Path
} catch {
    # ignore
}

$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue)
if (-not $ffmpeg) {
    throw "ffmpeg not found on PATH. Install it or run 'winget install Gyan.FFmpeg'."
}

$ffprobe = (Get-Command ffprobe -ErrorAction SilentlyContinue)
if (-not $ffprobe) {
    throw "ffprobe not found on PATH (it usually comes with ffmpeg). Ensure ffprobe is installed and on PATH."
}

$audioExts = @(".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")
$excludeOutputDir = $false
try {
    $srcResolved = (Resolve-Path -LiteralPath $src).Path
    if (-not [string]::IsNullOrWhiteSpace($dstResolved) -and ($dstResolved -ne $srcResolved)) {
        $excludeOutputDir = $true
    }
} catch {
    # If we can't resolve paths for any reason, err on the safe side and don't exclude.
    $excludeOutputDir = $false
}

$files = Get-ChildItem -Path $src -Recurse -File |
    Where-Object {
        ($audioExts -contains $_.Extension.ToLowerInvariant()) -and
        (-not $excludeOutputDir -or ($_.FullName -notlike ($dstResolved + "\\*")))
    }

# Skip files that already look like outputs from this script (prevents double-work/clutter)
if (-not [string]::IsNullOrWhiteSpace($Suffix)) {
    $files = $files | Where-Object { -not $_.BaseName.EndsWith($Suffix, [System.StringComparison]::OrdinalIgnoreCase) }
}

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

    $tags = $null
    if ($NameFromTags -or $FoldersFromTags -or $Layout -eq "tags") {
        $tags = Get-AudioTags -Path $f.FullName -FfprobePath $ffprobe.Source
    }

    $title = if ($NameFromTags -or $Layout -eq "tags") { Get-TagValue -Tags $tags -Key "title" } else { $null }
    $artist = if ($FoldersFromTags -or $Layout -eq "tags") { Get-TagValue -Tags $tags -Key "artist" } else { $null }
    $album = if ($FoldersFromTags -or $Layout -eq "tags") { Get-TagValue -Tags $tags -Key "album" } else { $null }

    $safeTitle = if ($title) { Get-SafeName -Name $title } else { Get-SafeName -Name $base }
    $safeArtist = if ($artist) { Get-SafeName -Name $artist } else { "unknown_artist" }
    $safeAlbum = if ($album) { Get-SafeName -Name $album } else { "unknown_album" }

    $targetDir = $dst
    if ($Layout -eq "mirror") {
        $relDir = [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetRelativePath($src, $f.FullName))
        if (-not [string]::IsNullOrWhiteSpace($relDir)) {
            $targetDir = Join-Path $dst $relDir
        }
    } elseif ($Layout -eq "tags" -or $FoldersFromTags) {
        $targetDir = Join-Path (Join-Path $dst $safeArtist) $safeAlbum
    }

    if (-not (Test-Path -LiteralPath $targetDir)) {
        if ($DryRun) {
            Write-Host "DRY RUN: mkdir '$targetDir'"
        } else {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }
    }

    $targetDirResolved = $targetDir
    try {
        $targetDirResolved = (Resolve-Path -LiteralPath $targetDir).Path
    } catch {
        # ignore
    }

    $outNameBase = $safeTitle
    if ($targetDirResolved -eq $f.DirectoryName) {
        if ([string]::IsNullOrWhiteSpace($Suffix)) {
            throw "OutDir is the same as the source file folder, but -Suffix is empty. Refusing to overwrite in-place. Provide -Suffix (e.g. _converted) or choose a different -OutDir."
        }
        $outNameBase = $outNameBase + $Suffix
    }
    $outPath = Join-Path $targetDir ($outNameBase + ".wav")
    if ((Test-Path -LiteralPath $outPath) -and (-not $Force)) {
        Write-Host "Skip (exists): $outPath"
        $skipped++
        continue
    }

    # If not forcing and the output path exists due to collisions, pick a unique name.
    if ((Test-Path -LiteralPath $outPath) -and $Force) {
        # We'll overwrite.
    } elseif (Test-Path -LiteralPath $outPath) {
        $outPath = Get-UniquePath -Path $outPath
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
