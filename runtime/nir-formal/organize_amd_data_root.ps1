param(
    [string]$Root = "D:\_AttentionData\Beijing-NIR\amd-directml"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath $Root).Path

function Is-Junction([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $item = Get-Item -LiteralPath $Path -Force
    return [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Move-Preserved([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source)) {
        return $false
    }
    if (Test-Path -LiteralPath $Destination) {
        Write-Warning "Both paths exist; refusing overwrite. source=$Source destination=$Destination"
        return $false
    }
    Ensure-Dir (Split-Path -Parent $Destination)
    try {
        Move-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
        Write-Host "[MOVE] $Source"
        Write-Host "    -> $Destination"
        return $true
    }
    catch {
        Write-Warning "Could not move; left untouched. source=$Source error=$($_.Exception.Message)"
        return $false
    }
}

function Ensure-HiddenJunction([string]$Alias, [string]$Target) {
    if (Test-Path -LiteralPath $Alias) {
        if (Is-Junction $Alias) {
            Write-Host "[OK compatibility alias] $Alias"
        }
        return
    }
    if (-not (Test-Path -LiteralPath $Target)) { return }
    New-Item -ItemType Junction -Path $Alias -Target $Target | Out-Null
    attrib +h $Alias 2>$null
    Write-Host "[HIDDEN compatibility alias] $Alias -> $Target"
}

Write-Host "=== Attention-Analysis AMD NIR data organizer ==="
Write-Host "Root: $Root"
Write-Host "Policy: MOVE/PRESERVE ONLY; NO SCIENTIFIC DATA DELETE."
Write-Host "Final readable folders: historical-yolo / final-topology / validation / runtime / archive"
Write-Host ""

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(.exe)?$' -and
    $_.CommandLine -match 'ritnet_fullclass|run_ritnet_fullclass'
}
if ($active) {
    $active | Select-Object ProcessId,Name,CommandLine | Format-List
    throw "A formal RITnet process is still running. Stop it before reorganizing."
}

# Safety gate: do not reorganize unless the known-good sub-032 Topology result exists.
$final32Candidates = @(
    (Join-Path $Root "final-topology\sub-032"),
    (Join-Path $Root "10-final-topology\sub-032"),
    (Join-Path $Root "ritnet-fullclass-final\sub-032")
)
$final32 = $final32Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $final32) { throw "Safety gate failed: completed formal sub-032 was not found." }
$completion32 = Join-Path $final32 "completion.json"
$manifest32 = Join-Path $final32 "manifest.json"
if (-not (Test-Path -LiteralPath $completion32) -or -not (Test-Path -LiteralPath $manifest32)) {
    throw "Safety gate failed: sub-032 completion.json or manifest.json is missing."
}
$completion = Get-Content -LiteralPath $completion32 -Raw | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $manifest32 -Raw | ConvertFrom-Json
if ($completion.status -ne "complete") { throw "Safety gate failed: sub-032 is not complete." }
if ($manifest.work_identity.analysis_domain_version -ne "source-backed-output-mask-v3-primary-pupil-topology") {
    throw "Safety gate failed: sub-032 is not the frozen Topology production result."
}
Write-Host "[SAFETY OK] sub-032 complete Topology result found."

# If the abandoned numbered naming was ever created, normalize it first.
$numberedFinal = Join-Path $Root "10-final-topology"
$finalTarget = Join-Path $Root "final-topology"
if ((Test-Path -LiteralPath $numberedFinal) -and -not (Test-Path -LiteralPath $finalTarget)) {
    [void](Move-Preserved $numberedFinal $finalTarget)
}
$numberedValidation = Join-Path $Root "20-validation"
if ((Test-Path -LiteralPath $numberedValidation) -and -not (Test-Path -LiteralPath (Join-Path $Root "validation"))) {
    [void](Move-Preserved $numberedValidation (Join-Path $Root "validation"))
}
$numberedRuntime = Join-Path $Root "90-runtime"
if ((Test-Path -LiteralPath $numberedRuntime) -and -not (Test-Path -LiteralPath (Join-Path $Root "runtime"))) {
    [void](Move-Preserved $numberedRuntime (Join-Path $Root "runtime"))
}
$numberedArchive = Join-Path $Root "99-archive"
if ((Test-Path -LiteralPath $numberedArchive) -and -not (Test-Path -LiteralPath (Join-Path $Root "archive"))) {
    [void](Move-Preserved $numberedArchive (Join-Path $Root "archive"))
}
$numberedSource = Join-Path $Root "00-source-yolo-historical"
if ((Test-Path -LiteralPath $numberedSource) -and -not (Test-Path -LiteralPath (Join-Path $Root "historical-yolo"))) {
    [void](Move-Preserved $numberedSource (Join-Path $Root "historical-yolo"))
}

# Current formal outputs. If Windows currently locks the legacy folder, leave it
# intact and continue organizing everything else; rerunning later will retry only
# this missing migration. Never create an empty final-topology that could be
# mistaken for the real cohort output.
$legacyFinal = Join-Path $Root "ritnet-fullclass-final"
if (-not (Test-Path -LiteralPath $finalTarget)) {
    if ((Test-Path -LiteralPath $legacyFinal) -and -not (Is-Junction $legacyFinal)) {
        [void](Move-Preserved $legacyFinal $finalTarget)
    }
}
if (Test-Path -LiteralPath $finalTarget) {
    Ensure-HiddenJunction $legacyFinal $finalTarget
    Write-Host "[OK formal] $finalTarget"
} elseif (Test-Path -LiteralPath $legacyFinal) {
    Write-Warning "Formal result folder is still at legacy path because Windows did not allow the move: $legacyFinal"
    Write-Host "[KEEP legacy final] No data was deleted; close Explorer/image viewers using this folder and rerun later."
} else {
    throw "Formal output folder disappeared from all recognized locations; refusing to continue."
}

# Active recovery checkpoints. Preserve partial sub-034 progress.
$runtimeTarget = Join-Path $Root "runtime"
Ensure-Dir $runtimeTarget
$checkpointTarget = Join-Path $runtimeTarget "checkpoints\final-topology"
$legacyWork = Join-Path $Root ".ritnet-fullclass-work"
if (-not (Test-Path -LiteralPath $checkpointTarget) -and (Test-Path -LiteralPath $legacyWork) -and -not (Is-Junction $legacyWork)) {
    [void](Move-Preserved $legacyWork $checkpointTarget)
}
Ensure-HiddenJunction $legacyWork $checkpointTarget
Ensure-Dir (Join-Path $runtimeTarget "logs")

# Scientific validation evidence.
$validationTarget = Join-Path $Root "validation\pupil-geometry"
if (Test-Path -LiteralPath (Join-Path $Root "_validation\pupil-geometry")) {
    if (-not (Test-Path -LiteralPath $validationTarget)) {
        [void](Move-Preserved (Join-Path $Root "_validation\pupil-geometry") $validationTarget)
    }
}
if (Test-Path -LiteralPath (Join-Path $Root "ritnet-fullclass-geometry-validation")) {
    if (-not (Test-Path -LiteralPath $validationTarget)) {
        [void](Move-Preserved (Join-Path $Root "ritnet-fullclass-geometry-validation") $validationTarget)
    } else {
        Write-Warning "Validation target already exists; legacy validation folder was left untouched rather than merged automatically."
    }
}

# Historical / failed / development material.
$archiveTarget = Join-Path $Root "archive"
$legacyArchive = Join-Path $Root "_archive"
if (-not (Test-Path -LiteralPath $archiveTarget) -and (Test-Path -LiteralPath $legacyArchive) -and -not (Is-Junction $legacyArchive)) {
    [void](Move-Preserved $legacyArchive $archiveTarget)
}
Ensure-Dir $archiveTarget
Ensure-HiddenJunction $legacyArchive $archiveTarget

[void](Move-Preserved `
    (Join-Path $Root ".ritnet-fullclass-work-geometry-validation-sub031-20260828") `
    (Join-Path $archiveTarget "old-checkpoints\geometry-validation-sub031-20260828"))
[void](Move-Preserved `
    (Join-Path $Root ".ritnet-fullclass-work-production-backup-20260828") `
    (Join-Path $archiveTarget "old-checkpoints\production-backup-20260828"))
[void](Move-Preserved `
    (Join-Path $Root "_smoke-workstore-archive") `
    (Join-Path $archiveTarget "development\smoke-workstore"))

# Historical YOLO sources. Real directories move under one parent; hidden root aliases
# preserve the current source-discovery contract without rerunning YOLO.
$sourceTarget = Join-Path $Root "historical-yolo"
Ensure-Dir $sourceTarget
$legacySources = Get-ChildItem -LiteralPath $Root -Directory -Force | Where-Object {
    $_.Name -match '^sub-\d+_formal_' -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint)
}
foreach ($source in $legacySources) {
    $dest = Join-Path $sourceTarget $source.Name
    if (-not (Test-Path -LiteralPath $dest)) {
        if (Move-Preserved $source.FullName $dest) {
            Ensure-HiddenJunction $source.FullName $dest
        }
    }
}

# Runtime logs.
$oldBatchSummary = Join-Path $Root "ritnet_fullclass_batch_summary.json"
if (Test-Path -LiteralPath $oldBatchSummary) {
    $dest = Join-Path $runtimeTarget "logs\last-fullclass-batch-summary.json"
    if (Test-Path -LiteralPath $dest) {
        $dest = Join-Path $runtimeTarget ("logs\fullclass-batch-summary-{0}.json" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    }
    [void](Move-Preserved $oldBatchSummary $dest)
}
$legacyBatchSummary = Join-Path $Root "batch_run_summary.json"
if (Test-Path -LiteralPath $legacyBatchSummary) {
    [void](Move-Preserved $legacyBatchSummary (Join-Path $archiveTarget "legacy-logs\batch-run-summary-legacy.json"))
}

# README creation is strictly conditional on the real formal directory existing.
# This prevents a locked formal folder from turning a successful partial cleanup
# into an irrelevant DirectoryNotFoundException at the end.
if (Test-Path -LiteralPath $finalTarget) {
@"
CURRENT FORMAL OUTPUT ONLY.
RITnet FP32 + primary-pupil-topology + OpenCV pupil ellipse.
A subject is complete only when completion.json validates.
"@ | Set-Content -LiteralPath (Join-Path $finalTarget "README.txt") -Encoding UTF8
}

@"
historical-yolo : completed historical YOLO formal runs; bbox/source only; never rerun YOLO.
final-topology  : current formal RITnet + Topology cohort outputs.
validation      : scientific method-validation evidence; not cohort output.
runtime         : interruption checkpoints and run logs; not scientific output.
archive         : failed, interrupted, legacy, backup, and development artifacts.

Hidden legacy junctions exist only for backward compatibility with the current Python paths.
"@ | Set-Content -LiteralPath (Join-Path $Root "README-DATA-LAYOUT.txt") -Encoding UTF8

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Use these folders:"
Write-Host "  historical-yolo   historical YOLO sources"
if (Test-Path -LiteralPath $finalTarget) {
    Write-Host "  final-topology    current formal outputs"
} else {
    Write-Host "  ritnet-fullclass-final   current formal outputs (temporary legacy name; move was locked)" -ForegroundColor Yellow
}
Write-Host "  validation        method validation"
Write-Host "  runtime           checkpoints/logs"
Write-Host "  archive           old/failed/backup"
