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

function Move-ToReadablePath([string]$Legacy, [string]$Readable, [switch]$KeepHiddenJunction) {
    if (Is-Junction $Legacy) {
        Write-Host "[OK alias] $Legacy"
        return
    }
    if (-not (Test-Path -LiteralPath $Legacy)) {
        if (Test-Path -LiteralPath $Readable) {
            Write-Host "[OK moved] $Readable"
        } else {
            Write-Host "[SKIP missing] $Legacy"
        }
        return
    }
    if (Test-Path -LiteralPath $Readable) {
        Write-Warning "Both legacy and readable paths exist; leaving both untouched: $Legacy / $Readable"
        return
    }
    Ensure-Dir (Split-Path -Parent $Readable)
    try {
        Move-Item -LiteralPath $Legacy -Destination $Readable
        Write-Host "[MOVE] $Legacy"
        Write-Host "    -> $Readable"
        if ($KeepHiddenJunction) {
            New-Item -ItemType Junction -Path $Legacy -Target $Readable | Out-Null
            attrib +h $Legacy 2>$null
            Write-Host "[HIDDEN compatibility alias] $Legacy"
        }
    } catch {
        Write-Warning "Could not move (likely Windows file lock); skipped without deleting anything: $Legacy :: $($_.Exception.Message)"
    }
}

Write-Host "=== Attention-Analysis AMD NIR data layout organizer ==="
Write-Host "Root: $Root"
Write-Host "Policy: MOVE ONLY, NO DELETE; legacy compatibility aliases are hidden."
Write-Host ""

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(.exe)?$' -and
    $_.CommandLine -match 'ritnet_fullclass|run_ritnet_fullclass'
}
if ($active) {
    Write-Host "Active formal RITnet process detected:" -ForegroundColor Red
    $active | Select-Object ProcessId,Name,CommandLine | Format-List
    throw "Stop the RITnet run before reorganizing the data root."
}

# Safety gate: never reorganize unless the successful Topology sub-032 is present.
$final32Candidates = @(
    (Join-Path $Root "10-final-topology\sub-032"),
    (Join-Path $Root "ritnet-fullclass-final\sub-032")
)
$final32 = $final32Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $final32) {
    throw "Safety gate failed: cannot find completed formal sub-032."
}
$completion32 = Join-Path $final32 "completion.json"
$manifest32 = Join-Path $final32 "manifest.json"
if (-not (Test-Path -LiteralPath $completion32) -or -not (Test-Path -LiteralPath $manifest32)) {
    throw "Safety gate failed: sub-032 completion/manifest missing under $final32"
}
$completion = Get-Content -LiteralPath $completion32 -Raw | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $manifest32 -Raw | ConvertFrom-Json
if ($completion.status -ne "complete") {
    throw "Safety gate failed: sub-032 status is not complete."
}
if ($manifest.work_identity.analysis_domain_version -ne "source-backed-output-mask-v3-primary-pupil-topology") {
    throw "Safety gate failed: sub-032 is not the frozen Topology production result."
}
Write-Host "[SAFETY OK] sub-032 complete Topology result found."
Write-Host ""

# 1) Current formal outputs. Old code keeps working through a hidden junction.
Move-ToReadablePath `
    (Join-Path $Root "ritnet-fullclass-final") `
    (Join-Path $Root "10-final-topology") `
    -KeepHiddenJunction

# 2) Active SQLite recovery checkpoints. This preserves partial sub-034/sub-036 progress.
Move-ToReadablePath `
    (Join-Path $Root ".ritnet-fullclass-work") `
    (Join-Path $Root "90-runtime\checkpoints\final-topology") `
    -KeepHiddenJunction

# 3) Scientific validation evidence.
if (Test-Path -LiteralPath (Join-Path $Root "_validation\pupil-geometry")) {
    Move-ToReadablePath `
        (Join-Path $Root "_validation\pupil-geometry") `
        (Join-Path $Root "20-validation\pupil-geometry")
}
Move-ToReadablePath `
    (Join-Path $Root "ritnet-fullclass-geometry-validation") `
    (Join-Path $Root "20-validation\pupil-geometry")

# 4) Historical/failed material. Runner keeps writing to _archive through a hidden junction.
Move-ToReadablePath `
    (Join-Path $Root "_archive") `
    (Join-Path $Root "99-archive") `
    -KeepHiddenJunction

Move-ToReadablePath `
    (Join-Path $Root ".ritnet-fullclass-work-geometry-validation-sub031-20260828") `
    (Join-Path $Root "99-archive\old-checkpoints\geometry-validation-sub031-20260828")
Move-ToReadablePath `
    (Join-Path $Root ".ritnet-fullclass-work-production-backup-20260828") `
    (Join-Path $Root "99-archive\old-checkpoints\production-backup-20260828")
Move-ToReadablePath `
    (Join-Path $Root "_smoke-workstore-archive") `
    (Join-Path $Root "99-archive\development\smoke-workstore")

# 5) Historical YOLO source runs. Real folders go under one readable parent.
# Hidden junctions remain in the root only for backwards-compatible discovery.
$sourceRoot = Join-Path $Root "00-source-yolo-historical"
Ensure-Dir $sourceRoot
$legacySources = Get-ChildItem -LiteralPath $Root -Directory -Force | Where-Object {
    $_.Name -match '^sub-\d+_formal_' -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint)
}
foreach ($source in $legacySources) {
    $dest = Join-Path $sourceRoot $source.Name
    Move-ToReadablePath $source.FullName $dest -KeepHiddenJunction
}

# 6) Runtime logs.
Ensure-Dir (Join-Path $Root "90-runtime\logs")
$oldBatchSummary = Join-Path $Root "ritnet_fullclass_batch_summary.json"
if (Test-Path -LiteralPath $oldBatchSummary) {
    $dest = Join-Path $Root "90-runtime\logs\last-fullclass-batch-summary.json"
    if (Test-Path -LiteralPath $dest) {
        $dest = Join-Path $Root ("90-runtime\logs\fullclass-batch-summary-{0}.json" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    }
    Move-ToReadablePath $oldBatchSummary $dest
}
$legacyBatchSummary = Join-Path $Root "batch_run_summary.json"
if (Test-Path -LiteralPath $legacyBatchSummary) {
    Move-ToReadablePath $legacyBatchSummary (Join-Path $Root "99-archive\legacy-logs\batch-run-summary-legacy.json")
}

@"
THIS IS THE ONLY CURRENT FORMAL RITNET COHORT OUTPUT.
Method: RITnet FP32 + primary-pupil-topology + OpenCV pupil ellipse.
A subject is complete only when its completion.json validates.
"@ | Set-Content -LiteralPath (Join-Path $Root "10-final-topology\README.txt") -Encoding UTF8

@"
00-source-yolo-historical : historical completed YOLO formal runs; bbox source only, never rerun YOLO.
10-final-topology         : current formal RITnet + Topology cohort outputs.
20-validation             : scientific validation evidence; not cohort output.
90-runtime                : SQLite interruption checkpoints and run logs; not scientific output.
99-archive                : failed, interrupted, legacy, backup, and development artifacts.

Hidden legacy junctions exist only so current code remains backward-compatible. Do not use them manually.
"@ | Set-Content -LiteralPath (Join-Path $Root "README-DATA-LAYOUT.txt") -Encoding UTF8

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Use these five folders only:"
Write-Host "  00-source-yolo-historical   historical YOLO sources"
Write-Host "  10-final-topology           current formal outputs"
Write-Host "  20-validation               method validation"
Write-Host "  90-runtime                  checkpoints/logs"
Write-Host "  99-archive                  old/failed/backup"
Write-Host ""
Write-Host "Hidden compatibility aliases remain for the current Python code; they can be removed only after a later code-path migration."