param(
    [string]$Root = "D:\_AttentionData\Beijing-NIR\amd-directml"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath $Root).Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

function Ensure-Parent([string]$Path) {
    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Move-Preserved([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Host "[SKIP missing] $Source"
        return
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "Refusing to overwrite existing destination: $Destination"
    }
    Ensure-Parent $Destination
    Move-Item -LiteralPath $Source -Destination $Destination
    Write-Host "[MOVE] $Source"
    Write-Host "    -> $Destination"
}

Write-Host "=== AMD NIR data-root organizer ==="
Write-Host "Root: $Root"
Write-Host "Policy: move/preserve only; no deletion"

# Hard safety gate: the successful sub-032 production result must remain exactly
# where downstream cohort analysis expects it, and it must be the frozen
# Topology production contract before any surrounding historical clutter moves.
$final32 = Join-Path $Root "ritnet-fullclass-final\sub-032"
$completion32 = Join-Path $final32 "completion.json"
$manifest32 = Join-Path $final32 "manifest.json"
if (-not (Test-Path -LiteralPath $completion32)) {
    throw "Safety gate failed: formal sub-032 completion.json is missing: $completion32"
}
if (-not (Test-Path -LiteralPath $manifest32)) {
    throw "Safety gate failed: formal sub-032 manifest.json is missing: $manifest32"
}
$completion = Get-Content -LiteralPath $completion32 -Raw | ConvertFrom-Json
$manifest = Get-Content -LiteralPath $manifest32 -Raw | ConvertFrom-Json
if ($completion.status -ne "complete") {
    throw "Safety gate failed: sub-032 completion status is '$($completion.status)', expected 'complete'"
}
$analysisDomain = $manifest.work_identity.analysis_domain_version
$expectedDomain = "source-backed-output-mask-v3-primary-pupil-topology"
if ($analysisDomain -ne $expectedDomain) {
    throw "Safety gate failed: sub-032 analysis domain is '$analysisDomain', expected '$expectedDomain'"
}
Write-Host "[KEEP formal] sub-032 complete Topology result: $final32"

# The interrupted sub-031 formal rerun is not the completed three-method
# validation result. If it has no completion marker, preserve it as an aborted
# attempt so ritnet-fullclass-final contains only completed formal subjects.
$final31 = Join-Path $Root "ritnet-fullclass-final\sub-031"
if (Test-Path -LiteralPath $final31) {
    $completion31 = Join-Path $final31 "completion.json"
    if (-not (Test-Path -LiteralPath $completion31)) {
        $dest31 = Join-Path $Root "_archive\ritnet-fullclass-final\sub-031\${stamp}__aborted-formal-rerun"
        Move-Preserved $final31 $dest31
    }
    else {
        Write-Host "[KEEP formal] sub-031 has completion.json; organizer will not move it"
    }
}

# Keep the scientifically important geometry-selection run visible as validation
# evidence rather than burying it among failed/legacy artifacts.
Move-Preserved `
    (Join-Path $Root "ritnet-fullclass-geometry-validation") `
    (Join-Path $Root "_validation\pupil-geometry")

# Historical/recovery-only workstores and smoke artifacts are not formal cohort
# outputs. Preserve them under explicit archive categories.
Move-Preserved `
    (Join-Path $Root ".ritnet-fullclass-work-geometry-validation-sub031-20260828") `
    (Join-Path $Root "_archive\workstores\.ritnet-fullclass-work-geometry-validation-sub031-20260828")
Move-Preserved `
    (Join-Path $Root ".ritnet-fullclass-work-production-backup-20260828") `
    (Join-Path $Root "_archive\workstores\.ritnet-fullclass-work-production-backup-20260828")
Move-Preserved `
    (Join-Path $Root "_smoke-workstore-archive") `
    (Join-Path $Root "_archive\development\_smoke-workstore-archive")
Move-Preserved `
    (Join-Path $Root "batch_run_summary.json") `
    (Join-Path $Root "_archive\legacy-run-summaries\batch_run_summary.json")

# The current production workstore remains in place because the current engine
# uses this exact path for interruption recovery. It is not a scientific output.
$currentWork = Join-Path $Root ".ritnet-fullclass-work"
if (Test-Path -LiteralPath $currentWork) {
    Write-Host "[KEEP runtime] $currentWork"
    Write-Host "    Active SQLite recovery checkpoint; not final scientific data."
}

$validationReadme = Join-Path $Root "_validation\README.txt"
Ensure-Parent $validationReadme
@"
_validation contains completed scientific validation evidence, not formal cohort outputs.

pupil-geometry\sub-031
  Three-method Legacy / Topology / EllSeg validation run used to select the final
  primary-pupil-topology production geometry. Preserve for method/QC evidence.
"@ | Set-Content -LiteralPath $validationReadme -Encoding UTF8

$archiveReadme = Join-Path $Root "_archive\README.txt"
Ensure-Parent $archiveReadme
@"
_archive contains preserved historical, failed, interrupted, or recovery-only artifacts.
Nothing here is the current formal cohort result.

ritnet-fullclass-final\
  Invalid/incomplete/contract-stale formal attempts automatically preserved by the runner.
workstores\
  Old SQLite recovery checkpoints and backups; not final scientific data.
development\
  Smoke/development artifacts.
legacy-run-summaries\
  Historical runner summaries.
"@ | Set-Content -LiteralPath $archiveReadme -Encoding UTF8

Write-Host ""
Write-Host "=== Organizer complete ==="
Write-Host "Current formal result:  $Root\ritnet-fullclass-final"
Write-Host "Validation evidence:    $Root\_validation"
Write-Host "Historical archive:     $Root\_archive"
Write-Host "Active recovery cache:  $Root\.ritnet-fullclass-work"
Write-Host "Historical YOLO sources remain untouched: $Root\sub-*_formal_*"
