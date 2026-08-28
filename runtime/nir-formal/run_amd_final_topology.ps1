param(
    [int]$MinSubject = 33,
    [string]$Root = "D:\_AttentionData\Beijing-NIR\amd-directml",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath $Root).Path
$RuntimeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RuntimeDir

Write-Host "=== AMD FINAL TOPOLOGY RUNNER ==="
Write-Host "Root: $Root"
Write-Host "Minimum subject: $MinSubject"
Write-Host "Policy: historical YOLO only; RITnet FP32 + Topology; fail-fast on any subject error."
Write-Host ""

$zotero = Get-Process zotero-mcp -ErrorAction SilentlyContinue
if ($zotero) {
    $zotero | Select-Object Id,ProcessName,Path | Format-Table -AutoSize
    throw "zotero-mcp is running. Disable/close it before formal DirectML inference to avoid system-memory OOM."
}

$sourceParent = Join-Path $Root "historical-yolo"
if (-not (Test-Path -LiteralPath $sourceParent)) {
    $sourceParent = Join-Path $Root "00-source-yolo-historical"
}
if (-not (Test-Path -LiteralPath $sourceParent)) {
    $sourceParent = $Root
}

$subjects = Get-ChildItem -LiteralPath $sourceParent -Directory -Force |
    ForEach-Object {
        if ($_.Name -match '^sub-(\d+)_formal_') {
            $n = [int]$matches[1]
            if ($n -ge $MinSubject -and $n -ne 9504) {
                "sub-$('{0:D3}' -f $n)"
            }
        }
    } |
    Sort-Object { [int]($_ -replace '^sub-','') } -Unique

if (-not $subjects) {
    throw "No historical formal source subjects found at or above sub-$('{0:D3}' -f $MinSubject)."
}

Write-Host ("Subjects ({0}): {1}" -f $subjects.Count, ($subjects -join ', '))
Write-Host ""

$logDir = Join-Path $Root "runtime\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$index = 0
foreach ($subject in $subjects) {
    $index += 1
    Write-Host ""
    Write-Host ("========== [{0}/{1}] {2} ==========" -f $index, $subjects.Count, $subject) -ForegroundColor Cyan

    $args = @(
        "run_ritnet_fullclass_batch.py",
        "--output", $Root,
        "--subjects", $subject,
        "--device", "0"
    )
    if ($DryRun) {
        $args += "--dry-run"
    }

    & python @args
    $code = $LASTEXITCODE

    $batchSummary = Join-Path $Root "ritnet_fullclass_batch_summary.json"
    if (Test-Path -LiteralPath $batchSummary) {
        $dest = Join-Path $logDir ("{0}-batch-summary.json" -f $subject)
        Move-Item -LiteralPath $batchSummary -Destination $dest -Force
        Write-Host "Batch summary -> $dest"
    }

    if ($code -ne 0) {
        Write-Host ""
        Write-Host "STOPPED: $subject returned exit code $code." -ForegroundColor Red
        Write-Host "No later subject was started. Fix/resume this subject first."
        exit $code
    }
}

Write-Host ""
if ($DryRun) {
    Write-Host "=== DRY RUN COMPLETE ===" -ForegroundColor Green
} else {
    Write-Host "=== ALL REQUESTED SUBJECTS COMPLETE/SKIPPED ===" -ForegroundColor Green
}
