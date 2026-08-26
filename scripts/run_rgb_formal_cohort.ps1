param(
    [string]$FaceModelDir = $env:ATTENTION_FACE_MODEL_DIR,

    [string[]]$Subjects,

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RgbEnv = "D:\CondaEnvs\attention-rgb"
$Config = "configs/rgb_analysis.yaml"
$OutputRoot = "D:\_AttentionData\Beijing-RGB"
$InventoryPath = Join-Path $OutputRoot "rgb_inventory.csv"
$StatusPath = Join-Path $OutputRoot "cohort_status.csv"
$CohortManifestPath = Join-Path $OutputRoot "cohort_manifest.json"

if (-not (Test-Path $RgbEnv)) {
    throw "RGB conda environment not found: $RgbEnv"
}
if ([string]::IsNullOrWhiteSpace($FaceModelDir)) {
    throw "FaceModelDir is required. Pass -FaceModelDir <dir> or set ATTENTION_FACE_MODEL_DIR."
}
if (-not (Test-Path $FaceModelDir)) {
    throw "Face model directory not found: $FaceModelDir"
}

Set-Location $RepoRoot

Write-Host "=== Refresh RGB inventory ==="
conda run -p $RgbEnv python scripts/rgb_analysis.py --config $Config --stage audit
if ($LASTEXITCODE -ne 0) {
    throw "RGB audit failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $InventoryPath)) {
    throw "RGB inventory not found after audit: $InventoryPath"
}

$inventory = Import-Csv $InventoryPath
$eligible = @($inventory | Where-Object { $_.analysis_eligible -eq "True" })
if ($Subjects -and $Subjects.Count -gt 0) {
    $wanted = @{}
    foreach ($subject in $Subjects) { $wanted[$subject] = $true }
    $eligible = @($eligible | Where-Object { $wanted.ContainsKey($_.subject) })
}
$eligible = @($eligible | Sort-Object subject)

if ($eligible.Count -eq 0) {
    throw "No eligible RGB subjects selected. Check rgb_inventory.csv or -Subjects."
}

Write-Host "Eligible subjects selected: $($eligible.Count)"
Write-Host "Output root: $OutputRoot"

$statusRows = New-Object System.Collections.Generic.List[object]
$runStarted = (Get-Date).ToUniversalTime().ToString("o")

foreach ($row in $eligible) {
    $subject = $row.subject
    $subjectDir = Join-Path $OutputRoot $subject
    $subjectManifest = Join-Path $subjectDir "$subject`_manifest.json"
    $started = (Get-Date).ToUniversalTime().ToString("o")
    $status = "pending"
    $errorText = ""

    if (-not $Force -and (Test-Path $subjectManifest)) {
        try {
            $existing = Get-Content $subjectManifest -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($existing.completion_status -eq "complete" -and $existing.extraction_complete -eq $true) {
                $status = "skipped_complete"
                Write-Host "`n=== $subject: already complete, skip ==="
            }
        }
        catch {
            # Invalid/incomplete final manifest is not considered complete; rerun the
            # subject orchestrator, whose individual stages already implement resume guards.
            $status = "pending"
        }
    }

    if ($status -eq "pending") {
        Write-Host "`n=== Run $subject ==="
        try {
            $forceArgs = @()
            if ($Force) { $forceArgs = @("-Force") }
            & powershell -ExecutionPolicy Bypass -File .\scripts\run_rgb_formal_subject.ps1 `
                -Subject $subject -FaceModelDir $FaceModelDir @forceArgs
            if ($LASTEXITCODE -ne 0) {
                throw "subject runner exited with code $LASTEXITCODE"
            }
            if (-not (Test-Path $subjectManifest)) {
                throw "final subject manifest missing after runner"
            }
            $final = Get-Content $subjectManifest -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($final.completion_status -ne "complete" -or $final.extraction_complete -ne $true) {
                throw "final validator did not mark extraction complete"
            }
            $status = "complete"
        }
        catch {
            $status = "failed"
            $errorText = $_.Exception.Message -replace "[\r\n]+", " "
            Write-Warning "$subject failed: $errorText"
        }
    }

    $finished = (Get-Date).ToUniversalTime().ToString("o")
    $statusRows.Add([pscustomobject]@{
        subject = $subject
        status = $status
        started_utc = $started
        finished_utc = $finished
        error = $errorText
        manifest = $subjectManifest
    })

    # Rewrite after every subject so an interrupted cohort run still leaves a useful
    # progress table. Resume is based on each subject's validated final manifest.
    $statusRows | Export-Csv $StatusPath -NoTypeInformation -Encoding UTF8
}

$completeCount = @($statusRows | Where-Object { $_.status -in @("complete", "skipped_complete") }).Count
$failedCount = @($statusRows | Where-Object { $_.status -eq "failed" }).Count
$runFinished = (Get-Date).ToUniversalTime().ToString("o")

$cohortManifest = [ordered]@{
    schema_version = "rgb-formal-cohort-manifest-v1.0"
    run_started_utc = $runStarted
    run_finished_utc = $runFinished
    selected_subjects = $eligible.Count
    complete_or_skipped = $completeCount
    failed = $failedCount
    status_csv = $StatusPath
    output_root = $OutputRoot
    completion_status = $(if ($failedCount -eq 0 -and $completeCount -eq $eligible.Count) { "complete" } else { "incomplete" })
}
$cohortManifest | ConvertTo-Json -Depth 5 | Set-Content $CohortManifestPath -Encoding UTF8

Write-Host "`n=== RGB cohort run finished ==="
Write-Host "Selected: $($eligible.Count)"
Write-Host "Complete/skipped: $completeCount"
Write-Host "Failed: $failedCount"
Write-Host "Status: $StatusPath"
Write-Host "Manifest: $CohortManifestPath"

if ($failedCount -gt 0) {
    exit 2
}
