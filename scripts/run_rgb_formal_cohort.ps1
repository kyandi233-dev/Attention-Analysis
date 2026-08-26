param(
    [string[]]$Subjects,

    [string]$CudaDevice = "cuda",

    [int]$FaceBatch = 0,

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RgbEnv = "D:\conda_envs\attention-rgb"
$RgbPython = Join-Path $RgbEnv "python.exe"
$Config = "configs/rgb_analysis.yaml"
$OutputRoot = "D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB"
$InventoryPath = Join-Path $OutputRoot "rgb_inventory.csv"
$StatusPath = Join-Path $OutputRoot "cohort_status.csv"
$CohortManifestPath = Join-Path $OutputRoot "cohort_manifest.json"

if (-not (Test-Path $RgbPython)) {
    throw "RGB Python not found: $RgbPython"
}
Set-Location $RepoRoot

Write-Host "=== Refresh NVIDIA RGB inventory ==="
& $RgbPython scripts/rgb_analysis.py --config $Config --stage audit
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

Write-Host "Eligible NVIDIA RGB subjects selected: $($eligible.Count)"
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
                Write-Host "`n=== ${subject}: already complete, skip ==="
            }
        }
        catch { $status = "pending" }
    }

    if ($status -eq "pending") {
        Write-Host "`n=== Run $subject ==="
        try {
            $runnerArgs = @(
                "-ExecutionPolicy", "Bypass", "-File", ".\scripts\run_rgb_formal_subject.ps1",
                "-Subject", $subject,
                "-CudaDevice", $CudaDevice
            )
            if ($FaceBatch -gt 0) { $runnerArgs += @("-FaceBatch", "$FaceBatch") }
            if ($Force) { $runnerArgs += "-Force" }
            & powershell @runnerArgs
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
            Write-Warning "${subject} failed: $errorText"
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
    $statusRows | Export-Csv $StatusPath -NoTypeInformation -Encoding UTF8
}

$completeCount = @($statusRows | Where-Object { $_.status -in @("complete", "skipped_complete") }).Count
$failedCount = @($statusRows | Where-Object { $_.status -eq "failed" }).Count
$runFinished = (Get-Date).ToUniversalTime().ToString("o")
$cohortManifest = [ordered]@{
    schema_version = "rgb-nvidia-formal-cohort-manifest-v1.0"
    run_started_utc = $runStarted
    run_finished_utc = $runFinished
    selected_subjects = $eligible.Count
    complete_or_skipped = $completeCount
    failed = $failedCount
    status_csv = $StatusPath
    output_root = $OutputRoot
    cuda_device = $CudaDevice
    face_batch_override = $(if ($FaceBatch -gt 0) { $FaceBatch } else { $null })
    completion_status = $(if ($failedCount -eq 0 -and $completeCount -eq $eligible.Count) { "complete" } else { "incomplete" })
}
$cohortManifest | ConvertTo-Json -Depth 5 | Set-Content $CohortManifestPath -Encoding UTF8

Write-Host "`n=== NVIDIA RGB cohort run finished ==="
Write-Host "Selected: $($eligible.Count)"
Write-Host "Complete/skipped: $completeCount"
Write-Host "Failed: $failedCount"
Write-Host "Status: $StatusPath"
Write-Host "Manifest: $CohortManifestPath"
if ($failedCount -gt 0) { exit 2 }
