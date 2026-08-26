param(
    [Parameter(Mandatory = $true)]
    [string]$Subject,

    [string]$CudaDevice = "cuda",

    [int]$FaceBatch = 0,

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RgbEnv = "D:\conda_envs\attention-rgb"
$FaceEnv = "D:\conda_envs\attention-face-cuda"
$RgbPython = Join-Path $RgbEnv "python.exe"
$FacePython = Join-Path $FaceEnv "python.exe"
$Config = "configs/rgb_analysis.yaml"
$OutputRoot = "D:\Project\厚粲杯\11_数据\04_Attention-Analysis_nvidia-cuda_RGB"
$SubjectDir = Join-Path $OutputRoot $Subject

if (-not (Test-Path $RgbPython)) {
    throw "RGB Python not found: $RgbPython"
}
if (-not (Test-Path $FacePython)) {
    throw "Face CUDA Python not found: $FacePython"
}
if ($CudaDevice -notmatch '^cuda(?::\d+)?$') {
    throw "CudaDevice must be cuda or cuda:<index>; got: $CudaDevice"
}

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path $SubjectDir | Out-Null
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogDir = Join-Path $SubjectDir "_runlogs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ForceArg = @()
if ($Force) { $ForceArg = @("--force") }

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$Python,
        [string[]]$Arguments
    )
    Write-Host "`n=== $Label ==="
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Quote-ProcessArg {
    param([string]$Value)
    if ($Value -match '\s') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Test-CompleteManifest {
    param([string]$Path)
    if (-not (Test-Path $Path -PathType Leaf)) { return $false }
    try {
        $data = Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json
        return $data.completion_status -eq "complete"
    }
    catch { return $false }
}

function Test-RawStageComplete {
    param([string]$Name)
    switch ($Name) {
        "motion" {
            $raw = Join-Path $SubjectDir "$Subject`_motion_raw.parquet"
            $manifest = Join-Path $SubjectDir "$Subject`_motion_manifest.json"
        }
        "pose" {
            $raw = Join-Path $SubjectDir "$Subject`_pose_landmarks.parquet"
            $manifest = Join-Path $SubjectDir "$Subject`_pose_manifest.json"
        }
        "face" {
            $raw = Join-Path $SubjectDir "$Subject`_face_raw.parquet"
            $manifest = Join-Path $SubjectDir "$Subject`_face_raw_manifest.json"
        }
        default { throw "Unknown raw stage: $Name" }
    }
    return ((Test-Path $raw -PathType Leaf) -and (Test-CompleteManifest $manifest))
}

function Start-RawStage {
    param(
        [string]$Name,
        [string]$Python,
        [string[]]$Arguments
    )
    $stdout = Join-Path $LogDir "$RunStamp-$Name.stdout.log"
    $stderr = Join-Path $LogDir "$RunStamp-$Name.stderr.log"
    $quotedArgs = @($Arguments | ForEach-Object { Quote-ProcessArg $_ })
    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList $quotedArgs `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -NoNewWindow `
        -PassThru
    return [pscustomobject]@{
        Name = $Name
        Process = $process
        StdOut = $stdout
        StdErr = $stderr
    }
}

# Fail fast if the Face environment is not the native CUDA environment expected by this branch.
Invoke-Checked "0/3 Verify native PyTorch CUDA Face environment" `
    $FacePython `
    @("-c", "import torch,importlib.metadata as m; assert torch.cuda.is_available(), 'CUDA unavailable'; assert m.version('py-feat')=='2.1.1', m.version('py-feat'); print('torch=',torch.__version__,'cuda=',torch.version.cuda,'gpu=',torch.cuda.get_device_name(0),'py-feat=',m.version('py-feat'))")

Invoke-Checked "1/3 Prepare formal Face frame manifest (15 Hz, baseline start -> Block2 end)" `
    $RgbPython `
    @("scripts/face_formal_prepare.py", "--config", $Config, "--subject", $Subject)

Write-Host "`n=== 2/3 Parallel raw extraction: Motion + Pose + native PyTorch/CUDA Face ==="
Write-Host "Motion: full-fps OpenCV raw"
Write-Host "Pose:   10 Hz MediaPipe landmark raw"
if ($FaceBatch -gt 0) {
    Write-Host "Face:   15 Hz Py-Feat 2.1.1 Detectorv2 native CUDA, batch override=$FaceBatch"
}
else {
    Write-Host "Face:   15 Hz Py-Feat 2.1.1 Detectorv2 native CUDA, batch=config"
}
Write-Host "Tracking / eyelid / Pose features / blink-PERCLOS / QC are downstream."
Write-Host "Logs: $LogDir"

$motionArgs = @(
    "scripts/rgb_formal_motion_pose.py", "--config", $Config,
    "--subject", $Subject, "--stage", "motion"
) + $ForceArg
$poseArgs = @(
    "scripts/rgb_formal_motion_pose.py", "--config", $Config,
    "--subject", $Subject, "--stage", "pose"
) + $ForceArg
$faceArgs = @(
    "scripts/face_formal_cuda.py", "--config", $Config,
    "--subject", $Subject, "--device", $CudaDevice
)
if ($FaceBatch -gt 0) {
    $faceArgs += @("--batch-size", "$FaceBatch")
}
$faceArgs += $ForceArg

$stages = @()
if (-not $Force -and (Test-RawStageComplete "motion")) {
    Write-Host "[resume] motion=skip_complete"
}
else {
    $stages += Start-RawStage "motion" $RgbPython $motionArgs
}
if (-not $Force -and (Test-RawStageComplete "pose")) {
    Write-Host "[resume] pose=skip_complete"
}
else {
    $stages += Start-RawStage "pose" $RgbPython $poseArgs
}
if (-not $Force -and (Test-RawStageComplete "face")) {
    Write-Host "[resume] face=skip_complete"
}
else {
    $stages += Start-RawStage "face" $FacePython $faceArgs
}

if ($stages.Count -gt 0) {
    $parallelStarted = Get-Date
    while ($true) {
        $running = @()
        $elapsed = (Get-Date) - $parallelStarted
        $states = @()
        foreach ($stage in $stages) {
            $stage.Process.Refresh()
            if ($stage.Process.HasExited) {
                $states += "$($stage.Name)=finished"
            }
            else {
                $running += $stage
                $states += "$($stage.Name)=running"
            }
        }
        Write-Host ("[parallel {0:hh\:mm\:ss}] {1}" -f $elapsed, ($states -join " | "))
        if ($running.Count -eq 0) { break }
        Start-Sleep -Seconds 15
    }

    $failed = @()
    foreach ($stage in $stages) {
        $stage.Process.WaitForExit()
        $stage.Process.Refresh()
        if (Test-RawStageComplete $stage.Name) {
            Write-Host "[parallel] $($stage.Name)=complete"
            continue
        }
        $failed += $stage.Name
        Write-Warning "$($stage.Name) incomplete (process exit code: $($stage.Process.ExitCode))"
        if (Test-Path $stage.StdErr) {
            Write-Host "--- $($stage.Name) stderr tail ---"
            Get-Content $stage.StdErr -Tail 60
        }
        if (Test-Path $stage.StdOut) {
            Write-Host "--- $($stage.Name) stdout tail ---"
            Get-Content $stage.StdOut -Tail 30
        }
    }
    if ($failed.Count -gt 0) {
        throw "Parallel raw extraction incomplete: $($failed -join ', '). Completed branches are kept for resume."
    }
}
else {
    Write-Host "[resume] all three raw stages already complete; no extraction process started."
}

Invoke-Checked "3/3 Validate raw extraction completeness and CUDA backend evidence" `
    $RgbPython `
    @("scripts/rgb_formal_validate.py", "--config", $Config, "--subject", $Subject)

$SubjectManifest = Join-Path $SubjectDir "$Subject`_manifest.json"
Write-Host "`n=== NVIDIA formal RGB raw extraction complete ==="
Write-Host "Subject:  $Subject"
Write-Host "Output:   $SubjectDir"
Write-Host "Manifest: $SubjectManifest"
Write-Host "Core completion = Motion raw + Pose landmark raw + native PyTorch/CUDA Py-Feat Face raw."
Write-Host "Derived tracking/eyelid/Pose features/blink/PERCLOS/QC do not block the next subject."
