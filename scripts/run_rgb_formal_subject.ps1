param(
    [Parameter(Mandatory = $true)]
    [string]$Subject,

    [string]$FaceModelDir = $env:ATTENTION_FACE_MODEL_DIR,

    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RgbEnv = "D:\CondaEnvs\attention-rgb"
$FaceEnv = "D:\CondaEnvs\attention-face-directml"
$RgbPython = Join-Path $RgbEnv "python.exe"
$FacePython = Join-Path $FaceEnv "python.exe"
$Config = "configs/rgb_analysis.yaml"
$OutputRoot = "D:\_AttentionData\Beijing-RGB"
$SubjectDir = Join-Path $OutputRoot $Subject

if (-not (Test-Path $RgbPython)) {
    throw "RGB Python not found: $RgbPython"
}
if (-not (Test-Path $FacePython)) {
    throw "Face DirectML Python not found: $FacePython"
}
if ([string]::IsNullOrWhiteSpace($FaceModelDir)) {
    throw "FaceModelDir is required. Pass -FaceModelDir <dir> or set ATTENTION_FACE_MODEL_DIR."
}
if (-not (Test-Path $FaceModelDir)) {
    throw "Face model directory not found: $FaceModelDir"
}

$RfModel = Join-Path $FaceModelDir "pyfeat211_retinaface_r34.onnx"
$MtModel = Join-Path $FaceModelDir "pyfeat211_multitask_scientific_core.onnx"
if (-not (Test-Path $RfModel)) { throw "Missing Face model: $RfModel" }
if (-not (Test-Path $MtModel)) { throw "Missing Face model: $MtModel" }

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

# Step 1 is intentionally quick and sequential because Face raw needs the 15 Hz
# frame manifest. After it exists, Motion, Pose and Py-Feat are independent readers
# of the same source AVI and can run concurrently.
Invoke-Checked "1/3 Prepare formal Face frame manifest (15 Hz, baseline start -> Block2 end)" `
    $RgbPython `
    @("scripts/face_formal_prepare.py", "--config", $Config, "--subject", $Subject)

Write-Host "`n=== 2/3 Parallel raw extraction: Motion + Pose + Py-Feat Face ==="
Write-Host "Motion: full-fps OpenCV raw"
Write-Host "Pose:   10 Hz MediaPipe landmark raw"
Write-Host "Face:   15 Hz Py-Feat DirectML raw (RetinaFace B8, multitask B16)"
Write-Host "Derived tracking / eyelid / Pose features are intentionally deferred."
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
    "scripts/face_formal_directml.py", "--config", $Config,
    "--subject", $Subject, "--model-dir", $FaceModelDir
) + $ForceArg

$stages = @(
    (Start-RawStage "motion" $RgbPython $motionArgs),
    (Start-RawStage "pose" $RgbPython $poseArgs),
    (Start-RawStage "face" $FacePython $faceArgs)
)

$parallelStarted = Get-Date
while ($true) {
    $running = @($stages | Where-Object { -not $_.Process.HasExited })
    $elapsed = (Get-Date) - $parallelStarted
    $states = @()
    foreach ($stage in $stages) {
        if ($stage.Process.HasExited) {
            $states += "$($stage.Name)=done($($stage.Process.ExitCode))"
        }
        else {
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
    if ($stage.Process.ExitCode -ne 0) {
        $failed += $stage.Name
        Write-Warning "$($stage.Name) failed with exit code $($stage.Process.ExitCode)"
        if (Test-Path $stage.StdErr) {
            Write-Host "--- $($stage.Name) stderr tail ---"
            Get-Content $stage.StdErr -Tail 40
        }
        if (Test-Path $stage.StdOut) {
            Write-Host "--- $($stage.Name) stdout tail ---"
            Get-Content $stage.StdOut -Tail 20
        }
    }
}
if ($failed.Count -gt 0) {
    throw "Parallel raw extraction failed: $($failed -join ', '). Completed branches are kept for resume."
}

Invoke-Checked "3/3 Validate raw extraction completeness" `
    $RgbPython `
    @("scripts/rgb_formal_validate.py", "--config", $Config, "--subject", $Subject)

$SubjectManifest = Join-Path $SubjectDir "$Subject`_manifest.json"
Write-Host "`n=== Formal RGB raw extraction complete ==="
Write-Host "Subject:  $Subject"
Write-Host "Output:   $SubjectDir"
Write-Host "Manifest: $SubjectManifest"
Write-Host "Core completion = Motion raw + Pose landmark raw + Py-Feat Face raw."
Write-Host "Tracking, eyelid, Pose features, blink/PERCLOS and QC are downstream and do not block the next subject."
