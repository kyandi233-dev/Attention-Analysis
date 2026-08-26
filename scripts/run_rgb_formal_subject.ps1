param(
    [Parameter(Mandatory = $true)]
    [string]$Subject,

    [string]$FaceModelDir = $env:ATTENTION_FACE_MODEL_DIR,

    [switch]$SharedDecode,

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

function Test-CompleteManifest {
    param([string]$Path)
    if (-not (Test-Path $Path -PathType Leaf)) {
        return $false
    }
    try {
        $data = Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json
        return $data.completion_status -eq "complete"
    }
    catch {
        return $false
    }
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
        default {
            throw "Unknown raw stage: $Name"
        }
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

Invoke-Checked "1/3 Prepare formal Face frame manifest (15 Hz, baseline start -> Block2 end)" `
    $RgbPython `
    @("scripts/face_formal_prepare.py", "--config", $Config, "--subject", $Subject)

if ($SharedDecode) {
    Write-Host "`n=== 2/3 Shared single-decode raw extraction ==="
    Write-Host "AVI decode: one sequential OpenCV reader"
    Write-Host "Motion: full-fps in shared reader"
    Write-Host "Pose:   10 Hz MediaPipe worker thread"
    Write-Host "Face:   15 Hz lossless raw-BGR pipe -> DirectML worker"
    Write-Host "Face defaults: RetinaFace B16, multitask B32, early 0.5 filter, overlapped CPU postprocess"
    Write-Host "This mode is experimental until representative parity/speed validation is complete."

    $sharedArgs = @(
        "scripts/rgb_formal_shared_decode.py", "--config", $Config,
        "--subject", $Subject,
        "--face-python", $FacePython,
        "--model-dir", $FaceModelDir
    ) + $ForceArg

    Invoke-Checked "2/3 Shared single-decode Motion + Pose + Py-Feat Face" `
        $RgbPython `
        $sharedArgs
}
else {
    Write-Host "`n=== 2/3 Parallel raw extraction: Motion + Pose + Py-Feat Face ==="
    Write-Host "Motion: full-fps OpenCV raw"
    Write-Host "Pose:   10 Hz MediaPipe landmark raw"
    Write-Host "Face:   15 Hz Py-Feat DirectML raw (RetinaFace B16, multitask B32)"
    Write-Host "Face optimizations: early 0.5 RetinaFace filter + overlapped CPU postprocess"
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

            $exitCode = $stage.Process.ExitCode
            $failed += $stage.Name
            Write-Warning "$($stage.Name) incomplete (process exit code: $exitCode)"
            if (Test-Path $stage.StdErr) {
                Write-Host "--- $($stage.Name) stderr tail ---"
                Get-Content $stage.StdErr -Tail 40
            }
            if (Test-Path $stage.StdOut) {
                Write-Host "--- $($stage.Name) stdout tail ---"
                Get-Content $stage.StdOut -Tail 20
            }
        }
        if ($failed.Count -gt 0) {
            throw "Parallel raw extraction incomplete: $($failed -join ', '). Completed branches are kept for resume."
        }
    }
    else {
        Write-Host "[resume] all three raw stages already complete; no extraction process started."
    }
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
