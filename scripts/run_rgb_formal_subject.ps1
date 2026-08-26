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
$Config = "configs/rgb_analysis.yaml"

if (-not (Test-Path $RgbEnv)) {
    throw "RGB conda environment not found: $RgbEnv"
}
if (-not (Test-Path $FaceEnv)) {
    throw "Face DirectML conda environment not found: $FaceEnv"
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
$ForceArg = @()
if ($Force) { $ForceArg = @("--force") }

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Host "`n=== $Label ==="
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked "1/4 Face formal frame preparation (15 Hz, full formal span)" {
    conda run -p $RgbEnv python scripts/face_formal_prepare.py `
        --config $Config --subject $Subject
}

Invoke-Checked "2/4 Motion + Pose full-span extraction" {
    conda run -p $RgbEnv python scripts/rgb_formal_motion_pose.py `
        --config $Config --subject $Subject --stage all @ForceArg
}

Invoke-Checked "3/4 Face DirectML full-span inference from original AVI" {
    conda run -p $FaceEnv python scripts/face_formal_directml.py `
        --config $Config --subject $Subject --model-dir $FaceModelDir @ForceArg
}

Invoke-Checked "4/4 Face tracking + primary face + eyelid derivation" {
    conda run -p $FaceEnv python scripts/face_formal_derive.py `
        --config $Config --subject $Subject @ForceArg
}

$SubjectDir = "D:\_AttentionData\Beijing-RGB\$Subject"
Write-Host "`n=== Formal subject pipeline complete ==="
Write-Host "Subject: $Subject"
Write-Host "Output:  $SubjectDir"
Write-Host "QC/gap review is downstream and does not block extraction completion."
