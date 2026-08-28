param(
    [string]$Output = "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR",
    [string]$Config = ".\config.yaml",
    [string]$Subjects = "",
    [string]$Device = "0",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RuntimeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "D:\Project\厚粲杯\08_算法\.venv_nir_gpu\Scripts\python.exe"
Set-Location -LiteralPath $RuntimeDir

Write-Host "=== NVIDIA FINAL TOPOLOGY RUNNER ==="
Write-Host "PythonExe: $PythonExe"
Write-Host "Output: $Output"
Write-Host "Policy: fixed .venv_nir_gpu; no PATH Python resolution; fail-closed."

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Required NVIDIA NIR Python executable not found: $PythonExe"
}

# Process-local CUDA DLL visibility only; never modify system/user PATH.
$NvidiaRoot = Join-Path (Split-Path -Parent (Split-Path -Parent $PythonExe)) "Lib\site-packages\nvidia"
$DllDirs = @(Get-ChildItem -LiteralPath $NvidiaRoot -Recurse -Directory -Filter bin -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty FullName)
if ($DllDirs.Count -gt 0) {
    $env:PATH = (($DllDirs -join ";") + ";" + $env:PATH)
    Write-Host ("Process-local CUDA DLL dirs: " + ($DllDirs -join ";"))
}

$Preflight = @'
import sys
import onnxruntime as ort
print("sys.executable:", sys.executable)
print("onnxruntime.__version__:", ort.__version__)
print("CUDAExecutionProvider available:", "CUDAExecutionProvider" in ort.get_available_providers())
if sys.executable.casefold() != r"D:\Project\厚粲杯\08_算法\.venv_nir_gpu\Scripts\python.exe".casefold():
    raise SystemExit("FAIL-CLOSED: wrong Python executable")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("FAIL-CLOSED: CUDAExecutionProvider unavailable")
'@
& $PythonExe -c $Preflight
if ($LASTEXITCODE -ne 0) { throw "NVIDIA NIR Python/CUDA preflight failed." }

$Args = @(
    ".\run_ritnet_fullclass_batch.py",
    "--output", $Output,
    "--config", $Config,
    "--device", $Device
)
if ($Subjects) { $Args += @("--subjects", $Subjects) }
if ($DryRun) { $Args += "--dry-run" }

Write-Host ("Command: & $PythonExe " + ($Args -join " "))
& $PythonExe @Args
exit $LASTEXITCODE
