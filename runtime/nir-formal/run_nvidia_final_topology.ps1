param(
    [string]$Output = "D:\Project\厚粲杯\11_数据\01_Attention-Analysis_nvidia-cuda_formal_NIR",
    [string]$Config = ".\config.yaml",
    [string]$Subjects = "",
    [string]$Device = "0",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RuntimeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "D:\CondaEnvs\nir-nvidia\python.exe"
$Config = (Resolve-Path -LiteralPath $Config).Path
Set-Location -LiteralPath $RuntimeDir

Write-Host "=== NVIDIA FINAL TOPOLOGY RUNNER ==="
Write-Host "PythonExe: $PythonExe"
Write-Host "Output: $Output"
Write-Host "Policy: fixed D:\CondaEnvs\nir-nvidia; no PATH Python resolution; fail-closed."

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Required NVIDIA NIR Python executable not found: $PythonExe"
}

# Process-local CUDA DLL visibility only; never modify system/user PATH.
# The NVIDIA pip wheels keep their runtime DLLs in one bin directory per
# package (cublas, cudnn, cufft, ...). Enumerate those direct package bins so
# every child process inherits the complete CUDA dependency path.
$EnvRoot = Split-Path -Parent $PythonExe
$NvidiaRoot = Join-Path $EnvRoot "Lib\site-packages\nvidia"
if (-not (Test-Path -LiteralPath $NvidiaRoot -PathType Container)) {
    throw "Required NVIDIA wheel root not found: $NvidiaRoot"
}

$DllDirs = @(
    $CondaBin = Join-Path $EnvRoot "Library\bin"
    if (Test-Path -LiteralPath $CondaBin -PathType Container) {
        (Resolve-Path -LiteralPath $CondaBin).Path
    }
    Get-ChildItem -LiteralPath $NvidiaRoot -Directory -ErrorAction Stop |
        Where-Object { $_.Name -ne "__pycache__" } |
        ForEach-Object {
            $bin = Join-Path $_.FullName "bin"
            if (Test-Path -LiteralPath $bin -PathType Container) {
                (Resolve-Path -LiteralPath $bin).Path
            }
        } |
        Sort-Object -Unique
)
if ($DllDirs.Count -eq 0) {
    throw "No NVIDIA wheel DLL directories found under: $NvidiaRoot"
}

$CufftDll = Join-Path $NvidiaRoot "cufft\bin\cufft64_11.dll"
if (-not (Test-Path -LiteralPath $CufftDll -PathType Leaf)) {
    throw "Required CUDA dependency missing: $CufftDll"
}

$env:PATH = (($DllDirs -join ";") + ";" + $env:PATH)
Write-Host ("Process-local CUDA DLL dirs (inherited by child): " + ($DllDirs -join ";"))
Write-Host ("Required CUDA DLL: " + $CufftDll)

$Preflight = @'
import sys
from pathlib import Path
import onnxruntime as ort
print("sys.executable:", sys.executable)
print("onnxruntime.__version__:", ort.__version__)
available = ort.get_available_providers()
print("onnxruntime providers:", available)
print("CUDAExecutionProvider available:", "CUDAExecutionProvider" in available)
if sys.executable.casefold() != r"D:\CondaEnvs\nir-nvidia\python.exe".casefold():
    raise SystemExit("FAIL-CLOSED: wrong Python executable")
if "g:\\program files python" in sys.executable.casefold():
    raise SystemExit("FAIL-CLOSED: disallowed G:\\Program Files Python executable")
if "CUDAExecutionProvider" not in available:
    raise SystemExit("FAIL-CLOSED: CUDAExecutionProvider unavailable")

model = Path("models/ritnet-b16-fp32.onnx").resolve()
if not model.is_file():
    raise SystemExit(f"FAIL-CLOSED: CUDA preflight model missing: {model}")
options = ort.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session = ort.InferenceSession(
    str(model),
    sess_options=options,
    providers=[("CUDAExecutionProvider", {"device_id": 0})],
)
active = session.get_providers()
print("CUDA preflight session providers:", active)
if not active or active[0] != "CUDAExecutionProvider":
    raise SystemExit(f"FAIL-CLOSED: CUDA session did not bind first: {active}")
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
