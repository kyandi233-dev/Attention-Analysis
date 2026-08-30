"""生成前台监控 ps1（GBK 编码，供 Windows PowerShell 5.1 直接读取）。

用法: python scripts/maintenance/gen_monitor_ps1.py
输出: scripts/maintenance/monitor_terminal_20260831.ps1（GBK）
"""
from __future__ import annotations

from pathlib import Path

SCRIPT = r'''while ($true) {
    $l = (Get-CimInstance Win32_Processor).LoadPercentage
    Write-Host ("=== " + (Get-Date -Format "HH:mm:ss") + "  CPU " + $l + "% ===")
    $procs = Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        $cmd = $p.CommandLine
        if ($cmd -and $cmd -notmatch "monitor_terminal") {
            $wp = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
            $el = [math]::Round(((Get-Date) - $p.CreationDate).TotalMinutes, 1)
            $cpu = [math]::Round($wp.CPU, 0)
            $mem = [math]::Round($wp.WorkingSet64 / 1MB, 0)
            if ($cmd.Length -gt 130) { $cmd = $cmd.Substring(0, 130) + "..." }
            Write-Host ("  pid=" + $p.ProcessId + " run=" + $el + "min cpu=" + $cpu + "s mem=" + $mem + "MB")
            Write-Host ("    " + $cmd)
        }
    }
    $mm = "D:\Project\厚粲杯\11_数据\_FormalAnalysis\MultiModal"
    if (Test-Path $mm) {
        $new = Get-ChildItem $mm -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 2
        foreach ($f in $new) { Write-Host ("  [new] " + $f.Name + " @ " + $f.LastWriteTime.ToString("HH:mm:ss")) }
    }
    Start-Sleep 15
}
'''

OUT = Path(__file__).with_name("monitor_terminal_20260831.ps1")
OUT.write_text(SCRIPT, encoding="gbk")
print(f"已生成: {OUT}")
