"""Isolated real-input benchmark for the CUDA compact cohort output contract."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from benchmark_nir_fullclass import extract_rois, load_rows, Monitor
from ritnet_fullclass_final_runtime import RitnetFullClassFinalRuntime
from ritnet_fullclass_metric_adapter import summarize_final_hard_metrics
from ritnet_fullclass_roi import valid_source_analysis_mask, fixed_aspect_roi_geometry
import yaml, cv2

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--max-eyes',type=int,default=1024); ap.add_argument('--config',type=Path,default=Path('runtime/nir-formal/config.yaml')); a=ap.parse_args()
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8')); video,rows=load_rows(a.run_dir.resolve(),a.max_eyes); rois=extract_rois(video,rows,cfg['fullclass']['roi']); runtime=RitnetFullClassFinalRuntime(Path('runtime/nir-formal')/cfg['models']['ritnet_fullclass_final'],device='0')
    tensor,valid,_=runtime.prepare_batch(rois[:16]); runtime.infer_prepared(tensor,valid)
    labels_seen=0; summary_ms=0.0
    with Monitor() as monitor:
        start=time.perf_counter()
        for off in range(0,len(rois),16):
            tensor,valid,_=runtime.prepare_batch(rois[off:off+16]); outputs,_=runtime.infer_prepared(tensor,valid)
            for label in outputs['labels']:
                t=time.perf_counter(); summarize_final_hard_metrics(label,None); summary_ms+=(time.perf_counter()-t)*1000
            labels_seen += valid
        wall=time.perf_counter()-start
    gpu=monitor.gpu; peak=max((x[1] for x in gpu),default=0); total=monitor.gpu_total
    frames=len({int(float(r['frame_idx'])) for r in rows})
    print(json.dumps({'implementation':'current-cuda-compact-cohort','provider':runtime.providers[0],'output_contract':'labels+class_probability-cohort','eyes':labels_seen,'frames':frames,'wall_sec':wall,'eyes_per_sec':labels_seen/wall,'frames_per_sec':frames/wall,'summary_ms':summary_ms,'gpu_avg_pct':float(np.mean([x[0] for x in gpu])) if gpu else None,'gpu_p95_pct':float(np.percentile([x[0] for x in gpu],95)) if gpu else None,'vram_peak_bytes':peak,'vram_total_bytes':total,'cpu_peak_pct':max(monitor.cpu,default=None),'ram_peak_bytes':max(monitor.ram,default=None)},ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
