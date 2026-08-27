"""Real-input CUDA parity check for compact cohort vs full uncertainty reduction."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, yaml
from benchmark_nir_fullclass import extract_rois, load_rows
from ritnet_fullclass_final_runtime import RitnetFullClassFinalRuntime
from ritnet_fullclass_uncertainty import summarize_uncertainty

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--config',type=Path,default=Path('runtime/nir-formal/config.yaml')); a=ap.parse_args()
    cfg=yaml.safe_load(a.config.read_text(encoding='utf-8')); video,rows=load_rows(a.run_dir.resolve(),16); rois=extract_rois(video,rows,cfg['fullclass']['roi']); rt=RitnetFullClassFinalRuntime(Path('runtime/nir-formal')/cfg['models']['ritnet_fullclass_final'],device='0'); tensor,valid,_=rt.prepare_batch(rois); compact,_=rt.infer_prepared(tensor,valid); full,_=rt._infer_full_prepared(tensor,valid); checks=[]
    for i in range(valid):
        mask=np.ones((400,640),dtype=bool); c=summarize_uncertainty(labels=compact['labels'][i],valid_source_mask=mask,class_probability=compact['class_probability'][i],max_probability=compact['max_probability'][i],top1_top2_margin=compact['top1_top2_margin'][i],entropy=compact['entropy'][i],inputs_validated=True); f=summarize_uncertainty(labels=full['labels'][i],valid_source_mask=mask,class_probability=full['class_probability'][i],max_probability=full['max_probability'][i],top1_top2_margin=full['top1_top2_margin'][i],entropy=full['entropy'][i]); checks.append((np.array_equal(compact['labels'][i],full['labels'][i]), all(np.isclose(c[k],f[k],rtol=0,atol=1e-6) for k in ('soft_background_fraction','soft_sclera_fraction','soft_iris_fraction','soft_pupil_fraction','ocular_max_probability_mean','ocular_top1_top2_margin_mean','ocular_entropy_mean'))))
    print(json.dumps({'provider':rt.providers[0],'eyes':valid,'labels_exact':all(x[0] for x in checks),'compact_scalars_equal_full':all(x[1] for x in checks),'checks':len(checks)},ensure_ascii=False,indent=2)); return 0 if all(all(x) for x in checks) else 1
if __name__=='__main__': raise SystemExit(main())
