"""
run_all.py -- Tüm projeyi baştan sona yeniden üretir (veri -> benchmark'lar -> figürler -> rapor).

Kullanım:
    py -3.14 run_all.py                 # her şey (veri önbellekteyse API'ye gidilmez)
    py -3.14 run_all.py --skip-data     # veri hazırsa: sadece benchmark + figür + rapor
    py -3.14 run_all.py --only phase3   # tek bir adım

Her adım ayrı bir Python süreci olarak çalışır: bir adımın Numba/BLAS/Ray
durumu diğerinin ölçümünü etkilemesin (adil benchmark).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("fetch", ["dataset/fetch_arxiv.py", "--pages", "3"], True),
    ("preprocess", ["dataset/preprocess.py"], True),
    ("embed", ["dataset/embed.py"], True),
    ("profile", ["phase1_baseline/profiling_template.py"], False),
    ("baseline", ["phase1_baseline/naive_search.py"], False),
    ("phase2a", ["phase2_matrix_ops/sparse_dense_compare.py"], False),
    ("phase2b", ["phase2_matrix_ops/svd_power_iteration.py"], False),
    ("phase2c", ["phase2_matrix_ops/vectorise_vs_loop.py"], False),
    ("phase3", ["phase3_parallelism/run_phase3.py"], False),
    ("phase4", ["phase4_final/run_phase4.py"], False),
    ("ann", ["bonus/ann_compare.py"], False),
    ("roofline", ["bonus/roofline.py"], False),
    ("hardware", ["-c", "import common; common.save_json(common.hardware_info(), 'hardware.json')"], False),
    ("report", ["report/build_report.py"], False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    for name, cmd, is_data in STEPS:
        if args.only and name != args.only:
            continue
        if args.skip_data and is_data:
            continue
        print(f"\n{'=' * 70}\n>>> {name}: {' '.join(cmd)}\n{'=' * 70}", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env)
        print(f"<<< {name}: {time.perf_counter() - t0:.0f} s (çıkış kodu {r.returncode})")
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
