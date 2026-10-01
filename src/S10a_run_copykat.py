#!/usr/bin/env python
"""run copyKAT per patient.

Runs copyKAT separately for each patient on raw counts, using annotated immune/stromal cells 
of that patient as the known diploid reference. Patients are run one by one so a failure in 
one does not stop the rest, and patients with existing results are skipped (re-run with --overwrite).

Outputs, per patient, in <OUTPUT_DIR>/<patient>/:
    copykat_result_<patient>_copykat_prediction.txt   copyKAT's own prediction table (read by jupyter notebook)
    copykat_result_<patient>_*                        other copyKAT outputs (heatmap, CNA matrix)
    <patient>_copykat.h5ad                            obs + CNV matrix only (no expression), for plotting
and, in <OUTPUT_DIR>/:
    copykat_run_params.json                           parameters and versions used
    failed_patients.txt                               patients that errored, with the message

Usage:
    python S10a_run_copykat.py                        # all patients
    python S10a_run_copykat.py --patients TN001 R001  # selected patients
    sbatch S10a_run_copykat.sbatch                    # on SLURM 
"""
import argparse
import json
import os
import sys
import traceback

import anndata as ad
import infercnvpy as cnv
import numpy as np
import scanpy as sc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_PATH = "./../snRNAseq/output/cellbender/cleanup/scVI_integrated_adata_clean_latent20_major_cell_type_inferCNV_add.h5ad"
OUTPUT_DIR = "./copyKAT"

PATIENT_COL = "ID"                      # one copyKAT run per patient (tumour + adjacent normal together)
LABEL_COL = "cell_type"
COUNTS_LAYER = "raw_counts"             # copyKAT needs raw UMI counts (not log-normalised)
NORMAL_CELL_TYPES = ["Myeloid", "Lymphoid", "Fibroblast", "Endothelial", "Non-adeno"]   # known diploid cells
MIN_CELLS = 50                          # patients with fewer cells are skipped

COPYKAT_PARAMS = dict(
    gene_ids="S",                       # gene symbols in var_names
    organism="human",
    segmentation_cut=0.05,              # KS.cut; smaller = stricter segmentation
    distance="euclidean",
    min_genes_chr=5,                    # ngene.chr; cells with fewer genes per chromosome are filtered
    window_size=25,                     # win.size (copyKAT default)
)
KEY = "cnv_copykat"


def run_patient(adata, pid, n_jobs, overwrite):
    out_dir = os.path.abspath(os.path.join(OUTPUT_DIR, pid))
    s_name = f"copykat_result_{pid}"
    pred_file = os.path.join(out_dir, f"{s_name}_copykat_prediction.txt") #read this later in step 10b
    if os.path.exists(pred_file) and not overwrite:
        print(f"[{pid}] result exists, skipped")
        return "skipped"

    mask = (adata.obs[PATIENT_COL].astype(str) == pid).to_numpy()
    if mask.sum() < MIN_CELLS:
        print(f"[{pid}] {mask.sum()} cells (< {MIN_CELLS}), skipped")
        return "too_few_cells"
    sub = adata[mask].to_memory() if adata.isbacked else adata[mask].copy()

    vals = sub.layers[COUNTS_LAYER][:500].data
    assert np.allclose(vals, np.round(vals)), f"layers['{COUNTS_LAYER}'] must contain raw counts"
    normals = sub.obs_names[sub.obs[LABEL_COL].isin(NORMAL_CELL_TYPES)].tolist()   # this patient's normals only
    print(f"[{pid}] {sub.n_obs} cells, {len(normals)} reference cells")
    if not normals:
        print(f"[{pid}] WARNING: no reference cells; copyKAT will estimate its own diploid baseline")

    os.makedirs(out_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(out_dir)                   # copyKAT writes its files to the working directory
    try:
        cnv.tl.copykat(sub, s_name=s_name, norm_cell_names=normals, key_added=KEY, inplace=True,
                       layer=COUNTS_LAYER, n_jobs=n_jobs, **COPYKAT_PARAMS)
    finally:
        os.chdir(cwd)

    small = ad.AnnData(obs=sub.obs[[PATIENT_COL, LABEL_COL, KEY]].copy(),
                       obsm={f"X_{KEY}": sub.obsm[f"X_{KEY}"]}, uns={KEY: sub.uns[KEY]})
    small.obs[KEY] = small.obs[KEY].astype(str)
    small.write(os.path.join(out_dir, f"{pid}_copykat.h5ad"))
    print(f"[{pid}] done: {small.obs[KEY].value_counts(dropna=False).to_dict()}")
    return "done"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patients", nargs="*", help="patient IDs to run (default: all)")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    parser.add_argument("--overwrite", action="store_true", help="re-run patients with existing results")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    adata = sc.read(INPUT_PATH, backed="r")          # load cells per patient, not the whole matrix
    patients = args.patients or sorted(adata.obs[PATIENT_COL].astype(str).unique())
    unknown = set(patients) - set(adata.obs[PATIENT_COL].astype(str))
    if unknown:
        sys.exit(f"unknown patients: {sorted(unknown)}")

    with open(os.path.join(OUTPUT_DIR, "copykat_run_params.json"), "w") as f:
        json.dump(dict(input=INPUT_PATH, patient_col=PATIENT_COL, normal_cell_types=NORMAL_CELL_TYPES,
                       counts_layer=COUNTS_LAYER, min_cells=MIN_CELLS, copykat=COPYKAT_PARAMS,
                       infercnvpy=cnv.__version__, n_jobs=args.n_jobs, patients=patients), f, indent=2)

    print(f"{len(patients)} patients, n_jobs = {args.n_jobs}")
    failed = {}
    for pid in patients:
        try:
            run_patient(adata, pid, args.n_jobs, args.overwrite)
        except Exception as e:                       # keep going; report at the end
            failed[pid] = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    if failed:
        with open(os.path.join(OUTPUT_DIR, "failed_patients.txt"), "w") as f:
            f.writelines(f"{p}\t{m}\n" for p, m in failed.items())
        sys.exit(f"{len(failed)} patient(s) failed: {sorted(failed)}")
    print("all patients finished")


if __name__ == "__main__":
    main()
