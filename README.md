# gr-cls-multitask

Code for the dual-stream multi-task CNN study:

**From Tasks to Topology: Advancing the Debate on the Origins of the Dual Visual Pathways**  
Reza, Jordan, Luo, Patel, Tang, Niemeier  

Preprint: [bioRxiv 10.1101/2025.11.16.688720](https://www.biorxiv.org/content/10.1101/2025.11.16.688720v2)

A shared AlexNet-based backbone is trained jointly for **object classification** and **grasp prediction**. Analyses ask whether dorsal/ventral-like specialization can emerge from task optimization, then relate model units (Shapley attributions, connectivity) to human EEG.

## Repository layout

| Path | Role |
|------|------|
| `multi_task_models/` | Multi-task architectures (`grcn_multi_alex.py` = primary) |
| `data_processing/` | Jacquard → `.npy` preprocess; `data_loader_v2` / `v3` |
| `utils/parameters.py` | Model name, epochs, paths, Shapley layers / top-k |
| `multi_train.py` | Train multi-task model |
| `multi_test.py` | Evaluate trained weights |
| `shap/`, `shap_arrays/` | Shapley run outputs / arrays |
| `get_top_shapley.py`, `shapley_analysis.py` | Top-k units, correlations / distributions |
| `chi_test.py`, `dissociation.py`, `graph_analysis_*.py` | Task-bias graphs and connectivity stats |
| `rsa.py`, `rsm.py`, `process_data.py` | Model–EEG RSA / RSM helpers |
| `trained-models/` | Checkpoints (see `.gitignore`; `*_final.pth` may be tracked) |
| `matlab_files/` | Experiment / analysis MATLAB (placeholder in public tree) |
| `vis/` | Visualization helpers |

Default multi-task config in `utils/parameters.py`: **`multiAlexMap_top5_v1.5`** (150 epochs, batch 5, RGB+depth, 224×224, top-5 object classes).

## Prerequisites

- Python 3.10+ recommended  
- CUDA GPU strongly preferred for training / Shapley  
- PyTorch (pinned in `requirements.txt`)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

If `skimage==0.0` fails, install `scikit-image` instead and continue.

## Dataset

Training images come from a **Jacquard**-derived top-5 object set (RGB + depth + grasp maps), preprocessed to `.npy`.

1. Place / generate data under paths expected by `Params` (defaults):
   - Train: `data/top_5_compressed/train`
   - Test: `data/top_5_compressed/test`
2. Preprocess (when starting from raw Jacquard-style folders):

```bash
python data_processing/data_preprocess.py
```

Large image / array assets are gitignored (`*.png`, `*.npy`, etc.). Obtain the processed dataset from the authors or your lab copy; do not commit raw EEG (`.bdf` / trial-level) without REB approval.

## Model training

Hyperparameters and checkpoint names live in **`utils/parameters.py`** (`MODEL_NAME`, `EPOCHS`, `LR`, `BATCH_SIZE`, data paths).

```bash
python multi_train.py
```

Weights and logs land under `trained-models/<MODEL_NAME>/` (e.g. `multiAlexMap_top5_v1.5_final.pth`).

Evaluate:

```bash
python multi_test.py
```

Architecture entry point: [`multi_task_models/grcn_multi_alex.py`](multi_task_models/grcn_multi_alex.py) (`Multi_AlexnetMap_v3`).

### Reported multi-task variants (from lab training table)

| Model | Div. heads | Epochs | Batch | Loss ratio (grasp:class) | Grasp (train/test) | Class (train/test) |
|-------|------------|--------|-------|--------------------------|--------------------|--------------------|
| multiAlexMap_top5_v1.5 | 4 layers | 150 | 5 | 1.5 : 0.5 | 83.65 / 81.5 | 99.02 / 85.0 |
| multiAlexMap_top5_v1.4 | 4 layers | 150 | 5 | 0.5 : 1.5 | 77.9 / 75.5 | 97.98 / 84.5 |
| multiAlexMap_top5_v1.3 | 4 layers | 130 | 5 | — | 79.95 / 79.5 | 98.17 / 82.75 |
| multiAlexMap_top5_v1.2 | 1 layer | 150 | 2 | — | 72.4 / 67.0 | 98.53 / 89.25 |
| multiAlexMap_top5_v1.1 | 1 layer | 150 | 5 | — | 72.22 / 75.75 | 98.5 / 82.75 |

## Neuron Shapley

Shapley attribution runs write under `shap/` (and related folders). Useful scripts:

- `get_top_shapley.py` — aggregate top-k filters per layer / task  
- `shapley_analysis.py` — layer-wise correlations and distributions  
- `chi_test.py` — connection-type / task-bias analyses (includes effect-size style  
  \(d = (\mathrm{Shap_{cls}}-\mathrm{Shap_{grasp}}) / (|\mathrm{Shap_{cls}}|+|\mathrm{Shap_{grasp}}|+\varepsilon)\))

Layers used by default (`Params.LAYERS`):  
`rgb_features.0`, `features.0`, `features.4`, `features.7`, `features.10` (`TOP_K = 5`).

## Graph / dissociation analyses

- `graph_analysis_shapley.py`, `graph_analysis_weights.py`, `graph_analysis_activity.py`, `graph_analysis_heads.py`  
- `dissociation.py` — task dissociation helpers  

See script docstrings for inputs (Shapley H5 / arrays under `shap/` and `shap_arrays/`).

## EEG / RSA

- `process_data.py` — load / filter EEG-derived matrices for RSMs  
- `rsa.py`, `rsa_v2.py`, `rsm.py` — representational similarity vs model features  

Participant inclusion for the companion EEG set is documented in the preprint Methods (incomplete / dual-ID / task-incomplete exclusions). **Deidentified aggregates only** in public releases unless ethics approval covers raw data.

## MATLAB

PsychToolbox / analysis MATLAB is intended under `matlab_files/`. The public clone may only contain placeholders; use the lab desktop / cluster copy for the full experiment code.

## Citation

```bibtex
@article{Reza2025TasksToTopology,
  title={From Tasks to Topology: Advancing the Debate on the Origins of the Dual Visual Pathways},
  author={Reza, Tahsin and Jordan, Ewan and Luo, Steven Tin Sui and Patel, Marvi and Tang, Zi and Niemeier, Matthias},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.11.16.688720}
}
```

## Notes for contributors

- Edit training / data paths in `utils/parameters.py` rather than hard-coding in scripts.  
- Do not commit secrets, raw EEG, or large untracked dumps; respect `.gitignore`.  
- Paper–code details (RDM metric, seed lists, lesion indices) should stay aligned with the manuscript before claiming full reproducibility.
