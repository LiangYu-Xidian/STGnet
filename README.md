# STGnet — Heterogeneous Graph Model for Cell-type Deconvolution

This repository provides an end-to-end pipeline on the **MOB** dataset to build a heterogeneous graph with **real spots / pseudo spots / genes**, and to train the **STGnet** model (a custom HeteroGAT-style architecture) to predict cell-type proportions for real spots.

## Code & files

The GitHub repository is organized into two main folders:

- `code/`: **all code and notebooks**
- `data/`: **all input data files**

Inside `code/`:

- **Data preparation & graph construction**
  - `code/main_mob.ipynb`: load MOB data from `data/` → select highly variable genes → generate pseudo spots → build adjacency matrices and export CSVs
  - `code/process.py`: pseudo-spot generation (e.g., `random_mix_with_dominant`)
  - `code/graph.py`: graph building utilities (real-real, real-pseudo, spot-gene adjacency)

- **Model training**
  - `code/MOB/500/base_v1.ipynb`: read adjacency CSVs + expression matrices → build a DGL heterograph → train the model → export predictions to `code/MOB/500/base_v1.csv`

## Environment setup (recommended: Conda)

Because this project uses **PyTorch + DGL + Scanpy**, we recommend creating a dedicated conda environment.

### 1) Create the environment

Use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate stgraph-mob
```

> Note: the CUDA/CPU builds of `torch` and `dgl` depend on your GPU and CUDA version. If installation fails, install PyTorch first (per the official instructions) and then install the matching DGL build (see FAQ).

### 2) (Optional) GPU installation notes

- **PyTorch**: install the build matching your CUDA version (follow official instructions)  
- **DGL**: install the DGL build matching your PyTorch/CUDA (e.g., `dgl-cu11x` / `dgl-cu12x`)

## Data inputs

By default, `code/main_mob.ipynb` expects the following files under `data/MOB/`:

- `data/MOB/spatial_count.csv`: spatial expression matrix (spot × gene)
- `data/MOB/spatial_location.csv`: spot coordinates (two columns: x/y)
- `data/MOB/sc_count.txt`: scRNA expression (indexed by `cell_id`; the notebook transposes it to cell × gene)
- `data/MOB/sc_labels.txt`: scRNA cell-type labels (single column)

Make sure file names and paths match. If the data are large, do not upload them to GitHub—use `.gitignore` (template included) to exclude them.

## Running the pipeline

### Step A: Generate pseudo spots & adjacency matrices (`code/main_mob.ipynb`)

Start Jupyter from the **code folder** so its relative paths resolve correctly:

```bash
cd code
jupyter lab
```

Open and run `main_mob.ipynb`. It will:

- load spatial / scRNA data, then normalize + log1p + select highly variable genes (default: 2000 HVGs)
- generate pseudo spots via `process.py::random_mix_with_dominant` (pseudo expression + label fractions)
- build three graphs via `graph.py`:
  - real ↔ real (fusing expression similarity and spatial proximity)
  - real ↔ pseudo (MNN / similarity-based)
  - spot ↔ gene (column-normalize then threshold edges)
- export key CSV files to `STgraph/MOB/500/` (example outputs):
  - `STgraph/MOB/500/pseudo_spot_expression.csv`
  - `STgraph/MOB/500/pseudo_spot_label_fractions.csv`
  - `STgraph/MOB/500/adj_real_real.csv`
  - `STgraph/MOB/500/adj_real_pseudo.csv`
  - `STgraph/MOB/500/adj_realspot_gene.csv`
  - `STgraph/MOB/500/adj_pseuspot_gene.csv`

### Step B: Train the heterograph model & predict (`code/MOB/500/base_v1.ipynb`)

`base_v1.ipynb` loads inputs using relative paths like `./adj_real_pseudo.csv`, so you must run it with **`code/MOB/500/` as the working directory**.

Recommended:

```bash
cd code/MOB/500
jupyter lab base_v1.ipynb
```

After training, it will write to `code/MOB/500/`:

- `base_v1.csv`: predicted cell-type fractions for real spots (rows = spots, columns = cell types)

## Outputs & reproducibility

- `base_v1.ipynb` sets random seeds (PyTorch/NumPy/CUDA) to improve reproducibility.
- Different machines / CUDA / cuDNN versions may still lead to small numerical differences.
