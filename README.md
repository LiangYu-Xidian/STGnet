# STGnet — Heterogeneous Graph Model for Cell-type Deconvolution

This repository provides an end-to-end pipeline on the **MOB** dataset to build a heterogeneous graph with **real spots / pseudo spots / genes**, and to train the **STGnet** model (a gene-aware HeteroGAT-style architecture) to predict cell-type proportions for real spots.

## Code & files

The GitHub repository is organized into two main folders:

- `code/`: **all code and notebooks**
- `data/`: **all input data files**

Inside `code/`:

- **Data preparation & graph construction**
  - `code/main_mob.ipynb`: load MOB data from `data/MOB/` → select highly variable genes → generate pseudo spots → build adjacency matrices → export CSVs to `data/MOB/500/`
  - `code/process.py`: pseudo-spot generation (e.g., `random_mix_with_dominant`)
  - `code/graph.py`: graph building utilities (real-real, real-pseudo, spot-gene adjacency)

- **Model training**
  - `code/base_v1.ipynb`: read adjacency and expression CSVs from `data/MOB/500/` → build a DGL heterograph → train the model → export predictions to `code/base_v1.csv`

- **Other**
  - `code/analysis.ipynb`: This is an example of obtaining a domain-specific gene based on the attention score learned by the model.

## Environment setup (recommended: Conda)

Because this project uses **PyTorch + DGL + Scanpy**, we recommend creating a dedicated conda environment.

### 1) Create the environment

Use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate stgraph-mob
```

> Note: the CUDA/CPU builds of `torch` and `dgl` depend on your GPU and CUDA version. If installation fails, install PyTorch first (per the official instructions) and then install the matching DGL build.

### 2) (Optional) GPU installation notes

- **PyTorch**: install the build matching your CUDA version (follow official instructions)  
- **DGL**: install the DGL build matching your PyTorch/CUDA (e.g., `dgl-cu11x` / `dgl-cu12x`)

## Data inputs

Both notebooks assume the repository root contains `code/` and `data/`. Paths in the code are relative to the **`code/`** directory (e.g. `../data/MOB/...`).

**Step A (`main_mob.ipynb`)** reads the following from `data/MOB/`:

- `data/MOB/spatial_count.csv`: spatial expression matrix (spot × gene)
- `data/MOB/spatial_location.csv`: spot coordinates (two columns: x/y)
- `data/MOB/sc_count.txt`: scRNA expression (indexed by `cell_id`; the notebook transposes it to cell × gene)
- `data/MOB/sc_labels.txt`: scRNA cell-type labels (single column)

Place these files accordingly. If the data are large, do not upload them to GitHub—use `.gitignore` (template included) to exclude them.

## Running the pipeline

### Step A: Generate pseudo spots & adjacency matrices (`code/main_mob.ipynb`)

Start Jupyter from the **repository root** (the folder that contains both `code/` and `data/`), then open and run `code/main_mob.ipynb` **with the current working directory set to `code/`** (e.g. in Jupyter: run the notebook from `code/`, or ensure your kernel’s cwd is `code/`). The notebook uses paths like `../data/MOB/...`.

```bash
cd /path/to/repo
cd code
jupyter lab
```

Then open `main_mob.ipynb`. It will:

- load spatial / scRNA data from `data/MOB/`, then normalize + log1p + select highly variable genes (default: 2000 HVGs)
- generate pseudo spots via `process.py::random_mix_with_dominant` (pseudo expression + label fractions)
- build three graphs via `graph.py`:
  - real ↔ real (fusing expression similarity and spatial proximity)
  - real ↔ pseudo (MNN / similarity-based)
  - spot ↔ gene (column-normalize then threshold edges)
- export CSV files to **`data/MOB/500/`**:
  - `data/MOB/500/pseudo_spot_expression.csv`
  - `data/MOB/500/pseudo_spot_label_fractions.csv`
  - `data/MOB/500/real_spot_shared_genes.csv`
  - `data/MOB/500/pseudo_spot_shared_genes.csv`
  - `data/MOB/500/adj_real_real.csv`
  - `data/MOB/500/adj_real_pseudo.csv`
  - `data/MOB/500/adj_realspot_gene.csv`
  - `data/MOB/500/adj_pseuspot_gene.csv`

### Step B: Train the heterograph model & predict (`code/base_v1.ipynb`)

Run `code/base_v1.ipynb` with the **current working directory set to `code/`** (same as Step A). It reads from `../data/MOB/500/` (adjacency and expression CSVs) and writes:

- **`code/base_v1.csv`**: predicted cell-type fractions for real spots (rows = spots, columns = cell types)

## Outputs & reproducibility

- `base_v1.ipynb` sets random seeds (PyTorch/NumPy/CUDA) to improve reproducibility.
- Different machines / CUDA / cuDNN versions may still lead to small numerical differences.




