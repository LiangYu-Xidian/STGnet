# STGnet

STGnet is a gene-aware heterogeneous graph framework for spatial cell-type deconvolution and domain-specific functional characterization in spatial transcriptomics.

## Code

- `process.py`  
  Generate pseudospots from single-cell RNA-seq data.

- `graph.py`  
  Construct real spot–real spot, real spot–pseudospot, and spot–gene graphs.

- `main_mob.ipynb`  
  Example preprocessing workflow on the mouse olfactory bulb (MOB) dataset, including pseudospot generation and graph construction.

- `base_v1.ipynb`  
  Train STGnet and predict cell-type proportions for spatial spots.

- `analysis.ipynb`  
  Perform downstream attention-based analysis to identify domain-specific genes.

## Environment

Create the environment using:

```bash
conda env create -f environment.yml
conda activate stgraph-mob
```

## Citation

If you use STGnet in your research, please cite:

> Accurate reconstruction of spatial cell-type maps and characterization of domain-specific functions based on a gene-aware heterogeneous network.  
> *Genome Research*.

