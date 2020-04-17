# TSSCpath Overview
TSSCpath is a python script (python 3) for cell trajectory construction by using k-nearest neighbors graph algorithm. It aims at time-series single-cell RNA-seq data. It is a developmental version.

# Installation
> 1. Download python scripts from github
> 2. Import the main function

```python
import sys
sys.path.append(path+'/TSSCpath/src/')
import tsscpath
```

# Quick Start
**Input file**: Time-series single cell RNA-seq data.

**Output file**: An inferenced cell trajectory.

1. Data preprocessing
```python
# Load data by pandas
>>> import pandas
>>> data = pd.read_csv(path, header=0, sep=',', index_col=0)
# Data QC
>>> tsscpath.plot_QC(data)
# Data filtering
>>> data = tsscpath.cell_filter(data, MT_ratio=0.1, min_genes=200, min_cells=10)
# Data normalization 
>>> data_norm = tsscpath.normalize_data(datalist) #  datalist is a list which contains multiple filtered data
>>> data_norm = tsscpath.log_transform(data_norm)
# Get variable genes and log transform
>>> data_hvg_norm = tsscpath.get_HVG(data_norm)
>>> data_hvg_norm = tsscpath.log_transform(data_hvg_norm)
```

2. Construct and filter KNN graph
```python
# Get KNN edges
>>> knn_inner = tsscpath.get_knn_innner(data_hvg_norm, pca_n=100, k=100)
>>> knn_link = tsscpath.get_knn_link(data_hvg_norm, data_norm, pca_n=100, k=100)
>>> knn_all_edges,knn_filter_edges = tsscpath.filter_knn_graph(knn_inner, knn_link, mutual=True)
# Get gaint graph cells
>>> nodes_gaint = tsscpath.get_Ggaint(knn_inner, knn_filter_edges)
```

3. Get cell type information
```python
# Get cell type information
# It can be given by user, just one column names celltype
celltype = tsscpath.cell_clusters(data, pca_n=100, k=20)
```

4. Coarse filtered KNN graph
```python
# Get coarsed cell trajectory graph
>>> cell_trajectoryG = tsscpath.get_cell_trajectory(celltype, nodes_gaint, knn_filter_edges)
# Draw cell trajectory
>>> tsscpath.plot_cell_trajectory(cell_trajectoryG) # This command will create a file named "cell.trajectory.pdf"
```











