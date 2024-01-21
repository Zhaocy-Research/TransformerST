#
import os
import scanpy as sc
import pandas as pd
from pathlib import Path
from scanpy.readwrite import read_visium
from scanpy._utils  import check_presence_download
# import SpaGCN as spg
import numpy as np
def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


def prefilter_cells(adata, min_counts=None, max_counts=None, min_genes=200, max_genes=None):
    """
    Filters cells from an AnnData object based on criteria such as minimum and maximum counts and genes, 
    a crucial step in cleaning and standardizing single-cell data.
    """
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[0], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, min_genes=min_genes)[0]) if min_genes is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, max_genes=max_genes)[0]) if max_genes is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw = sc.pp.log1p(adata, copy=True)  # check the rowname
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:", adata.raw.var_names.is_unique)


def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    """
    Filters genes based on their presence in a minimum number of cells or their expression count, aiding in reducing noise and focusing on relevant genetic information.
    """
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)
def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    """
    Excludes specific genes (like ERCC spike-ins or mitochondrial genes) that can skew analysis, ensuring more accurate biological interpretation.
    """
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)
def adata_preprocess(i_adata, min_cells=3, pca_n_comps=300):
    """
     Filters genes and cells based on minimum counts, normalizes total counts per cell, identifies highly variable genes, and applies log transformation. 

    """
    print('===== Preprocessing Data ')
    # print(i_adata)
    # prefilter_genes(i_adata, min_cells=3)  # avoiding all genes are zeros
    # prefilter_specialgenes(i_adata)
    # Normalize and take log for UMI
    # sc.pp.normalize_per_cell(i_adata)
    # sc.pp.log1p(i_adata)
    # adata_X=i_adata.uns['log1p']
    # print(adata_X)
    sc.pp.filter_cells(i_adata, min_counts=5)
    # print(i_adata)
    # sc.pp.filter_cells(i_adata, max_counts=35000)
    # i_adata = i_adata[adata.obs["pct_counts_mt"] < 20]
    # print(f"#cells after MT filter: {adata.n_obs}")
    sc.pp.filter_genes(i_adata, min_cells=5)
    # print(i_adata)
    # sc.pp.normalize_per_cell(i_adata,counts_per_cell_after=1)
    # print(i_adata,"wwwwwwwwwww")
    # adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    sc.pp.highly_variable_genes(i_adata, flavor="seurat_v3", n_top_genes=pca_n_comps)
    sc.pp.normalize_total(i_adata, target_sum=1e4, exclude_highly_expressed=False, inplace=True)
    # sc.pp.combat(i_adata, key='batch', covariates=None, inplace=True)
    # print(i_adata)
    sc.pp.log1p(i_adata)
    # sc.pp.log1p(i_adata)

    # print(i_adata)
    # sc.pp.scale(i_adata)
    # sc.pp.highly_variable_genes(i_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    i_adata = i_adata[:, i_adata.var.highly_variable]
    # sc.pp.regress_out(i_data, ['total_counts', 'pct_counts_mt'])
    # sc.pp.scale(i_adata, max_value=10)
    # sc.pp.normalize_per_cell(i_adata)
    # sc.pp.log1p(i_adata)
    # print(np.amax(i_adata.X),"wwwww")
    # adata_X = sc.pp.scale(i_adata)
    # sc.pp.pca(i_adata, n_comps=pca_n_comps)

    # print(i_adata.obsm['X_pca'].shape)
    # return i_adata.obsm['X_pca']
    return i_adata
def adata_preprocess1(i_adata, min_cells=3, pca_n_comps=300):
    """
     Similar to adata_preprocess, this function filters genes, normalizes, scales, and applies PCA. 
     However, it does not explicitly focus on identifying highly variable genes
    """
    print('===== Preprocessing Data ')
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)
    # print(type(i_adata.X))
    sc.pp.scale(i_adata)
    sc.pp.pca(i_adata, n_comps=pca_n_comps)
    # print(i_adata)
    # i_adata.X=adata_X
    return i_adata

def adata_preprocess_bc(i_adata, min_cells=3, pca_n_comps=300):
    """
     ocuses on normalizing the total counts per cell and scaling the data. 
     Unlike the other two functions, it does not apply PCA or filter genes based on their variability,
    """
    print('===== Preprocessing Data ')
    # print(i_adata)
    # prefilter_genes(i_adata, min_cells=3)  # avoiding all genes are zeros
    # prefilter_specialgenes(i_adata)
    # Normalize and take log for UMI
    # sc.pp.normalize_per_cell(i_adata)
    # sc.pp.log1p(i_adata)
    # adata_X=i_adata.uns['log1p']
    # print(adata_X)
    # sc.pp.filter_cells(i_adata, min_counts=5)
    # print(i_adata)
    # sc.pp.filter_cells(i_adata, max_counts=35000)
    # i_adata = i_adata[adata.obs["pct_counts_mt"] < 20]
    # print(f"#cells after MT filter: {adata.n_obs}")
    # sc.pp.filter_genes(i_adata, min_cells=5)
    # print(i_adata)
    # sc.pp.normalize_per_cell(i_adata,counts_per_cell_after=1)
    # print(i_adata,"wwwwwwwwwww")
    # adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    # sc.pp.highly_variable_genes(i_adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=True)
    # sc.pp.combat(i_adata, key='batch', covariates=None, inplace=True)
    # print(i_adata)
    # sc.pp.log1p(i_adata)
    # sc.pp.log1p(i_adata)

    # print(i_adata)
    sc.pp.scale(i_adata)
    # adata_X=sc.pp.pca(adata_X, n_comps=pca_n_comps)
    # sc.pp.highly_variable_genes(i_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # i_adata = i_adata[:, i_adata.var.highly_variable]
    # sc.pp.regress_out(i_data, ['total_counts', 'pct_counts_mt'])
    # sc.pp.scale(i_adata, max_value=10)
    # sc.pp.normalize_per_cell(i_adata)
    # sc.pp.log1p(i_adata)
    # print(np.amax(i_adata.X),"wwwww")
    # adata_X = sc.pp.scale(i_adata)
    # sc.pp.pca(i_adata, n_comps=pca_n_comps)

    # print(i_adata.obsm['X_pca'].shape)
    # return i_adata.obsm['X_pca']
    return i_adata
def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    """
    load spatial transcriptomics data including optional image data.
    """
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()
    # print(adata_h5)
    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['array_row','array_col','pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy()
        # print(adata_h5.obsm['spatial'].shape,"fdfsdfsad")
        adata_h5.obs.drop(columns=['barcode', 'array_row','array_col'], inplace=True)
        # print(adata_h5.obs,adata_h5.obsm)
    print('adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5
def load_ST_file_histology(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=False,file_Adj=None):
    """
    Specifically tailored for loading histology-related spatial transcriptomics data
    """
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()
    # print(adata_h5)
    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['array_row','array_col','pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy()
        # print(adata_h5.obsm['spatial'].shape,"fdfsdfsad")
        adata_h5.obs.drop(columns=['barcode'], inplace=True)
        # print(adata_h5.obs,adata_h5.obsm)
    print('adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5
def load_ST_file1(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    """
    load spatial transcriptomics data including optional image data.
    """
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()
    # print(adata_h5)
    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_row_in_fullres',
            'pxl_col_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres']].to_numpy()
        # print(adata_h5.obsm['spatial'].shape,"fdfsdfsad")
        adata_h5.obs.drop(columns=['barcode', 'array_row','array_col'], inplace=True)
        # print(adata_h5.obs,adata_h5.obsm)
    print('adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5
def load_ST_file_gai(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    """
    load spatial transcriptomics data
    """
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()
    # print(adata_h5)
    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['array_row','array_col','pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy()
        # print(adata_h5.obsm['spatial'].shape,"fdfsdfsad")
        adata_h5.obs.drop(columns=['barcode'], inplace=True)
        # print(adata_h5.obs,adata_h5.obsm)
    print('adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5

# from scanpy
def _download_visium_dataset(
    sample_id: str,
    spaceranger_version: str,
    base_dir='./data/',
):
    """
    Downloads and extracts spatial transcriptomics datasets from the 10x Genomics platform, facilitating easy access to public datasets.
    """
    import tarfile

    url_prefix = f'https://cf.10xgenomics.com/samples/spatial-exp/{spaceranger_version}/{sample_id}/'

    sample_dir = Path(mk_dir(os.path.join(base_dir, sample_id)))

    # Download spatial data
    tar_filename = f"{sample_id}_spatial.tar.gz"
    tar_pth = Path(os.path.join(sample_dir, tar_filename))
    check_presence_download(filename=tar_pth, backup_url=url_prefix + tar_filename)
    with tarfile.open(tar_pth) as f:
        for el in f:
            if not (sample_dir / el.name).exists():
                f.extract(el, sample_dir)

    # Download counts
    check_presence_download(
        filename=sample_dir / "filtered_feature_bc_matrix.h5",
        backup_url=url_prefix + f"{sample_id}_filtered_feature_bc_matrix.h5",
    )


def load_visium_sge(sample_id='V1_Breast_Cancer_Block_A_Section_1', save_path='./data/'):
    """
    A higher-level function that downloads and loads a specific Visium spatial gene expression dataset, 
    providing a streamlined way to access and begin analyzing these complex datasets.
    """
    if "V1_" in sample_id:
        spaceranger_version = "1.1.0"
    else:
        spaceranger_version = "1.2.0"
    _download_visium_dataset(sample_id, spaceranger_version, base_dir=save_path)
    adata = read_visium(os.path.join(save_path, sample_id))

    print('adata: (' + str(adata.shape[0]) + ', ' + str(adata.shape[1]) + ')')
    return adata
