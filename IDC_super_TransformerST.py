#
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from src.TransformerST_graph_func import TransformerST_graph_construction1 as graph_construction
# from src.TransformerST_graph_func import TansformerST_graph_construction_super
from src.TransformerST_graph_func import calculate_adj_matrix,search_l
from src.TransformerST_utils_func import mk_dir, adata_preprocess, load_ST_file1
from src.contour_util import *
from src.calculate_dis import *
from src.util import *
import anndata
from src.TransformerST_train_adaptive import TransformerST_Train
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc
import cv2
from pytictoc import TicToc
from src.util import *
from src.contour_util import *
from src.calculate_dis import *
from anndata import AnnData
from rpy2.robjects.packages import importr
from rpy2.robjects import r
# from scipy.io import savemat
# import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from sklearn.cluster import KMeans
importr('mclust')
pandas2ri.activate()
import tifffile
warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)



# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=3000, help='Dim of input genes')
parser.add_argument('--feat_hidden1', type=int, default=512, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=128, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=128, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=64, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=1, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=1, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.001, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.0001, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')

params = parser.parse_args()
params.device = device

# ################ Path setting
data_root = './data/IDC'
# all DLPFC folder list
# proj_list = ['A1', 'A2', 'A3', 'A4']
proj_list = ['1']
# proj_list = ['151507']
# set saving result path
save_root = './output/IDC_super_ST/'
def comp_tsne_km(adata,k=10):
    sc.pp.pca(adata)
    sc.tl.tsne(adata)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0).fit(adata.obsm['X_pca'])
    adata.obs['kmeans'] = kmeans.labels_.astype(str)
    return adata

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res
t = TicToc()

for proj_idx in range(len(proj_list)):
    t.tic()
    data_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx+1) + ' : ' + data_name)
    file_fold = f'{data_root}/'
    img_fold=f'{data_root}/V1_Human_Invasive_Ductal_Carcinoma_image.tif'
    im = tifffile.imread(img_fold)
    im=np.swapaxes(im,0,1)
    im = np.uint8(np.swapaxes(im, 1, 2))
    im=im[:,:,0]

    adata_h5 = load_ST_file1(file_fold=file_fold,load_images=False)
    adata_h5.var_names_make_unique()

    adata = adata_preprocess(AnnData(adata_h5), min_cells=5, pca_n_comps=params.cell_feat_dim)
    # print(adata)
    adata_X=adata.X.toarray()

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    pre_resolution=0.25
    sc.tl.leiden(adata, resolution=pre_resolution, key_added='expression_louvain_label')

    adata_TransformerST_super = super11(img=im, raw=adata,raw_feature=adata_X,
                                    genes=adata.var.index.tolist(), shape="None", s=1, k=2, num_nbs=1)
    adata_TransformerST_super.obsm['spatial'] = adata_TransformerST_super.obs[['x', 'y']].to_numpy()
    n_clusters = 10
    adata_TransformerST_super = comp_tsne_km(adata_TransformerST_super, n_clusters)

    adata_TransformerST_super.obs["pred"] = adata_TransformerST_super.obs['kmeans']
    adata_TransformerST_super.obs["pred"] = adata_TransformerST_super.obs["pred"].astype('category')

    adata_TransformerST_super.obs["x_pixel"] = adata_TransformerST_super.obsm["spatial"][:, 0]
    adata_TransformerST_super.obs["y_pixel"] = adata_TransformerST_super.obsm["spatial"][:, 1]

    sc.pl.spatial(adata_TransformerST_super, img=None, scale_factor=1, color='pred', spot_size=56, save='IDC_super.pdf')

