#
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
from src.TransformerST_graph_func import TransformerST_graph_construction1 as graph_construction
from src.TransformerST_graph_func import calculate_adj_matrix,search_l
from src.TransformerST_utils_func import mk_dir, adata_preprocess, load_ST_file
import anndata
from src.TransformerST_train_adaptive import TransformerST_Train
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc
import cv2
from pytictoc import TicToc

from anndata import AnnData
from rpy2.robjects.packages import importr
from rpy2.robjects import r
# from scipy.io import savemat
# import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
importr('mclust')
pandas2ri.activate()

warnings.filterwarnings('ignore')
torch.cuda.cudnn_enabled = False
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('===== Using device: ' + device)



# ################ Parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=20, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=3000, help='Dim of input genes')
parser.add_argument('--feat_hidden1', type=int, default=512, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=128, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=128, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=64, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
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
data_root = './data/Lung'
# all DLPFC folder list
# proj_list = ['A1', 'A2', 'A3', 'A4']
proj_list = ['A1']
# proj_list = ['151507']
# set saving result path
save_root = './output/Lung_adaptive/'


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
    file_fold = f'{data_root}/{data_name}/outs/'
    img_fold=f'{data_root}/{data_name}/outs/spatial/full_image.tif'
    # ################## Load data
    adata_h5 = load_ST_file(file_fold=file_fold,load_images=False)
    adata_h5.var_names_make_unique()
    # print(adata_h5.obsm['spatial'])
    adata = adata_preprocess(AnnData(adata_h5), min_cells=5, pca_n_comps=params.cell_feat_dim)
    adata_X=adata.X.toarray()
    # adata_X = adata.obsm['X_pca']
    # adata=AnnData(adata_h5)
    # sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    # pre_resolution = res_search_fixed_clus(adata, 5)
    pre_resolution=0.6
    sc.tl.leiden(adata, resolution=pre_resolution, key_added='expression_louvain_label')
    # nr, nc = adata_X.shape
    # x_sample_r = ro.r.matrix(adata_X, nrow=nr, ncol=nc)
    # label_r = r['Mclust'](x_sample_r, G=7, modelNames="EEE")
    # label = np.array(label_r.rx('classification'))
    # print(adata)
    # pre_labels = 'expression_louvain_label'
    # print(adata_X.shape,'ssssss')
    image = cv2.imread(img_fold)
    # print(img.shape)
    params.use_feature=0
    spatial_loc=adata_h5.obsm['spatial']
    # print(type(spatial_loc[0]))
    x_pixel=spatial_loc[:,2]
    y_pixel = spatial_loc[:,3]
    x=spatial_loc[:,0]
    y= spatial_loc[:, 1]
    # print(image.shape)
    # x=x_pixel
    # y=y_pixel
    beta=49
    alpha = 1
    beta_half = round(beta / 2)
    g = []
    # print(x_pixel.shape)
    ######################
    for i in range(spatial_loc.shape[0]):
        max_x = image.shape[0]
        max_y = image.shape[1]
        # print(x_pixel.shape,max(0, x_pixel[i] - beta_half),min(max_x, x_pixel[i] + beta_half + 1),max(0, y_pixel[i] - beta_half),min(max_y, y_pixel[i] + beta_half + 1))
        nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
              max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
        g.append(np.mean(np.mean(nbs, axis=0), axis=0))
    c0, c1, c2 = [], [], []
    for i in g:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    # print("Var of c0,c1,c2 = ", np.var(c0), np.var(c1), np.var(c2))
    c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
    c4 = (c3 - np.mean(c3)) / np.std(c3)
    z_scale = np.max([np.std(x), np.std(y)]) * alpha
    z = c4 * z_scale
    z = z.tolist()
    # print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
    X = np.array([x, y, z]).T.astype(np.float32)

    ###################
    graph_dict,data1= graph_construction(X, adata_h5.shape[0],adata.obs['expression_louvain_label'] , params)
    params.use_feature = 1
    graph_dict_prue, data1_prue = graph_construction(X, adata_h5.shape[0],
                                           adata.obs['expression_louvain_label'], params)
    params.save_path = mk_dir(f'{save_root}/{data_name}/TransformerST_adaptive')

    params.cell_num = adata_h5.shape[0]
    print('==== Graph Construction Finished')
    # ################## Model training
    TransformerST_net = TransformerST_Train(adata_X, graph_dict,data1,graph_dict_prue,data1_prue,adata_h5.obsm['spatial'],params)
    if params.using_dec:
        TransformerST_net.train_with_dec()
    else:
        TransformerST_net.train_without_dec()
    TransformerST_feat, _, _ = TransformerST_net.process()

    np.savez(f'{params.save_path}/TransformerST_result.npz', sedr_feat=TransformerST_feat, params=params)
    t.toc()
    # ################## Result plot
    adata_TransformerST = anndata.AnnData(TransformerST_feat)
    adata_h5 = load_ST_file(file_fold=file_fold,load_images=True)
    adata_h5.var_names_make_unique()
    adata_TransformerST.uns['spatial'] = adata_h5.uns['spatial']
    adata_TransformerST.obsm['spatial'] = adata_h5.obsm['spatial']
    sc.pp.neighbors(adata_TransformerST, n_neighbors=params.eval_graph_n)
    sc.tl.umap(adata_TransformerST)
    n_clusters = 5
    eval_resolution = res_search_fixed_clus(adata_TransformerST, n_clusters)
    # print(eval_resolution)
    sc.tl.leiden(adata_TransformerST, key_added="TransformerST_leiden", resolution=eval_resolution)
    # print(adata_sedr)
    sc.pl.spatial(adata_TransformerST, img_key="hires", color=['TransformerST_leiden'], show=False)
    plt.savefig(f'{params.save_path}/TransformerST_leiden_plot.jpg', bbox_inches='tight', dpi=150)
    df_meta = pd.read_csv(f'{data_root}/{data_name}/outs/metadata.tsv', sep='\t')
    df_meta['TransformerST'] = adata_TransformerST.obs['TransformerST_leiden'].tolist()
    df_meta.to_csv(f'{params.save_path}/metadata.tsv', sep='\t', index=False)

