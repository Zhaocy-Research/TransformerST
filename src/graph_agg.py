

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import src.network_utils as net_utils
from scipy.spatial import distance
import numpy as np
import pandas as pd
import cv2
from src.calculate_dis import *
from src.contour_util import *
from anndata import AnnData
# from models.lib import ops
# from config import cfg

def euclidean_distance(x,y):
    """
    Calculates the pairwise Euclidean distance between two sets of vectors. It is used as a distance metric for comparing feature embeddings
    """
    out = -2*torch.matmul(x, y)
    out += (x**2).sum(dim=-1, keepdim=True)
    out += (y**2).sum(dim=-2, keepdim=True)
    return out
def compute_distances(xe, ye, train=True):
    r"""
    Computes pairwise distances for all pairs of query items and
    potential neighbors.

    :param xe: BxNxE tensor of database (son) item embeddings
    :param ye: BxMxE tensor of query (father) item embeddings
    :param I: BxMxO index tensor that selects O potential neighbors in a window for each item in ye
    :param train: whether to use tensor comprehensions for inference (forward only)

    :return D: a BxMxO tensor of distances
    """

    # xe -> b n e
    # ye -> b m e
    # I  -> b m o
    b,n,e = xe.shape
    m = ye.shape[1]

    # if train or not cfg.NETWORK.WITH_WINDOW:
        # D_full -> b m n
    D = euclidean_distance(ye, xe.permute(0,2,1))
        # if cfg.NETWORK.WITH_WINDOW:
        #     # D -> b m o
        #     D = D.gather(dim=2, index=I) + 1e-5
    # else:
    #     o = I.shape[2]
    #     # xe_ind -> b m o e
    #     If = I.view(b, m*o,1).expand(b,m*o,e)
    #     # D -> b m o
    #     ye = ye.unsqueeze(3)
    #     D = -2*ops.indexed_matmul_1_efficient(xe, ye.squeeze(3), I).unsqueeze(3)
    #
    #     xe_sqs = (xe**2).sum(dim=-1, keepdim=True)
    #     xe_sqs_ind = xe_sqs.gather(dim=1, index=If[:,:,0:1]).view(b,m,o,1)
    #     D += xe_sqs_ind
    #     D += (ye**2).sum(dim=-2, keepdim=True) + 1e-5
    #
    #     D = D.squeeze(3)
        
    return D

def hard_knn(D, k):
    r"""
    Performs a k-nearest neighbors search, returning the indices and scores of the top k nearest neighbors in a dataset.
    input D: b m n
    output Idx: b m k
    """
    score, idx = torch.topk(D, k, dim=2, largest=False, sorted=True)
    # if cfg.NETWORK.WITH_WINDOW:
    #     idx = I.gather(dim=2, index=idx)

    return score, idx

class GraphConstruct(nn.Module):
    r"""
    Graph Construction,These classes create a graph structure from input data.
    """
    def __init__(self, scale, k, patchsize, stride, padding=None):
        r"""
        :param scale: downsampling factor
        :param indexer: function for creating index tensor
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphConstruct, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        # self.stride = stride
        # self.indexer = indexer
        self.k = k
        # self.padding = padding

    def graph_k(self, xe, ye):
        # xe -> b n e
        # ye -> b m e
        # I  -> b m o
        n = xe.shape[1]
        b, m, e = ye.shape
        k = self.k
        # print(xe.shape,ye.shape)
        # Euclidean Distance
        D = compute_distances(xe, ye,  train=self.training)

        # hard knn
        # return: b m k
        # print(D.shape,"wwwwwwww")
        score_k, idx_k = hard_knn(D, k)

        # xe -> b m e n
        # idx-> b m e k
        xe = xe.permute(0,2,1).contiguous()
        xe_e = xe.view(b,1,e,n).expand(b,m,e,n)
        idx_k_e = idx_k.view(b,m,1,k).expand(b,m,e,k)
        WITH_DIFF = True
        WITH_SCORE = True
        if WITH_DIFF:
            ye_e = ye.view(b,m,e,1).expand(b,m,e,k)
            diff_patch = ye_e-torch.gather(xe_e, dim=3, index=idx_k_e)
        else:
            diff_patch = None

        if WITH_SCORE:
            score_k = (-score_k/10.).exp()
        else:
            score_k = None

        # score_k: b m k
        # idx_k: b m k
        # diff_patch: b m e k
        return score_k, idx_k, diff_patch

    def forward(self, xe_patch, ye_patch):
        r"""
        :param xe: embedding of son features
        :param ye: embedding of father features

        :return score_k: similarity scores of top k nearest neighbors
        :return idx_k: indexs of top k nearest neighbors
        :return diff_patch: difference vectors between query and k nearest neighbors
        """
        # Convert everything to patches
        # H, W = ye.shape[2:]
        # xe_patch = ops.im2patch(xe, self.patchsize, self.stride, self.padding)
        # ye_patch, padding = ops.im2patch(ye, self.patchsize, self.stride, self.padding, returnpadding=True)
        xe_patch=torch.unsqueeze(xe_patch,0)
        # xe_patch = torch.unsqueeze(xe_patch, 3)
        ye_patch = torch.unsqueeze(ye_patch, 0)
        # print(xe_patch.shape,ye_patch.shape,"dddddd")
        # I = self.indexer(xe_patch, ye_patch)
        WITH_DIFF=True
        WITH_SCORE=True
        # if not self.training:
        #     index_neighbours_cache.clear()

        # bacth, channel, patchsize1, patchsize2, h, w
        # _,_,_,_,n1,n2 = xe_patch.shape
        # b,ce,e1,e2,m1,m2 = ye_patch.shape
        b,m,ce=ye_patch.shape
        k = self.k
        # n = n1*n2; m=m1*m2; e=ce*e1*e2
        # xe_patch = xe_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,e)
        # ye_patch = ye_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,e)

        # Get nearest neighbor volumes
        score_k, idx_k, diff_patch = self.graph_k(xe_patch, ye_patch)

        if WITH_DIFF:
            # diff_patch -> b,m,e,k      b m1*m2 ce e1*e2 k
            diff_patch = abs(diff_patch.view(b, m, ce, 1, k))
            diff_patch = torch.sum(diff_patch,dim=3,keepdim=True)
            diff_patch = diff_patch.expand(b, m, ce, self.scale, k)
            # print(diff_patch.shape,"wwwwwwww")
            # diff_patch: b m ce e1*s*e2*s k; e1==p1, e2==p2;
            # diff_patch -> b k ce e1*s*e2*s m
            diff_patch = diff_patch.permute(0,4,2,3,1).contiguous()
            # diff_patch -> b k*ce e1*s e2*s m1 m2 
            diff_patch = diff_patch.view(b,k*ce,self.scale,m)
            # padding_sr = [p*self.scale for p in padding]
            # z_sr -> b k*c_y H*s W*s
            # diff_patch = ops.patch2im(diff_patch, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
            # diff_patch = diff_patch.contiguous().view(b,k*ce,H*self.scale,W*self.scale)
            diff_patch=diff_patch.contiguous().view(b,k*ce,m*self.scale)
        if WITH_SCORE:
            # score_k: b,m,k --> b,k,e1*s,e2*s,m1,m2
            score_k = score_k.permute(0,2,1).contiguous().view(b,k,1,m)
            score_k = score_k.view(b,k,1,m).expand(b,k,self.scale,m)
            # padding_sr = [p*self.scale for p in padding]
            # score_k = ops.patch2im(score_k, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
            score_k = score_k.contiguous().view(b,k,m*self.scale)
        
        # score_k: b k H*s W*s
        # idx_k: b m k
        # diff_patch: b k*ce H*s W*s
        return score_k, idx_k, diff_patch


class GraphConstruct_spatial(nn.Module):
    r"""
    Graph Construction,These classes create a graph structure from input data with spatial transcriptomcis
    """

    def __init__(self, scale, k, patchsize, stride, padding=None):
        r"""
        :param scale: downsampling factor
        :param indexer: function for creating index tensor
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphConstruct_spatial, self).__init__()
        self.scale = scale
        # self.patchsize = patchsize
        # self.stride = stride
        # self.indexer = indexer
        self.k = k
        # self.padding = padding

    def graph_k(self, xe, ye,adj_coo):
        # xe -> b n e
        # ye -> b m e
        # I  -> b m o
        k=self.k
        knn_distanceType='euclidean'
        cell_num=xe.shape[1]
        D=np.zeros((xe.shape[1],xe.shape[1]))
        for node_idx in range(cell_num):
            tmp = adj_coo[node_idx, :].reshape(1, -1)
            distMat = distance.cdist(tmp, adj_coo, knn_distanceType)
            D[node_idx]=distMat
            # print(distMat.shape)
            # res = distMat.argsort()[:k * 2 + 1]
            # tmpdist = distMat[0, res[0][1:k * 2 + 1]]
            # tmp1 = xe[node_idx, :].reshape(1, -1)
            # distMat1 = distance.cdist(tmp1, xe[res[0][1:k * 2 + 1], :], knn_distanceType)
            # # print(distMat.shape)
            # # distMat1=distMat1[0, res[0][1:params.k*2 + 1]]
            # res1 = distMat1.argsort()[:k + 1]
            # tmpdist1 = distMat1[0, res1[0][1:k + 1]]
            # boundary = np.mean(tmpdist) + np.std(tmpdist)
            # boundary1 = np.mean(tmpdist1) + np.std(tmpdist1)
            # for j in np.arange(1, params.k + 1):
            #     if distMat1[0, res1[0][j]] <= boundary1:
            #         weight = 1.0
            #     else:
            #         weight = 0.0
            #     edgeList.append((node_idx, res[0, res1[0][j]], weight))

        # return edgeList




        n = xe.shape[1]
        b, m, e = ye.shape
        # k = self.k
        # print(xe.shape,ye.shape)
        # Euclidean Distance
        # D1 = compute_distances(xe, ye, train=self.training)
        # hard knn
        # return: b m k
        # print(D1.shape,"wweeee")
        D=torch.from_numpy(D).unsqueeze(0).cuda()
        # print(D.shape,"wwwwww")
        score_k, idx_k = hard_knn(D, k)

        # xe -> b m e n
        # idx-> b m e k
        xe = xe.permute(0, 2, 1).contiguous()
        xe_e = xe.view(b, 1, e, n).expand(b, m, e, n)
        idx_k_e = idx_k.view(b, m, 1, k).expand(b, m, e, k)
        WITH_DIFF = True
        WITH_SCORE = True
        if WITH_DIFF:
            ye_e = ye.view(b, m, e, 1).expand(b, m, e, k)
            diff_patch = ye_e - torch.gather(xe_e, dim=3, index=idx_k_e)
        else:
            diff_patch = None

        if WITH_SCORE:
            score_k = (-score_k / 10.).exp()
        else:
            score_k = None

        # score_k: b m k
        # idx_k: b m k
        # diff_patch: b m e k
        return score_k, idx_k, diff_patch

    def forward(self, xe_patch, ye_patch,spatial):
        r"""
        :param xe: embedding of son features
        :param ye: embedding of father features

        :return score_k: similarity scores of top k nearest neighbors
        :return idx_k: indexs of top k nearest neighbors
        :return diff_patch: difference vectors between query and k nearest neighbors
        """
        # Convert everything to patches
        # H, W = ye.shape[2:]
        # xe_patch = ops.im2patch(xe, self.patchsize, self.stride, self.padding)
        # ye_patch, padding = ops.im2patch(ye, self.patchsize, self.stride, self.padding, returnpadding=True)
        xe_patch = torch.unsqueeze(xe_patch, 0)
        # xe_patch = torch.unsqueeze(xe_patch, 3)
        ye_patch = torch.unsqueeze(ye_patch, 0)
        # print(xe_patch.shape,ye_patch.shape,"dddddd")
        # I = self.indexer(xe_patch, ye_patch)
        WITH_DIFF = True
        WITH_SCORE = True
        # if not self.training:
        #     index_neighbours_cache.clear()

        # bacth, channel, patchsize1, patchsize2, h, w
        # _,_,_,_,n1,n2 = xe_patch.shape
        # b,ce,e1,e2,m1,m2 = ye_patch.shape
        b, m, ce = ye_patch.shape
        k = self.k
        # n = n1*n2; m=m1*m2; e=ce*e1*e2
        # xe_patch = xe_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,e)
        # ye_patch = ye_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,e)

        # Get nearest neighbor volumes
        score_k, idx_k, diff_patch = self.graph_k(xe_patch, ye_patch,spatial)

        if WITH_DIFF:
            # diff_patch -> b,m,e,k      b m1*m2 ce e1*e2 k
            diff_patch = abs(diff_patch.view(b, m, ce, 1, k))
            diff_patch = torch.sum(diff_patch, dim=3, keepdim=True)
            diff_patch = diff_patch.expand(b, m, ce, self.scale, k)

            # diff_patch: b m ce e1*s*e2*s k; e1==p1, e2==p2;
            # diff_patch -> b k ce e1*s*e2*s m
            diff_patch = diff_patch.permute(0, 4, 2, 3, 1).contiguous()
            # diff_patch -> b k*ce e1*s e2*s m1 m2
            diff_patch = diff_patch.view(b, k * ce, self.scale, m)
            # padding_sr = [p*self.scale for p in padding]
            # z_sr -> b k*c_y H*s W*s
            # diff_patch = ops.patch2im(diff_patch, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
            # diff_patch = diff_patch.contiguous().view(b,k*ce,H*self.scale,W*self.scale)
            diff_patch = diff_patch.contiguous().view(b, k * ce, m * self.scale)
        if WITH_SCORE:
            # score_k: b,m,k --> b,k,e1*s,e2*s,m1,m2
            score_k = score_k.permute(0, 2, 1).contiguous().view(b, k, 1, m)
            score_k = score_k.view(b, k, 1, m).expand(b, k, self.scale, m)
            # padding_sr = [p*self.scale for p in padding]
            # score_k = ops.patch2im(score_k, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
            score_k = score_k.contiguous().view(b, k, m * self.scale)

        # score_k: b k H*s W*s
        # idx_k: b m k
        # diff_patch: b k*ce H*s W*s
        return score_k, idx_k, diff_patch


class GraphAggregation_spatial(nn.Module):
    r"""
    Graph Aggregation, These classes aggregate information across the constructed graphs in spatial transcriptomics. 
    They play a crucial role in synthesizing and summarizing the information captured in the graph structure
    """

    def __init__(self, scale=2, k=1, patchsize=1, stride=1, padding=None):
        r"""
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphAggregation_spatial, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        self.stride = stride
        self.k = k
        self.padding = padding

    def aggregation(self, yd, idx_k):
        r"""
        :param yd: database items, shape BxNxF
        :param idx_k: indexs of top k nearest neighbors

        :return: gathered features
        """
        # yd  -> b n f
        # I  -> b m o
        m = idx_k.shape[1]
        b, n, f = yd.shape
        k = self.k

        # yd -> b m f n
        # idx-> b m f k
        yd = yd.permute(0, 2, 1).contiguous()
        yd_e = yd.view(b, 1, f, n).expand(b, m, f, n)
        idx_k_e = idx_k.view(b, m, 1, k).expand(b, m, f, k)
        z = torch.gather(yd_e, dim=3, index=idx_k_e)

        # b m1*m2 c*p1*p2 k
        return z

    def forward(self, y_patch, yd_patch, idx_k):
        r"""
        :param y: query lr features
        :param yd: pixelshuffle_down features of y
        :param idx_k: indexs of top k nearest neighbors

        :return: aggregated hr features
        """
        # Convert everything to patches
        # y_patch, padding = ops.im2patch(y, self.patchsize, self.stride, self.padding, returnpadding=True)
        # yd_patch = ops.im2patch(yd, self.patchsize, self.stride, self.padding)

        # bacth, channel, patchsize1, patchsize2, h, w
        # _,_,H,W = y.shape
        y_patch = y_patch.unsqueeze(0)
        yd_patch = yd_patch.unsqueeze(0)
        yd_patch = yd_patch.view(1, y_patch.shape[1], -1)
        _, m, _ = y_patch.shape
        b, n, f = yd_patch.shape
        # print(yd_patch.shape,y_patch.shape,"wwsssw")
        # m = m1*m2; n = n1*n2; f=c*p1*p2
        k = self.k

        # y_patch  = y_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,f//self.scale**2)
        # yd_patch = yd_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,f)

        # Get nearest neighbor volumes
        # z_patch -> b m1*m2 c*p1*p2 k
        # print(yd_patch.shape,"ffff")
        z_patch = self.aggregation(yd_patch, idx_k)

        # Adaptive_instance_normalization
        # if cfg.NETWORK.WITH_ADAIN_NROM:
        #     reduce_scale = self.scale**2
        #     y_patch_norm = y_patch.view(b,m,c//reduce_scale,p1*p2)
        #     z_patch_norm = z_patch.view(b,m,c//reduce_scale,reduce_scale*p1*p2,k)
        #     z_patch = net_utils.adaptive_instance_normalization(y_patch_norm,z_patch_norm).view(*z_patch.size())

        # z_patch -> b k*c p1 p2 m1 m2
        z_patch = z_patch.permute(0, 3, 2, 1).contiguous()

        z_patch_sr = z_patch.view(b, k, f // self.scale, self.scale, m)
        z_sr = z_patch_sr
        # z_patch_sr = z_patch_sr.view(b,k*(c//self.scale**2),p1*self.scale,p2*self.scale,m1,m2)
        # padding_sr = [p*self.scale for p in padding]
        # z_sr -> b k*c_y H*s W*s
        # z_sr = ops.patch2im(z_patch_sr, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
        z_sr = z_sr.contiguous().view(b, k * (f // self.scale), m * self.scale)

        return z_sr
class GraphAggregation(nn.Module):
    r"""
    Graph Aggregation,These classes aggregate information across the constructed graphs. 
    They play a crucial role in synthesizing and summarizing the information captured in the graph structure
    """
    def __init__(self, scale, k, patchsize, stride, padding=None):
        r"""
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphAggregation, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        self.stride = stride
        self.k = k
        self.padding = padding

    def aggregation(self, yd, idx_k):
        r"""
        :param yd: database items, shape BxNxF
        :param idx_k: indexs of top k nearest neighbors

        :return: gathered features
        """
        # yd  -> b n f
        # I  -> b m o
        m = idx_k.shape[1]
        b, n, f = yd.shape
        k = self.k

        # yd -> b m f n
        # idx-> b m f k
        yd = yd.permute(0,2,1).contiguous()
        yd_e = yd.view(b,1,f,n).expand(b,m,f,n)
        idx_k_e = idx_k.view(b,m,1,k).expand(b,m,f,k)
        z = torch.gather(yd_e, dim=3, index=idx_k_e)

        # b m1*m2 c*p1*p2 k
        return z

    def forward(self, y_patch, yd_patch, idx_k):
        r"""
        :param y: query lr features
        :param yd: pixelshuffle_down features of y
        :param idx_k: indexs of top k nearest neighbors

        :return: aggregated hr features 
        """
        # Convert everything to patches
        # y_patch, padding = ops.im2patch(y, self.patchsize, self.stride, self.padding, returnpadding=True)
        # yd_patch = ops.im2patch(yd, self.patchsize, self.stride, self.padding)
      
        # bacth, channel, patchsize1, patchsize2, h, w
        # _,_,H,W = y.shape
        y_patch=y_patch.unsqueeze(0)
        yd_patch=yd_patch.unsqueeze(0)
        yd_patch=yd_patch.view(1,y_patch.shape[1],-1)
        _,m,_ = y_patch.shape
        b,n,f = yd_patch.shape
        # print(yd_patch.shape,y_patch.shape,"wwsssw")
        # m = m1*m2; n = n1*n2; f=c*p1*p2
        k = self.k

        # y_patch  = y_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,f//self.scale**2)
        # yd_patch = yd_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,f)

        # Get nearest neighbor volumes
        # z_patch -> b m1*m2 c*p1*p2 k
        # print(yd_patch.shape,"ffff")
        z_patch = self.aggregation(yd_patch, idx_k)

        # Adaptive_instance_normalization
        # if cfg.NETWORK.WITH_ADAIN_NROM:
        #     reduce_scale = self.scale**2
        #     y_patch_norm = y_patch.view(b,m,c//reduce_scale,p1*p2)
        #     z_patch_norm = z_patch.view(b,m,c//reduce_scale,reduce_scale*p1*p2,k)
        #     z_patch = net_utils.adaptive_instance_normalization(y_patch_norm,z_patch_norm).view(*z_patch.size())
        
        # z_patch -> b k*c p1 p2 m1 m2
        z_patch = z_patch.permute(0,3,2,1).contiguous()

        z_patch_sr = z_patch.view(b,k,f//self.scale,self.scale,m)
        z_sr=z_patch_sr
        # z_patch_sr = z_patch_sr.view(b,k*(c//self.scale**2),p1*self.scale,p2*self.scale,m1,m2)
        # padding_sr = [p*self.scale for p in padding]
        # z_sr -> b k*c_y H*s W*s
        # z_sr = ops.patch2im(z_patch_sr, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
        z_sr = z_sr.contiguous().view(b,k*(f//self.scale),m*self.scale)

        return z_sr


# index_neighbours_cache = {}
# def index_neighbours(xe_patch, ye_patch, window_size, scale):
#     r"""
#     This function generates the indexing tensors that define neighborhoods for each query patch in (father) features
#     It selects a neighborhood of window_size x window_size patches around each patch in xe (son) features
#     Index tensors get cached in order to speed up execution time
#     """
#     if cfg.NETWORK.WITH_WINDOW == False:
#         return None
#         # dev = xe_patch.get_device()
#         # key = "{}_{}_{}_{}_{}_{}".format(n1,n2,m1,m2,s,dev)
#         # if not key in index_neighbours_cache:
#         #     I = torch.tensor(range(n), device=dev, dtype=torch.int64).view(1,1,n)
#         #     I = I.repeat(b, m, 1)
#         #     index_neighbours_cache[key] = I
#
#         # I = index_neighbours_cache[key]
#         # return Variable(I, requires_grad=False)
#
#     b,_,n = xe_patch.shape
#     s = window_size
#
#     # if s>=n1 and s>=n2:
#     #     cfg.NETWORK.WITH_WINDOW = False
#     #     return None
#
#     s = min(s, n)
#     o = s
#     b,_,m= ye_patch.shape
#
#     dev = xe_patch.get_device()
#     key = "{}_{}_{}_{}".format(n,m,s,dev)
#     if not key in index_neighbours_cache:
#         I = torch.empty(1, m, o, device=dev, dtype=torch.int64)
#
#         ih = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1,1,s)
#         # iw = torch.tensor(range(s), device=dev, dtype=torch.int64).view(1,1,1,s)*n2
#
#         i = torch.tensor(range(m1), device=dev, dtype=torch.int64).view(m,1,1)
#         # j = torch.tensor(range(m2), device=dev, dtype=torch.int64).view(1,m2,1,1)
#
#         i_s = (torch.tensor(range(m1), device=dev, dtype=torch.int64).view(m,1,1)//2.0).long()
#         # j_s = (torch.tensor(range(m2), device=dev, dtype=torch.int64).view(1,m2,1,1)//2.0).long()
#
#         ch = (i_s-s//scale).clamp(0,n1-s)
#         # cw = (j_s-s//scale).clamp(0,n2-s)
#
#         cidx = ch
#         mI = cidx + ih
#         mI = mI.view(m,-1)
#         I[0,:,:] = mI
#
#         index_neighbours_cache[key] = I
#
#     I = index_neighbours_cache[key]
#     I = I.repeat(b,1,1)
#
#     return Variable(I, requires_grad=False)
class GraphConstruct_spatial_gai(nn.Module):
    r"""
    Graph Construction
    """

    def __init__(self, scale, k, patchsize, stride, padding=None):
        r"""
        :param scale: downsampling factor
        :param indexer: function for creating index tensor
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphConstruct_spatial_gai, self).__init__()
        self.scale = scale
        # self.patchsize = patchsize
        # self.stride = stride
        # self.indexer = indexer
        self.k = k
        # self.padding = padding

    def graph_k(self, xe, ye,adj_coo):
        # xe -> b n e
        # ye -> b m e
        # I  -> b m o
        xe_numpy=np.squeeze(xe.detach().cpu().numpy(), axis=0)
        ye_numpy= np.squeeze(ye.detach().cpu().numpy(), axis=0)
        k=self.k
        knn_distanceType='euclidean'
        cell_num=xe.shape[1]
        D=np.zeros((xe.shape[1],xe.shape[1]))
        for node_idx in range(cell_num):
            tmp = xe_numpy[node_idx, :].reshape(1, -1)
            distMat = distance.cdist(tmp, ye_numpy, knn_distanceType)
            D[node_idx]=distMat
            # print(distMat.shape)
            # res = distMat.argsort()[:k * 2 + 1]
            # tmpdist = distMat[0, res[0][1:k * 2 + 1]]
            # tmp1 = xe[node_idx, :].reshape(1, -1)
            # distMat1 = distance.cdist(tmp1, xe[res[0][1:k * 2 + 1], :], knn_distanceType)
            # # print(distMat.shape)
            # # distMat1=distMat1[0, res[0][1:params.k*2 + 1]]
            # res1 = distMat1.argsort()[:k + 1]
            # tmpdist1 = distMat1[0, res1[0][1:k + 1]]
            # boundary = np.mean(tmpdist) + np.std(tmpdist)
            # boundary1 = np.mean(tmpdist1) + np.std(tmpdist1)
            # for j in np.arange(1, params.k + 1):
            #     if distMat1[0, res1[0][j]] <= boundary1:
            #         weight = 1.0
            #     else:
            #         weight = 0.0
            #     edgeList.append((node_idx, res[0, res1[0][j]], weight))

        # return edgeList



        # xe=np.expand_dims(xe, axis=0)
        # ye = np.expand_dims(ye, axis=0)
        n = xe.shape[1]
        b,m, e = ye.shape
        # k = self.k
        # print(xe.shape,ye.shape)
        # Euclidean Distance
        # D1 = compute_distances(xe, ye, train=self.training)
        # hard knn
        # return: b m k
        # print(D1.shape,"wweeee")
        D=torch.from_numpy(D).unsqueeze(0).cuda()
        # print(D.shape,"wwwwww")
        score_k, idx_k = hard_knn(D, k)

        # xe -> b m e n
        # idx-> b m e k
        xe = xe.permute(0, 2, 1).contiguous()
        xe_e = xe.view(b, 1, e, n).expand(b, m, e, n)
        idx_k_e = idx_k.view(b, m, 1, k).expand(b, m, e, k)
        WITH_DIFF = True
        WITH_SCORE = True
        if WITH_DIFF:
            ye_e = ye.view(b, m, e, 1).expand(b, m, e, k)
            diff_patch = ye_e - torch.gather(xe_e, dim=3, index=idx_k_e)
        else:
            diff_patch = None

        if WITH_SCORE:
            score_k = (-score_k / 10.).exp()
        else:
            score_k = None

        # score_k: b m k
        # idx_k: b m k
        # diff_patch: b m e k
        return score_k, idx_k, diff_patch

    def forward(self, xe_patch, ye_patch,spatial):
        r"""
        :param xe: embedding of son features
        :param ye: embedding of father features

        :return score_k: similarity scores of top k nearest neighbors
        :return idx_k: indexs of top k nearest neighbors
        :return diff_patch: difference vectors between query and k nearest neighbors
        """
        # Convert everything to patches
        # H, W = ye.shape[2:]
        # xe_patch = ops.im2patch(xe, self.patchsize, self.stride, self.padding)
        # ye_patch, padding = ops.im2patch(ye, self.patchsize, self.stride, self.padding, returnpadding=True)
        xe_patch = torch.unsqueeze(xe_patch, 0)
        # xe_patch = torch.unsqueeze(xe_patch, 3)
        ye_patch = torch.unsqueeze(ye_patch, 0)
        # print(xe_patch.shape,ye_patch.shape,"dddddd")
        # I = self.indexer(xe_patch, ye_patch)
        WITH_DIFF = True
        WITH_SCORE = True
        # if not self.training:
        #     index_neighbours_cache.clear()

        # bacth, channel, patchsize1, patchsize2, h, w
        # _,_,_,_,n1,n2 = xe_patch.shape
        # b,ce,e1,e2,m1,m2 = ye_patch.shape
        b, m, ce = ye_patch.shape
        k = self.k
        # n = n1*n2; m=m1*m2; e=ce*e1*e2
        # xe_patch = xe_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,e)
        # ye_patch = ye_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,e)

        # Get nearest neighbor volumes
        score_k, idx_k, diff_patch = self.graph_k(xe_patch, ye_patch,spatial)

        if WITH_DIFF:
            # diff_patch -> b,m,e,k      b m1*m2 ce e1*e2 k
            diff_patch = abs(diff_patch.view(b, m, ce, 1, k))
            diff_patch = torch.sum(diff_patch, dim=3, keepdim=True)
            diff_patch = diff_patch.expand(b, m, ce, self.scale, k)

            # diff_patch: b m ce e1*s*e2*s k; e1==p1, e2==p2;
            # diff_patch -> b k ce e1*s*e2*s m
            diff_patch = diff_patch.permute(0, 4, 2, 3, 1).contiguous()
            # diff_patch -> b k*ce e1*s e2*s m1 m2
            diff_patch = diff_patch.view(b, k * ce, self.scale, m)
            # padding_sr = [p*self.scale for p in padding]
            # z_sr -> b k*c_y H*s W*s
            # diff_patch = ops.patch2im(diff_patch, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
            # diff_patch = diff_patch.contiguous().view(b,k*ce,H*self.scale,W*self.scale)
            diff_patch = diff_patch.contiguous().view(b, k * ce, m * self.scale)
        if WITH_SCORE:
            # score_k: b,m,k --> b,k,e1*s,e2*s,m1,m2
            score_k = score_k.permute(0, 2, 1).contiguous().view(b, k, 1, m)
            score_k = score_k.view(b, k, 1, m).expand(b, k, self.scale, m)
            # padding_sr = [p*self.scale for p in padding]
            # score_k = ops.patch2im(score_k, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
            score_k = score_k.contiguous().view(b, k, m * self.scale)

        # score_k: b k H*s W*s
        # idx_k: b m k
        # diff_patch: b k*ce H*s W*s
        return score_k, idx_k, diff_patch


class GraphAggregation_spatial_gai(nn.Module):
    r"""
    Graph Aggregation
    """

    def __init__(self, scale, k, patchsize, stride, padding=None):
        r"""
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphAggregation_spatial_gai, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        self.stride = stride
        self.k = k
        self.padding = padding

    def aggregation(self, yd, idx_k):
        r"""
        :param yd: database items, shape BxNxF
        :param idx_k: indexs of top k nearest neighbors

        :return: gathered features
        """
        # yd  -> b n f
        # I  -> b m o
        m = idx_k.shape[1]
        b, n, f = yd.shape
        k = self.k

        # yd -> b m f n
        # idx-> b m f k
        yd = yd.permute(0, 2, 1).contiguous()
        yd_e = yd.view(b, 1, f, n).expand(b, m, f, n)
        idx_k_e = idx_k.view(b, m, 1, k).expand(b, m, f, k)
        z = torch.gather(yd_e, dim=3, index=idx_k_e)

        # b m1*m2 c*p1*p2 k
        return z

    def forward(self, y_patch, yd_patch, idx_k):
        r"""
        :param y: query lr features
        :param yd: pixelshuffle_down features of y
        :param idx_k: indexs of top k nearest neighbors

        :return: aggregated hr features
        """
        # Convert everything to patches
        # y_patch, padding = ops.im2patch(y, self.patchsize, self.stride, self.padding, returnpadding=True)
        # yd_patch = ops.im2patch(yd, self.patchsize, self.stride, self.padding)

        # bacth, channel, patchsize1, patchsize2, h, w
        # _,_,H,W = y.shape
        y_patch = y_patch.unsqueeze(0)
        yd_patch = yd_patch.unsqueeze(0)
        yd_patch = yd_patch.view(1, y_patch.shape[1], -1)
        _, m, _ = y_patch.shape
        b, n, f = yd_patch.shape
        # print(yd_patch.shape,y_patch.shape,idx_k.shape,"wwsssw")
        # m = m1*m2; n = n1*n2; f=c*p1*p2
        k = self.k

        # y_patch  = y_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,f//self.scale**2)
        # yd_patch = yd_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,f)

        # Get nearest neighbor volumes
        # z_patch -> b m1*m2 c*p1*p2 k
        # print(yd_patch.shape,"ffff")
        z_patch = self.aggregation(yd_patch, idx_k)

        # Adaptive_instance_normalization
        # if cfg.NETWORK.WITH_ADAIN_NROM:
        #     reduce_scale = self.scale**2
        #     y_patch_norm = y_patch.view(b,m,c//reduce_scale,p1*p2)
        #     z_patch_norm = z_patch.view(b,m,c//reduce_scale,reduce_scale*p1*p2,k)
        #     z_patch = net_utils.adaptive_instance_normalization(y_patch_norm,z_patch_norm).view(*z_patch.size())

        # z_patch -> b k*c p1 p2 m1 m2
        z_patch = z_patch.permute(0, 3, 2, 1).contiguous()

        z_patch_sr = z_patch.view(b, k, f // self.scale, self.scale, m)
        z_sr = z_patch_sr
        # z_patch_sr = z_patch_sr.view(b,k*(c//self.scale**2),p1*self.scale,p2*self.scale,m1,m2)
        # padding_sr = [p*self.scale for p in padding]
        # z_sr -> b k*c_y H*s W*s
        # z_sr = ops.patch2im(z_patch_sr, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
        z_sr = z_sr.contiguous().view(b, k * (f // self.scale), m * self.scale)

        return z_sr
class GraphConstruct_TransformerST(nn.Module):
    r"""
    Graph Construction, specialized components of the TransformerST framework. 
    This class is responsible for constructing a graph that encapsulates both spatial and feature information inherent in spatial transcriptomics data. 
    It integrates spatial coordinates and gene expression data (or other molecular features) to 
    create a graph where nodes represent spatial locations (like individual cells or defined spatial areas) 
    and edges represent spatial relationships or similarities in feature space. 
    """

    def __init__(self, scale=2, k=1, patchsize=1, stride=1, padding=None):
        r"""
        :param scale: downsampling factor
        :param indexer: function for creating index tensor
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphConstruct_TransformerST, self).__init__()
        self.scale = scale
        # self.patchsize = patchsize
        # self.stride = stride
        # self.indexer = indexer
        self.k = k
        # self.padding = padding

    def graph_k(self, xe, ye,adj_coo):
        # xe -> b n e
        # ye -> b m e
        # I  -> b m o
        xe_numpy=np.squeeze(xe.detach().cpu().numpy(), axis=0)
        ye_numpy= np.squeeze(ye.detach().cpu().numpy(), axis=0)
        k=self.k
        knn_distanceType='euclidean'
        cell_num=xe.shape[1]
        D=np.zeros((xe.shape[1],xe.shape[1]))
        for node_idx in range(cell_num):
            tmp = xe_numpy[node_idx, :].reshape(1, -1)
            distMat = distance.cdist(tmp, ye_numpy, knn_distanceType)
            D[node_idx]=distMat
            # print(distMat.shape)
            # res = distMat.argsort()[:k * 2 + 1]
            # tmpdist = distMat[0, res[0][1:k * 2 + 1]]
            # tmp1 = xe[node_idx, :].reshape(1, -1)
            # distMat1 = distance.cdist(tmp1, xe[res[0][1:k * 2 + 1], :], knn_distanceType)
            # # print(distMat.shape)
            # # distMat1=distMat1[0, res[0][1:params.k*2 + 1]]
            # res1 = distMat1.argsort()[:k + 1]
            # tmpdist1 = distMat1[0, res1[0][1:k + 1]]
            # boundary = np.mean(tmpdist) + np.std(tmpdist)
            # boundary1 = np.mean(tmpdist1) + np.std(tmpdist1)
            # for j in np.arange(1, params.k + 1):
            #     if distMat1[0, res1[0][j]] <= boundary1:
            #         weight = 1.0
            #     else:
            #         weight = 0.0
            #     edgeList.append((node_idx, res[0, res1[0][j]], weight))

        # return edgeList



        # xe=np.expand_dims(xe, axis=0)
        # ye = np.expand_dims(ye, axis=0)
        n = xe.shape[1]
        b,m, e = ye.shape
        # k = self.k
        # print(xe.shape,ye.shape)
        # Euclidean Distance
        # D1 = compute_distances(xe, ye, train=self.training)
        # hard knn
        # return: b m k
        # print(D1.shape,"wweeee")
        D=torch.from_numpy(D).unsqueeze(0).cuda()
        # print(D.shape,"wwwwww")
        score_k, idx_k = hard_knn(D, k)

        # xe -> b m e n
        # idx-> b m e k
        xe = xe.permute(0, 2, 1).contiguous()
        xe_e = xe.view(b, 1, e, n).expand(b, m, e, n)
        idx_k_e = idx_k.view(b, m, 1, k).expand(b, m, e, k)
        WITH_DIFF = True
        WITH_SCORE = True
        if WITH_DIFF:
            ye_e = ye.view(b, m, e, 1).expand(b, m, e, k)
            diff_patch = ye_e - torch.gather(xe_e, dim=3, index=idx_k_e)
        else:
            diff_patch = None

        if WITH_SCORE:
            score_k = (-score_k / 10.).exp()
        else:
            score_k = None

        # score_k: b m k
        # idx_k: b m k
        # diff_patch: b m e k
        return score_k, idx_k, diff_patch

    def forward(self, raw_feature, img,raw,genes,num):
        binary = np.zeros((img.shape[0:2]), dtype=np.uint8)
        cnt = cv2_detect_contour(img, apertureSize=5, L2gradient=True)
        cnt_enlarged = scale_contour(cnt, 1.05)
        binary_enlarged = np.zeros(img.shape[0:2])
        cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
        x_max = np.amax(raw.obsm['spatial'][:, 3])
        y_max = np.amax(raw.obsm['spatial'][:, 2])
        x_min = np.min(raw.obsm['spatial'][:, 3])
        y_min = np.min(raw.obsm['spatial'][:, 2])
        # x_max, y_max=img.shape[0], img.shape[1]
        res = 100
        x_list = list(range(int(x_min), int(x_max), int(res)))
        y_list = list(range(int(y_min), int(y_max), int(res)))
        x = np.repeat(x_list, len(y_list)).tolist()
        y = y_list * len(x_list)
        super = pd.DataFrame({"x": x, "y": y})
        super = super[super.index.isin([i for i in super.index if (binary_enlarged[super.x[i], super.y[i]] != 0)])]
        b = res
        # sudo["color"]=
        super["color"] = extract_color(x_pixel=super.x.tolist(), y_pixel=super.y.tolist(), image=img, beta=b, RGB=False)
        s=1
        z_scale = np.max([np.std(super.x), np.std(super.y)]) * s
        super["z"] = (super["color"] - np.mean(super["color"])) / np.std(super["color"]) * z_scale
        super = super.reset_index(drop=True)
        # ------------------------------------Known points---------------------------------#
        raw_adata = raw[:, raw.var.index.isin(genes)]
        raw_adata.obs["x"] = raw_adata.obsm['spatial'][:, 3].astype(int).tolist()
        raw_adata.obs["y"] = raw_adata.obsm['spatial'][:, 2].astype(int).tolist()
        # print(raw_adata)
        raw_adata.obs["color"] = extract_color(x_pixel=raw_adata.obs["x"].astype(int).tolist(),
                                                 y_pixel=raw_adata.obs["y"].astype(int).tolist(), image=img, beta=b,
                                                 RGB=False)
        raw_adata.obs["z"] = (raw_adata.obs["color"] - np.mean(raw_adata.obs["color"])) / np.std(
            raw_adata.obs["color"]) * z_scale
        # -----------------------Distance matrix between sudo and known points-------------#
        # start_time = time.time()
        dis = np.zeros((super.shape[0], raw_adata.shape[0]))
        x_super, y_super, z_super = super["x"].values, super["y"].values, super["z"].values
        x_raw, y_raw, z_raw = raw_adata.obs["x"].values, raw_adata.obs["y"].values, raw_adata.obs[
            "z"].values
        for i in range(super.shape[0]):
            cord1 = np.array([x_super[i], y_super[i], z_super[i]])
            for j in range(raw_adata.shape[0]):
                cord2 = np.array([x_raw[j], y_raw[j], z_raw[j]])
                dis[i][j] = distance(cord1, cord2)
        dis = pd.DataFrame(dis, index=super.index, columns=raw_adata.obs.index)
        # print(super.shape[0],'fffff')
        super_adata = AnnData(np.zeros((super.shape[0], 3000)))
        super_adata.obs = super
        super_adata.var = raw_adata.var
        # # Convert everything to patches
        # # H, W = ye.shape[2:]
        # # xe_patch = ops.im2patch(xe, self.patchsize, self.stride, self.padding)
        # # ye_patch, padding = ops.im2patch(ye, self.patchsize, self.stride, self.padding, returnpadding=True)
        # xe_patch = torch.unsqueeze(xe_patch, 0)
        # # xe_patch = torch.unsqueeze(xe_patch, 3)
        # ye_patch = torch.unsqueeze(ye_patch, 0)
        # # print(xe_patch.shape,ye_patch.shape,"dddddd")
        # # I = self.indexer(xe_patch, ye_patch)
        # WITH_DIFF = True
        # WITH_SCORE = True
        # # if not self.training:
        # #     index_neighbours_cache.clear()
        #
        # # bacth, channel, patchsize1, patchsize2, h, w
        # # _,_,_,_,n1,n2 = xe_patch.shape
        # # b,ce,e1,e2,m1,m2 = ye_patch.shape
        # b, m, ce = ye_patch.shape
        # k = self.k
        # # n = n1*n2; m=m1*m2; e=ce*e1*e2
        # # xe_patch = xe_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,e)
        # # ye_patch = ye_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,e)
        #
        # # Get nearest neighbor volumes
        # score_k, idx_k, diff_patch = self.graph_k(xe_patch, ye_patch,spatial)
        #
        # if WITH_DIFF:
        #     # diff_patch -> b,m,e,k      b m1*m2 ce e1*e2 k
        #     diff_patch = abs(diff_patch.view(b, m, ce, 1, k))
        #     diff_patch = torch.sum(diff_patch, dim=3, keepdim=True)
        #     diff_patch = diff_patch.expand(b, m, ce, self.scale, k)
        #
        #     # diff_patch: b m ce e1*s*e2*s k; e1==p1, e2==p2;
        #     # diff_patch -> b k ce e1*s*e2*s m
        #     diff_patch = diff_patch.permute(0, 4, 2, 3, 1).contiguous()
        #     # diff_patch -> b k*ce e1*s e2*s m1 m2
        #     diff_patch = diff_patch.view(b, k * ce, self.scale, m)
        #     # padding_sr = [p*self.scale for p in padding]
        #     # z_sr -> b k*c_y H*s W*s
        #     # diff_patch = ops.patch2im(diff_patch, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
        #     # diff_patch = diff_patch.contiguous().view(b,k*ce,H*self.scale,W*self.scale)
        #     diff_patch = diff_patch.contiguous().view(b, k * ce, m * self.scale)
        # if WITH_SCORE:
        #     # score_k: b,m,k --> b,k,e1*s,e2*s,m1,m2
        #     score_k = score_k.permute(0, 2, 1).contiguous().view(b, k, 1, m)
        #     score_k = score_k.view(b, k, 1, m).expand(b, k, self.scale, m)
        #     # padding_sr = [p*self.scale for p in padding]
        #     # score_k = ops.patch2im(score_k, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
        #     score_k = score_k.contiguous().view(b, k, m * self.scale)
        #
        # # score_k: b k H*s W*s
        # # idx_k: b m k
        # # diff_patch: b k*ce H*s W*s
        return super_adata,dis


class GraphAggregation_TransformerST(nn.Module):
    r"""
    Graph Aggregation specialized components of the TransformerST framework. 
    Once a graph is constructed, this class aggregates information across the graph. 
    It synthesizes data from various nodes (representing spatial locations) to create a comprehensive representation of the spatial transcriptomics data. 
    """

    def __init__(self, scale=2, k=1, patchsize=1, stride=1, padding=None):
        r"""
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphAggregation_TransformerST, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        self.stride = stride
        self.k = k
        self.padding = padding

    # def aggregation(self, yd, idx_k):
    #
    #     return z

    def forward(self, weights,adata_x):
        r"""
        :return: aggregated hr features
        """
        return np.dot(weights, adata_x)
        # # Convert everything to patches
        # # y_patch, padding = ops.im2patch(y, self.patchsize, self.stride, self.padding, returnpadding=True)
        # # yd_patch = ops.im2patch(yd, self.patchsize, self.stride, self.padding)
        #
        # # bacth, channel, patchsize1, patchsize2, h, w
        # # _,_,H,W = y.shape
        # y_patch = y_patch.unsqueeze(0)
        # yd_patch = yd_patch.unsqueeze(0)
        # yd_patch = yd_patch.view(1, y_patch.shape[1], -1)
        # _, m, _ = y_patch.shape
        # b, n, f = yd_patch.shape
        # # print(yd_patch.shape,y_patch.shape,idx_k.shape,"wwsssw")
        # # m = m1*m2; n = n1*n2; f=c*p1*p2
        # k = self.k
        #
        # # y_patch  = y_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,f//self.scale**2)
        # # yd_patch = yd_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,f)
        #
        # # Get nearest neighbor volumes
        # # z_patch -> b m1*m2 c*p1*p2 k
        # # print(yd_patch.shape,"ffff")
        # z_patch = self.aggregation(yd_patch, idx_k)
        #
        # # Adaptive_instance_normalization
        # # if cfg.NETWORK.WITH_ADAIN_NROM:
        # #     reduce_scale = self.scale**2
        # #     y_patch_norm = y_patch.view(b,m,c//reduce_scale,p1*p2)
        # #     z_patch_norm = z_patch.view(b,m,c//reduce_scale,reduce_scale*p1*p2,k)
        # #     z_patch = net_utils.adaptive_instance_normalization(y_patch_norm,z_patch_norm).view(*z_patch.size())
        #
        # # z_patch -> b k*c p1 p2 m1 m2
        # z_patch = z_patch.permute(0, 3, 2, 1).contiguous()
        #
        # z_patch_sr = z_patch.view(b, k, f // self.scale, self.scale, m)
        # z_sr = z_patch_sr
        # # z_patch_sr = z_patch_sr.view(b,k*(c//self.scale**2),p1*self.scale,p2*self.scale,m1,m2)
        # # padding_sr = [p*self.scale for p in padding]
        # # z_sr -> b k*c_y H*s W*s
        # # z_sr = ops.patch2im(z_patch_sr, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
        # z_sr = z_sr.contiguous().view(b, k * (f // self.scale), m * self.scale)

        # return z_sr
class GraphAggregation_spatial_gai1(nn.Module):
    r"""
    Graph Aggregation, other variant of graph aggregation tailored for spatial data, 
    """

    def __init__(self, scale, k, patchsize, stride, padding=None):
        r"""
        :param k: number of neighbors
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        """
        super(GraphAggregation_spatial_gai1, self).__init__()
        self.scale = scale
        self.patchsize = patchsize
        self.stride = stride
        self.k = k
        self.padding = padding

    def aggregation(self, yd, idx_k):
        r"""
        :param yd: database items, shape BxNxF
        :param idx_k: indexs of top k nearest neighbors

        :return: gathered features
        """
        # yd  -> b n f
        # I  -> b m o
        m = idx_k.shape[1]
        b, n, f = yd.shape
        k = self.k

        # yd -> b m f n
        # idx-> b m f k
        yd = yd.permute(0, 2, 1).contiguous()
        yd_e = yd.view(b, 1, f, n).expand(b, m, f, n)
        idx_k_e = idx_k.view(b, m, 1, k).expand(b, m, f, k)
        z = torch.gather(yd_e, dim=3, index=idx_k_e)

        # b m1*m2 c*p1*p2 k
        return z

    def forward(self, y_patch, yd_patch, idx_k):
        r"""
        :param y: query lr features
        :param yd: pixelshuffle_down features of y
        :param idx_k: indexs of top k nearest neighbors

        :return: aggregated hr features
        """
        # Convert everything to patches
        # y_patch, padding = ops.im2patch(y, self.patchsize, self.stride, self.padding, returnpadding=True)
        # yd_patch = ops.im2patch(yd, self.patchsize, self.stride, self.padding)

        # bacth, channel, patchsize1, patchsize2, h, w
        # _,_,H,W = y.shape
        y_patch = y_patch.unsqueeze(0)
        yd_patch = yd_patch.unsqueeze(0)
        yd_patch = yd_patch.view(1, y_patch.shape[1], -1)
        _, m, _ = y_patch.shape
        b, n, f = yd_patch.shape
        # print(yd_patch.shape,y_patch.shape,"wwsssw")
        # m = m1*m2; n = n1*n2; f=c*p1*p2
        k = self.k

        # y_patch  = y_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,f//self.scale**2)
        # yd_patch = yd_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,f)

        # Get nearest neighbor volumes
        # z_patch -> b m1*m2 c*p1*p2 k
        # print(yd_patch.shape,"ffff")
        z_patch = self.aggregation(yd_patch, idx_k)

        # Adaptive_instance_normalization
        # if cfg.NETWORK.WITH_ADAIN_NROM:
        #     reduce_scale = self.scale**2
        #     y_patch_norm = y_patch.view(b,m,c//reduce_scale,p1*p2)
        #     z_patch_norm = z_patch.view(b,m,c//reduce_scale,reduce_scale*p1*p2,k)
        #     z_patch = net_utils.adaptive_instance_normalization(y_patch_norm,z_patch_norm).view(*z_patch.size())

        # z_patch -> b k*c p1 p2 m1 m2
        z_patch = z_patch.permute(0, 3, 2, 1).contiguous()

        z_patch_sr = z_patch.view(b, k, f // self.scale, self.scale, m)
        z_sr = z_patch_sr
        # z_patch_sr = z_patch_sr.view(b,k*(c//self.scale**2),p1*self.scale,p2*self.scale,m1,m2)
        # padding_sr = [p*self.scale for p in padding]
        # z_sr -> b k*c_y H*s W*s
        # z_sr = ops.patch2im(z_patch_sr, self.patchsize*self.scale, self.stride*self.scale, padding_sr)
        z_sr = z_sr.contiguous().view(b, k * (f // self.scale), m * self.scale)

        return z_sr
