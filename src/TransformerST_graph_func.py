#
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from scipy.spatial import distance
import numba
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
	sum=0
	for i in range(t1.shape[0]):
		sum+=(t1[i]-t2[i])**2
	return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
	n=X.shape[0]
	adj=np.empty((n, n), dtype=np.float32)
	for i in numba.prange(n):
		for j in numba.prange(n):
			adj[i][j]=euclid_dist(X[i], X[j])
	return adj
def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
#x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
        print("Calculateing adj matrix using histology image...")
    #beta to control the range of neighbourhood when calculate grey vale for one spot
#alpha to control the color scale
        beta_half=round(beta/2)
        g=[]
        for i in range(len(x_pixel)):
            max_x=image.shape[0]
            max_y=image.shape[1]
            nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
            g.append(np.mean(np.mean(nbs,axis=0),axis=0))
        c0, c1, c2=[], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0=np.array(c0)
        c1=np.array(c1)
        c2=np.array(c2)
        print("Var of c0,c1,c2 = ", np.var(c0),np.var(c1),np.var(c2))
        c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
        c4=(c3-np.mean(c3))/np.std(c3)
        z_scale=np.max([np.std(x), np.std(y)])*alpha
        z=c4*z_scale
        # print(x.shape, y.shape, z.shape, "wwwwwwwwwwwwwww")
        z = z.tolist()
        print("Var of x,y,z = ", np.var(x),np.var(y),np.var(z))

        X=np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculateing adj matrix using xy only...")
        X=np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)

# edgeList to edgeDict
def edgeList2edgeDict(edgeList, nodesize):
    graphdict = {}
    tdict = {}
    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

    # check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []

    return graphdict


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)
def calculate_p(adj, l):
    adj_exp=np.exp(-1*(adj**2)/(2*(l**2)))
    return np.mean(np.sum(adj_exp,1))-1
def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    run=0
    p_low=calculate_p(adj, start)
    p_high=calculate_p(adj, end)
    if p_low>p+tol:
        print("l not found, try smaller start point.")
        return None
    elif p_high<p-tol:
        print("l not found, try bigger end point.")
        return None
    elif  np.abs(p_low-p) <=tol:
        print("recommended l = ", str(start))
        return start
    elif  np.abs(p_high-p) <=tol:
        print("recommended l = ", str(end))
        return end
    while (p_low+tol)<p<(p_high-tol):
        run+=1
        print("Run "+str(run)+": l ["+str(start)+", "+str(end)+"], p ["+str(p_low)+", "+str(p_high)+"]")
        if run >max_run:
            print("Exact l not found, closest values are:\n"+"l="+str(start)+": "+"p="+str(p_low)+"\nl="+str(end)+": "+"p="+str(p_high))
            return None
        mid=(start+end)/2
        p_mid=calculate_p(adj, mid)
        if np.abs(p_mid-p)<=tol:
            print("recommended l = ", str(mid))
            return mid
        if p_mid<=p:
            start=mid
            p_low=p_mid
        else:
            end=mid
            p_high=p_mid
# ====== Graph construction
def graph_computing1(adj_coo, cell_num, img,params):
    edgeList = []
    # print(adj_coo.shape)
    # b=49
    # s=5
    # p=0.5
    # adj_coo=calculate_adj_matrix(x=adj_coo[:,0],y=adj_coo[:,1], x_pixel=adj_coo[:,0], y_pixel=adj_coo[:,1], image=img, beta=b, alpha=s, histology=True)
    # l = search_l(p, adj_coo, start=0.01, end=10000, tol=0.01, max_run=100)
    # adj_coo = np.exp(-1 * (adj_coo ** 2) / (2 * (l ** 2)))
    # print(adj_coo.shape)
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, params.knn_distanceType)
        # print(distMat.shape)
        res = distMat.argsort()[:params.k + 1]
        tmpdist = distMat[0, res[0][1:params.k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, params.k + 1):
            if distMat[0, res[0][j]] <= boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((node_idx, res[0][j], weight))

    return edgeList
def graph_computing_adaptive(adj_coo, cell_num, init_label,params):
    edgeList = []
    # print(adj_coo.shape)
    # b=49
    # s=5
    # p=0.5
    # adj_coo=calculate_adj_matrix(x=adj_coo[:,0],y=adj_coo[:,1], x_pixel=adj_coo[:,0], y_pixel=adj_coo[:,1], image=img, beta=b, alpha=s, histology=True)
    # l = search_l(p, adj_coo, start=0.01, end=10000, tol=0.01, max_run=100)
    # adj_coo = np.exp(-1 * (adj_coo ** 2) / (2 * (l ** 2)))
    # print(adj_coo.shape)
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, params.knn_distanceType)
        # print(distMat.shape)
        if params.use_feature == 1:
            res = distMat.argsort()[:params.k  + 1]
            tmpdist = distMat[0, res[0][1:params.k + 1]]
            # tmp1 = x_feature[node_idx, :].reshape(1, -1)
            # distMat1 = distance.cdist(tmp1, x_feature[res[0][1:params.k * 2 + 1], :], params.knn_distanceType)
            # print(distMat.shape)
            # distMat1=distMat1[0, res[0][1:params.k*2 + 1]]
            # res1 = distMat1.argsort()[:params.k + 1]
            # tmpdist1 = distMat1[0, res1[0][1:params.k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            # boundary1 = np.mean(tmpdist1) + np.std(tmpdist1)
            for j in np.arange(1, params.k + 1):
                if distMat[0, res[0][j]] <= boundary and init_label[node_idx]==init_label[res[0][j]]:
                    weight = distMat[0, res[0][j]]
                # elif distMat1[0, res1[0][j]] <= boundary1 or distMat[0, res[0][j]] <= boundary:
                #     weight = 0.5
                else:
                    weight = 0.0
                edgeList.append((node_idx, res[0][j], weight))
        else:
            res = distMat.argsort()[:params.k + 1]
            tmpdist = distMat[0, res[0][1:params.k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            for j in np.arange(1, params.k + 1):
                if distMat[0, res[0][j]] <= boundary:
                    weight = distMat[0, res[0][j]]
                else:
                    weight = 0.0
                edgeList.append((node_idx, res[0][j], weight))
            # tmp1 = x_feature[node_idx, :].reshape(1, -1)
    return edgeList
def graph_computing_adaptive_histology(adj_coo, cell_num, init_label,histology,params):
    edgeList = []
    # print(adj_coo.shape)
    # b=49
    # s=5
    # p=0.5
    # adj_coo=calculate_adj_matrix(x=adj_coo[:,0],y=adj_coo[:,1], x_pixel=adj_coo[:,0], y_pixel=adj_coo[:,1], image=img, beta=b, alpha=s, histology=True)
    # l = search_l(p, adj_coo, start=0.01, end=10000, tol=0.01, max_run=100)
    # adj_coo = np.exp(-1 * (adj_coo ** 2) / (2 * (l ** 2)))
    # print(adj_coo.shape)
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, params.knn_distanceType)
        # print(distMat.shape)
        if params.use_feature == 1:
            res = distMat.argsort()[:params.k  + 1]
            tmpdist = distMat[0, res[0][1:params.k + 1]]
            # tmp1 = x_feature[node_idx, :].reshape(1, -1)
            # distMat1 = distance.cdist(tmp1, x_feature[res[0][1:params.k * 2 + 1], :], params.knn_distanceType)
            # print(distMat.shape)
            # distMat1=distMat1[0, res[0][1:params.k*2 + 1]]
            # res1 = distMat1.argsort()[:params.k + 1]
            # tmpdist1 = distMat1[0, res1[0][1:params.k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            # boundary1 = np.mean(tmpdist1) + np.std(tmpdist1)
            for j in np.arange(1, params.k + 1):
                if distMat[0, res[0][j]] <= boundary and init_label[node_idx]==init_label[res[0][j]]:
                    weight = distMat[0, res[0][j]]
                # elif distMat1[0, res1[0][j]] <= boundary1 or distMat[0, res[0][j]] <= boundary:
                #     weight = 0.5
                else:
                    weight = 0.0
                edgeList.append((node_idx, res[0][j], weight))
        else:
            res = distMat.argsort()[:params.k + 1]
            tmpdist = distMat[0, res[0][1:params.k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            for j in np.arange(1, params.k + 1):
                if distMat[0, res[0][j]] <= boundary:
                    weight = distMat[0, res[0][j]]
                else:
                    weight = 0.0
                edgeList.append((node_idx, res[0][j], weight))
            # tmp1 = x_feature[node_idx, :].reshape(1, -1)
    return edgeList
def graph_computing_new(adj_coo, cell_num, x_feature,params):
    edgeList = []
    # print(adj_coo.shape)
    # b=49
    # s=5
    # p=0.5
    # adj_coo=calculate_adj_matrix(x=adj_coo[:,0],y=adj_coo[:,1], x_pixel=adj_coo[:,0], y_pixel=adj_coo[:,1], image=img, beta=b, alpha=s, histology=True)
    # l = search_l(p, adj_coo, start=0.01, end=10000, tol=0.01, max_run=100)
    # adj_coo = np.exp(-1 * (adj_coo ** 2) / (2 * (l ** 2)))
    # print(adj_coo.shape)
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, params.knn_distanceType)
        # print(distMat.shape)
        res = distMat.argsort()[:params.k*2 + 1]
        tmpdist = distMat[0, res[0][1:params.k*2 + 1]]
        tmp1 = x_feature[node_idx, :].reshape(1, -1)
        distMat1 = distance.cdist(tmp1, x_feature[res[0][1:params.k*2 + 1],:], params.knn_distanceType)
        # print(distMat.shape)
        # distMat1=distMat1[0, res[0][1:params.k*2 + 1]]
        res1 = distMat1.argsort()[:params.k + 1]
        tmpdist1= distMat1[0, res1[0][1:params.k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        boundary1 = np.mean(tmpdist1) + np.std(tmpdist1)
        for j in np.arange(1, params.k + 1):
            if distMat1[0, res1[0][j]] <= boundary1:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((node_idx, res[0,res1[0][j]], weight))

    return edgeList
def graph_computing_new1(adj_coo, cell_num, x_feature,params):
    edgeList = []
    # print(adj_coo.shape)
    # b=49
    # s=5
    # p=0.5
    # adj_coo=calculate_adj_matrix(x=adj_coo[:,0],y=adj_coo[:,1], x_pixel=adj_coo[:,0], y_pixel=adj_coo[:,1], image=img, beta=b, alpha=s, histology=True)
    # l = search_l(p, adj_coo, start=0.01, end=10000, tol=0.01, max_run=100)
    # adj_coo = np.exp(-1 * (adj_coo ** 2) / (2 * (l ** 2)))
    # print(adj_coo.shape)
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, params.knn_distanceType)
        # print(distMat.shape)
        if params.use_feature==1:
            res = distMat.argsort()[:params.k*2 + 1]
            tmpdist = distMat[0, res[0][1:params.k*2 + 1]]
            tmp1 = x_feature[node_idx, :].reshape(1, -1)
            distMat1 = distance.cdist(tmp1, x_feature[res[0][1:params.k*2 + 1],:], params.knn_distanceType)
        # print(distMat.shape)
        # distMat1=distMat1[0, res[0][1:params.k*2 + 1]]
            res1 = distMat1.argsort()[:params.k + 1]
            tmpdist1= distMat1[0, res1[0][1:params.k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            boundary1 = np.mean(tmpdist1) + np.std(tmpdist1)
            for j in np.arange(1, params.k + 1):
                if distMat1[0, res1[0][j]] <= boundary1 and distMat[0,res[0][j]]<=boundary:
                    weight = 1.0
                elif distMat1[0, res1[0][j]] <= boundary1 or distMat[0,res[0][j]]<=boundary:
                    weight = 0.5
                else:
                    weight = 0.0
                edgeList.append((node_idx, res[0,res1[0][j]], weight))
        else:
            res = distMat.argsort()[:params.k + 1]
            tmpdist = distMat[0, res[0][1:params.k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            for j in np.arange(1, params.k + 1):
                if distMat[0,res[0][j]]<=boundary:
                    weight = 1.0
                else:
                    weight = 0.0
                edgeList.append((node_idx, res[0][j], weight))
            # tmp1 = x_feature[node_idx, :].reshape(1, -1)
    return edgeList
def graph_computing_super(x_feature,adj_coo, cell_num, init_label,params):
    edgeList = []
    # print(adj_coo.shape)
    # b=49
    # s=5
    # p=0.5
    # adj_coo=calculate_adj_matrix(x=adj_coo[:,0],y=adj_coo[:,1], x_pixel=adj_coo[:,0], y_pixel=adj_coo[:,1], image=img, beta=b, alpha=s, histology=True)
    # l = search_l(p, adj_coo, start=0.01, end=10000, tol=0.01, max_run=100)
    # adj_coo = np.exp(-1 * (adj_coo ** 2) / (2 * (l ** 2)))
    # print(adj_coo.shape)
    #     if params.use_feature == 1:
    #         res = distMat.argsort()[:params.k  + 1]
    #         tmpdist = distMat[0, res[0][1:params.k + 1]]
    #         # tmp1 = x_feature[node_idx, :].reshape(1, -1)
    #         # distMat1 = distance.cdist(tmp1, x_feature[res[0][1:params.k * 2 + 1], :], params.knn_distanceType)
    #         # print(distMat.shape)
    #         # distMat1=distMat1[0, res[0][1:params.k*2 + 1]]
    #         # res1 = distMat1.argsort()[:params.k + 1]
    #         # tmpdist1 = distMat1[0, res1[0][1:params.k + 1]]
    #         boundary = np.mean(tmpdist) + np.std(tmpdist)
    #         # boundary1 = np.mean(tmpdist1) + np.std(tmpdist1)
    #         for j in np.arange(1, params.k + 1):
    #             if distMat[0, res[0][j]] <= boundary and init_label[node_idx]==init_label[res[0][j]]:
    #                 weight = distMat[0, res[0][j]]
    #             # elif distMat1[0, res1[0][j]] <= boundary1 or distMat[0, res[0][j]] <= boundary:
    #             #     weight = 0.5
    #             else:
    #                 weight = 0.0
    #             edgeList.append((node_idx, res[0][j], weight))
    #     else:
    #         res = distMat.argsort()[:params.k + 1]
    #         tmpdist = distMat[0, res[0][1:params.k + 1]]
    #         boundary = np.mean(tmpdist) + np.std(tmpdist)
    #         for j in np.arange(1, params.k + 1):
    #             if distMat[0, res[0][j]] <= boundary:
    #                 weight = distMat[0, res[0][j]]
    #             else:
    #                 weight = 0.0
    #             edgeList.append((node_idx, res[0][j], weight))
    #         # tmp1 = x_feature[node_idx, :].reshape(1, -1)
    # return edgeList

    x_feature_all=np.zeros((cell_num,params.k,x_feature.shape[1]))
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, params.knn_distanceType)
        # tmp1 = x_feature[node_idx, :].reshape(1, -1)
        # print(distMat.shape)
        if params.use_feature == 1:
            res = distMat.argsort()[:params.k + 1]
            # print(res.shape,"wwwwwwwwwwww")
            tmpdist = distMat[0, res[0][1:params.k + 1]]
            x_feature_all[node_idx] = x_feature[res[0][1:params.k + 1]]
        # distMat1 = distance.cdist(tmp1, x_feature[res[0][1:params.k*2 + 1],:], params.knn_distanceType)
        # print(distMat.shape)
        # distMat1=distMat1[0, res[0][1:params.k*2 + 1]]
        # res1 = distMat1.argsort()[:params.k + 1]
        # tmpdist1= distMat1[0, res1[0][1:params.k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            for j in np.arange(1, params.k + 1):
                if distMat[0, res[0][j]] <= boundary and init_label[node_idx] == init_label[res[0][j]]:
                    weight = distMat[0, res[0][j]]
                # elif distMat1[0, res1[0][j]] <= boundary1 or distMat[0, res[0][j]] <= boundary:
                #     weight = 0.5
                else:
                    weight = 0.0
                edgeList.append((node_idx, res[0][j], weight))
        else:
            res = distMat.argsort()[:params.k + 1]
            tmpdist = distMat[0, res[0][1:params.k + 1]]
            boundary = np.mean(tmpdist) + np.std(tmpdist)
            x_feature_all[node_idx] = x_feature[res[0][1:params.k + 1]]
            for j in np.arange(1, params.k + 1):
                if distMat[0, res[0][j]] <= boundary:
                    weight = distMat[0, res[0][j]]
                else:
                    weight = 0.0
                edgeList.append((node_idx, res[0][j], weight))





        # boundary1 = np.mean(tmpdist1) + np.std(tmpdist1)
        #     for j in np.arange(1, params.k + 1):
        #         if distMat[0, res[0][j]] <= boundary:
        #             weight = 1.0
        #         else:
        #             weight = 0.0
        #         edgeList.append((node_idx, res[0][j], weight))
        # else:
        #     res = distMat.argsort()[:params.k + 1]
        #     tmpdist = distMat[0, res[0][1:params.k + 1]]
        #     boundary = np.mean(tmpdist) + np.std(tmpdist)
        #     for j in np.arange(1, params.k + 1):
        #         if distMat[0,res[0][j]]<=boundary:
        #             weight = 1.0
        #         else:
        #             weight = 0.0
        #         edgeList.append((node_idx, res[0][j], weight))
        #     # tmp1 = x_feature[node_idx, :].reshape(1, -1)
    return edgeList,x_feature_all
def graph_computing(adj_coo, cell_num, params):
    edgeList = []
    # b=49
    # s=1
    # adj_coo=calculate_adj_matrix(x=adj_coo[0],y=adj_coo[1], x_pixel=adj_coo[0], y_pixel=adj_coo[1], image=img, beta=b, alpha=s, histology=True)
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, params.knn_distanceType)
        res = distMat.argsort()[:params.k + 1]
        tmpdist = distMat[0, res[0][1:params.k + 1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, params.k + 1):
            if distMat[0, res[0][j]] <= boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((node_idx, res[0][j], weight))

    return edgeList
def TransformerST_graph_construction1(adj_coo, cell_N, img,params):
    adata_Adj = graph_computing_adaptive(adj_coo, cell_N, img,params)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    data1 = from_networkx(nx.from_dict_of_lists(graphdict))
    # print(data1.batch)
    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_org
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_label_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
    # print(adj_norm_m1.shape)
    graph_dict = {
        "adj_org": adj_org,
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1
    }

    # mask is binary matrix for semi-supervised/multi-dataset (1-valid edge, 0-unknown edge)
    if params.using_mask is True:
        graph_dict["adj_mask"] = torch.ones(cell_N, cell_N)

    return graph_dict,data1
def graph_construction_super(x,adj_coo, cell_N, img,params):
    adata_Adj,x_feature = graph_computing_super(x,adj_coo, cell_N, img,params)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    data1 = from_networkx(nx.from_dict_of_lists(graphdict))
    # print(data1.batch)
    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_org
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_label_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
    # print(adj_norm_m1.shape)
    graph_dict = {
        "adj_org": adj_org,
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1
    }

    # mask is binary matrix for semi-supervised/multi-dataset (1-valid edge, 0-unknown edge)
    if params.using_mask is True:
        graph_dict["adj_mask"] = torch.ones(cell_N, cell_N)

    return graph_dict,data1,x_feature
def graph_construction(adj_coo, cell_N, params):
    adata_Adj = graph_computing(adj_coo, cell_N, params)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))

    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_org
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_label_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)

    graph_dict = {
        "adj_org": adj_org,
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1
    }

    # mask is binary matrix for semi-supervised/multi-dataset (1-valid edge, 0-unknown edge)
    if params.using_mask is True:
        graph_dict["adj_mask"] = torch.ones(cell_N, cell_N)

    return graph_dict
def TransformerST_graph_construction_histology(adj_coo, cell_N, img,histology,params):
    adata_Adj = graph_computing_adaptive_histology(adj_coo, cell_N, img,histology,params)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    data1 = from_networkx(nx.from_dict_of_lists(graphdict))
    # print(data1.batch)
    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_org
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()

    # Some preprocessing
    adj_norm_m1 = preprocess_graph(adj_m1)
    adj_label_m1 = adj_m1 + sp.eye(adj_m1.shape[0])
    adj_label_m1 = torch.FloatTensor(adj_label_m1.toarray())
    norm_m1 = adj_m1.shape[0] * adj_m1.shape[0] / float((adj_m1.shape[0] * adj_m1.shape[0] - adj_m1.sum()) * 2)
    # print(adj_norm_m1.shape)
    graph_dict = {
        "adj_org": adj_org,
        "adj_norm": adj_norm_m1,
        "adj_label": adj_label_m1,
        "norm_value": norm_m1
    }

    # mask is binary matrix for semi-supervised/multi-dataset (1-valid edge, 0-unknown edge)
    if params.using_mask is True:
        graph_dict["adj_mask"] = torch.ones(cell_N, cell_N)

    return graph_dict,data1
def combine_graph_dict(dict_1, dict_2):
    # TODO add adj_org
    tmp_adj_norm = torch.block_diag(dict_1['adj_norm'].to_dense(), dict_2['adj_norm'].to_dense())
    graph_dict = {
        "adj_norm": tmp_adj_norm.to_sparse(),
        "adj_label": torch.block_diag(dict_1['adj_label'], dict_2['adj_label']),
        "norm_value": np.mean([dict_1['norm_value'], dict_2['norm_value']])
    }
    return graph_dict



