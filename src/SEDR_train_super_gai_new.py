#
import time
import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from progress.bar import Bar
from sklearn.cluster import KMeans
from src.SEDR_model import SEDR,SEDR_GAT,SEDR_GIN,SEDR_SAGE,SEDR1,SEDR_Transformer,SEDR_gcn_cluster,SEDR_GATv2,SEDR_GAT_topic,SEDR_GATv2_adaptive,SEDR_Transformer_adaptive,SEDR_GAT_adaptive,SEDR_Transformer_adaptive_super_gai,SEDR_Transformer_super_gai
import scanpy as sc
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from src.graph_func import graph_construction1 as graph_construction
import torch.optim as optim
def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn

def min_max_normalization(tensor,min_value,max_value):
    min_tensor=tensor.min()
    tensor=(tensor-min_tensor)
    max_tensor=tensor.max()
    tensor=tensor/max_tensor
    tensor=tensor*(max_value-min_value)+min_value
    return tensor
def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
    if mask is not None:
        preds = preds * mask
        labels = labels * mask

    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    # print(cost,KLD,"wwwww")
    return cost + KLD
def gcn_loss_attention(preds, labels, norm, mask=None):
    if mask is not None:
        preds = preds
        labels = labels

    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 / n_nodes * torch.mean(torch.sum(
    #     1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    # print(cost,KLD,"wwwww")
    return cost

class SEDR_Train_new:
    def __init__(self, node_X,neighbor_x, org_X,spatial,params):
        self.params = params
        self.device = params.device
        self.epochs = params.epochs
        self.node_X = torch.FloatTensor(node_X.copy()).to(self.device)
        self.org_X = torch.FloatTensor(org_X.copy()).to(self.device)
        self.neighbor_x = torch.FloatTensor(neighbor_x.copy()).to(self.device)
        # print(torch.min(self.node_X),torch.max(self.node_X))
        # self.node_X=min_max_normalization(self.node_X,0,1)
        # print(torch.min(self.node_X), torch.max(self.node_X))
        # self.adj_norm= data.edge_index.long().to(self.device)
        # self.adj_norm_prue = data_prue.edge_index.long().to(self.device)
        # # self.adj_norm =graph_dict["adj_norm"].to(self.device)
        # self.adj_label = graph_dict["adj_label"].to(self.device)
        # self.adj_label_prue = graph_dict_prue["adj_label"].to(self.device)
        # # print(self.adj_label.shape,"sssssssssssss")
        # self.adj_label=(self.adj_label+self.adj_label_prue)/2
        self.spatial=spatial
        # self.data=data
        # print(self.adj_label.shape,self.adj_norm.shape)
        # self.norm_value = graph_dict["norm_value"]
        # self.norm_value_prue = graph_dict_prue["norm_value"]
        # if params.using_mask is True:
        #     self.adj_mask = graph_dict["adj_mask"].to(self.device)
        # else:
        #     self.adj_mask = None
        # params.A_size=self.adj_norm.shape
        self.model = SEDR_Transformer_super_gai(self.params.cell_feat_dim, self.params).to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                          lr=self.params.gcn_lr, weight_decay=self.params.gcn_decay)
        # self.optimizer=torch.optim.SGD(params=list(self.model.parameters()),
        #                                   lr=self.params.gcn_lr, momentum=0.9)
    def train_without_dec(self):
        self.model.train()
        training=True
        bar = Bar('GNN model train without DEC: ', max=self.epochs)
        bar.check_tty = False

        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            _,de_feat,_= self.model(self.node_X, self.neighbor_x,self.spatial,training)
            # latent_z_prue, mu_prue, logvar_prue, de_feat_prue, _, feat_x_prue, _ = self.model(self.node_X, self.adj_norm_prue, training)
            # loss_gcn = gcn_loss_attention(preds=self.model.dc(latent_z), labels=self.adj_label, norm=self.norm_value, mask=self.adj_label)
            # loss_gcn_prue = gcn_loss(preds=self.model.dc(latent_z_prue), labels=self.adj_label_prue, mu=mu_prue,
            #                    logvar=logvar_prue, n_nodes=self.params.cell_num, norm=self.norm_value_prue, mask=self.adj_label_prue)
            loss_rec = reconstruction_loss(de_feat, self.org_X)
            # print(loss_rec,"2222222222",loss_gcn,torch.isnan(de_feat))
            loss = loss_rec
            loss.backward()
            self.optimizer.step()

            end_time = time.time()
            batch_time = end_time - start_time
            bar_str = '{} / {} | Left time: {batch_time:.2f} mins| Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch + 1, self.epochs,
                                        batch_time=batch_time * (self.epochs - epoch) / 60, loss=loss.item())
            bar.next()
        bar.finish()

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self):
        self.model.eval()
        training=False
        latent_z,x_rec,x_hr = self.model(self.node_X, self.neighbor_x,self.spatial,training)
        latent_z = latent_z.data.cpu().numpy()
        # q = q.data.cpu().numpy()
        # feat_x = feat_x.data.cpu().numpy()
        x_rec=x_rec.data.cpu().numpy()
        x_hr=x_hr.data.cpu().numpy()
        # hr_z=hr_z.data.cpu().numpy()
        # atten = atten.data.cpu().numpy()
        # gnn_z = gnn_z.data.cpu().numpy()
        # print(x_hr.shape,"eeeeeeeeeee")
        return latent_z,x_rec,x_hr

    def train_with_dec(self):
        n_neighbors = 10
        res = 0.4
        training=True
        # initialize cluster parameter
        self.train_without_dec()
        # scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.1)
        kmeans = KMeans(n_clusters=self.params.dec_cluster_n, n_init=self.params.dec_cluster_n * 2, random_state=0)
        test_z,x_rec,x_hr= self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))
        # bgm = GaussianMixture(n_components=10, random_state=0).fit(test_z)
        # y_pred_last = np.copy(bgm.fit_predict(test_z))
        # adata = sc.AnnData(test_z)
        # sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        # sc.tl.leiden(adata, resolution=res)
        # y_pred = adata.obs['leiden'].astype(int).to_numpy()
        # self.n_clusters = len(np.unique(y_pred))
        # # # ----------------------------------------------------------------
        # # y_pred_last = y_pred
        # # # self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
        # # # X = torch.FloatTensor(X)
        # # # adj = torch.FloatTensor(adj)
        # # # self.trajectory.append(y_pred)
        # features = pd.DataFrame(test_z, index=np.arange(0, test_z.shape[0]))
        # Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        # Mergefeature = pd.concat([features, Group], axis=1)
        # cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        # y_pred_last = y_pred
        # self.model.cluster_layer.data = torch.tensor(cluster_centers).float().to(self.device)
        # self.model.cluster_layer.data = torch.tensor(bgm.means_).float().to(self.device)
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        # print(self.model.cluster_layer.shape,'fffff')
        self.model.train()
        training=True
        bar = Bar('Training Graph Net with DEC loss: ', max=self.epochs)
        bar.check_tty = False
        for epoch_id in range(self.epochs*2):
            # DEC clustering update
            if epoch_id % self.params.dec_interval == 0:
                ######################
                # self.params.use_feature=0
                # graph_dict, data1 = graph_construction(self.spatial, self.params.cell_num, test_z, self.params)
                # self.node_X = torch.FloatTensor(node_X.copy()).to(self.device)
                # print(torch.min(self.node_X),torch.max(self.node_X))
                # self.node_X=min_max_normalization(self.node_X,0,1)
                # print(torch.min(self.node_X), torch.max(self.node_X))
                # self.adj_norm = data1.edge_index.long().to(self.device)
                # self.adj_norm =graph_dict["adj_norm"].to(self.device)
                # self.adj_label = graph_dict["adj_label"].to(self.device)
                # # self.data=data
                # # print(self.adj_label.shape,self.adj_norm.shape)
                # self.norm_value = graph_dict["norm_value"]
                # if self.params.using_mask is True:
                #     self.adj_mask = graph_dict["adj_mask"].to(self.device)
                # else:
                #     self.adj_mask = None
                # self.params.use_feature = 0
                ######################
                _, tmp_q, _ ,_,_,_= self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                # print(delta_label)
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < self.params.dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', self.params.dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # training model
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            latent_z, de_feat, out_q, _,_,_,_= self.model(self.node_X, self.neighbor_x,self.adj_norm,self.adj_norm_prue,self.spatial,training)
            loss_gcn = gcn_loss_attention(preds=self.model.dc(latent_z), labels=self.adj_label, norm=self.norm_value, mask=self.adj_label)
            loss_rec = reconstruction_loss(de_feat, self.node_X)
            # clustering KL loss
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            loss = self.params.gcn_w * loss_gcn + self.params.dec_kl_w * loss_kl + self.params.feat_w * loss_rec
            loss.backward()
            self.optimizer.step()
            # scheduler.step()
            bar_str = '{} / {} | Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch_id + 1, self.epochs, loss=loss.item())
            bar.next()
        bar.finish()


