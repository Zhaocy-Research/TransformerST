#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GraphConv, graclus, max_pool, global_mean_pool, JumpingKnowledge
import torch_geometric as tg
from src.gat_v2conv import GATv2Conv
from src.model_utils import InputPartlyTrainableLinear, PartlyTrainableParameter2D, get_fully_connected_layers, get_kl
import anndata
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union
import numpy as np
import logging
from scipy.sparse import spmatrix
import networkx as nx
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal, Independent
from src.decoder import BasisODEDecoder,BasisDecoder
from torch.nn.functional import softplus
from src.submodules import *
from src.quantize_1d import VectorQuantizer
import torch_geometric as pyg
# from scETM.logging_utils import log_arguments
# from src.BaseCellModel import BaseCellModel
def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.1, eps=0.0001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


# GCN Layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj 


class SEDR(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2

        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))

        # GCN layers
        self.gc1 = GraphConvolution(params.feat_hidden2, params.gcn_hidden1, params.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.edge_importance = Parameter(torch.Tensor(params.A_size))
    def encode(self, x, adj):
        feat_x = self.encoder(x)
        # print(adj.shape,self.edge_importance.shape)
        # adj=adj* (self.edge_importance)
        hidden1 = self.gc1(feat_x, adj)
        x = F.relu(hidden1)
        hidden1 = F.dropout(x, training=True)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,training):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, mu, logvar, de_feat, q, feat_x, gnn_z
class SEDR1(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR1, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2

        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))

        # GCN layers
        self.gc1 = GraphConv(params.feat_hidden2, params.gcn_hidden1, aggr='add')
        self.gc2 = GraphConv(params.gcn_hidden1, params.gcn_hidden2,  aggr='add')
        self.gc3 = GraphConv(params.gcn_hidden1, params.gcn_hidden2,  aggr='add')
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        x = F.relu(hidden1)
        hidden1 = F.dropout(x, training=True)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    # def reset_parameters(self):
    #     self.conv1.reset_parameters()
    #     for conv in self.convs:
    #         conv.reset_parameters()
    #     self.jump.reset_parameters()
    #     self.lin1.reset_parameters()
    #     self.lin2.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,training):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, mu, logvar, de_feat, q, feat_x, gnn_z
class SEDR_GATv2_super(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR_GATv2_super, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        self.layer_num=3
        scale=params.k
        self.params=params
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))

        # GCN layers
        self.gc1 = GATv2Conv(params.feat_hidden2, params.gcn_hidden1)
        self.conv_hidden = nn.ModuleList([GATv2Conv(params.gcn_hidden1, params.gcn_hidden1) for i in range(self.layer_num - 2)])
        self.gc2 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2)
        self.gc3 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2)
        self.graph = Graph(scale, k=params.k, patchsize=1, stride=1,
                          window_size=1, in_channels=256, embedcnn=None)
        self.gcn = GCNBlock(scale, k=params.k, patchsize=1, stride=1)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(adj.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        x = F.elu(hidden1)
        x = F.dropout(x, training=True)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, adj)
            x = F.elu(x)
            x = F.dropout(x, training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,x_hr,training):
        mu, logvar, feat_x = self.encode(x, adj,training)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        # score_k, idx_k, diff_patch = self.graph(z, z)
        # idx_k = idx_k.detach()
        # diff_patch = diff_patch.detach()
        # x_h=np.zeros((z.shape[0],self.params.k,z.shape[1]))
        # x1=z.detach().cpu().numpy()
        # x_hr=x_hr.numpy().astype(int)
        # for i in range(x.shape[0]):
        #     x_h[i]=x1[x_hr[i]]
        # x_h=torch.from_numpy(x_h).float().cuda()
        x1_lr=z
        # print(z.shape,"www")
        # x1_lr, _ = self.gcn(z,x_h,idx_k, diff_patch)
        # x1_lr=torch.swapaxes(x1_lr, 1, 2).squeeze(0)
        # x1_lr=torch.cat((z, x1_lr),1)
        # print(x1_lr.shape)
        de_feat = self.decoder(x1_lr)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return x1_lr, mu, logvar, de_feat, q, feat_x, gnn_z
class SEDR_SAGE(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR_SAGE, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        self.layer_num=3
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))

        # GCN layers
        self.gc1 = tg.nn.SAGEConv(params.feat_hidden2, params.gcn_hidden1)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(params.gcn_hidden1, params.gcn_hidden1) for i in range(self.layer_num - 2)])
        self.gc2 = tg.nn.SAGEConv(params.gcn_hidden1, params.gcn_hidden2)
        self.gc3 = tg.nn.SAGEConv(params.gcn_hidden1, params.gcn_hidden2)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        x = F.relu(hidden1)
        x = F.dropout(x, training=True)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, adj)
            x = F.relu(x)
            x = F.dropout(x, training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,training):
        mu, logvar, feat_x = self.encode(x, adj,training)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, mu, logvar, de_feat, q, feat_x, gnn_z
class SEDR_GAT_topic(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR_GAT_topic, self).__init__()
        self.alpha = 1.0

        self.layer_num=3
        self.n_topics: int = 50
        self.trainable_gene_emb_dim: int = input_dim
        self.hidden_sizes: Sequence[int] = (128,)
        self.bn: bool = True
        self.dropout_prob: float = 0.2
        self.normalize_beta: False
        self.normed_loss: bool = True
        self.norm_cells: bool = True
        self.input_batch_id: bool = False
        self.enable_batch_bias: bool = True
        self.enable_global_bias: bool = False
        self.n_batches=1
        self.n_fixed_genes=0
        rho_fixed_emb= None
        rho_fixed_gene= None
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_trainable_genes=input_dim
        self.max_logsigma = 10
        self.min_logsigma = -10
        if self.n_batches <= 1:
            # _logger.warning(f'n_batches == {self.n_batches}, disabling batch bias')
            self.enable_batch_bias = False
            self.input_batch_id = False
        # feature autoencoder
        # self.encoder = nn.Sequential()
        # self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        # self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))
        self.latent_dim = params.gcn_hidden2 + self.n_topics
        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        hidden_dim = self.hidden_sizes[-1]
        # GCN layers
        self.gc1 = GATv2Conv(hidden_dim, params.gcn_hidden1)
        self.conv_hidden = nn.ModuleList([GATv2Conv(params.gcn_hidden1, params.gcn_hidden1) for i in range(self.layer_num - 2)])
        self.gc2 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2)
        self.gc3 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+self.n_topics))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.q_delta: nn.Sequential = get_fully_connected_layers(
            n_trainable_input=self.n_trainable_genes + ((self.n_batches - 1) if self.input_batch_id else 0),
            hidden_sizes=self.hidden_sizes,
            bn=self.bn,
            dropout_prob=self.dropout_prob,
            n_fixed_input=self.n_fixed_genes
        )

        self.mu_q_delta: nn.Linear = nn.Linear(hidden_dim, self.n_topics, bias=True)
        self.logsigma_q_delta: nn.Linear = nn.Linear(hidden_dim, self.n_topics, bias=True)

        self.rho_fixed_emb: Union[None, torch.Tensor] = None
        self.rho_trainable_emb: Union[None, PartlyTrainableParameter2D] = None
        self._init_rho_trainable_emb()
        if self.trainable_gene_emb_dim > 0 and self.n_fixed_genes > 0 and rho_fixed_gene is not None:
            assert rho_fixed_gene.shape == (self.trainable_gene_emb_dim, self.n_fixed_genes)
            self.rho_trainable_emb.fixed = torch.FloatTensor(rho_fixed_gene)
        if rho_fixed_emb is not None:
            assert rho_fixed_emb.shape[1] == self.n_fixed_genes + self.n_trainable_genes
            self.rho_fixed_emb = torch.FloatTensor(rho_fixed_emb).to(device)

        self.alpha1: nn.Parameter = nn.Parameter(torch.randn(self.n_topics, self.trainable_gene_emb_dim + (
            self.rho_fixed_emb.shape[0] if self.rho_fixed_emb is not None else 0)))
        self._init_batch_and_global_biases()
    @property
    def rho(self) -> torch.Tensor:
        """The fixed and trainable combined gene embedding rho.

        This is a read-only property. To modify the gene embeddings, please
        change self.rho_fixed_emb and self.rho_trainable_emb.
        """

        rho = [param for param in (self.rho_fixed_emb, self.rho_trainable_emb.get_param()) if param is not None]
        rho = torch.cat(rho, dim=0) if len(rho) > 1 else rho[0]
        return rho

    def _init_encoder_first_layer(self) -> None:
        """Initializes the first layer of the encoder given the constant
        attributes.
        """

        trainable_dim = self.n_trainable_genes + ((self.n_batches - 1) if self.input_batch_id else 0)
        if self.n_fixed_genes > 0:
            self.q_delta[0] = InputPartlyTrainableLinear(self.n_fixed_genes, self.hidden_sizes[0], trainable_dim)
        else:
            self.q_delta[0] = nn.Linear(trainable_dim, self.hidden_sizes[0])

    def _init_rho_trainable_emb(self) -> None:
        """Initializes self.rho_trainable_emb given the constant attributes."""

        if self.trainable_gene_emb_dim > 0:
            self.rho_trainable_emb = PartlyTrainableParameter2D(self.trainable_gene_emb_dim, self.n_fixed_genes,
                                                                self.n_trainable_genes)

    def _init_batch_and_global_biases(self) -> None:
        """Initializes batch and global biases given the constant attributes."""

        if self.enable_batch_bias:
            self.batch_bias: nn.Parameter = nn.Parameter(
                torch.randn(self.n_batches, self.n_fixed_genes + self.n_trainable_genes))

        self.global_bias: nn.Parameter = nn.Parameter(
            torch.randn(1, self.n_fixed_genes + self.n_trainable_genes)) if self.enable_global_bias else None

    def decode(self,
               theta: torch.Tensor,
               batch_indices: Union[None, torch.Tensor]
               ) -> torch.Tensor:
        """Decodes the topic proportions (theta) to gene expression profiles.

        Args:
            theta: the topic proportions for cells in the current batch.
            batch_indices: the batch indices of cells in the current batch.

        Returns:
            Log of decoded gene expression profile reconstructions.
        """

        beta = self.alpha1 @ self.rho

        if self.normalize_beta:
            recon = torch.mm(theta, F.softmax(beta, dim=-1))
            recon_log = (recon + 1e-30).log()
        else:
            recon_logit = torch.mm(theta, beta)  # [batch_size, n_genes]
            if self.enable_global_bias:
                recon_logit += self.global_bias
            if self.enable_batch_bias:
                recon_logit += self.batch_bias[batch_indices]
            recon_log = F.log_softmax(recon_logit, dim=-1)
        return recon_log
    def get_all_embeddings_and_nll(self,
        adata: anndata.AnnData,
        batch_size: int = 2000,
        emb_names: Union[str, Iterable[str], None] = None,
        batch_col: str = 'batch_indices',
        inplace: bool = True,
        writer= None
    ) -> Union[Union[None, float], Tuple[Mapping[str, np.ndarray], Union[None, float]]]:
        """Calculates cell, gene, topic embeddings and nll for the dataset.

        If inplace, cell embeddings will be stored to adata.obsm. You can
        reference them by the keys in self.emb_names. Gene embeddings will be
        stored to adata.varm with the key "rho". Topic embeddings will be
        stored to adata.uns with the key "alpha".

        Args:
            adata: the test dataset. adata.n_vars must equal to #genes of this
                model.
            batch_size: batch size for test data input.
            emb_names: names of the embeddings to be returned or stored to
                adata.obsm. Must be a subset of self.emb_names. If None,
                default to self.emb_names.
            batch_col: a key in adata.obs to the batch column. Only used when
                self.need_batch is True.
            inplace: whether embeddings will be stored to adata or returned.
            writer: an initialized SummaryWriter.

        Returns:
            If inplace, only the test nll. Otherwise, return the cell, gene and
            topic embeddings as a dict and also the test nll.
        """

        result = super().get_cell_embeddings_and_nll(adata, batch_size=batch_size, emb_names=emb_names, batch_col=batch_col, inplace=inplace)
        if writer is not None:
            self.write_topic_gene_embeddings_to_tensorboard(writer, adata.var_names)
        if inplace:
            adata.varm['rho'] = self.rho.T.detach().cpu().numpy()
            adata.uns['alpha'] = self.alpha1.detach().cpu().numpy()
            return result
        else:
            result_dict, nll = result
            result_dict['rho'] = self.rho.T.detach().cpu().numpy()
            result_dict['alpha'] = self.alpha1.detach().cpu().numpy()
            return result_dict, nll
    def encode(self, x, adj,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # feat_x = self.encoder(x)
        # print(x.shape)
        q_delta = self.q_delta(x)
        # print(q_delta.shape,"wwwwww")
        mu_q_delta = self.mu_q_delta(q_delta)
        logsigma_q_delta = self.logsigma_q_delta(q_delta).clamp(self.min_logsigma, self.max_logsigma)
        q_delta1 = Independent(Normal(
            loc=mu_q_delta,
            scale=logsigma_q_delta.exp()
        ), 1)

        delta = q_delta1.rsample()
        # print(delta.shape)
        # theta = F.softmax(delta, dim=-1)  # [batch_size, n_topics]
        hidden1 = self.gc1(q_delta, adj)
        x = F.relu(hidden1)
        x = F.dropout(x, training=True)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, adj)
            x = F.relu(x)
            x = F.dropout(x, training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), delta

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,training):
        # print(x.shape)
        mu, logvar, feat_x = self.encode(x, adj,training)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        # print(z.shape)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, mu, logvar, de_feat, q, feat_x, gnn_z
class SEDR_GATv2_basis(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR_GATv2_basis, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.feat_hidden2
        self.layer_num=3
        self.z_dim=params.feat_hidden2
        # feature autoencoder
        self.encoder = nn.Sequential(torch.nn.Linear(input_dim, params.feat_hidden1),
            torch.nn.ReLU(),
            torch.nn.Linear(params.feat_hidden1, 2*params.feat_hidden2))
        # self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        # self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))
        hidden_dim=64
        # self.decoder = nn.Sequential()
        # self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        self.decoder=BasisODEDecoder(input_dim, hidden_dim, self.latent_dim, 3,
                       likelihood="Gaussian",
                       nonlinearity = torch.nn.ReLU,
                       alpha=0.1,
                       x0init=params.x0init,
                       device="gpu")
        # self.decoder=BasisDecoder(input_dim, hidden_dim, self.latent_dim, 5,
        #                           scale_invariance=True,
        #                           translation_invariance=True,
        #                           likelihood="ZINB",
        #                           nonlinearity=torch.nn.ReLU,
        #                           # max_delta=2.0,
        #                           alpha=0.1,
        #                           inference="collapsed")
        # GCN layers
        self.gc1 = GATv2Conv(params.feat_hidden2, params.gcn_hidden1)
        self.conv_hidden = nn.ModuleList([GATv2Conv(params.gcn_hidden1, params.gcn_hidden1) for i in range(self.layer_num - 2)])
        self.gc2 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2)
        self.gc3 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # out = self.mapping(Y)
        #
        # mu = out[:, 0:self.z_dim]
        # sigma = 1e-6 + softplus(out[:, self.z_dim:(2 * self.z_dim)])
        # print(x.shape)
        feat_x = self.encoder(x)
        mu = feat_x[:, 0:self.z_dim]
        sigma = 1e-6 + softplus(feat_x[:, self.z_dim:(2 * self.z_dim)])
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        hidden1 = self.gc1(z, adj)
        x = F.relu(hidden1)
        x = F.dropout(x, training=True)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, adj)
            x = F.relu(x)
            x = F.dropout(x, training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), z,mu,sigma

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,training):
        mu, logvar, feat_x,mu_z,sigma_z = self.encode(x, adj,training)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        # print(z.shape,"wwwwww")
        de_feat, dropout_prob_logit, theta = self.decoder.forward(feat_x)

        # de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, mu, logvar, de_feat, q, feat_x, gnn_z,dropout_prob_logit, theta,mu_z,sigma_z
class SEDR_GAT(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR_GAT, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        self.layer_num=3
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))

        # GCN layers
        self.gc1 = tg.nn.GATConv(params.feat_hidden2, params.gcn_hidden1)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(params.gcn_hidden1, params.gcn_hidden1) for i in range(self.layer_num - 2)])
        self.gc2 = tg.nn.GATConv(params.gcn_hidden1, params.gcn_hidden2)
        self.gc3 = tg.nn.GATConv(params.gcn_hidden1, params.gcn_hidden2)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        x = F.relu(hidden1)
        x = F.dropout(x, training=True)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, adj)
            x = F.relu(x)
            x = F.dropout(x, training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,training):
        mu, logvar, feat_x = self.encode(x, adj,training)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, mu, logvar, de_feat, q, feat_x, gnn_z
class SEDR_GATv2_adaptive(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR_GATv2_adaptive, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        self.layer_num=3
        self.at=0.5
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))

        # GCN layers
        self.gc1 = GATv2Conv(params.feat_hidden2, params.gcn_hidden1)
        self.conv_hidden = nn.ModuleList([GATv2Conv(params.gcn_hidden1, params.gcn_hidden1) for i in range(self.layer_num - 2)])
        self.gc2 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2)
        # self.gc3 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2,dropout = params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,adj_prue,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(x.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        # print(torch.isinf(feat_x).any(), "1111")
        hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
        # atten1=atten
        # g=nx.Graph(atten[0],atten[1])
        atten=pyg.utils.remove_self_loops(atten[0],atten[1])
        atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
        atten=atten.squeeze(0).squeeze(-1)
        # if torch.isnan(atten).any():
        #     print(atten1)
        # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
        # adj_org = nx.adjacency_matrix(atten)
        # print(adj_org.shape)
        hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
        # print(torch.isnan(hidden1).any(),"2")
        x = F.relu(hidden1)
        # x = F.relu(hidden1_prue)
        x = F.dropout(x, p = 0.2,training=True)
        for i in range(self.layer_num - 2):
            x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
            x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
            atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
            atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
            atten_prue = atten_prue.squeeze(0).squeeze(-1)
            atten = pyg.utils.remove_self_loops(atten[0], atten[1])
            atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
            atten = atten.squeeze(0).squeeze(-1)
            x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
            x = F.relu(x)
            x = F.dropout(x, p = 0.2,training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        # print(torch.isnan(hidden1).any(),"3")
        mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
        # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
        mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
        # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
        atten = pyg.utils.remove_self_loops(atten[0], atten[1])
        atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
        atten = atten.squeeze(0).squeeze(-1)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
        # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
        # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
        # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
        # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
        # atten_var = atten_var.squeeze(0).squeeze(-1)
        mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
        # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
        return mu,feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,adj_prue,training):
        mu,feat_x = self.encode(x, adj,adj_prue,training)
        # gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, mu), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, de_feat, q, feat_x
class SEDR_GAT_adaptive(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR_GAT_adaptive, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        self.layer_num=3
        self.at=0.3
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L3', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, params.feat_hidden1, params.p_drop))
        self.decoder.add_module('decoder_L1', full_block(params.feat_hidden1, input_dim, params.p_drop))

        # GCN layers
        self.gc1 = tg.nn.GATConv(params.feat_hidden2, params.gcn_hidden1)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(params.gcn_hidden1, params.gcn_hidden1) for i in range(self.layer_num - 2)])
        self.gc2 = tg.nn.GATConv(params.gcn_hidden1, params.gcn_hidden2)
        # self.gc3 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2,dropout = params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,adj_prue,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(x.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        # print(torch.isinf(feat_x).any(), "1111")
        hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
        # atten1=atten
        # g=nx.Graph(atten[0],atten[1])
        atten=pyg.utils.remove_self_loops(atten[0],atten[1])
        atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
        atten=atten.squeeze(0).squeeze(-1)
        # if torch.isnan(atten).any():
        #     print(atten1)
        # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
        # adj_org = nx.adjacency_matrix(atten)
        # print(adj_org.shape)
        hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
        # print(torch.isnan(hidden1).any(),"2")
        x = F.relu(hidden1)
        # x = F.relu(hidden1_prue)
        x = F.dropout(x, p = 0.2,training=True)
        for i in range(self.layer_num - 2):
            x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
            x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
            atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
            atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
            atten_prue = atten_prue.squeeze(0).squeeze(-1)
            atten = pyg.utils.remove_self_loops(atten[0], atten[1])
            atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
            atten = atten.squeeze(0).squeeze(-1)
            x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
            x = F.relu(x)
            x = F.dropout(x, p = 0.2,training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        # print(torch.isnan(hidden1).any(),"3")
        mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
        # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
        mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
        # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
        atten = pyg.utils.remove_self_loops(atten[0], atten[1])
        atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
        atten = atten.squeeze(0).squeeze(-1)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
        # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
        # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
        # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
        # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
        # atten_var = atten_var.squeeze(0).squeeze(-1)
        mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
        # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
        return mu,feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,adj_prue,training):
        mu,feat_x = self.encode(x, adj,adj_prue,training)
        # gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, mu), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, de_feat, q, feat_x
class SEDR_GATv2(nn.Module):
    def __init__(self, input_dim, params):
        super(SEDR_GATv2, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        self.layer_num=3
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))

        # GCN layers
        self.gc1 = GATv2Conv(params.feat_hidden2, params.gcn_hidden1)
        self.conv_hidden = nn.ModuleList([GATv2Conv(params.gcn_hidden1, params.gcn_hidden1) for i in range(self.layer_num - 2)])
        self.gc2 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2)
        self.gc3 = GATv2Conv(params.gcn_hidden1, params.gcn_hidden2)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(adj.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        x = F.relu(hidden1)
        x = F.dropout(x, training=True)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, adj)
            x = F.relu(x)
            x = F.dropout(x, training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,training):
        mu, logvar, feat_x = self.encode(x, adj,training)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, mu, logvar, de_feat, q, feat_x, gnn_z
class ST_Transformer(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        self.layer_num=3
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1,heads=1,dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList([tg.nn.TransformerConv(params.gcn_hidden1*1, params.gcn_hidden1,heads=1,dropout=params.p_drop) for i in range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1*1, params.gcn_hidden2,dropout=params.p_drop)
        self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1*1, params.gcn_hidden2,dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        x = F.relu(hidden1)
        # print(x.shape)
        # x = F.dropout(x, training=True)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, adj)
            x = F.relu(x)
            # x = F.dropout(x, training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,training):
        mu, logvar, feat_x = self.encode(x, adj,training)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, mu, logvar, de_feat, q, feat_x, gnn_z
class ST_Transformer_adaptive(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer_adaptive, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+64
        self.layer_num=3
        self.at=0.5
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))
        self.quantize = VectorQuantizer(self.latent_dim, 64, beta=0.25)
        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L1', full_block(self.latent_dim, input_dim, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1, heads=1, dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden1, heads=1, dropout=params.p_drop) for i in
             range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,adj_prue,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(x.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        # print(feat_x.shape,"wwwwwwwwwwww")
        # print(torch.isinf(feat_x).any(), "1111")
        hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
        # print(atten.shape,"wwwwwwwww")
        # atten1=atten
        # g=nx.Graph(atten[0],atten[1])
        atten=pyg.utils.remove_self_loops(atten[0],atten[1])
        atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
        atten=atten.squeeze(0).squeeze(-1)
        # print(torch.sum(atten,1),"wwwwwwww")
        # if torch.isnan(atten).any():
        #     print(atten1)
        # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
        # adj_org = nx.adjacency_matrix(atten)
        # print(adj_org.shape)
        hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # print(atten.shape,hidden1.shape,"fdsfsdfsadf")
        hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
        # print(torch.isnan(hidden1).any(),"2")
        x = F.relu(hidden1)
        # x = F.relu(hidden1_prue)
        # x = F.dropout(x, p = 0.2,training=True)
        for i in range(self.layer_num - 2):
            x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
            x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
            atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
            atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
            atten_prue = atten_prue.squeeze(0).squeeze(-1)
            atten = pyg.utils.remove_self_loops(atten[0], atten[1])
            atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
            atten = atten.squeeze(0).squeeze(-1)
            x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
            x = F.relu(x)
            # x = F.dropout(x, p = 0.2,training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        # print(torch.isnan(hidden1).any(),"3")
        mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
        # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
        mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
        # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
        atten = pyg.utils.remove_self_loops(atten[0], atten[1])
        atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
        atten = atten.squeeze(0).squeeze(-1)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
        # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
        # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
        # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
        # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
        # atten_var = atten_var.squeeze(0).squeeze(-1)
        mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
        # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
        return mu,feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,adj_prue,training):
        mu,feat_x = self.encode(x, adj,adj_prue,training)
        # gnn_z = self.reparameterize(mu, logvar)
        quant, _, info = self.quantize(feat_x)
        z = torch.cat((quant, mu), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, de_feat, q, feat_x
class ST_Transformer_adaptive_super_gai_new(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer_adaptive_super_gai_new, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        self.layer_num=3
        self.at=0.5
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1, heads=1, dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden1, heads=1, dropout=params.p_drop) for i in
             range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,adj_prue,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(x.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        # print(feat_x.shape,"wwwwwwwwwwww")
        # print(torch.isinf(feat_x).any(), "1111")
        hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
        # print(atten.shape,"wwwwwwwww")
        # atten1=atten
        # g=nx.Graph(atten[0],atten[1])
        atten=pyg.utils.remove_self_loops(atten[0],atten[1])
        atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
        atten=atten.squeeze(0).squeeze(-1)
        # print(torch.sum(atten,1),"wwwwwwww")
        # if torch.isnan(atten).any():
        #     print(atten1)
        # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
        # adj_org = nx.adjacency_matrix(atten)
        # print(adj_org.shape)
        hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
        # print(torch.isnan(hidden1).any(),"2")
        x = F.relu(hidden1)
        # x = F.relu(hidden1_prue)
        # x = F.dropout(x, p = 0.2,training=True)
        for i in range(self.layer_num - 2):
            x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
            x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
            atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
            atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
            atten_prue = atten_prue.squeeze(0).squeeze(-1)
            atten = pyg.utils.remove_self_loops(atten[0], atten[1])
            atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
            atten = atten.squeeze(0).squeeze(-1)
            x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
            x = F.relu(x)
            # x = F.dropout(x, p = 0.2,training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        # print(torch.isnan(hidden1).any(),"3")
        mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
        # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
        mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
        # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
        atten = pyg.utils.remove_self_loops(atten[0], atten[1])
        atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
        atten = atten.squeeze(0).squeeze(-1)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
        # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
        # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
        # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
        # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
        # atten_var = atten_var.squeeze(0).squeeze(-1)
        mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
        # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
        return mu,feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, enhanced_weights,enhanced_index,adj,adj_prue,training):
        mu,feat_x = self.encode(x, adj,adj_prue,training)
        # gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, mu), 1)
        
        if training:
            z_super = torch.zeros(6 * z.shape[0], z.shape[1]).cuda()
        # print(z.shape,enhanced_weights.shape,enhanced_index.shape,'werewr')
            for i in range(enhanced_weights.shape[0]):
                index=enhanced_index[i].squeeze(0)
            # print(z[index,:].shape)
                z_super[i]=torch.matmul(enhanced_weights[i],z[index,:])
            z_super=z_super.view(6,z.shape[0],z.shape[1])
            z=torch.mean(z_super,0)
            z_super = z_super.view(6*z.shape[0], z.shape[1])
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        if training:
            return z, de_feat, q, feat_x,z_super
        else:
            return z, de_feat, q, feat_x,z
class ST_Transformer_adaptive_super1(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer_adaptive_super1, self).__init__()
        self.alpha = 1.0
        self.latent_dim = int(params.gcn_hidden2+params.feat_hidden2)
        # self.latent_dim = int(params.gcn_hidden2/6)
        self.layer_num=3
        self.at=0.5
        self.k=10
        self.scale = 6
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim*6, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1, heads=1, dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden1, heads=1, dropout=params.p_drop) for i in
             range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.graph = Graph_spatial(self.scale, k=self.k, patchsize=1, stride=1,
        #                    window_size=1, in_channels=256, embedcnn=None)
        # self.gcn = GCNBlock_spatial(self.scale, k=self.k, patchsize=1, stride=1)
        # self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,adj_prue,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(x.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        # print(feat_x.shape,"wwwwwwwwwwww")
        # print(torch.isinf(feat_x).any(), "1111")
        hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
        # print(atten.shape,"wwwwwwwww")
        # atten1=atten
        # g=nx.Graph(atten[0],atten[1])
        atten=pyg.utils.remove_self_loops(atten[0],atten[1])
        atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
        atten=atten.squeeze(0).squeeze(-1)
        # print(torch.sum(atten,1),"wwwwwwww")
        # if torch.isnan(atten).any():
        #     print(atten1)
        # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
        # adj_org = nx.adjacency_matrix(atten)
        # print(adj_org.shape)
        hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
        # print(torch.isnan(hidden1).any(),"2")
        x = F.relu(hidden1)
        # x = F.relu(hidden1_prue)
        # x = F.dropout(x, p = 0.2,training=True)
        for i in range(self.layer_num - 2):
            x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
            x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
            atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
            atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
            atten_prue = atten_prue.squeeze(0).squeeze(-1)
            atten = pyg.utils.remove_self_loops(atten[0], atten[1])
            atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
            atten = atten.squeeze(0).squeeze(-1)
            x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
            x = F.relu(x)
            # x = F.dropout(x, p = 0.2,training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        # print(torch.isnan(hidden1).any(),"3")
        mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
        # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
        mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
        # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
        atten = pyg.utils.remove_self_loops(atten[0], atten[1])
        atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
        atten = atten.squeeze(0).squeeze(-1)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
        # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
        # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
        # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
        # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
        # atten_var = atten_var.squeeze(0).squeeze(-1)
        mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
        # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
        return mu,feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, x_neighbor,adj,adj_prue,spatial,training):
        mu,feat_x = self.encode(x, adj,adj_prue,training)
        mu=mu.view(mu.shape[0],6,-1)
        mu1= torch.mean(mu, 1)
        # mu_z =mu.view(mu.shape[0]*6, -1)
        mu = mu.view(mu.shape[0], -1)

        # mu_neighbor, feat_x_neighbor = self.encode(x_neighbor, adj, adj_prue, training)
        # gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, mu), 1)
        z1= torch.cat((feat_x, mu1), 1)
        # z_neighbor = torch.cat((feat_x_neighbor, mu_neighbor), 1)
        # # print(z.shape,"wwwwwwwwww")
        # score_k, idx_k, diff_patch = self.graph(z_neighbor, z,spatial)
        # score_k=score_k.squeeze(0)
        # idx_k=idx_k.squeeze(0)
        # # diff_patch=diff_patch.squeeze(0)
        # # print(score_k.shape,idx_k.shape,diff_patch.shape)
        # idx_k = idx_k.detach()
        # diff_patch = diff_patch.detach()
        # # print(diff_patch.shape,"eee")
        # x_h=np.zeros((z.shape[0],self.scale,z.shape[1]))
        # x1=z_neighbor.detach().cpu().numpy()
        # x_hr=idx_k.cpu().numpy().astype(int)
        # for i in range(x.shape[0]):
        #     # print(x1.shape,x_hr.shape,"eeeeeeeeeee")
        #     x_h[i]=x1[x_hr[i,:self.scale]]
        # x_h=torch.from_numpy(x_h).float().cuda()
        # idx_k = idx_k.unsqueeze(0)
        # # print(x_h.shape,"wwwwwwwwww")
        # # x1_lr = z
        # # print(z.shape,"www",x_h.shape,"wwwwwwww")
        # x1_lr, x1_hr = self.gcn(z,x_h,idx_k, diff_patch)
        # # print(x1_hr.shape,"wwwwwweeeee")
        # x1_lr=torch.swapaxes(x1_lr, 1, 2).squeeze(0)
        # x1_hr = torch.swapaxes(x1_hr, 1, 2).squeeze(0)
        # # z=torch.cat((z, x1_lr),1)
        # z=x1_lr
        # x1_hr = x1_hr.view(z.shape[0], 6, -1)
        # # for ii in range(6):
        # #     x1_hr[:,ii,:]=x1_lr
        # x1_hr = x1_hr.view(z.shape[0]*6, -1)
        # print(x1_hr.shape,"eeeeeee")
        # print(x1_lr.shape,z.shape,"aaaaaaaaaaaaa")
        # z=z.view(z.shape[0],6,-1)
        # z1= torch.mean(z, 1)
        # z = z.view(z.shape[0]*6, -1)
        # de_feat_hr= self.decoder(z)
        # de_feat_hr=de_feat_hr.view(z1.shape[0],6,-1)
        de_feat_hr = self.decoder(z)
        de_feat_hr = de_feat_hr.view(mu1.shape[0], 6, -1)
        de_feat = torch.mean(de_feat_hr, 1)
        # de_feat_hr = self.decoder(x1_hr)
        # # de_feat_lr = self.decoder(x1_lr)
        # de_feat_hr=de_feat_hr.view(z.shape[0],6,-1)
        # # for ii in range(6):
        # #     de_feat_hr[:,ii,:]=de_feat_lr
        # de_feat=torch.mean(de_feat_hr,1)
        # print(de_feat_hr.shape)
        # print(gnn_z.shape)
        # DEC clustering
        # print(z.shape,self.cluster_layer.shape,"]]]]]]]]]]]")
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        de_feat_hr = de_feat_hr.view(z1.shape[0]*6, -1)
        # print(q.shape,"wwwwwwwwwwwww")
        return z1, de_feat, q, feat_x,de_feat_hr,z
class ST_Transformer_adaptive_super0(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer_adaptive_super0, self).__init__()
        self.alpha = 1.0
        self.latent_dim = int(params.gcn_hidden2+params.feat_hidden2)
        # self.latent_dim = int(params.gcn_hidden2/6)
        self.layer_num=3
        self.at=0.5
        self.k=10
        self.scale = 6
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1, heads=1, dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden1, heads=1, dropout=params.p_drop) for i in
             range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.graph = Graph_spatial(self.scale, k=self.k, patchsize=1, stride=1,
        #                    window_size=1, in_channels=256, embedcnn=None)
        # self.gcn = GCNBlock_spatial(self.scale, k=self.k, patchsize=1, stride=1)
        # self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,adj_prue,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(x.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        # print(feat_x.shape,"wwwwwwwwwwww")
        # print(torch.isinf(feat_x).any(), "1111")
        hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
        # print(atten.shape,"wwwwwwwww")
        # atten1=atten
        # g=nx.Graph(atten[0],atten[1])
        atten=pyg.utils.remove_self_loops(atten[0],atten[1])
        atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
        atten=atten.squeeze(0).squeeze(-1)
        # print(torch.sum(atten,1),"wwwwwwww")
        # if torch.isnan(atten).any():
        #     print(atten1)
        # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
        # adj_org = nx.adjacency_matrix(atten)
        # print(adj_org.shape)
        hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
        # print(torch.isnan(hidden1).any(),"2")
        x = F.relu(hidden1)
        # x = F.relu(hidden1_prue)
        # x = F.dropout(x, p = 0.2,training=True)
        for i in range(self.layer_num - 2):
            x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
            x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
            atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
            atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
            atten_prue = atten_prue.squeeze(0).squeeze(-1)
            atten = pyg.utils.remove_self_loops(atten[0], atten[1])
            atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
            atten = atten.squeeze(0).squeeze(-1)
            x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
            x = F.relu(x)
            # x = F.dropout(x, p = 0.2,training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        # print(torch.isnan(hidden1).any(),"3")
        mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
        # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
        mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
        # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
        atten = pyg.utils.remove_self_loops(atten[0], atten[1])
        atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
        atten = atten.squeeze(0).squeeze(-1)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
        # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
        # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
        # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
        # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
        # atten_var = atten_var.squeeze(0).squeeze(-1)
        mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
        # print(torch.argmax(atten[0]),torch.argmax(atten[1]))
        # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
        return mu,feat_x,atten_prue

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, x_neighbor,adj,adj_prue,spatial,training):
        mu,feat_x,atten= self.encode(x, adj,adj_prue,training)
        # mu=mu.view(mu.shape[0],6,-1)
        # mu1= torch.mean(mu, 1)
        # mu_z =mu.view(mu.shape[0]*6, -1)
        # mu = mu.view(mu.shape[0], -1)

        # mu_neighbor, feat_x_neighbor = self.encode(x_neighbor, adj, adj_prue, training)
        # gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, mu), 1)
        # print(z.shape,"[[[[[[[[")
        # z1= torch.cat((feat_x, mu1), 1)
        # z_neighbor = torch.cat((feat_x_neighbor, mu_neighbor), 1)
        # # print(z.shape,"wwwwwwwwww")
        # score_k, idx_k, diff_patch = self.graph(z_neighbor, z,spatial)
        # score_k=score_k.squeeze(0)
        # idx_k=idx_k.squeeze(0)
        # # diff_patch=diff_patch.squeeze(0)
        # # print(score_k.shape,idx_k.shape,diff_patch.shape)
        # idx_k = idx_k.detach()
        # diff_patch = diff_patch.detach()
        # # print(diff_patch.shape,"eee")
        # x_h=np.zeros((z.shape[0],self.scale,z.shape[1]))
        # x1=z_neighbor.detach().cpu().numpy()
        # x_hr=idx_k.cpu().numpy().astype(int)
        # for i in range(x.shape[0]):
        #     # print(x1.shape,x_hr.shape,"eeeeeeeeeee")
        #     x_h[i]=x1[x_hr[i,:self.scale]]
        # x_h=torch.from_numpy(x_h).float().cuda()
        # idx_k = idx_k.unsqueeze(0)
        # # print(x_h.shape,"wwwwwwwwww")
        # # x1_lr = z
        # # print(z.shape,"www",x_h.shape,"wwwwwwww")
        # x1_lr, x1_hr = self.gcn(z,x_h,idx_k, diff_patch)
        # # print(x1_hr.shape,"wwwwwweeeee")
        # x1_lr=torch.swapaxes(x1_lr, 1, 2).squeeze(0)
        # x1_hr = torch.swapaxes(x1_hr, 1, 2).squeeze(0)
        # # z=torch.cat((z, x1_lr),1)
        # z=x1_lr
        # x1_hr = x1_hr.view(z.shape[0], 6, -1)
        # # for ii in range(6):
        # #     x1_hr[:,ii,:]=x1_lr
        # x1_hr = x1_hr.view(z.shape[0]*6, -1)
        # print(x1_hr.shape,"eeeeeee")
        # print(x1_lr.shape,z.shape,"aaaaaaaaaaaaa")
        # z=z.view(z.shape[0],6,-1)
        # z1= torch.mean(z, 1)
        # z = z.view(z.shape[0]*6, -1)
        # de_feat_hr= self.decoder(z)
        # de_feat_hr=de_feat_hr.view(z1.shape[0],6,-1)
        de_feat = self.decoder(z)
        # de_feat_hr = de_feat_hr.view(mu1.shape[0], 6, -1)
        # de_feat = torch.mean(de_feat_hr, 1)
        # de_feat_hr = self.decoder(x1_hr)
        # # de_feat_lr = self.decoder(x1_lr)
        # de_feat_hr=de_feat_hr.view(z.shape[0],6,-1)
        # # for ii in range(6):
        # #     de_feat_hr[:,ii,:]=de_feat_lr
        # de_feat=torch.mean(de_feat_hr,1)
        # print(de_feat_hr.shape)
        # print(gnn_z.shape)
        # DEC clustering
        # print(z.shape,self.cluster_layer.shape,"]]]]]]]]]]]")
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # de_feat_hr = de_feat_hr.view(z1.shape[0]*6, -1)
        # print(q.shape,"wwwwwwwwwwwww")
        return z, de_feat, q, feat_x,de_feat,z,atten
class ST_Transformer_adaptive_super00(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer_adaptive_super00, self).__init__()
        self.alpha = 1.0
        self.latent_dim = int(params.gcn_hidden2+params.feat_hidden2)
        # self.latent_dim = int(params.gcn_hidden2/6)
        self.layer_num=3
        self.at=0.5
        self.k=10
        self.scale = 9
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1, heads=1, dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden1, heads=1, dropout=params.p_drop) for i in
             range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        self.graph = Graph_spatial(self.scale, k=self.k, patchsize=1, stride=1,
                           window_size=1, in_channels=256, embedcnn=None)
        self.gcn = GCNBlock_spatial(self.scale, k=self.k, patchsize=1, stride=1)
        # self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,adj_prue,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(x.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        # print(feat_x.shape,"wwwwwwwwwwww")
        # print(torch.isinf(feat_x).any(), "1111")
        hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
        # print(atten.shape,"wwwwwwwww")
        # atten1=atten
        # g=nx.Graph(atten[0],atten[1])
        atten=pyg.utils.remove_self_loops(atten[0],atten[1])
        atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
        atten=atten.squeeze(0).squeeze(-1)
        # print(torch.sum(atten,1),"wwwwwwww")
        # if torch.isnan(atten).any():
        #     print(atten1)
        # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
        # adj_org = nx.adjacency_matrix(atten)
        # print(adj_org.shape)
        hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
        # print(torch.isnan(hidden1).any(),"2")
        x = F.relu(hidden1)
        # x = F.relu(hidden1_prue)
        # x = F.dropout(x, p = 0.2,training=True)
        for i in range(self.layer_num - 2):
            x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
            x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
            atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
            atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
            atten_prue = atten_prue.squeeze(0).squeeze(-1)
            atten = pyg.utils.remove_self_loops(atten[0], atten[1])
            atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
            atten = atten.squeeze(0).squeeze(-1)
            x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
            x = F.relu(x)
            # x = F.dropout(x, p = 0.2,training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        # print(torch.isnan(hidden1).any(),"3")
        mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
        # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
        mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
        # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
        atten = pyg.utils.remove_self_loops(atten[0], atten[1])
        atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
        atten = atten.squeeze(0).squeeze(-1)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
        # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
        # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
        # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
        # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
        # atten_var = atten_var.squeeze(0).squeeze(-1)
        mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
        # print(torch.argmax(atten[0]),torch.argmax(atten[1]))
        # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
        return mu,feat_x,atten_prue

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, x_neighbor,adj,adj_prue,spatial,training):
        mu,feat_x,atten= self.encode(x, adj,adj_prue,training)
        # mu=mu.view(mu.shape[0],6,-1)
        # mu1= torch.mean(mu, 1)
        # mu_z =mu.view(mu.shape[0]*6, -1)
        # mu = mu.view(mu.shape[0], -1)

        mu_neighbor, feat_x_neighbor,_ = self.encode(x_neighbor, adj, adj_prue, training)
        # gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, mu), 1)
        # print(z.shape,"[[[[[[[[")
        # z1= torch.cat((feat_x, mu1), 1)
        z_neighbor = torch.cat((feat_x_neighbor, mu_neighbor), 1)
        # # print(z.shape,"wwwwwwwwww")
        score_k, idx_k, diff_patch = self.graph(z_neighbor, z,spatial)
        score_k=score_k.squeeze(0)
        idx_k=idx_k.squeeze(0)
        # diff_patch=diff_patch.squeeze(0)
        # print(score_k.shape,idx_k.shape,diff_patch.shape)
        idx_k = idx_k.detach()
        diff_patch = diff_patch.detach()
        # print(diff_patch.shape,"eee")
        x_h=np.zeros((z.shape[0],self.scale,z.shape[1]))
        x1=z_neighbor.detach().cpu().numpy()
        x_hr=idx_k.cpu().numpy().astype(int)
        for i in range(x.shape[0]):
            # print(x1.shape,x_hr.shape,"eeeeeeeeeee")
            x_h[i]=x1[x_hr[i,:self.scale]]
        x_h=torch.from_numpy(x_h).float().cuda()
        idx_k = idx_k.unsqueeze(0)
        # # print(x_h.shape,"wwwwwwwwww")
        # # x1_lr = z
        # # print(z.shape,"www",x_h.shape,"wwwwwwww")
        x1_lr, x1_hr = self.gcn(z,x_h,idx_k, diff_patch)
        # # print(x1_hr.shape,"wwwwwweeeee")
        x1_lr=torch.swapaxes(x1_lr, 1, 2).squeeze(0)
        x1_hr = torch.swapaxes(x1_hr, 1, 2).squeeze(0)
        x1_hr = torch.swapaxes(x1_hr, 0, 1)
        # # z=torch.cat((z, x1_lr),1)
        z=x1_lr
        # x1_hr = x1_hr.view(z.shape[0], 6, -1)
        # # for ii in range(6):
        # #     x1_hr[:,ii,:]=x1_lr
        x1_hr = x1_hr.view(z.shape[0]*9, -1)
        # print(x1_hr.shape,"eeeeeee")
        # print(x1_lr.shape,z.shape,"aaaaaaaaaaaaa")
        # z=z.view(z.shape[0],6,-1)
        # z1= torch.mean(z, 1)
        # z = z.view(z.shape[0]*6, -1)
        # de_feat_hr= self.decoder(z)
        # de_feat_hr=de_feat_hr.view(z1.shape[0],6,-1)
        de_feat = self.decoder(z)
        # de_feat_hr = de_feat_hr.view(mu1.shape[0], 6, -1)
        # de_feat = torch.mean(de_feat_hr, 1)
        # de_feat_hr = self.decoder(x1_hr)
        # # de_feat_lr = self.decoder(x1_lr)
        # de_feat_hr=de_feat_hr.view(z.shape[0],6,-1)
        # # for ii in range(6):
        # #     de_feat_hr[:,ii,:]=de_feat_lr
        # de_feat=torch.mean(de_feat_hr,1)
        # print(de_feat_hr.shape)
        # print(gnn_z.shape)
        # DEC clustering
        # print(z.shape,self.cluster_layer.shape,"]]]]]]]]]]]")
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # de_feat_hr = de_feat_hr.view(z1.shape[0]*6, -1)
        # print(q.shape,"wwwwwwwwwwwww")
        # print(x1_hr.shape,"[[[[[[[[[[")
        return z, de_feat, q, feat_x,de_feat,x1_hr,atten
class ST_Transformer_adaptive_super_gai(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer_adaptive_super_gai, self).__init__()
        self.alpha = 1.0
        self.latent_dim = int(params.gcn_hidden2+params.feat_hidden2)
        # self.latent_dim = int(params.gcn_hidden2/6)
        self.layer_num=3
        self.at=0.5
        self.k=10
        self.scale = 9
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1, heads=1, dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden1, heads=1, dropout=params.p_drop) for i in
             range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        self.graph = Graph_spatial(self.scale, k=self.k, patchsize=1, stride=1,
                           window_size=1, in_channels=256, embedcnn=None)
        self.gcn = GCNBlock_spatial(self.scale, k=self.k, patchsize=1, stride=1)
        # self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,adj_prue,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(x.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        # print(feat_x.shape,"wwwwwwwwwwww")
        # print(torch.isinf(feat_x).any(), "1111")
        hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
        # print(atten.shape,"wwwwwwwww")
        # atten1=atten
        # g=nx.Graph(atten[0],atten[1])
        atten=pyg.utils.remove_self_loops(atten[0],atten[1])
        atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
        atten=atten.squeeze(0).squeeze(-1)
        # print(torch.sum(atten,1),"wwwwwwww")
        # if torch.isnan(atten).any():
        #     print(atten1)
        # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
        # adj_org = nx.adjacency_matrix(atten)
        # print(adj_org.shape)
        hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
        # print(torch.isnan(hidden1).any(),"2")
        x = F.relu(hidden1)
        # x = F.relu(hidden1_prue)
        # x = F.dropout(x, p = 0.2,training=True)
        for i in range(self.layer_num - 2):
            x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
            x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
            atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
            atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
            atten_prue = atten_prue.squeeze(0).squeeze(-1)
            atten = pyg.utils.remove_self_loops(atten[0], atten[1])
            atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
            atten = atten.squeeze(0).squeeze(-1)
            x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
            x = F.relu(x)
            # x = F.dropout(x, p = 0.2,training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        # print(torch.isnan(hidden1).any(),"3")
        mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
        # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
        mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
        # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
        atten = pyg.utils.remove_self_loops(atten[0], atten[1])
        atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
        atten = atten.squeeze(0).squeeze(-1)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
        # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
        # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
        # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
        # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
        # atten_var = atten_var.squeeze(0).squeeze(-1)
        mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
        # print(torch.argmax(atten[0]),torch.argmax(atten[1]))
        # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
        return mu,feat_x,atten_prue

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, x_neighbor,adj,adj_prue,spatial,training):
        mu,feat_x,atten= self.encode(x, adj,adj_prue,training)
        # mu=mu.view(mu.shape[0],6,-1)
        # mu1= torch.mean(mu, 1)
        # mu_z =mu.view(mu.shape[0]*6, -1)
        # mu = mu.view(mu.shape[0], -1)

        # mu_neighbor, feat_x_neighbor,_ = self.encode(x_neighbor, adj, adj_prue, training)
        # gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, mu), 1)
        # print(z.shape,"[[[[[[[[")
        # z1= torch.cat((feat_x, mu1), 1)
        # z_neighbor = torch.cat((feat_x_neighbor, mu_neighbor), 1)
        # # print(z.shape,"wwwwwwwwww")
        score_k, idx_k, diff_patch = self.graph(z, z,spatial)
        score_k=score_k.squeeze(0)
        idx_k=idx_k.squeeze(0)
        # diff_patch=diff_patch.squeeze(0)
        # print(score_k.shape,idx_k.shape,diff_patch.shape)
        idx_k = idx_k.detach()
        diff_patch = diff_patch.detach()
        # print(diff_patch.shape,"eee")
        x_h=np.zeros((z.shape[0],self.scale,z.shape[1]))
        x1=z.detach().cpu().numpy()
        x_hr=idx_k.cpu().numpy().astype(int)
        for i in range(x.shape[0]):
            # print(x1.shape,x_hr.shape,"eeeeeeeeeee")
            x_h[i]=x1[x_hr[i,:self.scale]]
        x_h=torch.from_numpy(x_h).float().cuda()
        # idx_k = idx_k.unsqueeze(0)
        # # print(x_h.shape,"wwwwwwwwww")
        # # x1_lr = z
        # # print(z.shape,"www",x_h.shape,"wwwwwwww")
        ###################################
        # x1_lr, x1_hr = self.gcn(z,x_h,idx_k, diff_patch)
        # print(z.shape,x1_hr.shape,"wwwwwweeeee")
        # x1_lr=torch.swapaxes(x1_lr, 1, 2).squeeze(0)
        # x1_hr = torch.swapaxes(x1_hr, 1, 2).squeeze(0)
        # x1_hr = torch.swapaxes(x1_hr, 0, 1)
        # # # z=torch.cat((z, x1_lr),1)
        # z=x1_lr
        ## x1_hr = x1_hr.view(z.shape[0], 6, -1)
        ##for ii in range(6):
        ##    x1_hr[:,ii,:]=x1_lr
        #x1_hr = x1_hr.view(z.shape[0]*9, -1)
        #################################
        ## print(x1_hr.shape,"eeeeeee")
        ## print(x1_lr.shape,z.shape,"aaaaaaaaaaaaa")
        ## z=z.view(z.shape[0],6,-1)
        ## z1= torch.mean(z, 1)
        # z = z.view(z.shape[0]*6, -1)
        # de_feat_hr= self.decoder(z)
        # de_feat_hr=de_feat_hr.view(z1.shape[0],6,-1)
        z=torch.mean(x_h,1)
        # print(z.shape)
        de_feat = self.decoder(z)
        # de_feat_hr = de_feat_hr.view(mu1.shape[0], 6, -1)
        # de_feat = torch.mean(de_feat_hr, 1)
        # de_feat_hr = self.decoder(x1_hr)
        # # de_feat_lr = self.decoder(x1_lr)
        # de_feat_hr=de_feat_hr.view(z.shape[0],6,-1)
        # # for ii in range(6):
        # #     de_feat_hr[:,ii,:]=de_feat_lr
        # de_feat=torch.mean(de_feat_hr,1)
        # print(de_feat_hr.shape)
        # print(gnn_z.shape)
        # DEC clustering
        # print(z.shape,self.cluster_layer.shape,"]]]]]]]]]]]")
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # de_feat_hr = de_feat_hr.view(z1.shape[0]*6, -1)
        # print(q.shape,"wwwwwwwwwwwww")
        # print(x1_hr.shape,"[[[[[[[[[[")
        return z, de_feat, q, feat_x,de_feat,x_h,atten
class ST_Transformer_super_gai(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer_super_gai, self).__init__()
        self.alpha = 1.0
        self.latent_dim = int(params.gcn_hidden2+params.feat_hidden2)
        # # self.latent_dim = int(params.gcn_hidden2/6)
        self.layer_num=3
        # self.at=0.5
        self.k=10
        self.scale = 6
        # # feature autoencoder
        # self.encoder = nn.Sequential()
        # self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        # self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1, heads=1, dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden1, heads=1, dropout=params.p_drop) for i in
             range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        self.graph = Graph_spatial(self.scale, k=self.k, patchsize=1, stride=1,
                           window_size=1, in_channels=256, embedcnn=None)
        self.gcn = GCNBlock_spatial(self.scale, k=self.k, patchsize=1, stride=1)
        # self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    # def encode(self, x, adj,adj_prue,training):
    #     # x, edge_index = data.x, data.edge_index
    #     # edge_index=adj
    #     # x=self.add_noise(x)
    #     # print(x.shape,"wwwwwwwwwwwwwwwwwww")
    #     feat_x = self.encoder(x)
    #     # print(feat_x.shape,"wwwwwwwwwwww")
    #     # print(torch.isinf(feat_x).any(), "1111")
    #     hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
    #     # print(atten.shape,"wwwwwwwww")
    #     # atten1=atten
    #     # g=nx.Graph(atten[0],atten[1])
    #     atten=pyg.utils.remove_self_loops(atten[0],atten[1])
    #     atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
    #     atten=atten.squeeze(0).squeeze(-1)
    #     # print(torch.sum(atten,1),"wwwwwwww")
    #     # if torch.isnan(atten).any():
    #     #     print(atten1)
    #     # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
    #     # adj_org = nx.adjacency_matrix(atten)
    #     # print(adj_org.shape)
    #     hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
    #     atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
    #     atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
    #     atten_prue = atten_prue.squeeze(0).squeeze(-1)
    #     hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
    #     # print(torch.isnan(hidden1).any(),"2")
    #     x = F.relu(hidden1)
    #     # x = F.relu(hidden1_prue)
    #     # x = F.dropout(x, p = 0.2,training=True)
    #     for i in range(self.layer_num - 2):
    #         x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
    #         x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
    #         atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
    #         atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
    #         atten_prue = atten_prue.squeeze(0).squeeze(-1)
    #         atten = pyg.utils.remove_self_loops(atten[0], atten[1])
    #         atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
    #         atten = atten.squeeze(0).squeeze(-1)
    #         x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
    #         x = F.relu(x)
    #         # x = F.dropout(x, p = 0.2,training=True)
    #     # x = F.normalize(x, p=2, dim=-1)
    #     hidden1=x
    #     # print(torch.isnan(hidden1).any(),"3")
    #     mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
    #     # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
    #     mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
    #     # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
    #     atten = pyg.utils.remove_self_loops(atten[0], atten[1])
    #     atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
    #     atten = atten.squeeze(0).squeeze(-1)
    #     atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
    #     atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
    #     atten_prue = atten_prue.squeeze(0).squeeze(-1)
    #     # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
    #     # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
    #     # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
    #     # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
    #     # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
    #     # atten_var = atten_var.squeeze(0).squeeze(-1)
    #     mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
    #     # print(torch.argmax(atten[0]),torch.argmax(atten[1]))
    #     # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
    #     return mu,feat_x,atten_prue

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, x_neighbor,spatial,training):
        x_neighbor=x_neighbor.view(self.scale,x.shape[0],x.shape[1])
        x_neighbor=torch.swapaxes(x_neighbor, 0, 1)
        # print(x_neighbor.shape)
        x_neighbor=torch.reshape(x_neighbor,(x.shape[0],x_neighbor.shape[1]*x_neighbor.shape[2]))
        # mu,feat_x,atten= self.encode(x, adj,adj_prue,training)
        # mu=mu.view(mu.shape[0],6,-1)
        # mu1= torch.mean(mu, 1)
        # mu_z =mu.view(mu.shape[0]*6, -1)
        # mu = mu.view(mu.shape[0], -1)

        # mu_neighbor, feat_x_neighbor,_ = self.encode(x_neighbor, adj, adj_prue, training)
        # gnn_z = self.reparameterize(mu, logvar)
        # z = torch.cat((feat_x, mu), 1)
        # print(z.shape,"[[[[[[[[")
        # z1= torch.cat((feat_x, mu1), 1)
        # z_neighbor = torch.cat((feat_x_neighbor, mu_neighbor), 1)
        # # print(z.shape,"wwwwwwwwww")
        score_k, idx_k, diff_patch = self.graph(x, x,spatial)
        score_k=score_k.squeeze(0)
        # idx_k=idx_k.squeeze(0)
        # diff_patch=diff_patch.squeeze(0)
        # print(score_k.shape,idx_k.shape,diff_patch.shape)
        idx_k = idx_k.detach()
        z=x
        diff_patch = diff_patch.detach()
        # print(diff_patch.shape,"eee")
        # x_h=np.zeros((z.shape[0],self.scale,z.shape[1]))
        # x1=z.detach().cpu().numpy()
        # x_hr=idx_k.cpu().numpy().astype(int)
        # for i in range(x.shape[0]):
        #     # print(x1.shape,x_hr.shape,"eeeeeeeeeee")
        #     x_h[i]=x1[x_hr[i,:self.scale]]
        # x_h=torch.from_numpy(x_h).float().cuda()
        # idx_k = idx_k.unsqueeze(0)
        # # print(x_h.shape,"wwwwwwwwww")
        # # x1_lr = z
        # # print(z.shape,"www",x_h.shape,"wwwwwwww")
        # print(idx_k.shape,z.shape,diff_patch.shape,x_neighbor.shape,"[]]][][][][][[]")
        ###################################
        x1_lr, x1_hr = self.gcn(z,x_neighbor,idx_k, diff_patch)
        # print(z.shape,x1_hr.shape,"wwwwwweeeee")
        x1_lr=torch.swapaxes(x1_lr, 1, 2).squeeze(0)
        x1_hr = torch.swapaxes(x1_hr, 1, 2).squeeze(0)
        # x1_hr = torch.swapaxes(x1_hr, 0, 1)
        # # # z=torch.cat((z, x1_lr),1)
        z=x1_lr.squeeze(0)
        x1_hr=x1_hr.squeeze(0)
        # print(x1_hr.shape)
        # x1_hr=x1_hr.view(x1_lr.shape[0],self.scale,x1_lr.shape[1])
        # x1_hr = torch.swapaxes(x1_hr, 0, 1)
        # x1_hr = x1_hr.view(x1_lr.shape[0]*self.scale, x1_lr.shape[1])
        ## x1_hr = x1_hr.view(z.shape[0], 6, -1)
        ##for ii in range(6):
        ##    x1_hr[:,ii,:]=x1_lr
        #x1_hr = x1_hr.view(z.shape[0]*9, -1)
        #################################
        ## print(x1_hr.shape,"eeeeeee")
        ## print(x1_lr.shape,z.shape,"aaaaaaaaaaaaa")
        ## z=z.view(z.shape[0],6,-1)
        ## z1= torch.mean(z, 1)
        # z = z.view(z.shape[0]*6, -1)
        # de_feat_hr= self.decoder(z)
        # de_feat_hr=de_feat_hr.view(z1.shape[0],6,-1)
        # z=torch.mean(x_h,1)
        # print(z.shape)
        de_feat = self.decoder(z)
        # de_feat_hr = de_feat_hr.view(mu1.shape[0], 6, -1)
        # de_feat = torch.mean(de_feat_hr, 1)
        # de_feat_hr = self.decoder(x1_hr)
        # # de_feat_lr = self.decoder(x1_lr)
        # de_feat_hr=de_feat_hr.view(z.shape[0],6,-1)
        # # for ii in range(6):
        # #     de_feat_hr[:,ii,:]=de_feat_lr
        # de_feat=torch.mean(de_feat_hr,1)
        # print(de_feat_hr.shape)
        # print(gnn_z.shape)
        # DEC clustering
        # print(z.shape,self.cluster_layer.shape,"]]]]]]]]]]]")
        # q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        # q = q.pow((self.alpha + 1.0) / 2.0)
        # q = (q.t() / torch.sum(q, 1)).t()
        # de_feat_hr = de_feat_hr.view(z1.shape[0]*6, -1)
        # print(q.shape,"wwwwwwwwwwwww")
        # print(x1_hr.shape,"[[[[[[[[[[")
        return z, de_feat, x1_hr

