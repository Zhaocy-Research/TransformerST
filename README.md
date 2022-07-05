# TransformerST
Transformer Enables Reference Free And Unsupervised Analysis of Spatial Transcriptomics at Single Cell Resolution

We develop a novel Transformer based framework (TransformerST) for associating the heterogeneity of local gene expression properties and revealing the dependency of structural relationship at near single cell resolution.TransformerST-super consists of three components, the conditional transformer based variational autoencoder, the adaptive graph Transformer model with multi-head attention, and the cross-scale super-resolved gene expression reconstruction model.

Framework

image

The code is licensed under the MIT license.
1. Requirements
1.1 Operating systems:

The code in python has been tested on Linux (Ubuntu 20.04.1 LTS).  

1.2 Required packages in python:

anndata==0.7.6   
numpy==1.20.2  
opencv-python==4.5.3.56  
pandas==1.2.4  
python==3.9.4  
python-louvain==0.15  
rpy2==3.4.5  
scanpy==1.7.2  
scipy==1.6.3  
seaborn==0.11.1  
torch==1.8.1  
torch-geometric==1.7.0  
torchvision==0.9.1  
tqdm==4.59.0  
umap-learn==0.5.1  

2. Instructions: Demo on mouse lung data.   
 
2.1 Raw data

Raw data should be placed in the folder data.

we take the mouse lung data for example, which is in data/Lung/A1. 


2.2 Cell type identification at spot resolution with Mouse Lung data

The TransformerST model is implemented in Mouse_Lung_histology.py. When running TransformerST, the data path should be specified, please modify the --data_root  and --proj_list here. In addition, the parameter --save_root should also be modified to save the experimental results.

The defination of each argument in Mouse_Lung_histology.py is listed below.

'--k', type=int, default=20, help='parameter k in spatial graph'  
'--knn_distanceType', type=str, default='euclidean',help='graph distance type: euclidean/cosine/correlation'  
'--epochs', type=int, default=1000, help='Number of epochs to train.'  
'--cell_feat_dim', type=int, default=3000, help='Dim of input genes'  
'--feat_hidden1', type=int, default=512, help='Dim of DNN hidden 1-layer.'  
'--feat_hidden2', type=int, default=128, help='Dim of DNN hidden 2-layer.'  
'--gcn_hidden1', type=int, default=128, help='Dim of GCN hidden 1-layer.'  
'--gcn_hidden2', type=int, default=64, help='Dim of GCN hidden 2-layer.'  
'--p_drop', type=float, default=0.2, help='Dropout rate.'  
'--using_dec', type=bool, default=True, help='Using DEC loss.'  
'--using_mask', type=bool, default=False, help='Using mask for multi-dataset.'  
'--feat_w', type=float, default=1, help='Weight of DNN loss.'  
'--gcn_w', type=float, default=1, help='Weight of GCN loss.'  
'--dec_kl_w', type=float, default=1, help='Weight of DEC loss.'  
'--gcn_lr', type=float, default=0.001, help='Initial GNN learning rate.'  
'--gcn_decay', type=float, default=0.0001, help='Initial decay rate.'  
'--dec_cluster_n', type=int, default=10, help='DEC cluster number.'  
'--dec_interval', type=int, default=20, help='DEC interval nnumber.'  
'--dec_tol', type=float, default=0.00, help='DEC tol.'  
'--eval_resolution', type=int, default=1, help='Eval cluster number.'  
'--eval_graph_n', type=int, default=20, help='Eval graph kN tol.' 
  
3. Instructions: Demo on IDC data. 
 
3.1 Raw data

Raw data should be placed in the folder data.

we take the IDC data for example, which is in data/IDC. 


3.2 Spatial transcriptomics super resolution with IDC data

The TransformerST model is implemented in IDC_super_TransformerST.py. When running TransformerST, the data path should be specified, please modify the --data_root  and --proj_list here. In addition, the parameter --save_root should also be modified to save the experimental results.

The defination of each argument in IDC_super_TransformerST.py is listed below.

'--k', type=int, default=10, help='parameter k in spatial graph'  
'--knn_distanceType', type=str, default='euclidean', help='graph distance type: euclidean/cosine/correlation'  
'--epochs', type=int, default=1000, help='Number of epochs to train.'  
'--cell_feat_dim', type=int, default=3000, help='Dim of input genes'  
'--feat_hidden1', type=int, default=512, help='Dim of DNN hidden 1-layer.'  
'--feat_hidden2', type=int, default=128, help='Dim of DNN hidden 2-layer.'  
'--gcn_hidden1', type=int, default=128, help='Dim of GCN hidden 1-layer.'  
'--gcn_hidden2', type=int, default=64, help='Dim of GCN hidden 2-layer.'  
'--p_drop', type=float, default=0.1, help='Dropout rate.'  
'--using_dec', type=bool, default=True, help='Using DEC loss.'  
'--using_mask', type=bool, default=False, help='Using mask for multi-dataset.'  
'--feat_w', type=float, default=1, help='Weight of DNN loss.'  
'--gcn_w', type=float, default=1, help='Weight of GCN loss.'  
'--dec_kl_w', type=float, default=1, help='Weight of DEC loss.'  
'--gcn_lr', type=float, default=0.001, help='Initial GNN learning rate.'  
'--gcn_decay', type=float, default=0.0001, help='Initial decay rate.'  
'--dec_cluster_n', type=int, default=10, help='DEC cluster number.'  
'--dec_interval', type=int, default=20, help='DEC interval nnumber.'  
'--dec_tol', type=float, default=0.00, help='DEC tol.'  
'--eval_resolution', type=int, default=1, help='Eval cluster number.'  
'--eval_graph_n', type=int, default=20, help='Eval graph kN tol.'  

3. All datasets used in TransformerST:

(1) LIBD human dorsolateral pre-frontal cortex data (DLPFC) (http://research.libd.org/spatialLIBD/); 

(2) Melanoma ST data (https://www.spatialresearch.org/wp-content/uploads/
2019/03/ST-Melanoma-Datasets 1.zip); 

(3) Human epidermal growth factor
receptor(HER) 2 amplified (HER+) invasive ductal carcinoma (IDC) sample;

Zhao, E. et al. Spatial transcriptomics at subspot resolution with bayesspace. Nature Biotechnology 1–10 (2021).

(4)HER2+ breast cancer data;
He, B. et al. Integrating spatial gene expression and breast tumour morphology via deep learning. Nature biomedical engineering 4 (8), 827–834 (2020) 

(5) Mouse lung data available upon reasonable request.
