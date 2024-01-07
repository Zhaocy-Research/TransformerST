library(mclust)
library(ggplot2)
library(patchwork)
library(Seurat)
library(mclust)
options(bitmapType = 'cairo')

args <- commandArgs(trailingOnly = TRUE)
sample <- args[1]
#sample <- "151507"
sp_data <- readRDS(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/Seurat/Seurat_final.rds'))

##### SpatialDimPlot
metadata <- read.table(file.path('/media/cyzhao/New_Volume/data/DLPFC/', sample, 'metadata.tsv'), sep='\t', header=TRUE)

#spagcn_cluster <- read.table(file.path('./output/DLPFC/', sample, '/SpaGCN/metadata.tsv'), sep='\t', header=TRUE, row.names =1)
#sedr_cluster$sed_labels <- sedr_cluster$leiden_fixed_clusCount

# seurat_cluster <- read.table(file.path('./output/DLPFC/', sample, '/Seurat/metadata.tsv'), sep='\t', header=TRUE)
Our_cluster <- read.table(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/TransformerST/metadata.tsv'), sep='\t', header=TRUE)
BayesSpace_cluster <- read.table(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/BayesSpace/metadata.tsv'), sep='\t', header=TRUE)
spaGCN_cluster <- read.table(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/SpaGCN/metadata.tsv'), sep='\t', header=TRUE)
Giotto_cluster <- read.table(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/Giotto/metadata.tsv'), sep='\t', header=TRUE)
row.names(Giotto_cluster) <- Giotto_cluster$cell_ID
stLearn_cluster <- read.table(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/stLearn/metadata.tsv'), sep='\t', header=TRUE)
ccst_cluster <-read.table(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/lambdaI0.3/metadata.tsv'), sep='\t', header=TRUE)
STAGATE_cluster <-read.table(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/STAGATE1/metadata.tsv'), sep='\t', header=TRUE)
CONST_cluster <-read.table(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/conST/metadata.tsv'), sep='\t', header=TRUE)
DEEPST_cluster <-read.table(file.path('/media/cyzhao/New_Volume/output/DLPFC/', sample, '/DEEPST/metadata.tsv'), sep='\t', header=TRUE)
truth <- as.factor(metadata$layer_guess)
truth <- factor(truth, levels=c('WM', 'nan', 'Layer6', 'Layer5', 'Layer4', 'Layer3', 'Layer2', 'Layer1'))
sp_data <- AddMetaData(sp_data, truth, col.name = 'layer_guess')
sp_data <- AddMetaData(sp_data, spaGCN_cluster$refined_pred, col.name = 'SpaGCN')
# sp_data <- AddMetaData(sp_data, seurat_cluster$seurat_clusters, col.name = 'Seurat')
sp_data <- AddMetaData(sp_data, Our_cluster$TransformerST, col.name = 'TransformerST')
sp_data <- AddMetaData(sp_data, BayesSpace_cluster$spatial.cluster, col.name = 'BayesSpace')
sp_data <- AddMetaData(sp_data, Giotto_cluster[, 'HMRF_cluster', drop=F], col.name = 'Giotto')
sp_data <- AddMetaData(sp_data, stLearn_cluster$X_pca_kmeans, col.name = 'stLearn')
sp_data <- AddMetaData(sp_data, BayesSpace_cluster$kmeans, col.name = 'Kmeans')
sp_data <- AddMetaData(sp_data, BayesSpace_cluster$louvain, col.name = 'Louvain')
sp_data <- AddMetaData(sp_data, BayesSpace_cluster$mclust, col.name = 'Mclust')
sp_data <- AddMetaData(sp_data, STAGATE_cluster$STAGATE, col.name = 'STAGATE')
sp_data <- AddMetaData(sp_data, ccst_cluster$CCST, col.name = 'CCST')
sp_data <- AddMetaData(sp_data, CONST_cluster$conST_refine, col.name = 'CONST')
sp_data <- AddMetaData(sp_data, DEEPST_cluster$DEEPST, col.name = 'DEEPSTST')
SpaGCN_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$SpaGCN)
# Seurat_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$Seurat)
Our_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$TransformerST)
BayesSpace_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$BayesSpace)
Giotto_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$Giotto)
stLearn_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$stLearn)
Kmeans_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$Kmeans)
Mclust_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$Mclust)
Louvain_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$Louvain)
CCST_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$CCST)
CONST_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$CONST)
DEEPST_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$DEEPST)
STAGATE_ARI = adjustedRandIndex(sp_data@meta.data$layer_guess, sp_data@meta.data$STAGATE)
df_clusters <- data.frame(layer_guess = sp_data@meta.data$layer_guess,
                          SpaGCN = as.factor(sp_data@meta.data$SpaGCN),
                          # Seurat = as.factor(sp_data@meta.data$Seurat),
                          TransformerST = as.factor(sp_data@meta.data$TransformerST),
                          BayesSpace = as.factor(sp_data@meta.data$BayesSpace),
                          Kmeans = as.factor(sp_data@meta.data$Kmeans),
                          Mclust = as.factor(sp_data@meta.data$Mclust),
                          Louvain = as.factor(sp_data@meta.data$Louvain),
                          Giotto = as.factor(sp_data@meta.data$Giotto),
                          stLearn = as.factor(sp_data@meta.data$stLearn),
                          CCST = as.factor(sp_data@meta.data$CCST),
                          STAGATE = as.factor(sp_data@meta.data$STAGATE),
                          CONST= as.factor(sp_data@meta.data$CONST),
                          DEEPST= as.factor(sp_data@meta.data$DEEPST)
                          )

df <- sp_data@images$slice1@coordinates
df <- cbind(df, df_clusters)
p0 <- ggplot(df, aes(imagecol, imagerow, color=layer_guess)) + geom_point(stroke=0, size=1.1) + ggtitle('Ground Truth') +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))

# p1 <- ggplot(df, aes(imagecol, imagerow, color=Seurat)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('Seurat: ARI=', round(Seurat_ARI, 3))) +
#   coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)
p1 <- ggplot(df, aes(imagecol, imagerow, color=Mclust)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('Mclust: ARI=', round(Mclust_ARI, 3))) +
    coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")
p2 <- ggplot(df, aes(imagecol, imagerow, color=Kmeans)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('Kmeans: ARI=', round(Kmeans_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")
p3 <- ggplot(df, aes(imagecol, imagerow, color=Louvain)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('Louvain: ARI=', round(Louvain_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")
p4 <- ggplot(df, aes(imagecol, imagerow, color=Giotto)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('Giotto: ARI=', round(Giotto_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")

p5 <- ggplot(df, aes(imagecol, imagerow, color=stLearn)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('StLearn: ARI=', round(stLearn_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")

p10 <- ggplot(df, aes(imagecol, imagerow, color=TransformerST)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('TransformerST: ARI=', round(Our_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))

p7 <- ggplot(df, aes(imagecol, imagerow, color=BayesSpace)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('BayesSpace: ARI=', round(BayesSpace_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")

p6 <- ggplot(df, aes(imagecol, imagerow, color=SpaGCN)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('SpaGCN: ARI=', round(SpaGCN_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")
p8 <- ggplot(df, aes(imagecol, imagerow, color=CCST)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('CCST: ARI=', round(CCST_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")
p9 <- ggplot(df, aes(imagecol, imagerow, color=STAGATE)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('STAGATE: ARI=', round(STAGATE_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")
p11 <- ggplot(df, aes(imagecol, imagerow, color=CONST)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('CONST: ARI=', round(CONST_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")
p12 <- ggplot(df, aes(imagecol, imagerow, color=DEEPST)) + geom_point(stroke=0, size=1.1) + ggtitle(paste('DEEPST: ARI=', round(DEEPST_ARI, 3))) +
  coord_fixed() + scale_y_reverse() + theme_void() + viridis::scale_color_viridis(option="plasma", discrete = T)+theme(plot.title = element_text(hjust = 0.5))+ theme(legend.position="none")
#p1 +p2+p3+p4+ p5 + p6 +p7+p8+ p9+p10+theme(legend.key.size = unit(0.1, 'cm'),legend.title = element_blank())+p11+p12+plot_layout(ncol = 5, widths = c(1,1,1,1,1), heights = c(1,1,1))
p0+plot_layout(ncol = 5, widths = c(1,1,1,1,1), heights = c(1,1,1))


dir.output <- file.path('./output/DLPFC/', sample, '/Comparison/')
if(!dir.exists(file.path(dir.output))){
  dir.create(file.path(dir.output), recursive = TRUE)
}


ggsave(filename = file.path(dir.output, 'comparison2.png'), width=11, height=5.5)
ggsave(filename = file.path(dir.output,  'comparison2.pdf'), width=11, height=5.5)

write.table(df, file.path(dir.output, 'comparison2.tsv'), sep='\t', quote=FALSE)

