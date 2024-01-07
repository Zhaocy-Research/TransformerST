args = commandArgs(trailingOnly=TRUE)
sample.name <- args[1]
n_clusters <- as.numeric(args[2])

library(BayesSpace)
library(ggplot2)

dir.input <- file.path('./data/DLPFC/', sample.name)
dir.output <- file.path('./output/DLPFC/', sample.name, '/BayesSpace/')

if(!dir.exists(file.path(dir.output))){
  dir.create(file.path(dir.output), recursive = TRUE)
}


dlpfc <- getRDS("2020_maynard_prefrontal-cortex", sample.name)

set.seed(101)
dec <- scran::modelGeneVar(dlpfc)
top <- scran::getTopHVGs(dec, n = 2000)

set.seed(102)
dlpfc <- scater::runPCA(dlpfc, subset_row=top)

## Add BayesSpace metadata
dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)


##### Clustering with BayesSpace
q <- n_clusters  # Number of clusters
d <- 15  # Number of PCs

## Run BayesSpace clustering
set.seed(104)
dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium',
                        nrep=5000, gamma=3, save.chain=TRUE)

labels <- dlpfc$spatial.cluster

## View results
clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
  scale_fill_viridis_d(option = "A", labels = 1:7) +
  labs(title="BayesSpace")

ggsave(file.path(dir.output, 'clusterPlot.png'), width=5, height=5)
Y1.2 <- reducedDim(dlpfc, "PCA")[, seq_len(d)]

## mclust (BayesSpace initialization)
library(mclust)
set.seed(100)
mclust.labels <- Mclust(Y1.2, q, "EEE")$classification
# sample.name="ST_mel1_rep2"
# dir.output = file.path('./output/ST/', sample.name, 'MCLUST')
# write.table(df_meta, file = file.path(dir.output, 'metadata.tsv'), sep='\t', quote=F, row.names = F)
## K-means
set.seed(103)
km.labels <- kmeans(Y1.2, centers = q)$cluster
# spagcn.label<-as.numeric(unlist(spagcn_cluster$refined_pred))
## Louvain
set.seed(100)
g.jaccard <- scran::buildSNNGraph(dlpfc, use.dimred="PCA", type="jaccard")
louvain.labels <- igraph::cluster_louvain(g.jaccard)$membership
# giotto.fname <- system.file("extdata", "2018_thrane_melanoma", "ST_mel1_rep2.Giotto_HMRF.csv", package = "BayesSpace")
# giotto.labels <- read.csv(giotto.fname)$HMRF_PCA_k4_b.2
dlpfc@colData@listData$mclust<-mclust.labels 
dlpfc@colData@listData$kmeans<-km.labels 
dlpfc@colData@listData$louvain<-louvain.labels

write.table(colData(dlpfc), file=file.path(dir.output, 'metadata.tsv'), sep='\t', quote=FALSE)

