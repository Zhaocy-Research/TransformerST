library(dplyr)
library(Giotto)
library(Seurat)
library(ggplot2)
library(patchwork)
library(ggthemes)
# library(ggpubr)
library(mclust)
options(bitmapType = 'cairo')

list.samples <- c("151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676")

list.methods <- c( "Mclust","Kmeans", "Louvain","Giotto","stLearn", "SpaGCN", "BayesSpace","CCST","STAGATE","CONST","DEEPST", "TransformerST")

##### Generate data
c1 <- c()
c2 <- c()
c3 <- c()

for (sample in list.samples) {
  file.results <- file.path('./output/DLPFC/', sample, '/Comparison/comparison.tsv')
  df.results <- read.table(file.results, sep='\t', header=T)
  for (method in list.methods){
    cluster <- df.results  %>% select(c(method))
    ARI <- adjustedRandIndex(x = df.results$layer_guess, y = cluster[, 1])
    
    c1 <- c(c1, method)
    c2 <- c(c2, sample)
    c3 <- c(c3, ARI)
  }
}

df.comp <- data.frame(method = c1,
                      sample = c2,
                      ARI = c3)


##### Plot results
df.comp$method <- as.factor(df.comp$method)
df.comp$method <- factor(df.comp$method, 
                         levels = c("Mclust","Kmeans", "Louvain","Giotto","stLearn", "SpaGCN", "BayesSpace", "CCST","STAGATE","CONST","DEEPST","TransformerST"))


ggplot(df.comp, aes(method, ARI)) + 
  geom_boxplot(width=0.5) + 
  geom_jitter(width = 0.1, size=1) +
  theme_bw() + 
  theme(panel.background = element_blank(),
        panel.grid = element_blank(), 
        axis.title.y = element_blank(), 
        axis.text = element_text(angle = 90,size=10)
        )

ggsave('./output/DLPFC/ARI_violin.png', width=10, height=4)
ggsave('./output/DLPFC/ARI_violin.pdf', width=10, height=4)


