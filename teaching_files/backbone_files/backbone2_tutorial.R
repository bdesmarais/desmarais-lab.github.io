#### Setup ####
set.seed(1)
library(igraph) #Used for visualizations
library(latex2exp) #Used for visualizations
library(poweRlaw) #To evaluate degree distributions
load("backbone2_tutorial.Rdata")  #Load example data

#### Install and Load ####
#install.packages("backbone")  #Install the package
library(backbone) #Load the package

#### Backbone from Bipartite Projection (toy example) ####
B <- rbind(cbind(matrix(rbinom(250,1,.8),10),  #Generate toy bipartite network,
                 matrix(rbinom(250,1,.2),10),  #with embedded communities
                 matrix(rbinom(250,1,.2),10)),
           cbind(matrix(rbinom(250,1,.2),10),
                 matrix(rbinom(250,1,.8),10),
                 matrix(rbinom(250,1,.2),10)),
           cbind(matrix(rbinom(250,1,.2),10),
                 matrix(rbinom(250,1,.2),10),
                 matrix(rbinom(250,1,.8),10)))
sum(B[1,])  #Agent 1's degree
sum(B[,1])  #Artifact 1's degree

P <- B %*% t(B) #Construct bipartite projection
P[1,2]  #Edge weight for agents in the same group
P[1,20]  #Edge weight for agents in different groups
min(P)  #Smallest edge weight

weighted <- graph_from_adjacency_matrix(P, mode = "undirected", weighted = TRUE, diag = FALSE) #Weighted projection
disparity <- fdsm(B, alpha = 0.05, class = "igraph",trials=5000) #Disparity backbone
backbone <- sdsm(B, alpha = 0.05, class = "igraph") #SDSM backbone

pdf("Figures/bipartite_toy.pdf", height = 2, width = 6) #Plot in manuscript
par(mfrow=c(1,3), mar = c(0, 0, 3, 0)) 
plot(weighted, vertex.label = NA, edge.width = ((E(weighted)$weight)^2)/100)
title("Weighted Original", line = 1)
plot(disparity, vertex.label = NA)
title("FDSM Backbone", line = 1)
plot(backbone, vertex.label = NA)
title("SDSM Backbone", line = 1)
dev.off()

#### Backbone from Bipartite Projection (empirical example, senate) ####
senate[1:2,1:2]  #First two rows and columns
sum(senate["Sen. Stabenow, Debbie [D-MI]",])  #Sponsorships by Sen. Stabenow
sum(senate[,"S.1006"])  #Sponsors of Equality Act

P <- senate %*% t(senate) #Construct bipartite projection
P["Sen. Stabenow, Debbie [D-MI]", "Sen. Peters, Gary C. [D-MI]"]
P["Sen. Stabenow, Debbie [D-MI]", "Sen. Cruz, Ted [R-TX]"]

weighted <- graph_from_adjacency_matrix(P, mode = "undirected", weighted = TRUE, diag = FALSE) #Weighted projection
k_w <- round(mean(strength(weighted)),2) #Mean degree
m_w <- round(modularity(weighted, party, weights = E(weighted)$weight),2) #Modularity

backbone_fdsm <- fdsm(senate, alpha = 0.05, class = "igraph", narrative = TRUE,trials=10000) #Disparity Backbone
k_d <- round(mean(degree(disparity)),2) #Mean degree
m_d <- round(modularity(disparity, party),2) #Modularity

backbone_sdsm <- sdsm(senate, alpha = 0.05, class = "igraph", narrative = TRUE) #SDSM Backbone
k_b <- round(mean(degree(backbone_sdsm)),2) #Mean degree
m_b <- round(modularity(backbone_sdsm, party),2) #Modularity

pdf("Figures/bipartite_empirical.pdf", height = 2, width = 6) #Plot in manuscript
color <- party
color[which(color==1)] <- rgb(1,0,0,.5)
color[which(color==2)] <- rgb(0,0,1,.5)
color[which(color==3)] <- rgb(0,1,0,.5)
par(mfrow=c(1,3), mar = c(0, 0, 3, 0)) 
plot(weighted, vertex.label = NA, vertex.color = color, vertex.frame.color = NA, vertex.size = 10, edge.color = rgb(0,0,0,.25), edge.width = (E(weighted)$weight/100)^2)
title("Weighted Original", line = 1.5)
title(TeX(paste0("$\\langle\\textit{k}\\rangle$ = ", k_w, ", \\textit{Q} = ", m_w)), line = .5)
plot(backbone_fdsm, vertex.label = NA, vertex.color = color, vertex.frame.color = NA, vertex.size = 10, edge.color = rgb(0,0,0,.25))
title("FDSM Backbone", line = 1.5)
title(TeX(paste0("$\\langle\\textit{k}\\rangle$ = ", k_d, ", \\textit{Q} = ", m_d)), line = .5)
plot(backbone_sdsm, vertex.label = NA, vertex.color = color, vertex.frame.color = NA, vertex.size = 10, edge.color = rgb(0,0,0,.25))
title("SDSM Backbone", line = 1.5)
title(TeX(paste0("$\\langle\\textit{k}\\rangle$ = ", k_b, ", \\textit{Q} = ", m_b)), line = .5)
dev.off()

#### Obtaining p-values, using backbone.extract() ####
backbone <- fdsm(senate, alpha = NULL, trials = 1000)

bb_net001 <- backbone.extract(backbone, alpha = 0.001)
bb_net01 <- backbone.extract(backbone, alpha = 0.01)



