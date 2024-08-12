##########################
#  Community Detection   #
##########################

# Creating Network Objects: Defense Pacts (edgelist) (2000)
# gathered from the MIDs project
# published in 

# Cranmer, Skyler J., Bruce A. Desmarais, and Justin H. Kirkland. "Toward a network theory of alliance formation." 
# International Interactions 38, no. 3 (2012): 295-324.

# Read in vertex dataset
allyV <- read.csv("allyVLD.csv",stringsAsFactors=F)

# Read in edgelist
allyEL <- read.csv("allyEL.csv", stringsAsFactors=F)

# Read in contiguity
contig <- read.csv("contiguity.csv",stringsAsFactors=F,row.names=1)

require(network)
# (1) Initialize network
# store number of vertices
n <- nrow(allyV)
AllyNet <- network.initialize(n,dir=F)

# (2) Set vertex labels
network.vertex.names(AllyNet)  <- allyV$stateabb

# (3) Add in the edges
# Note, edgelist must match vertex labels
AllyNet[as.matrix(allyEL)]  <- 1

# (4) Store country code attribute
set.vertex.attribute(x=AllyNet,             # Network in which to store
                     "ccode",            # What to name the attribute
                     allyV$ccode)            # Values to put in

# (5) Store year attribute
set.vertex.attribute(AllyNet,"created",allyV$styear)

# (6) Store network attribute
set.network.attribute(AllyNet,"contiguous",as.matrix(contig))

# igraph is a great all-around networks package
# best option for community detection algorithms
library(igraph,quietly=T)

# Convert into a graph
allygr <- graph.adjacency(AllyNet[,],mode="undirected")

# find the best algorithm
comm_fg <- cluster_fast_greedy(allygr)
comm_le <- cluster_leading_eigen(allygr)
comm_wt <- cluster_walktrap(allygr)
comm_eb <- cluster_edge_betweenness(allygr)

modularity(comm_fg)
modularity(comm_le)
modularity(comm_wt)
modularity(comm_eb)

mem <- comm_wt$membership

# Check number of communities
max(mem)

# Number of nodes in each community
cbind(table(mem))


# Get memberships and plot
detach("package:igraph")
set.seed(567)
ccols <- sample(colors(), # 
                max(mem))
## Now plot
# Simple plot
plot(AllyNet,displaylabels=T,vertex.col=ccols[mem],label.cex=.5,edge.col=rgb(150,150,150,100,maxColorValue=255),displayisolates=F)

# add memberships to vertex-level data
allyV$community_membership <- mem

# save updated data
write.csv(allyV,file="./saved_objects/updated_state_vld.csv",row.names=F)

# check if older countries are in larger communities
# first need to create a variable that counts the size of the community
# luckily, community numbers match integers 1-63
c_size <- table(mem)[mem]
# double check that that worked
cbind(mem,c_size)
table(mem)

cor.test(c_size,allyV$styear)
# negative relationship means older states are in larger communities

# check if that result is robust to using a different algorithm
mem_eb <- comm_eb$membership
c_size_eb <- table(mem_eb)[mem_eb]
cor.test(c_size_eb,allyV$styear)

# Why do country codes vary significantly by community?
boxplot(allyV$ccode[allyV$community_membership <= 6] ~ 
          allyV$community_membership[allyV$community_membership <= 6])

# independent exercises, time permitting

# 1. Look through the igraph package documentation and find some other 
# community detection functions.
# See if you can find one that produces higher-modularity communities

# 2. Do community detection with the advice network.






