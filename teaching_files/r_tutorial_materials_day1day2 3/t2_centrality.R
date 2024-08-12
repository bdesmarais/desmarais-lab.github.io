# install all of the R packages we will need for today
# install.packages(c("igraph","statnet","GGally"))

#################
#   Centrality  #
#################

# Read in data
advice <- read.csv("Advice.csv", header=F)

reportsto <- read.csv("ReportsTo.csv", header = F)

# Read in vertex attribute data
attributes <- read.csv("KrackhardtVLD.csv")
# Read in the library for network analysis
library(network)
adviceNet <- network(as.matrix(advice))

# Add the vertex attributes into the network
set.vertex.attribute(adviceNet,names(attributes),attributes)



# Creating Network Objects: Defense Pacts (edgelist) (2000)
# gathered from the MIDs project
# published in 

# Cranmer, Skyler J., Bruce A. Desmarais, and Justin H. Kirkland. "Toward a network theory of alliance formation." 
# International Interactions 38, no. 3 (2012): 295-324.

# Read in vertex dataset
allyV <- read.csv("allyVLD.csv", # file to read
                  stringsAsFactors=F) # indicates to not convert to factor vars

# Read in edgelist
allyEL <- read.csv("allyEL.csv", stringsAsFactors=F)

# Read in contiguity
contig <- read.csv("contiguity.csv",stringsAsFactors=F,row.names=1)

require(network)
# (1) Initialize network
# store number of vertices
n <- nrow(allyV)
AllyNet <- network.initialize(n, # number of nodes
                              dir=F) # network is not directed

# (2) Set vertex labels
network.vertex.names(AllyNet)  <- allyV$stateabb 

# (3) Add in the edges
# Note, edgelist must match vertex labels
AllyNet[as.matrix(allyEL)]  <- 1

# (4) Store country code attribute
set.vertex.attribute(x=AllyNet,   # Network in which to store
                     "ccode",     # What to name the attribute
                     allyV$ccode) # Values to put in

# (5) Store year attribute
set.vertex.attribute(AllyNet,"created",allyV$styear)

# (6) Store network attribute
set.network.attribute(AllyNet,"contiguous",as.matrix(contig))


require(sna)
# (in-) Degree Centrality is the number of in-connections by node
dc <- degree(adviceNet, cmode="indegree")

# Store in vertex level data frame
attributes$dc <- dc

# Plot degree centrality against tenure
## Make a simple scatter plot
plot(attributes$Tenure,attributes$dc)
## Add a trend (i.e., regression) line
abline(lm(attributes$dc ~ attributes$Tenure))

# Plot network with node size proportional to Degree Centrality
## First normalize degree 
ndc <- dc/max(dc)
## Set random number seed so the plot is replicable
set.seed(5)
## Colors based on department
five_colors <- c("white","lightblue","yellow","azure3","brown1")
vert.cols <- five_colors[attributes$Department+1]
## Now plot
plot(adviceNet,displaylabels=T,label=get.vertex.attribute(adviceNet,"Level"),vertex.cex=3*ndc,label.cex=1,edge.col=rgb(150,150,150,100,maxColorValue=255),label.pos=5,vertex.col=vert.cols)


# Eigenvector Centrality Recursively Considers Neighbors' Centrality
# need to create an undirected version of the network
advice_undirected <- network(as.matrix(advice),dir=F)
# now calculate eigenvector centrality
ec <- evcent(advice_undirected)

# Store in vertex level data frame
attributes$ec <- ec

# Plot eigenvector centrality against tenure
## Make a simple scatter plot
plot(attributes$Tenure,attributes$ec)
## Add a trend (i.e., regression) line
abline(lm(attributes$ec ~ attributes$Tenure))

# Plot network with node size proportional to eigenvector centrality
## First normalize
nec <- ec/max(ec)
## Set random number seed so the plot is replicable
set.seed(5)
## Now plot
plot(adviceNet,displaylabels=T,label=get.vertex.attribute(adviceNet,"Level"),vertex.cex=3*nec,label.cex=1,edge.col=rgb(150,150,150,100,maxColorValue=255),label.pos=5,vertex.col=vert.cols)

# Betweenness Centrality Considers unlikely connections
# Proportion of shortest paths that pass through a vertex
bc <- betweenness(adviceNet,rescale=T)

# Store in vertex level data frame
attributes$bc <- bc

# Plot betweenness centrality against tenure
## Make a simple scatter plot
plot(attributes$Tenure,attributes$bc)
## Add a trend (i.e., regression) line
abline(lm(attributes$bc ~ attributes$Tenure))

# Plot network with node size proportional to betweenness centrality
## First normalize
nbc <- bc/max(bc)
## Set random number seed so the plot is replicable
set.seed(5)
## Now plot
plot(adviceNet,displaylabels=T,label=get.vertex.attribute(adviceNet,"Level"),vertex.cex=3*nbc,label.cex=1,edge.col=rgb(150,150,150,100,maxColorValue=255),label.pos=5,vertex.col=vert.cols)

# Closeness Centrality Considers global reach across the network
# Inverse average geodesic distance between i and all other nodes
cc <- closeness(adviceNet)

# Store in vertex level data frame
attributes$cc <- cc

# Plot betweenness centrality against tenure
## Make a simple scatter plot
plot(attributes$Tenure,attributes$cc)
## Add a trend (i.e., regression) line
abline(lm(attributes$cc ~ attributes$Tenure))

# Plot network with node size proportional to closeness centrality
## First normalize
ncc <- cc/max(cc)
## Set random number seed so the plot is replicable
set.seed(5)
## Now plot
plot(adviceNet,displaylabels=T,label=get.vertex.attribute(adviceNet,"Level"),vertex.cex=3*ncc,label.cex=1,edge.col=rgb(150,150,150,100,maxColorValue=255),label.pos=5,vertex.col=vert.cols)


# view relationships among all of the centrality measures
centrality_measures <- data.frame(dc,ec,bc,cc)
plot(centrality_measures)

# correlations between centrality measures
cor(centrality_measures)

# independent exercises, time permitting
# calculate centrality measures for the alliance network
# what's going on with closeness centrality? See the arg cmode="suminvundir".
# How are the relationships among them?



