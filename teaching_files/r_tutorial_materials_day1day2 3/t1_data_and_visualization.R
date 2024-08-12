# install all of the R packages we will need for today
# install.packages(c("igraph","statnet","GGally"))

# data for the first example comes from http://networkdata.ics.uci.edu/netdata/html/krackHighTech.html. 
# It is a famous network dataset of 21 managers at a high tech firm that includes networks of advice, where edge i->j indicates
# that i reported getting advice from j, and reporting structure where i->j indicates that
# i reports to j. There is also attribute data that includes Age, Tenure at the company, level in the 
# org chart, and a department indicator. This was published in 

# Krackhardt, David. Assessing the Political Landscape: Structure, Cognition, and Power in Organizations.
# Administrative Science Quarterly Vol. 35, No. 2 (Jun., 1990), pp. 342-369

# Read in adjacency matrices
## read.csv creates a data frame object from a CSV file
advice <- read.csv("Advice.csv", # file name
                   header=F) # no header row

reportsto <- read.csv("ReportsTo.csv", header = F)

# Read in vertex attribute data
attributes <- read.csv("KrackhardtVLD.csv")
# Read in the library for network analysis
library(network)
# advice is a data frame, but needs to be a matrix to use in network()
adviceNet <- network(as.matrix(advice))

# Add the vertex attributes into the network
set.vertex.attribute(adviceNet, # name of the network object
                     names(attributes),# names of the variables
                     attributes) # values of the variables

# Add the organizational chart as a network variable
set.network.attribute(adviceNet, # name of the network object
                      "reportsto", # what to call the network object
                      reportsto) # values for the network object

# Simple plot
## Set random number seed so the plot is replicable
set.seed(5)
## Plot the network
plot(adviceNet, # name of network object
     displaylabels=T, # print labels on nodes
     label=get.vertex.attribute(adviceNet,"Level"), # use attribute called level
     vertex.cex=2, # make nodes twice as large as the default
     edge.col=rgb(150,150,150,100,maxColorValue=255), # use even mix of R/G/B
     label.pos=5, # put labels in the center of nodes
     vertex.col="lightblue") # color them blue

# check out all the options
?plot.network


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

# Simple plot
set.seed(5)
plot(AllyNet,displaylabels=T,label.cex=.5,edge.col=rgb(150,150,150,100,maxColorValue=255))

# may want to take out the isolates
set.seed(5)
plot(AllyNet,displayisolates=F,displaylabels=T,label.cex=.5,edge.col=rgb(150,150,150,100,maxColorValue=255))

# save this plot
pdf("./saved_objects/alliance_net_plot.pdf") # open pdf file to which to write
set.seed(5)
plot(AllyNet,displayisolates=F,displaylabels=T,label.cex=.5,edge.col=rgb(150,150,150,100,maxColorValue=255))
dev.off() # stop writing to that file

# visualization using ggplot 
# ggplot is a high-powered language-within-a-language for visualization in R
library(GGally)

# need to make character attribute to generate legend
# make a character version of level
cLevel <- as.character(get.vertex.attribute(adviceNet,'Level'))

set.vertex.attribute(adviceNet, # network
                     "cLevel", # name of new attribute
                     cLevel) # value

set.seed(5)
ggnet2(adviceNet, color = "cLevel", legend.size = 12,arrow.size=14,color.legend="Level",
       color.palette = "Dark2")  +  # use ggplot style
  theme(legend.position = "bottom") 

# Independent exercises if we have time. 
# Plot the nodes in the advice network with vertices sized based on tenure.
# Plot the alliance network using the Kamada-Kawai layout algorithm





