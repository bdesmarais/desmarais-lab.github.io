
# # 1. Introduction
# This tutorial provides an introduction to network analysis using R. We will cover the basics of network terminology and how to perform various network analyses using R packages.

# ## 1.1 What is a Network?
# A network is a collection of nodes (also called vertices) and the connections between them (called edges or links). Networks can represent many different types of relationships, such as social networks, communication networks, and transportation networks.

# ## 1.2 Why Use Network Analysis?
# Network analysis allows us to understand the structure and behavior of complex systems by examining the relationships between their components. It can reveal patterns and insights that are not apparent from analyzing individual components in isolation.

# ## 1.3 Getting Started with R
# Before we begin, make sure you have R and RStudio installed on your computer. You will also need to install the following R packages: igraph, sna, and ergm.

# ```{r setup, include=FALSE}
# knitr::opts_chunk$set(echo = TRUE)
# ```

# Load necessary libraries
library(igraph)
library(sna)
library(ergm)

# ## 2. Basic Network Analysis

# ### 2.1 Creating a Network
# You can create a network in R using the `graph_from_edgelist` or `graph_from_data_frame` functions from the igraph package.

# ```{r}
# Create an example network
edges <- matrix(c(1, 2, 2, 3, 3, 4, 4, 1), byrow = TRUE, ncol = 2)
g <- graph_from_edgelist(edges, directed = TRUE)
# Plot the network
plot(g)
# ```

# ### 2.2 Basic Network Measures
# We can calculate various network measures such as degree, betweenness, and closeness centrality using igraph functions.

# ```{r}
# Calculate degree centrality
degree_centrality <- degree(g)
print(degree_centrality)

# Calculate betweenness centrality
betweenness_centrality <- betweenness(g)
print(betweenness_centrality)

# Calculate closeness centrality
closeness_centrality <- closeness(g)
print(closeness_centrality)
# ```

# ### 2.3 Visualizing Networks
# Network visualization is an important part of network analysis. We can customize the appearance of network plots using various igraph functions.

# ```{r}
# Customize the network plot
plot(g, vertex.size = degree_centrality * 10, vertex.label = V(g)$name,
     edge.arrow.size = 0.5, edge.curved = 0.2)
# ```

# ### 2.4 Advanced Network Analysis
# For more advanced network analysis, we can use the sna and ergm packages. These packages provide functions for analyzing network structure and modeling network formation processes.

# ```{r}
# Load the sna and ergm packages
library(sna)
library(ergm)
# ```

# ### 2.5 Saving and Loading Networks
# We can save and load networks using the save and load functions in R.

# ```{r}
# Save the network to a file
save(g, file = "network.RData")

# Load the network from a file
load("network.RData")
# ```

# ## 3. Introduction to Networks

# ### 3.1 Network Terminology and the Basics
# * Units in the network: **Nodes, actors, or vertices**
# * Relationships between nodes: **edges, links, or ties**
# * Pairs of actors: **Dyads**
# * Direction: **Directed vs. Undirected (digraph vs. graph)**
# * Tie value: **Dichotomous/Binary, Valued/Weighted**
# * Ties to Self: **Self-loops**

# Analysis Content
# set the working directory
setwd("~/Dropbox/professional/Teaching/Consulting/UNCDataMatters/DataMattersMaterials2015/KrackhardtManagerData/")

# Read in adjacency matrices
## read.csv creates a data frame object from a CSV file
## Need to indicate that there's no header row in the CSV
advice <- read.csv("Advice.csv", header=F)

reportsto <- read.csv("ReportsTo.csv", header = F)

# Read in vertex attribute data
attributes <- read.csv("KrackhardtVLD.csv")

# Read in the library required for ERGM analysis
library(ergm)

# Use the advice network dataset to create network object
adviceNet <- network(advice)

# Add the vertex attributes into the network
set.vertex.attribute(adviceNet,names(attributes),attributes)

# Add the organizational chart as an edge variable
set.network.attribute(adviceNet,"reportsto",as.matrix(reportsto))

## Just basic covariate model
spec0 <- ergm(adviceNet~edges+edgecov("reportsto")+nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")+nodeicov("Age")+nodeocov("Age")+absdiff("Age"))
# See if the estimates are reliable
# How does it fit?
gf0 <- gof(spec0)
# Make a panel of four plots
par(mfrow=c(2,2))
# plot goodness of fit results
plot(gf0)
# Check out results
summary(spec0)


# Adding reciprocity
# Set the seed for replication
set.seed(5)
spec1 <-ergm(adviceNet~edges+mutual+edgecov("reportsto")+nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")+nodeicov("Age")+nodeocov("Age")+absdiff("Age"),control=control.ergm(MCMC.samplesize=2000,MCMLE.maxit=10))
# See if the MCMLE Converged
mcmc.diagnostics(spec1)
# See if its degenerate
gf.degeneracy <- gof(spec1, GOF=~model)
summary(gf.degeneracy)
# How does it fit?
gf1 <- gof(spec1)
# Make a panel of four plots
par(mfrow=c(2,2))
# plot goodness of fit results
plot(gf1)
# Check out results
summary(spec1)


# Account for varied activity
# Set the seed for replication
set.seed(5)
spec2 <- ergm(adviceNet~edges+mutual+ostar(2:3)+edgecov("reportsto")+nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")+nodeicov("Age")+nodeocov("Age")+absdiff("Age"), control=control.ergm(MCMC.samplesize=2000,MCMLE.maxit=10))
# See if the MCMC converged
mcmc.diagnostics(spec2)
# Need longer burnin to converge
spec2 <- ergm(adviceNet~edges+mutual+ostar(2:3)+edgecov("reportsto")+nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")+nodeicov("Age")+nodeocov("Age")+absdiff("Age"), control=control.ergm(MCMC.burnin=20000,MCMC.samplesize=4000,MCMLE.maxit=10))
# See if the MCMC converged
mcmc.diagnostics(spec2)
# See if its degenerate
gf.degeneracy <- gof(spec2, GOF=~model)
summary(gf.degeneracy)
# How does it fit?
gf2 <- gof(spec2)
# Make a panel of four plots
par(mfrow=c(2,2))
# plot goodness of fit results
plot(gf2)
# Check out results
summary(spec2)


# Now adding transitivity?
# Set the seed for replication
set.seed(5)
spec3 <- ergm(adviceNet~edges+mutual+ostar(2:3)+transitive+edgecov("reportsto")+nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")+nodeicov("Age")+nodeocov("Age")+absdiff("Age"), control=control.ergm(MCMC.samplesize=2000,MCMLE.maxit=10))


# Use GWESP Instead
# Set the seed for replication
set.seed(5)
spec4 <- ergm(adviceNet~edges+mutual+ostar(2:3)+gwesp(1)+edgecov("reportsto")+nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")+nodeicov("Age")+nodeocov("Age")+absdiff("Age"), control=control.ergm(MCMC.burnin=20000,MCMC.samplesize=4000,MCMLE.maxit=10))
# See if the MCMC converged
mcmc.diagnostics(spec4)
# See if its degenerate
gf.degeneracy <- gof(spec4, GOF=~model)
summary(gf.degeneracy)
# How does it fit?
gf4 <- gof(spec4)
# Make a panel of four plots
par(mfrow=c(2,2))
# plot goodness of fit results
plot(gf4)
# Check out results
summary(spec4)

# Maybe we can use simple form of GWESP
# Set the seed for replication
set.seed(5)
spec5 <- ergm(adviceNet~edges+mutual+ostar(2:3)+gwesp(0,fixed=T)+edgecov("reportsto")+nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")+nodeicov("Age")+nodeocov("Age")+absdiff("Age"), control=control.ergm(MCMC.burnin=20000,MCMC.samplesize=4000,MCMLE.maxit=10))
# See if the MCMC converged
mcmc.diagnostics(spec5)
# See if its degenerate
gf.degeneracy <- gof(spec5, GOF=~model)
summary(gf.degeneracy)
# How does it fit?
gf5 <- gof(spec5)
# Make a panel of four plots
par(mfrow=c(2,2))
# plot goodness of fit results
plot(gf5)
# Check out results
summary(spec5)

### See which model fits best
BIC(spec0,spec1,spec2,spec4,spec5)
AIC(spec0,spec1,spec2,spec4,spec5)

### Consider whether ERGM changes logit-based results
summary(spec0)
summary(spec5)

## Are tenure and age the same thing?
plot(get.vertex.attribute(adviceNet,"Age"),get.vertex.attribute(adviceNet,"Tenure"))



# Quadratic Assignment Procedure (QAP)
library(sna)

# Create an example network and covariate matrix
set.seed(123)
net <- rgraph(10, tprob = 0.2, mode = "graph", diag = FALSE)
covariate <- matrix(rnorm(100), ncol = 10)

# Perform QAP correlation test
qap_test <- netlm(net, covariate)
summary(qap_test)

# Latent Space Modeling using the latentnet package
library(latentnet)

# Generate example data
set.seed(123)
network_data <- network(net, directed = FALSE)

# 2D Euclidean Latent Space Model
euclidean_model <- ergmm(network_data ~ edges + nodematch("covariate") + euclidean(d = 2))
summary(euclidean_model)
plot(euclidean_model)

# 2D Bilinear Latent Space Model
bilinear_model <- ergmm(network_data ~ edges + nodematch("covariate") + bilinear(d = 2))
summary(bilinear_model)
plot(bilinear_model)
