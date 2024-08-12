# Read in adjacency matrices
## read.csv creates a data frame object from a CSV file
## Need to indicate that there's no header row in the CSV
advice <- read.csv("Advice.csv", header=F)

reportsto <- read.csv("ReportsTo.csv", header = F)

# Read in vertex attribute data
attributes <- read.csv("KrackhardtVLD.csv")



# Quadratic Assignment Procedure (QAP)
library(sna)

# Assuming the dataframe 'attributes' has columns 'Age' and 'Tenure'

# Create Distance Matrix for Age homophily
ageDist <- as.matrix(dist(attributes$Age))

# Create Sender covariate for Age
# Building matrix column by column
# element ij is i's Age value
ageSend <- matrix(attributes$Age, nrow(attributes), nrow(attributes), byrow = FALSE)

# Create Receiver covariate for Age
# Building matrix row by row
# element ij is j's Age value
ageRec <- matrix(attributes$Age, nrow(attributes), nrow(attributes), byrow = TRUE)

# Create Distance Matrix for Tenure homophily
tenureDist <- as.matrix(dist(attributes$Tenure))

# Create Sender covariate for Tenure
# Building matrix column by column
# element ij is i's Tenure value
tenureSend <- matrix(attributes$Tenure, nrow(attributes), nrow(attributes), byrow = FALSE)

# Create Receiver covariate for Tenure
# Building matrix row by row
# element ij is j's Tenure value
tenureRec <- matrix(attributes$Tenure, nrow(attributes), nrow(attributes), byrow = TRUE)

# for QAP, need to make a list of covariates
covlist <- list(as.matrix(reportsto),
                tenureDist,
                tenureSend,
                tenureRec,
                ageDist,
                ageSend,
                ageRec)

set.seed(5)
qap_res <- netlogit(as.matrix(advice),covlist,reps=200) # reps should be higher
qap_res

# Read in the library required for ERGM analysis
library(ergm)

# Use the advice network dataset to create network object
adviceNet <- network(as.matrix(advice))

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
spec2 <- ergm(adviceNet~edges+mutual+ostar(2:3)+edgecov("reportsto")+nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")+nodeicov("Age")+nodeocov("Age")+absdiff("Age"), control=control.ergm(MCMC.samplesize=2000,MCMLE.maxit=20))
# See if the MCMC converged
mcmc.diagnostics(spec2)
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
spec4 <- ergm(adviceNet~edges+mutual+ostar(2:3)+gwesp(0,fixed=T)+edgecov("reportsto")+nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")+nodeicov("Age")+nodeocov("Age")+absdiff("Age"), control=control.ergm(MCMC.burnin=20000,MCMC.samplesize=4000,MCMLE.maxit=10))
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

### See which model fits best
BIC(spec0,spec1,spec2,spec4)

### Consider whether ERGM changes logit-based results
summary(spec0)
summary(spec2)

# Latent Space Modeling using the latentnet package
library(latentnet)

# latent sapce model with 2d euclidean space
set.seed(5)
spec5 <- ergmm(adviceNet~euclidean(d=2)+edgecov("reportsto")
               +nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")
               +nodeicov("Age")+nodeocov("Age")+absdiff("Age"))
gf5 <- gof(spec5)
# Make a panel of four plots
par(mfrow=c(2,2))
# plot goodness of fit results
plot(gf5)
summary(spec5)

# latent sapce model with 2d euclidean space
set.seed(5)
spec6 <- ergmm(adviceNet~bilinear(d=2)+edgecov("reportsto")
               +nodeicov("Tenure")+nodeocov("Tenure")+absdiff("Tenure")
               +nodeicov("Age")+nodeocov("Age")+absdiff("Age"))
gf6 <- gof(spec6)
# Make a panel of four plots
par(mfrow=c(2,2))
# plot goodness of fit results
plot(gf6)
summary(spec6)

# what if we really wanted to know about the transitivity? Let's use CUG for that
# first controlling just for graph density
cug_edges <- cug.test(adviceNet,
                        gtrans,
                        mode="digraph",
                        cmode="edges")
cug_edges

# now also controlling for the level of transitivity
cug_dyad <- cug.test(adviceNet,
                        gtrans,
                        mode="digraph",
                        cmode="dyad.census")
cug_dyad

# independent exercise
# Try to think of another ERGM term to include (?ergm.terms)
# Test for a different structural feature with CUG



