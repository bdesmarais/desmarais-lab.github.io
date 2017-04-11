# width = 750
# height = 107

library(sna)

setwd("/Users/brucedesmarais/Desktop/Research/PredictTerror/Data")
library(sna)

states <- read.csv("system2008.csv",stringsAsFactors=F)
iterate <- read.csv("IteratePredict.csv", stringsAsFactors=F)
locations <- c(read.csv("IterateLocations.csv",stringsAsFactors=F, header=F))[[1]]
LocNum <- as.numeric(substr(c(locations),1,3))
LocName <- substr(locations,4,nchar(locations))

# Non Cow States and Numbers (to be added to each state system)
# NATO, International Organizations, Unspecified Foreign Nations, irrelevant, unknown, Indeterminate African Nation, Indeterminate Arabs Palestine, Indeterminate Latin American Nation, Northern Ireland, Corsica, Puerto Rico, Kurdistan
nc <- c(995,996,997,998,999,599,667,99,204,219,6,646)


# Integrating Link prediction proximity stuff into TERGM
# Not just a problem of predicting the development of new links
# Many problems require the forecasting of a future instance of a network
# Network evolution, not link completion 
# Decision theory is easier given the probability of an attack
# Host of problems with covariates
## Collection costs
## Inter-temporal comparison
## Endogeneity
## Cannot Integrate different types of actors
## Variation in availability/quality

## First Cut
# Network of "terrorist nationality" -> "LocationEnd"
# Get statistics on the network from 1994-1998
# Train the Parameters on the network in 1999
# Predict the network in 2000


# Create List of networks and IDs
attacks <- list()
nodes <- list()
for(t in 1977:2002){
	cct <- subset(states, states$Year ==t)$CCode
	cct <- c(cct,nc)
	netmat <- matrix(0,length(cct),length(cct))
	subiter <- subset(iterate, iterate$Year == t)
	send <- subiter$tnat1
	recv <- subiter$LocationEnd
	ws2 <- which(subiter$tnat2 < 998)
	send <- c(send,subiter$tnat2[ws2])
	recv <- c(recv,subiter$LocationEnd[ws2])
	ws3 <- which(subiter$tnat3 < 998)
	send <- c(send,subiter$tnat3[ws3])
	recv <- c(recv,subiter$LocationEnd[ws3])
	ind1 <- match(send,cct)
	ind2 <- match(recv,cct)
	inds <- na.omit(cbind(ind1,ind2))
	netmat[inds] <- 1
	netmat[inds] <- 1
	attacks[[t-1976]] <- netmat
	nodes[[t-1976]] <- cbind(cct,LocName[match(cct,LocNum)])
}


# 25 = 2001
set.seed(500000)
xy <- gplot(attacks[[25]],displayisolates=F,label=nodes[[25]][,2],label.cex=.75)

xyp <- xy[-isolates(attacks[[25]]),]

xyp[,1] <- -xyp[,1]

cols <- paste("grey",93-round((1:21)^2.5/22),sep="")
cols <- cols[rank(xyp[,1])]

nv <- .85

noise <- NULL
colrs <- NULL
for(i in 1:nrow(xyp)){
	xi <- rnorm(25,mean=xyp[i,1],sd=nv)
	yi <- rnorm(25,mean=xyp[i,2],sd=nv)
	noise <- rbind(noise,cbind(xi,yi))
	colrs <- c(colrs,rep(cols[i],25))
}

net <- attacks[[25]][-isolates(attacks[[25]]),-isolates(attacks[[25]])]

ecol <- net

for(i in 1:nrow(net)){
	for(j in 1:nrow(net)){
		ecol[i,j] <- cols[i]
	}
}

jpeg(filename = "/Users/brucedesmarais/Desktop/CVandWebsite/Website/ExpWeb/top.jpg",width=750,height=107,quality=100)
par(mar=c(.01,.01,.01,.01))

plot(noise,col=colrs,pch=20,cex=.75, bty="n", xlim=c(-40,27),xaxt="n",yaxt="n")	

gplot(net,new=F,coord=xyp,vertex.col=cols,vertex.border=cols,edge.col=ecol, edge.lwd=.9)
text(-43,37,"Bruce A. Desmarais",cex=3,pos=4)
text(-42.5,28.5,"Political Science",cex=2,pos=4,col="grey35")
text(-42.5,21.5,"Computational Social Science",cex=2,pos=4,col="grey35")
dev.off()

##### Navigation Bar Graphic #######

# <area shape="rectangle" coords="300,0,359,26" alt="home" href="index.html">
# <area shape="rectangle" coords="372,0,444,26" alt="people" href="people/index.html">
# <area shape="rectangle" coords="457,0,534,26" alt="projects" href="projects/index.html">
# <area shape="rectangle" coords="547,0,657,26" alt="publications" href="publications/index.html">
# <area shape="rectangle" coords="669,0,750,26" alt="contact" href="contact/index.html">


jpeg(filename = "/Users/brucedesmarais/Desktop/CVandWebsite/Website/ExpWeb/nav_bar.jpg",width=750,height=26,quality=100)
par(mar=c(0,0,0,0),bg="#D8D8D8")
plot(1:5,1:5,col="#C8C8C8", bty="n", xlim=c(0,760),ylim=c(0,26),xaxt="n",yaxt="n")
text(300,12,pos=4,labels="news | publications | presentations | teaching | vita",cex=1.6)
dev.off()	

# Find the points at which to break the rectangles
links <- "news | publications | presentations | teaching | vita"
clink <- strsplit(links,split="")
posit <- which(clink[[1]]=="|")
ptseq <- seq(300,745,length=length(clink[[1]]))













