library(ina)
data(ScottishSchool)
girls_el <- as.matrix(Girls,"edgelist")
girls_el[,1] <- girls_el[,1] - 1
girls_el[,2] <- girls_el[,2] - 1
colnames(girls_el) <- c("sen","rec")
write.csv(girls_el,"girls_el.csv",row.names=F)

library(network)

girls_am <- as.matrix(Girls,"adjacency")
write.csv(girls_am,"girls_am.csv")

smoke <- 1*(get.vertex.attribute(Girls,"smoke") > 1)
drugs <- get.vertex.attribute(Girls,"drugs")
node <- colnames(girls_am)

node_data <- data.frame(node,smoke,drugs) 

write.csv(node_data,"nodes.csv",row.names=F)

drugs_data <- data.frame(node,drugs) 

write.csv(drugs_data,"drugs.csv")


