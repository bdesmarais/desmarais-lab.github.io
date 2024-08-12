# Adapted from https://towardsdatascience.com/graph-neural-networks-in-python-c310c7c18c83

# To operate on graphs in Python, we will use the highly popular networkx library [1]. We start by creating an empty directed graph H:
import matplotlib.pyplot as plt
import networkx as nx
import torch as torch
import pandas as pd
import dgl.function as fn
H = nx.DiGraph()


# We will then add 4 nodes to the graph. Each node has 2 features attached to it, color and size. It is common for graphs in machine learning problems to have nodes with features, such as the name or age of a person in a social network, which can then be used by the model to infer complex relations and make predictions. Networkx comes with a built in utility function for filling a graph with nodes as a list, in addition to their features:
H.add_nodes_from([
  (0, {"color": "gray", "size": 450}),
  (1, {"color": "yellow", "size": 700}),
  (2, {"color": "red", "size": 250}),
  (3, {"color": "pink", "size": 500})
])
for node in H.nodes(data=True):
  print(node)

# An edge in the graph is defined as a tuple containing the origin and target node, so for example the edge (2, 3) connects node 2 to node 3. Since we have a directed graph, there can also be an edge (3, 2) which points in the opposite direction. Multiple edges can be added to the graph as part of a list in a similar manner as nodes can:

H.add_edges_from([
  (0, 1),
  (1, 2),
  (2, 0),
  (2, 3),
  (3, 2)
])
print(H.edges())

# Now that we have created a graph, let’s define a function to display some information about it. We validate that the graph is indeed directed and that it has the correct number of nodes as well as edges.

def print_graph_info(graph):
  print("Directed graph:", graph.is_directed())
  print("Number of nodes:", graph.number_of_nodes())
  print("Number of edges:", graph.number_of_edges())
print_graph_info(H)

# It can also be very helpful to plot a graph that you are working with. This can be achieved using nx.draw. We use the nodes’ features to color each node and give each of them their own size in the plot. Since node attributes come as dictionaries, and the draw function only accepts lists we will have to convert them first. The resulting graph looks like it is supposed to with 4 nodes, 5 edges and the correct node features.

node_colors = nx.get_node_attributes(H, "color").values()
colors = list(node_colors)
node_sizes = nx.get_node_attributes(H, "size").values()
sizes = list(node_sizes)
plt.figure()
nx.draw(H, with_labels=True, node_color=colors, node_size=sizes)
plt.show()

edges = pd.read_csv('girls_el.csv')
nodes = pd.read_csv('nodes.csv')
am = pd.read_csv('girls_am.csv',index_col=0)

nodes = pd.read_csv('nodes.csv')

smo = nodes['smoke'].to_numpy()
dru = nodes['drugs'].to_numpy()
sen = edges['sen'].to_numpy()
rec = edges['rec'].to_numpy()

y = torch.tensor(smo,dtype=torch.float)
x = torch.tensor(dru,dtype=torch.float)
edge_index = torch.tensor([sen,rec],dtype=torch.long)

from torch_geometric.data import Data
data = Data(y=y, x=x,edge_index=edge_index,num_nodes=50)

from torch_geometric.transforms import RandomNodeSplit as masking
msk=masking(split="train_rest", num_splits = 1, num_val = 0.3, num_test= 0.6)
data = msk(data)

from networkx import from_pandas_dataframe
G = nx.from_pandas_adjacency(am,create_using=nx.DiGraph)

nx.set_node_attributes(G, pd.Series(nodes.smoke, index=nodes.node).to_dict(), 'x')
nx.set_node_attributes(G, pd.Series(nodes.drugs, index=nodes.node).to_dict(), 'y')

from torch_geometric.utils import from_networkx
dataset = from_networkx(G,group_node_attrs=['x','y'])

# The Karate Club dataset is available through PyTorch Geometric (PyG ) [3]. The PyG library contains all sorts of methods for deep learning on graphs and other irregular structures. We begin by inspecting some of the properties of the dataset. It seems to only contain one graph, which is expected since it depicts one club. Furthermore, each node in the dataset is assigned a 34 dimensional feature vector that uniquely represents every node. Every member of the club is part of one of 4 factions, or classes in machine learning terms.
from torch_geometric.datasets import KarateClub
dataset = KarateClub()
print("Dataset:", dataset)
print("# Graphs:", len(dataset))
print("# Features:", dataset.num_features)
print("# Classes:", dataset.num_classes)

# We can further explore the only graph in the dataset. We see that the graph is undirected, and it has 34 nodes, each with 34 features as mentioned before. The edges are represented as tuples, and there are 156 of them. However, in PyG undirected edges are represented as two tuples, one for each direction, also known as bi-diretional, meaning that there are 78 unique edges in the Karate Club graph. PyG only include entries in A which are non-zero, which is why edges are represented like this. This type of representation is known as coordinate format, which is commonly used for sparse matrices. Each node has a label, y, that holds information about which class the corresponding node is part of. The data also contains a train_mask that has the indices of the nodes we know the ground truth labels for during training. There are 4 truth nodes, one for each faction, and the task at hand is then to infer the faction for the rest of the nodes.
data = dataset[0]
print(data)
print("Training nodes:", data.train_mask.sum().item())
print("Is directed:", data.is_directed())



# We convert the Karate Club Network to a Networkx graph, which allows us to use the nx.draw function to visualize it. The nodes are colored according to the class (or faction) they belong to.
from torch_geometric.utils import to_networkx
G = to_networkx(data, to_undirected=True)
plt.figure()
nx.draw(G, node_size=150)
plt.show()

# When training a model to perform node classification it can be referred to as semi-supervised machine learning, which is the general term used for models that combine labeled and unlabeled data during training. In the case of node classification we have access to all the nodes in the graph, even those belonging to the test set. The only information missing is the labels of the test nodes.

# Graph Convolutional Networks (GCNs) will be used to classify nodes in the test set. To give a brief theoretical introduction, a layer in a graph neural network can be written as a non-linear function f:
from torch.nn import Linear
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
  def __init__(self):
    super(GCN, self).__init__()
    torch.manual_seed(42)
    self.conv1 = GCNConv(data.num_features, 4)
    self.conv2 = GCNConv(4, 4)
    self.conv3 = GCNConv(4, 2)
    #self.classifier = Linear(2, dataset.num_classes)
  def forward(self, x, edge_index):
    h = self.conv1(x, edge_index)
    h = h.tanh()
    h = self.conv2(h, edge_index)
    h = h.tanh()
    h = self.conv3(h, edge_index)
    h = h.tanh()
    #out = self.classifier(h)
    return out, h
    #return h
model = GCN()
print(model)

# We use cross-entropy as loss functions since it is well suited for multi-class classification problems, and initialize Adam as a stochastic gradient optimizer. We create a standard PyTorch training loop, and let it run for 300 epochs. Note that while all nodes do indeed get updates to their node embeddings, the loss is only calculated for nodes in the training set. The loss is drastically decreased during training, meaning that the classification works well. The 2 dimensional embeddings from the last GCN layer are stored as a list so that we can animate the evolution of the embeddings during training, giving some insight into the latent space of the model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
def train(data):
  optimizer.zero_grad()
  optimizer.zero_grad()
  h = model(data.x, data.edge_index)
  loss = criterion(target=data.y[data.train_mask])
  loss.backward()
  optimizer.step()
  return loss, h
epochs = range(1, 301)
losses = []
embeddings = []
for epoch in epochs:
  loss, h = train(data)
  losses.append(loss)
  embeddings.append(h)
  print(f"Epoch: {epoch}\tLoss: {loss:.4f}")
  
print(embeddings[301])
  

# Matplotlib can be used to animate a scatter plot of the node embeddings where every dot is colored according to the faction they belong to. For every frame we display the epoch in addition to the training loss value for that epoch. Finally, the animation is converted to a GIF which is visible below.
import matplotlib.animation as animation
def animate(i):
  ax.clear()
  h = embeddings[i]
  h = h.detach().numpy()
  ax.scatter(h[:, 0], h[:, 1], c=data.y, s=100)
  ax.set_title(f'Epoch: {epochs[i]}, Loss: {losses[i].item():.4f}')
  ax.set_xlim([-1.1, 1.1])
  ax.set_ylim([-1.1, 1.1])
fig = plt.figure(figsize=(6, 6))
ax = plt.axes()
anim = animation.FuncAnimation(fig, animate, frames=epochs)
plt.show()
gif_writer = animation.PillowWriter(fps=20)
anim.save('embeddings.gif', writer=gif_writer)


  


import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

# Define a custom dataset class
class CustomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        
    def process(self):
        # Read adjacency matrix
        adjacency_df = pd.read_csv('girls_am.csv')
        adjacency_matrix = torch.tensor(adjacency_df.values, dtype=torch.float)
        
        # Read node-level dataset
        nodes_df = pd.read_csv('nodes.csv')
        node_features = torch.tensor(nodes_df['smoke'].values, dtype=torch.float).view(-1, 1)  # Smoke column
        
        # Create node indices
        node_indices = torch.arange(len(node_features), dtype=torch.long).view(-1, 1)
        
        # Create the graph data object
        data = Data(x=node_features, edge_index=None, edge_attr=adjacency_matrix, y=node_indices)
        
        # Save the graph data object
        torch.save(data, self.processed_paths[0])

    def len(self):
        return 1
    
    def get(self, idx):
        data = torch.load(self.processed_paths[0])
        return data

# Define the dataset root directory
dataset_root = './graph_dataset'

# Initialize the dataset and process the data
dataset = CustomDataset(dataset_root)
dataset.process()






###### DGL Tutorial #########
members = pd.read_csv("nodes.csv")
members.head()

interactions = pd.read_csv("girls_el.csv")
interactions.head()

import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch

import dgl
from dgl.data import DGLDataset


class exDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="ex_data")

    def process(self):
        nodes_data = pd.read_csv("nodes.csv")
        edges_data = pd.read_csv("girls_el.csv")
        node_features = torch.from_numpy(nodes_data["drugs"].to_numpy())
        node_labels = torch.from_numpy(
            nodes_data["smoke"].astype("category").cat.codes.to_numpy()
        )
        edges_src = torch.from_numpy(edges_data["sen"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["rec"].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=nodes_data.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


dataset = exDataset()
g = dataset[0]

print(graph)


import torch.nn as nn
import torch.nn.functional as F

# Define the message and reduce function
# NOTE: We ignore the GCN's normalization constant c_ij for this tutorial.
def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# Define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = inputs
        # trigger message passing on all edges
        g.send_and_recv(g.edges(), gcn_message,fn.copy_u('x', 'm'))
        # trigger aggregation at all nodes
        g.send_and_recv(g.nodes(), gcn_reduce,fn.sum('m', 'h'))
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        return self.linear(h)

# Define a 2-layer GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h
# The first layer transforms input features of size of 34 to a hidden size of 5.
# The second layer transforms the hidden layer and produces output features of
# size 2, corresponding to the two groups of the karate club.
net = GCN(34, 5, 2)

inputs = torch.eye(50)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(30):
    logits = net(g, inputs)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    
    

# now with nexgcn

import NexGCN.NexGCN as venom



  
import networkx as nx
from networkx import karate_club_graph, to_numpy_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten,Embedding,Dropout
from keras.models import Sequential, Model
from keras import initializers, regularizers,activations,constraints
import keras.backend as k
from tensorflow.keras.layers import Layer,Input
from keras.optimizers import Adam
import numpy as np
from networkx import to_numpy_matrix, degree_centrality, betweenness_centrality, shortest_path_length,in_degree_centrality,out_degree_centrality,eigenvector_centrality,katz_centrality,closeness_centrality
import matplotlib.pyplot as plt
import NexGCN.NexGCN as venom


Gr = nx.gnm_random_graph(70,140)

Gr = nx.gnm_random_graph(70,140)
exp=venom.ExperimentalGCN()
kernel=venom.feature_kernels()

#X=kernel.centrality_kernel(katz_centrality,Gr)
X=kernel.feature_random_weight_kernel(34,Gr)
#X=kernel.feature_distributions(np.random.poisson(4,9),Gr)

exp.create_network(Gr,X,True)
