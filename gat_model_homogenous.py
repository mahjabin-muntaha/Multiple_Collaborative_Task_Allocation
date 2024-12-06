import torch
import torch.nn as nn
import torch.nn.functional as F

import global_variables
from global_variables import device

"""
This neural network model is based on graph attention networks. 
"""


class GATQN_Homogenous(nn.Module):
    def __init__(self, embed_dim, node_attribute_size, edge_attribute_size):
        """
        Initialize the neural network
        :param embed_dim: the embedded dimension
        :param node_attribute_size: the dimension of the node attribute
        :param edge_attribute_size: the dimension of the edge attribute
        """
        super(GATQN_Homogenous, self).__init__()

        if global_variables.lrelu_neg_slope == 0:
            self.activate_fn = nn.ReLU()
        else:
            self.activate_fn = nn.LeakyReLU(negative_slope=global_variables.lrelu_neg_slope)
        if global_variables.mix_activate_tanh:
            self.mix_activate_fn = nn.Tanh()
        else:
            self.mix_activate_fn = self.activate_fn

        self.embed_dim = embed_dim
        self.node_attribute_size = node_attribute_size
        self.edge_attribute_size = edge_attribute_size

        self.mu_node = nn.Linear(self.node_attribute_size, self.embed_dim)
        self.mu_edge = nn.Linear(self.edge_attribute_size, self.embed_dim)

        self.mu_attention_node = nn.Linear(self.embed_dim * 3, self.embed_dim, bias=False)
        self.mu_attention_edge = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mlp_node = nn.Linear(self.embed_dim, 1, bias=False)
        self.mlp_edge = nn.Linear(self.embed_dim, 1, bias=False)
        self.mu_agg_node = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.final_mu_agg_node = nn.Linear(self.embed_dim * 2, self.embed_dim)

    def forward(self, adj, node_attribute, edge_attribute):
        batch_size = node_attribute.shape[0]  # [batch, nodes, node_feats]
        num_nodes = node_attribute.shape[1]
        adj = adj.unsqueeze(dim=-1)

        # initial feature transformation
        node_vector = self.activate_fn(self.mu_node(node_attribute))
        edge_vector = self.activate_fn(self.mu_edge(edge_attribute))

        # construct the attention factor, which includes the node vectors of two connected nodes and
        # the edge vector of the corresponding edge
        node_vector_extend_same = node_vector.unsqueeze(dim=2).expand(batch_size, num_nodes, num_nodes,
                                                                      self.embed_dim)
        node_vector_extend_diff = node_vector.unsqueeze(dim=1).expand(batch_size, num_nodes, num_nodes,
                                                                      self.embed_dim)
        attention_factor = torch.cat((node_vector_extend_same, edge_vector, node_vector_extend_diff), dim=-1)

        # measure the importance of nodes to nodes
        mu_attention_nodes = self.mix_activate_fn(self.mu_attention_node(attention_factor))
        mu_attention_nodes = adj * torch.exp(self.mlp_node(mu_attention_nodes))
        mu_attention_nodes = mu_attention_nodes / (torch.sum(mu_attention_nodes, dim=-2, keepdim=True).clamp(min=global_variables.eps_cnst))

        # measure the importance of edges
        mu_attention_edge = self.mix_activate_fn(self.mu_attention_edge(edge_vector))
        mu_attention_edge = adj * torch.exp(self.mlp_edge(mu_attention_edge))
        mu_attention_edge = mu_attention_edge / (torch.sum(mu_attention_edge, dim=-2, keepdim=True).clamp(min=global_variables.eps_cnst))

        # aggregate the information of adjacent nodes
        sum_adj_node = torch.matmul(mu_attention_nodes.squeeze(dim=-1), node_vector)

        # aggregate the information of adjacent edges
        sum_adj_edge = torch.sum(mu_attention_edge * edge_vector, dim=-2)

        # construct the new node vectors
        node_plus_edge = torch.cat((node_vector, sum_adj_node, sum_adj_edge), dim=-1)
        node_vector = self.activate_fn(self.mu_agg_node(node_plus_edge))

        # # construct the new edge vectors
        # edge_vector = self.activate_fn(self.mu_agg_edge(torch.cat((edge_vector, sum_adj_edge), dim=-1)))

        # highlight the importance of initial information in node vectors
        # node_vector_temp = torch.cat((node_attribute, node_vector), dim=-1)
        # node_vector = self.activate_fn(self.final_mu_agg_node(node_vector_temp))

        return node_vector
