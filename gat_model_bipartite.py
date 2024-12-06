import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from global_variables import device, lrelu_neg_slope, mix_activate_tanh, eps_cnst


class GATQN_Bipartite(nn.Module):
    def __init__(self, embed_dim, worker_attribute_size, task_attribute_size, edge_attribute_size):
        """
        This class represents a Graph Attention Network (GAT) model for bipartite graphs.

        Attributes:
            embed_dim (int): The embedded dimension.
            worker_attribute_size (int): The dimension of the worker node attribute.
            task_attribute_size (int): The dimension of the task node attribute.
            edge_attribute_size (int): The dimension of the edge attribute.
        """
        super(GATQN_Bipartite, self).__init__()

        if lrelu_neg_slope == 0:
            self.activate_fn = nn.ReLU()
        else:
            self.activate_fn = nn.LeakyReLU(negative_slope=lrelu_neg_slope)
        if mix_activate_tanh:
            self.mix_activate_fn = nn.Tanh()
        else:
            self.mix_activate_fn = self.activate_fn

        self.embed_dim = embed_dim
        self.worker_attribute_size = worker_attribute_size
        self.task_attribute_size = task_attribute_size
        self.edge_attribute_size = edge_attribute_size

        self.mu_worker = torch.nn.Linear(self.worker_attribute_size, self.embed_dim)
        self.mu_task = torch.nn.Linear(self.task_attribute_size, self.embed_dim)
        self.mu_edge = torch.nn.Linear(self.edge_attribute_size, self.embed_dim)

        self.mu_agg_worker = torch.nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.mu_agg_task = torch.nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.mu_attention_worker = torch.nn.Linear(self.embed_dim * 3, self.embed_dim, bias=False)
        self.mu_attention_task = torch.nn.Linear(self.embed_dim * 3, self.embed_dim, bias=False)
        self.mu_attention_edge = torch.nn.Linear(self.embed_dim * 3, self.embed_dim, bias=False)
        self.mlp_worker = torch.nn.Linear(self.embed_dim, 1, bias=False)
        self.mlp_task = torch.nn.Linear(self.embed_dim, 1, bias=False)
        self.mlp_edge = torch.nn.Linear(self.embed_dim, 1, bias=False)

    def forward(self, adj, worker_node_attribute, task_node_attribute, edge_attribute):
        batch_size = worker_node_attribute.shape[0]
        num_workers = worker_node_attribute.shape[1]
        num_tasks = task_node_attribute.shape[1]
        adj = adj.unsqueeze(dim=-1)

        # initial feature transformation
        worker_node_vector = self.activate_fn(self.mu_worker(worker_node_attribute))
        task_node_vector = self.activate_fn(self.mu_task(task_node_attribute))
        edge_vector = self.activate_fn(self.mu_edge(edge_attribute))

        worker_node_attr_extend = worker_node_vector.unsqueeze(dim=2).expand(batch_size, num_workers, num_tasks,
                                                                             self.embed_dim)
        task_node_attr_extend = task_node_attribute.unsqueeze(dim=1).expand(batch_size, num_workers, num_tasks,
                                                                            self.embed_dim)
        attention_factor = torch.cat((worker_node_attr_extend, edge_vector, task_node_attr_extend), dim=-1)

        edge_info_worker = self.mix_activate_fn(self.mu_attention_worker(attention_factor))
        edge_info_worker = adj * torch.exp(self.mlp_worker(edge_info_worker))
        edge_info_worker = edge_info_worker / torch.sum(edge_info_worker, dim=2, keepdim=True).clamp(min=eps_cnst)

        edge_info_task = self.mix_activate_fn(self.mu_attention_task(attention_factor))
        edge_info_task = adj * torch.exp(self.mlp_task(edge_info_task))
        edge_info_task = edge_info_task / torch.sum(edge_info_task, dim=1, keepdim=True).clamp(min=eps_cnst)

        edge_info_edge = self.mix_activate_fn(self.mu_attention_edge(attention_factor))
        edge_info_edge = adj * torch.exp(self.mlp_edge(edge_info_edge))
        edge_info_attention_worker = edge_info_edge / torch.sum(edge_info_edge, dim=2, keepdim=True).clamp(min=eps_cnst)
        edge_info_attention_task = edge_info_edge / torch.sum(edge_info_edge, dim=1, keepdim=True).clamp(min=eps_cnst)

        worker_vector_temp = torch.matmul(edge_info_worker.squeeze(-1), task_node_vector)
        task_vector_temp = torch.matmul(edge_info_task.squeeze(-1).transpose(1, 2), worker_node_vector)

        weight_vector_worker = edge_info_attention_worker * edge_vector
        weight_vector_worker = torch.sum(weight_vector_worker, dim=2)
        weight_vector_task = edge_info_attention_task * edge_vector
        weight_vector_task = torch.sum(weight_vector_task, dim=1)

        worker_agg = torch.cat((worker_node_vector, worker_vector_temp, weight_vector_worker), dim=2)
        task_agg = torch.cat((task_node_vector, task_vector_temp, weight_vector_task), dim=2)
        worker_node_vector = self.activate_fn(self.mu_agg_worker(worker_agg))
        task_node_vector = self.activate_fn(self.mu_agg_task(task_agg))

        return worker_node_vector, task_node_vector
