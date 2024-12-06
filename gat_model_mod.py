import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from global_variables import device, lrelu_neg_slope, mix_activate_tanh, eps_cnst
from gat_model_homogenous import GATQN_Homogenous
from gat_model_bipartite import GATQN_Bipartite

"""
This neural network model is based on graph attention networks. 
"""


class GATQN(nn.Module):
    def __init__(self, hidden_dim, embed_dim, embed_iter, worker_node_attribute_size,
                 worker_to_worker_edge_size, task_node_attribute_size, task_to_task_edge_size,
                 worker_to_task_edge_size):
        """
        Initialize the neural network
        :param hidden_dim: the number of neurons in the hidden layer.
        :param embed_dim: the embedded dimension
        :param embed_iter: rounds of graph info aggregation
        :param worker_node_attribute_size: the dimension of the worker node attribute
        :param worker_to_worker_edge_size: the dimension of the worker-worker edge attribute
        :param
        :param
        :param
        :param
        """
        super(GATQN, self).__init__()

        if lrelu_neg_slope == 0:
            self.activate_fn = nn.ReLU()
        else:
            self.activate_fn = nn.LeakyReLU(negative_slope=lrelu_neg_slope)
        if mix_activate_tanh:
            self.mix_activate_fn = nn.Tanh()
        else:
            self.mix_activate_fn = self.activate_fn

        self.embed_iter = embed_iter
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.gat_homogenous_worker = nn.ModuleList()
        self.gat_homogenous_task = nn.ModuleList()
        self.gat_bipartite = nn.ModuleList()

        for i in range(embed_iter):
            self.gat_homogenous_worker.append(GATQN_Homogenous(embed_dim=self.embed_dim,
                                                               node_attribute_size=self.embed_dim,
                                                               edge_attribute_size=self.embed_dim).to(device))

            self.gat_homogenous_task.append(GATQN_Homogenous(embed_dim=self.embed_dim,
                                                             node_attribute_size=self.embed_dim,
                                                             edge_attribute_size=self.embed_dim).to(device))

            self.gat_bipartite.append(GATQN_Bipartite(embed_dim=self.embed_dim,
                                                      worker_attribute_size=self.embed_dim,
                                                      task_attribute_size=self.embed_dim,
                                                      edge_attribute_size=self.embed_dim).to(device))

        self.worker_node_attribute_size = worker_node_attribute_size
        self.worker_to_worker_edge_size = worker_to_worker_edge_size
        self.task_node_attribute_size = task_node_attribute_size
        self.task_to_task_edge_size = task_to_task_edge_size
        self.worker_to_task_edge_size = worker_to_task_edge_size

        self.mu_worker_node = nn.Linear(self.worker_node_attribute_size, self.embed_dim)
        self.mu_worker_edge = nn.Linear(self.worker_to_worker_edge_size, self.embed_dim)

        self.mu_task_node = nn.Linear(self.task_node_attribute_size, self.embed_dim)
        self.mu_task_edge = nn.Linear(self.task_to_task_edge_size, self.embed_dim)

        self.mu_worker_node_bip = nn.Linear(self.worker_node_attribute_size, self.embed_dim)
        self.mu_task_node_bip = nn.Linear(self.task_node_attribute_size, self.embed_dim)
        self.mu_worker_task_edge = nn.Linear(self.worker_to_task_edge_size, self.embed_dim)

        self.initial_mu_worker = nn.Linear(self.worker_node_attribute_size, self.embed_dim)
        self.initial_mu_task = nn.Linear(self.task_node_attribute_size, self.embed_dim)
        self.initial_mu_worker_bip = nn.Linear(self.worker_node_attribute_size, self.embed_dim)
        self.initial_mu_task_bip = nn.Linear(self.task_node_attribute_size, self.embed_dim)

        self.final_mu_agg_worker = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.final_mu_agg_task = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.final_mu_agg_worker_bip = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.final_mu_agg_task_bip = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.mu_attention_all_nodes = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.mlp_all_nodes = nn.Linear(self.embed_dim, 1, bias=False)

        self.mix_attn = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.mix_mlp = nn.Linear(self.embed_dim, 1, bias=False)

        self.mu_action = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.q_hidden = nn.Linear(2 * self.embed_dim, self.hidden_dim)
        self.q_out = nn.Linear(self.hidden_dim, 1)

    def embed(self, location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix,
              worker_task_adj_matrix, distance_matrix, worker_dist_matrix, task_dist_matrix, worker_task_dist_matrix,
              worker_unstopped, task_incomplete, rest_capacity, current_time, worker_revenue, time_per_dis,
              task_comp_windows, task_budgets, num_workers_per_task, task_quality_req, worker_quality_req,
              jaccard_values, paths, action_space, current_partial_profit):

        # size info
        num_node_worker = worker_location.shape[1]
        num_node_task = task_location.shape[1]


        if worker_task_adj_matrix.shape[-1] == action_space.shape[-1]:
            worker_task_adj_matrix = worker_task_adj_matrix * (1 - action_space)
        else:
            worker_task_adj_matrix = worker_task_adj_matrix * (1 - action_space[..., :-1])

        # normalization
        worker_node_attribute = torch.cat((worker_location,
                                           rest_capacity.unsqueeze(dim=-1),
                                           current_time.unsqueeze(dim=-1),
                                           worker_revenue.unsqueeze(dim=-1),
                                           time_per_dis.unsqueeze(dim=-1),
                                           worker_unstopped.unsqueeze(dim=-1)), dim=-1)
        # worker_node_attribute = F.normalize(worker_node_attribute, dim=-1)
        worker_node_attribute[..., :-1] = F.normalize(worker_node_attribute[..., :-1], dim=-1)

        worker_edge_attribute = torch.cat((worker_dist_matrix.unsqueeze(dim=-1),
                                           jaccard_values.unsqueeze(dim=-1)), dim=-1).float()
        worker_edge_attribute = F.normalize(worker_edge_attribute, dim=-1)

        task_node_attribute = torch.cat((task_location,
                                         task_comp_windows.unsqueeze(dim=-1),
                                         task_budgets.unsqueeze(dim=-1),
                                         task_quality_req.unsqueeze(dim=-1),
                                         num_workers_per_task.unsqueeze(dim=-1),
                                         task_incomplete.unsqueeze(dim=-1),
                                         current_partial_profit.unsqueeze(dim=-1)), dim=-1)
        # task_node_attribute = F.normalize(task_node_attribute, dim=-1)
        task_node_attribute[..., :-1] = F.normalize(task_node_attribute[..., :-1], dim=-1)

        task_edge_attribute = task_dist_matrix.unsqueeze(dim=-1).float()
        task_edge_attribute = F.normalize(task_edge_attribute, dim=-1)
        worker_task_unselected = (paths < 0).float()  # unselected edges
        worker_assigned = torch.any(paths >= 0, dim=-1).float()  # assigned to any task or not
        # NOTE: broadcast to task dim before multiplication; 0-unassigned; 1-last assigned
        worker_task_last_pos = worker_assigned[..., None] * F.one_hot(paths.argmax(dim=-1), paths.shape[-1])

        worker_task_edge_attribute = torch.cat((worker_task_dist_matrix.unsqueeze(dim=-1),
                                                worker_quality_req.unsqueeze(dim=-1),
                                                worker_task_unselected.unsqueeze(dim=-1),
                                                worker_task_last_pos.unsqueeze(dim=-1)), dim=-1).float()

        # normalize edge features
        # worker_task_edge_attribute = F.normalize(worker_task_edge_attribute, dim=-1)
        worker_task_edge_attribute[..., :-1] = F.normalize(worker_task_edge_attribute[..., :-1], dim=-1)

        # in embedding layer, we obtain the initial node and edge vectors
        worker_node_homogenous = self.activate_fn(self.mu_worker_node(worker_node_attribute))
        worker_edge_attribute = self.activate_fn(self.mu_worker_edge(worker_edge_attribute))
        task_node_homogenous = self.activate_fn(self.mu_task_node(task_node_attribute))
        task_edge_attribute = self.activate_fn(self.mu_task_edge(task_edge_attribute))

        worker_node_bipartite = self.activate_fn(self.mu_worker_node_bip(worker_node_attribute))
        task_node_bipartite = self.activate_fn(self.mu_task_node_bip(task_node_attribute))
        worker_task_edge_attribute = self.activate_fn(self.mu_worker_task_edge(worker_task_edge_attribute))

        for i in range(self.embed_iter):
            worker_node_homogenous = self.gat_homogenous_worker[i](worker_adj_matrix, worker_node_homogenous,
                                                                   worker_edge_attribute)
            task_node_homogenous = self.gat_homogenous_task[i](task_adj_matrix, task_node_homogenous,
                                                               task_edge_attribute)

            worker_node_bipartite, task_node_bipartite = self.gat_bipartite[i](worker_task_adj_matrix,
                                                                               worker_node_bipartite,
                                                                               task_node_bipartite,
                                                                               worker_task_edge_attribute)

        # highlight the local node information in node vectors
        initial_node_vector_worker = self.activate_fn(self.initial_mu_worker(worker_node_attribute))
        initial_node_vector_task = self.activate_fn(self.initial_mu_task(task_node_attribute))
        initial_node_vector_worker_bip = self.activate_fn(self.initial_mu_worker_bip(worker_node_attribute))
        initial_node_vector_task_bip = self.activate_fn(self.initial_mu_task_bip(task_node_attribute))

        worker_node_homogenous = self.activate_fn(self.final_mu_agg_worker(
            torch.cat((initial_node_vector_worker, worker_node_homogenous), dim=-1)))
        task_node_homogenous = self.activate_fn(self.final_mu_agg_task(
            torch.cat((initial_node_vector_task, task_node_homogenous), dim=-1)))
        worker_node_bipartite = self.activate_fn(self.final_mu_agg_worker_bip(
            torch.cat((initial_node_vector_worker_bip, worker_node_bipartite), dim=-1)))
        task_node_bipartite = self.activate_fn(self.final_mu_agg_task_bip(
            torch.cat((initial_node_vector_task_bip, task_node_bipartite), dim=-1)))

        # merge embeddings from different graphs
        worker_node_homogenous = F.pad(worker_node_homogenous, (0, 0, 0, num_node_task), "constant", 0)
        task_node_homogenous = F.pad(task_node_homogenous, (0, 0, num_node_worker, 0), "constant", 0)
        all_nodes_combined = torch.stack([worker_node_homogenous, task_node_homogenous,
                                          torch.cat((worker_node_bipartite, task_node_bipartite), dim=1)], dim=1)

        mu_attention_all_nodes = self.mix_activate_fn(self.mu_attention_all_nodes(all_nodes_combined))
        mu_attention_all_nodes = torch.exp(self.mlp_all_nodes(mu_attention_all_nodes).squeeze(-1).mean(dim=-1))
        mu_attention_all_nodes = mu_attention_all_nodes / (
            torch.sum(mu_attention_all_nodes, dim=1, keepdim=True).clamp(min=eps_cnst))

        mu_attention_all_nodes = mu_attention_all_nodes.unsqueeze(dim=-1).unsqueeze(dim=-1)
        mu_attention_all_nodes = mu_attention_all_nodes.expand(-1, -1, all_nodes_combined.shape[-2],
                                                               all_nodes_combined.shape[-1])
        node_vector = torch.sum(mu_attention_all_nodes * all_nodes_combined, dim=1)

        return node_vector

    def embed_state_action(self, location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix,
                           worker_task_adj_matrix, distance_matrix, worker_dist_matrix, task_dist_matrix,
                           worker_task_dist_matrix, worker_unstopped, task_incomplete, rest_capacity, current_time,
                           worker_revenue, time_per_dis, task_comp_windows, task_budgets, num_workers_per_task,
                           task_quality_req, worker_quality_req, jaccard_values, paths, action_space, current_partial_profit):

        # size info
        num_node_worker = worker_location.shape[1]
        num_node_task = task_location.shape[1]
        num_nodes = num_node_worker + num_node_task
        batch_size = location_matrix.shape[0]

        node_vector = self.embed(location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix,
                                 worker_task_adj_matrix, distance_matrix, worker_dist_matrix, task_dist_matrix,
                                 worker_task_dist_matrix, worker_unstopped, task_incomplete, rest_capacity,
                                 current_time, worker_revenue, time_per_dis, task_comp_windows, task_budgets,
                                 num_workers_per_task, task_quality_req, worker_quality_req, jaccard_values,
                                 paths, action_space, current_partial_profit)
        node_vector_worker = node_vector[:, :num_node_worker, ...]
        node_vector_task = node_vector[:, -num_node_task:, ...]

        # represent the graph state with all nodes mixed
        node_attn = self.mix_activate_fn(self.mix_attn(node_vector))
        node_attn = self.mix_mlp(node_attn).squeeze(dim=-1)
        attn_factor = torch.softmax(node_attn, dim=-1).unsqueeze(dim=-2)
        state = torch.matmul(attn_factor, node_vector)  # [batch, nodes, embed] --> [batch, 1, embed]

        #  construct the action vector
        node_vector_worker = node_vector_worker.unsqueeze(dim=2).expand(batch_size, num_node_worker, num_node_task,
                                                                        self.embed_dim)
        node_vector_task = node_vector_task.unsqueeze(dim=1).expand(batch_size, num_node_worker, num_node_task,
                                                                    self.embed_dim)
        edge_action = torch.cat((node_vector_worker, node_vector_task), dim=-1)
        # condense info in action vector after concatenation
        action = self.activate_fn(self.mu_action(edge_action))

        # construct the state-action vector
        state_ext = state.unsqueeze(dim=1).expand(action.shape)
        state_action = torch.cat((state_ext, action), dim=-1)

        return state, state_action

    def forward(self, location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix,
                worker_task_adj_matrix, distance_matrix, worker_dist_matrix, task_dist_matrix, worker_task_dist_matrix,
                worker_unstopped, task_incomplete, rest_capacity, current_time, worker_revenue, time_per_dis,
                task_comp_windows, task_budgets, num_workers_per_task, task_quality_req, worker_quality_req,
                jaccard_values, paths, action_space, current_partial_profit):

        batch_size = location_matrix.shape[0]
        state, state_action = self.embed_state_action(
            location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix,
            worker_task_adj_matrix, distance_matrix, worker_dist_matrix, task_dist_matrix, worker_task_dist_matrix,
            worker_unstopped, task_incomplete, rest_capacity, current_time, worker_revenue, time_per_dis,
            task_comp_windows, task_budgets, num_workers_per_task, task_quality_req, worker_quality_req,
            jaccard_values, paths, action_space, current_partial_profit)

        # obtain the q-values of state-action pairs
        out_hidden = self.activate_fn(self.q_hidden(state_action))
        out_value = self.q_out(out_hidden).squeeze(dim=-1)  # [batch, workers, tasks]

        # append q-values of stopping actions (i.e., 0)
        out_value_ext = F.pad(out_value, pad=(0, 1), value=0).reshape((batch_size, -1))  # [batch, actions]
        # out_value_ext = out_value.reshape((batch_size, -1))  # [batch, actions]
        return out_value_ext
