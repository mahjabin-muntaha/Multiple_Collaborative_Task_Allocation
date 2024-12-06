import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

import global_variables
from environment import Environment
from global_variables import device


class DQNAgent:
    def __init__(self, models, replay_memory, params=None):
        super(DQNAgent, self).__init__()

        self.policy_net = models[0].to(device)
        self.target_net = models[1].to(device)

        if params is not None:
            self.epsilon = params['epsilon']
        else:
            self.epsilon = 1
        self.epsilon_min = global_variables.epsilon_min
        self.epsilon_decay = global_variables.epsilon_decay
        self.learning_rate = global_variables.learning_rate
        self.gamma = global_variables.gamma
        self.n_step = global_variables.n_step
        self.batch_size = global_variables.batch_size
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.tau = global_variables.tau
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = replay_memory
        self.update_steps = 0  # NOTE: num of updates to policy model

        self.batch_wgts = None
        self.batch_tree_pos = None

    def set_env(self, env):
        """
        Change the environment that the agent will run.
        :param env: the environment
        :return: None
        """
        self.env = env

    def fit(self, old_value, new_target):
        """
        This function is used to fit the model.
        :param old_value: this is drawn from direct perdition of the batch of state action pairs
        :param new_target: this is drawn from the rule of updating q_state_action values. (Q=r+gamma**n_step*Q_max)
        :return: None
        """
        loss_train = self.loss_fn(old_value, new_target)
        self.optimizer.zero_grad()
        loss_train.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return True, loss_train

    def derive_action(self, state_helper, data_no, is_training_mode=True, is_stop_guide=False):
        (paths, rest_capacity, current_time, num_workers_per_task, task_budget, action_space, worker_stopped,
         task_completed, current_partial_profit) = state_helper

        if not is_training_mode:  # NOTE: always allow stop_action during testing
            is_stop_guide = True
        if not is_stop_guide:
            action_space[:, -1] = 1  # disable stop_action

        if is_training_mode and np.random.random_sample() < self.epsilon:
            # randomly choose an action
            is_random_action = True
            # obtain available actions except for stopping action
            available_actions = np.argwhere(action_space.astype(int) == 0).tolist()
            action_index = np.random.choice(len(available_actions))
            chosen_action = available_actions[action_index]
        else:
            (location_matrix_tensor,
             worker_location_tensor,
             task_location_tensor,
             worker_adj_matrix_tensor,
             task_adj_matrix_tensor,
             worker_task_adj_matrix_tensor,
             distance_matrix_tensor,
             worker_dist_matrix_tensor,
             task_dist_matrix_tensor,
             worker_task_dist_matrix_tensor,
             worker_unstoppped_tensor,
             task_incomplete_tensor,
             rest_capacity_tensor,
             current_time_tensor,
             worker_revenue_tensor,
             time_per_dis_tensor,
             task_comp_windows_tensor,
             budget_tensor,
             num_workers_per_task_tensor,
             task_quality_req_tensor,
             worker_quality_req_tensor,
             jaccard_values_tensor,
             paths_tensor,
             action_space_tensor,
             current_partial_profit) = self.get_model_parameter(action_space, worker_stopped, task_completed, paths,
                                                                rest_capacity, current_time, num_workers_per_task,
                                                                task_budget, current_partial_profit, data_no)

            q_actions = self.policy_net(location_matrix_tensor,
                                        worker_location_tensor,
                                        task_location_tensor,
                                        worker_adj_matrix_tensor,
                                        task_adj_matrix_tensor,
                                        worker_task_adj_matrix_tensor,
                                        distance_matrix_tensor,
                                        worker_dist_matrix_tensor,
                                        task_dist_matrix_tensor,
                                        worker_task_dist_matrix_tensor,
                                        worker_unstoppped_tensor,
                                        task_incomplete_tensor,
                                        rest_capacity_tensor,
                                        current_time_tensor,
                                        worker_revenue_tensor,
                                        time_per_dis_tensor,
                                        task_comp_windows_tensor,
                                        budget_tensor,
                                        num_workers_per_task_tensor,
                                        task_quality_req_tensor,
                                        worker_quality_req_tensor,
                                        jaccard_values_tensor,
                                        paths_tensor,
                                        action_space_tensor,
                                        current_partial_profit)

            is_random_action = False
            act_feas_flat = (action_space.astype(int).flatten() == 0)  # NOTE: 0=feasible
            feas_idx = act_feas_flat.nonzero()[0]
            feas_q_acts = q_actions[0, act_feas_flat]  # [batch=1, actions]
            temp_act_idx = torch.argmax(feas_q_acts).item()
            chosen_action = feas_idx[temp_act_idx]
            chosen_action = np.unravel_index(chosen_action, action_space.shape)

        return chosen_action, is_random_action

    def get_max_value(self, next_location_matrix_tens,
                      next_worker_location_tens, next_task_location_tens,
                      next_worker_adj_matrix_tens, next_task_adj_matrix_tens, next_worker_task_adj_matrix_tens,
                      next_distance_matrix_tens,
                      next_worker_dist_matrix_tens, next_task_dist_matrix_tens,
                      next_worker_task_dist_matrix_tens, next_worker_unstopped_tens, next_task_incomplete_tens,
                      next_rest_capacity_tens, next_current_time_tens, next_worker_revenue_tens,
                      next_time_per_dis_tens,
                      next_task_comp_windows_tens, next_budget_tens, next_num_workers_per_task_tens,
                      next_task_quality_req_tens, next_worker_quality_req_tens, next_jaccard_values_tens,
                      next_paths_tens, next_action_space_tens, next_partial_profit_tens):
        """
        Get the next state-action pair with the maximum value and the corresponding value.
        :param location_mtx_tens: the tensor of the location matrix
        :param distance_mtx_tens: the tensor of the distance matrix
        :param next_position_tens: the tensor to denote the last positions of workers
        :param next_worker_unstopped_tens: the tensor to denote the unstopped workers
        :param next_rest_capacity_tens: the tensor of the rest capacities of workers
        :param next_time_tens: the tensor of the current time of workers
        :param next_money_per_dis_tens: the tensor of the money per distance of workers
        :param next_time_per_dis_tens: the tensor of the time per distance of workers
        :param next_time_window_tens: the tensor of the time windows of tasks
        :param next_budget_tens: the tensor of the budgets of tasks
        :param next_action_spaces: the action space of the state
        :return: the next state-action pair with the maximum value and the corresponding value
        """
        # get the values of the next state-action pairs with the online network
        values_model = self.policy_net(next_location_matrix_tens,
                                       next_worker_location_tens,
                                       next_task_location_tens,
                                       next_worker_adj_matrix_tens, next_task_adj_matrix_tens,
                                       next_worker_task_adj_matrix_tens,
                                       next_distance_matrix_tens,
                                       next_worker_dist_matrix_tens,
                                       next_task_dist_matrix_tens,
                                       next_worker_task_dist_matrix_tens,
                                       next_worker_unstopped_tens,
                                       next_task_incomplete_tens,
                                       next_rest_capacity_tens, next_current_time_tens,
                                       next_worker_revenue_tens,
                                       next_time_per_dis_tens,
                                       next_task_comp_windows_tens, next_budget_tens,
                                       next_num_workers_per_task_tens,
                                       next_task_quality_req_tens,
                                       next_worker_quality_req_tens,
                                       next_jaccard_values_tens,
                                       next_paths_tens,
                                       next_action_space_tens,
                                       next_partial_profit_tens)

        values_model = values_model.detach()
        values_model[next_action_space_tens.flatten(start_dim=1).type(torch.int) == 1] = float(
            "-inf")  # NOTE: mask infeasible actions

        # get the action recommended by the online network
        max_actions = torch.argmax(values_model, dim=1, keepdim=True)

        # get the values of the next state-action pairs with the target network
        values_target_model = self.target_net(next_location_matrix_tens,
                                              next_worker_location_tens,
                                              next_task_location_tens,
                                              next_worker_adj_matrix_tens,
                                              next_task_adj_matrix_tens,
                                              next_worker_task_adj_matrix_tens,
                                              next_distance_matrix_tens,
                                              next_worker_dist_matrix_tens,
                                              next_task_dist_matrix_tens,
                                              next_worker_task_dist_matrix_tens,
                                              next_worker_unstopped_tens,
                                              next_task_incomplete_tens,
                                              next_rest_capacity_tens, next_current_time_tens,
                                              next_worker_revenue_tens,
                                              next_time_per_dis_tens,
                                              next_task_comp_windows_tens, next_budget_tens,
                                              next_num_workers_per_task_tens,
                                              next_task_quality_req_tens,
                                              next_worker_quality_req_tens,
                                              next_jaccard_values_tens,
                                              next_paths_tens,
                                              next_action_space_tens,
                                              next_partial_profit_tens)
        values_target_model[next_action_space_tens.flatten(start_dim=1).type(torch.int) == 1] = float("-inf")

        # get the values of the next state-action pairs with the actions recommended by the online network
        values_target_model = torch.gather(values_target_model.squeeze(dim=-1), dim=1,
                                           index=max_actions.long())
        values_target_model[torch.isinf(values_target_model)] = 0  # NOTE: correct values of infeasible actions

        return values_target_model.detach(), max_actions.detach()

    def replay(self):
        """
        In this function, the agent derives samples and train the model.
        :return: None
        """

        batch_indices, mini_batch, weights = self.memory.sample(batch_size=self.batch_size, rnd=False, beta=0.4)

        # get a batch of memory samples represented by tensors
        location_matrix_tens, distance_matrix_tens, worker_location_tens, task_location_tens, \
            worker_adj_matrix_tens, task_adj_matrix_tens, worker_task_adj_matrix_tens, \
            worker_dist_matrix_tens, task_dist_matrix_tens, worker_task_dist_matrix_tens, worker_unstopped_tens, \
            task_incomplete_tens, rest_capacity_tens, current_time_tens, worker_revenue_tens, time_per_dis_tens, \
            task_comp_windows_tens, budget_tens, num_workers_per_task_tens, task_quality_req_tens, \
            worker_quality_req_tens, jaccard_values_tens, paths_tens, next_location_matrix_tens, \
            next_worker_location_tens, next_task_location_tens, next_worker_adj_matrix_tens, next_task_adj_matrix_tens, \
            next_worker_task_adj_matrix_tens, next_distance_matrix_tens, next_worker_dist_matrix_tens, next_task_dist_matrix_tens, \
            next_worker_task_dist_matrix_tens, next_worker_unstopped_tens, next_task_incomplete_tens, next_rest_capacity_tens, \
            next_current_time_tens, next_worker_revenue_tens, next_time_per_dis_tens, next_task_comp_windows_tens, next_budget_tens, \
            next_num_workers_per_task_tens, next_task_quality_req_tens, next_worker_quality_req_tens, \
            next_jaccard_values_tens, next_paths_tens, \
            action_tens, done_tens, reward_tens, action_space_tens, next_action_space_tens, \
            current_partial_profit_tens, next_partial_profit_tens = self.get_samples(batch_size=self.batch_size,
                                                                                     rnd=True)  # FIX ME: action_tens and action_space_tens

        # get the max action values of the next states and corresponding actions
        max_action_values, max_actions = self.get_max_value(next_location_matrix_tens,
                                                            next_worker_location_tens,
                                                            next_task_location_tens,
                                                            next_worker_adj_matrix_tens, next_task_adj_matrix_tens,
                                                            next_worker_task_adj_matrix_tens,
                                                            next_distance_matrix_tens,
                                                            next_worker_dist_matrix_tens, next_task_dist_matrix_tens,
                                                            next_worker_task_dist_matrix_tens,
                                                            next_worker_unstopped_tens, next_task_incomplete_tens,
                                                            next_rest_capacity_tens, next_current_time_tens,
                                                            next_worker_revenue_tens,
                                                            next_time_per_dis_tens,
                                                            next_task_comp_windows_tens, next_budget_tens,
                                                            next_num_workers_per_task_tens,
                                                            next_task_quality_req_tens, next_worker_quality_req_tens,
                                                            next_jaccard_values_tens, next_paths_tens,
                                                            next_action_space_tens,
                                                            next_partial_profit_tens)

        # get the bootstrapped target values
        reward_tens_scaled = reward_tens / global_variables.max_value_scale
        target = reward_tens_scaled + self.gamma ** self.n_step * done_tens * max_action_values

        target = target.detach()

        # get the Q-values directly from the model
        original = self.policy_net(location_matrix_tens, worker_location_tens,
                                   task_location_tens,
                                   worker_adj_matrix_tens, task_adj_matrix_tens,
                                   worker_task_adj_matrix_tens, distance_matrix_tens,
                                   worker_dist_matrix_tens, task_dist_matrix_tens,
                                   worker_task_dist_matrix_tens, worker_unstopped_tens,
                                   task_incomplete_tens, rest_capacity_tens,
                                   current_time_tens,
                                   worker_revenue_tens, time_per_dis_tens,
                                   task_comp_windows_tens,
                                   budget_tens, num_workers_per_task_tens,
                                   task_quality_req_tens,
                                   worker_quality_req_tens, jaccard_values_tens,
                                   paths_tens, action_space_tens,
                                   current_partial_profit_tens)

        # soft update the target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # extract the Q-values of the actions taken
        original = torch.gather(original, dim=1, index=action_tens.long())

        # train the model
        max_loss_batch = torch.argmax(torch.abs(original - target))
        # print('max_loss_batch', max_loss_batch)
        # flag, loss = self.fit(original, target)
        # Compute loss with importance sampling weights
        loss = self.loss_fn(original, target)
        weighted_loss = (loss * torch.tensor(weights).to(loss.device)).mean()

        # Backward propagate gradient
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # Calculate the absolute TD errors to update priorities
        batch_priorities = (np.abs((original - target).detach().cpu().numpy())).squeeze()
        self.memory.update_priorities(batch_indices, batch_priorities)

        self.update_steps += 1
        return weighted_loss.detach().to('cpu').item()

    def add_stop_task(self, location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix,
                      worker_task_adj_matrix,
                      distance_matrix, worker_capacity, worker_revenue,
                      time_per_dis, task_budget_original,
                      task_comp_windows, initial_paths, optimal_profit, optimal_path, acs_profit, acs_path,
                      worker_list, task_list, initial_action_space, num_workers_per_task,
                      jaccard_values, task_quality_req, worker_quality_req, task_completed, task_budget,
                      workers_req_per_task, current_partial_profit):

        location_matrix = np.vstack((location_matrix, np.array([0, 0])))
        task_location = np.vstack((task_location, np.array([0, 0])))
        task_adj_matrix = np.r_[task_adj_matrix, np.full((1, task_adj_matrix.shape[1]), 1)]
        task_adj_matrix = np.c_[task_adj_matrix, np.full((task_adj_matrix.shape[0], 1), 1)]
        task_adj_matrix[-1][-1] = 0
        worker_task_adj_matrix = np.c_[worker_task_adj_matrix, np.full((worker_task_adj_matrix.shape[0], 1), 1)]
        distance_matrix = np.c_[distance_matrix, np.zeros((distance_matrix.shape[0], 1))]
        distance_matrix = np.r_[distance_matrix, np.zeros((1, distance_matrix.shape[1]))]
        task_completed = np.append(task_completed, 0)
        task_budget = np.append(task_budget, 0)
        task_budget_original = np.append(task_budget_original, 0)
        task_comp_windows = np.append(task_comp_windows, 0)
        task_list = np.append(task_list, len(task_list) + len(worker_list))
        num_workers_per_task = np.append(num_workers_per_task, 0)
        workers_req_per_task = np.append(workers_req_per_task, 0)
        task_quality_req = np.append(task_quality_req, 0)
        current_partial_profit = np.append(current_partial_profit, 0)
        worker_quality_req = np.c_[worker_quality_req, np.zeros((worker_quality_req.shape[0], 1))]

        return location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix, \
            worker_task_adj_matrix, \
            distance_matrix, worker_capacity, worker_revenue, \
            time_per_dis, task_budget_original, \
            task_comp_windows, initial_paths, optimal_profit, optimal_path, acs_profit, acs_path, \
            worker_list, task_list, initial_action_space, num_workers_per_task, \
            jaccard_values, task_quality_req, worker_quality_req, task_completed, task_budget, \
            workers_req_per_task, current_partial_profit

    def get_model_parameter(self, action_space, worker_stopped, task_completed, paths, rest_capacity, current_time,
                            workers_req_per_task, task_budget, current_partial_profit, data_no):
        """
        Transfer the current state with problem instance indexed by data_no to raw features
        :param task_budget: the remaining budget of each task
        :param workers_req_per_task: the remaining number of workers required by each task
        :param paths: the current paths of the workers
        :param rest_capacity: the current rest capacities of the workers
        :param current_time: the current time of the workers
        :param data_no: the index/pointer of the problem instance
        :return: the tensors of raw features (check the forward function of the model for the details)
        """
        (location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix, worker_task_adj_matrix,
         distance_matrix, worker_capacity, worker_revenue,
         time_per_dis, task_budget_original,  # NOTE: Original task budget without costs deducted
         task_comp_windows, initial_paths, optimal_profit, optimal_path, acs_profit, acs_path,
         worker_list, task_list, initial_action_space, num_workers_per_task,
         jaccard_values, task_quality_req, worker_quality_req) = self.env.get_data_sample(data_no)

        # (location_matrix, worker_location, task_location, worker_adj_matrix,
        #  task_adj_matrix, worker_task_adj_matrix, distance_matrix, worker_capacity,
        #  worker_revenue, time_per_dis, task_budget_original, task_comp_windows, initial_paths,
        #  optimal_profit, optimal_path, acs_profit, acs_path, worker_list, task_list, initial_action_space,
        #  num_workers_per_task, jaccard_values, task_quality_req, worker_quality_req,
        #  task_completed, task_budget, workers_req_per_task, current_partial_profit) = self.add_stop_task(
        #     location_matrix, worker_location,
        #     task_location, worker_adj_matrix,
        #     task_adj_matrix, worker_task_adj_matrix,
        #     distance_matrix, worker_capacity,
        #     worker_revenue,
        #     time_per_dis, task_budget_original,
        #     task_comp_windows, initial_paths,
        #     optimal_profit, optimal_path, acs_profit,
        #     acs_path, worker_list, task_list,
        #     initial_action_space, num_workers_per_task,
        #     jaccard_values, task_quality_req,
        #     worker_quality_req, task_completed, task_budget,
        #     workers_req_per_task, current_partial_profit)

        num_workers = len(worker_list)
        num_tasks = len(task_list)  # NOTE: num of tasks including stop action

        location_matrix_tensor = torch.from_numpy(location_matrix).float().to(device=device)
        distance_matrix_tensor = torch.from_numpy(distance_matrix).float().to(device=device)
        worker_location_tensor = torch.from_numpy(worker_location).float().to(device=device)
        task_location_tensor = torch.from_numpy(task_location).float().to(device=device)
        worker_adj_matrix_tensor = torch.from_numpy(worker_adj_matrix).float().to(device=device)
        task_adj_matrix_tensor = torch.from_numpy(task_adj_matrix).float().to(device=device)
        worker_task_adj_matrix_tensor = torch.from_numpy(worker_task_adj_matrix).float().to(
            device=device)

        jaccard_values_tensor = torch.from_numpy(jaccard_values).float().to(device=device)
        worker_quality_req_tensor = torch.from_numpy(worker_quality_req).float().to(device=device)
        num_nodes = len(worker_list) + len(task_list)

        worker_dist_matrix_tensor = torch.from_numpy(
            Environment.get_distance_matrix(worker_list, worker_list, distance_matrix)).to(device=device)
        task_dist_matrix_tensor = torch.from_numpy(
            Environment.get_distance_matrix(task_list, task_list, distance_matrix)).to(device=device)
        worker_task_dist_matrix_tensor = torch.from_numpy(
            Environment.get_distance_matrix(worker_list, task_list, distance_matrix)).to(device=device)

        paths_tensor = torch.from_numpy(paths).float().to(device=device)
        action_space_tensor = torch.from_numpy(action_space).float().to(device=device)

        worker_unstopped_tensor = torch.from_numpy(1 - worker_stopped).float().to(device=device)
        task_incomplete_tensor = torch.from_numpy(1 - task_completed).float().to(device=device)

        rest_capacity_tensor = torch.zeros_like(worker_unstopped_tensor, device=device)
        current_time_tensor = torch.zeros_like(worker_unstopped_tensor, device=device)
        worker_revenue_tensor = torch.zeros_like(worker_unstopped_tensor, device=device)
        time_per_dis_tensor = torch.zeros_like(worker_unstopped_tensor, device=device)
        for i, worker_index in enumerate(paths):
            if np.max(worker_index) > -1:
                last_pos_index = task_list[np.argmax(worker_index)]
                for j in range(num_workers):
                    worker_index_temp = worker_list[j]
                    worker_dist_matrix_tensor[i][j] = distance_matrix[last_pos_index][
                        worker_index_temp]  # NOTE: updated worker-worker dist
                for j in range(num_tasks):
                    task_index_temp = task_list[j]
                    worker_task_dist_matrix_tensor[i][j] = distance_matrix[last_pos_index][
                        task_index_temp]  # NOTE: updated worker-task dist
            else:
                last_pos_index = worker_list[i]
            rest_capacity_tensor[i] = rest_capacity[i]
            current_time_tensor[i] = current_time[i]
            worker_revenue_tensor[i] = worker_revenue[i]
            time_per_dis_tensor[i] = time_per_dis[i]
            worker_location_tensor[i] = location_matrix_tensor[last_pos_index]  # NOTE: updated worker location

        task_quality_req_tensor = torch.zeros_like(task_incomplete_tensor, device=device)
        task_comp_windows_tensor = torch.zeros_like(task_incomplete_tensor, device=device)
        budget_tensor = torch.zeros_like(task_incomplete_tensor, device=device)
        num_workers_per_task_tensor = torch.zeros_like(task_incomplete_tensor, device=device)
        current_partial_profit_tensor = torch.zeros_like(task_incomplete_tensor, device=device)

        for task_index, task in enumerate(task_list):
            task_comp_windows_tensor[task_index] = task_comp_windows[task_index]
            budget_tensor[task_index] = task_budget[task_index]  # FIXME: original or current?
            task_quality_req_tensor[task_index] = task_quality_req[task_index]
            num_workers_per_task_tensor[task_index] = workers_req_per_task[task_index]
            current_partial_profit_tensor[task_index] = current_partial_profit[task_index]

        paths_tensor = paths_tensor[:, :-1]  # NOTE: remove stop action

        return location_matrix_tensor.unsqueeze(0), \
            worker_location_tensor.unsqueeze(0), \
            task_location_tensor.unsqueeze(0), \
            worker_adj_matrix_tensor.unsqueeze(0), \
            task_adj_matrix_tensor.unsqueeze(0), \
            worker_task_adj_matrix_tensor.unsqueeze(0), \
            distance_matrix_tensor.unsqueeze(0), \
            worker_dist_matrix_tensor.unsqueeze(0), \
            task_dist_matrix_tensor.unsqueeze(0), \
            worker_task_dist_matrix_tensor.unsqueeze(0), \
            worker_unstopped_tensor.unsqueeze(0), \
            task_incomplete_tensor.unsqueeze(0), \
            rest_capacity_tensor.unsqueeze(0), \
            current_time_tensor.unsqueeze(0), \
            worker_revenue_tensor.unsqueeze(0), \
            time_per_dis_tensor.unsqueeze(0), \
            task_comp_windows_tensor.unsqueeze(0), \
            budget_tensor.unsqueeze(0), \
            num_workers_per_task_tensor.unsqueeze(0), \
            task_quality_req_tensor.unsqueeze(0), \
            worker_quality_req_tensor.unsqueeze(0), \
            jaccard_values_tensor.unsqueeze(0), \
            paths_tensor.unsqueeze(0), \
            action_space_tensor.unsqueeze(0), \
            current_partial_profit_tensor.unsqueeze(0)

    def memorize(self, state, action, reward, next_state, done, data_no, memory=None):
        """
        Memorize state, action, reward, next_state, done and environment for memory_replay methods.
        :param state: the current state
        :param action: the action that is taken at the current state
        :param reward: the reward (or n-step return) gotten from the state-action pair
        :param next_state: the state after bootstrap steps
        :param done: a flag that denotes if the episode ends
        :param data_no: the index of the current data sample
        :return: None
        """
        data_nos = data_no
        (paths, rest_capacity, current_time, num_workers_per_task, task_budget, action_space, worker_stopped,
         task_completed, current_partial_profit) = state

        # transform the current state into tensors to fit the model
        (location_matrix_tensor,
         worker_location_tensor,
         task_location_tensor,
         worker_adj_matrix_tensor,
         task_adj_matrix_tensor,
         worker_task_adj_matrix_tensor,
         distance_matrix_tensor,
         worker_dist_matrix_tensor,
         task_dist_matrix_tensor,
         worker_task_dist_matrix_tensor,
         worker_unstopped_tensor,
         task_incomplete_tensor,
         rest_capacity_tensor,
         current_time_tensor,
         worker_revenue_tensor,
         time_per_dis_tensor,
         task_comp_windows_tensor,
         budget_tensor,
         num_workers_per_task_tensor,
         task_quality_req_tensor,
         worker_quality_req_tensor,
         jaccard_values_tensor,
         paths_tensor,
         action_space_tensor,
         current_partial_profit_tensor) = self.get_model_parameter(action_space, worker_stopped, task_completed, paths,
                                                                   rest_capacity, current_time, num_workers_per_task,
                                                                   task_budget, current_partial_profit, data_no)

        location_matrix_tens = location_matrix_tensor
        worker_location_tens = worker_location_tensor
        task_location_tens = task_location_tensor
        worker_adj_matrix_tens = worker_adj_matrix_tensor
        task_adj_matrix_tens = task_adj_matrix_tensor
        worker_task_adj_matrix_tens = worker_task_adj_matrix_tensor
        distance_matrix_tens = distance_matrix_tensor
        worker_dist_matrix_tens = worker_dist_matrix_tensor
        task_dist_matrix_tens = task_dist_matrix_tensor
        worker_task_dist_matrix_tens = worker_task_dist_matrix_tensor
        worker_unstopped_tens = worker_unstopped_tensor
        task_incomplete_tens = task_incomplete_tensor
        rest_capacity_tens = rest_capacity_tensor
        current_time_tens = current_time_tensor
        worker_revenue_tens = worker_revenue_tensor
        time_per_dis_tens = time_per_dis_tensor
        task_comp_windows_tens = task_comp_windows_tensor
        budget_tens = budget_tensor
        num_workers_per_task_tens = num_workers_per_task_tensor
        task_quality_req_tens = task_quality_req_tensor
        worker_quality_req_tens = worker_quality_req_tensor
        jaccard_values_tens = jaccard_values_tensor
        paths_tens = paths_tensor  # purpose?
        action_space_tens = action_space_tensor
        current_partial_profit_tens = current_partial_profit_tensor
        paths_batch = paths  # purpose?

        # transform the next state into tensors to fit the model
        (next_paths, next_rest_capacity, next_time, next_num_workers_per_task, next_task_budget,
         next_action_space, next_worker_stopped, next_task_completed, next_partial_profit) = next_state

        next_location_matrix_tensor, next_worker_location_tensor, next_task_location_tensor, next_worker_adj_matrix_tensor, \
            next_task_adj_matrix_tensor, next_worker_task_adj_matrix_tensor, next_distance_matrix_tensor, \
            next_worker_dist_matrix_tensor, next_task_dist_matrix_tensor, next_worker_task_dist_matrix_tensor, next_worker_unstopped_tensor, \
            next_task_incomplete_tensor, next_rest_capacity_tensor, next_current_time_tensor, next_worker_revenue_tensor, \
            next_time_per_dis_tensor, next_task_comp_windows_tensor, next_budget_tensor, next_num_workers_per_task_tensor, \
            next_task_quality_req_tensor, next_worker_quality_req_tensor, next_jaccard_values_tensor, \
            next_paths_tensor, next_action_space_tensor, next_partial_profit_tensor = \
            self.get_model_parameter(next_action_space, next_worker_stopped, next_task_completed, next_paths,
                                     next_rest_capacity, next_time, next_num_workers_per_task,
                                     next_task_budget, next_partial_profit, data_no)

        next_action_space_tens = next_action_space_tensor
        next_location_matrix_tens = next_location_matrix_tensor
        next_worker_location_tens = next_worker_location_tensor
        next_task_location_tens = next_task_location_tensor
        next_worker_adj_matrix_tens = next_worker_adj_matrix_tensor
        next_task_adj_matrix_tens = next_task_adj_matrix_tensor
        next_worker_task_adj_matrix_tens = next_worker_task_adj_matrix_tensor
        next_distance_matrix_tens = next_distance_matrix_tensor
        next_worker_dist_matrix_tens = next_worker_dist_matrix_tensor
        next_task_dist_matrix_tens = next_task_dist_matrix_tensor
        next_worker_task_dist_matrix_tens = next_worker_task_dist_matrix_tensor
        next_worker_unstopped_tens = next_worker_unstopped_tensor
        next_task_incomplete_tens = next_task_incomplete_tensor
        next_rest_capacity_tens = next_rest_capacity_tensor
        next_current_time_tens = next_current_time_tensor
        next_worker_revenue_tens = next_worker_revenue_tensor
        next_time_per_dis_tens = next_time_per_dis_tensor
        next_task_comp_windows_tens = next_task_comp_windows_tensor
        next_budget_tens = next_budget_tensor
        next_num_workers_per_task_tens = next_num_workers_per_task_tensor
        next_task_quality_req_tens = next_task_quality_req_tensor
        next_worker_quality_req_tens = next_worker_quality_req_tensor
        next_jaccard_values_tens = next_jaccard_values_tensor
        next_paths_tens = next_paths_tensor  # purpose?
        next_action_space_tens = next_action_space_tensor
        next_partial_profit_tens = next_partial_profit_tensor

        next_rest_capacity_batch = next_rest_capacity
        next_paths_batch = next_paths
        next_time_batch = next_time
        next_num_workers_per_task_batch = next_num_workers_per_task
        next_task_budget_batch = next_task_budget

        # transform the action, done flag and reward into tensors
        action = np.ravel_multi_index(action, next_action_space.shape)
        action_tens = torch.tensor([[float(action)]], device=device)
        done_tens = torch.tensor([[float(done)]], device=device)
        reward_tens = torch.tensor([[float(reward)]], device=device)

        memory_sample = (location_matrix_tens, distance_matrix_tens, worker_location_tens, task_location_tens,
                         worker_adj_matrix_tens, task_adj_matrix_tens, worker_task_adj_matrix_tens,
                         worker_dist_matrix_tens, task_dist_matrix_tens,
                         worker_task_dist_matrix_tens, worker_unstopped_tens, task_incomplete_tens,
                         rest_capacity_tens, current_time_tens, worker_revenue_tens, time_per_dis_tens,
                         task_comp_windows_tens, budget_tens, num_workers_per_task_tens, task_quality_req_tens,
                         worker_quality_req_tens, jaccard_values_tens, paths_tens, next_location_matrix_tens,
                         next_worker_location_tens, next_task_location_tens,
                         next_worker_adj_matrix_tens, next_task_adj_matrix_tens, next_worker_task_adj_matrix_tens,
                         next_distance_matrix_tens, next_worker_dist_matrix_tens, next_task_dist_matrix_tens,
                         next_worker_task_dist_matrix_tens, next_worker_unstopped_tens, next_task_incomplete_tens,
                         next_rest_capacity_tens, next_current_time_tens, next_worker_revenue_tens,
                         next_time_per_dis_tens, next_task_comp_windows_tens, next_budget_tens,
                         next_num_workers_per_task_tens,
                         next_task_quality_req_tens, next_worker_quality_req_tens, next_jaccard_values_tens,
                         next_paths_tens, action_tens, done_tens, action_space_tens, next_action_space_tens,
                         reward_tens,
                         paths_batch, next_paths_batch, next_rest_capacity, next_time_batch, next_rest_capacity_batch,
                         next_num_workers_per_task_batch, next_task_budget_batch, current_partial_profit_tens,
                         next_partial_profit_tens,
                         data_nos)

        # delete the tensors that are not used anymore to save memory (pretty sure this is not necessary, you can keep or delete it)

        del next_state
        gc.collect()

        if memory is None:
            used_memory = self.memory  # use agent's default memory
        else:
            used_memory = memory  # use given memory
        mem_pos = used_memory.push(memory_sample)

        return mem_pos

    def get_samples(self, batch_size, rnd=False):
        """
        Derive a batch of samples from memory.
        :return: a batch that includes the batch size of sample
        """

        # get a batch of samples randomly
        batch_indices, mini_batch, weights = self.memory.sample(batch_size=self.batch_size, rnd=False, beta=0.4)
        # store the tensors of the current state
        location_matrix_tens = []
        worker_location_tens = []
        task_location_tens = []
        worker_adj_matrix_tens = []
        task_adj_matrix_tens = []
        worker_task_adj_matrix_tens = []
        distance_matrix_tens = []
        worker_dist_matrix_tens = []
        task_dist_matrix_tens = []
        worker_task_dist_matrix_tens = []
        worker_unstopped_tens = []
        task_incomplete_tens = []
        rest_capacity_tens = []
        current_time_tens = []
        worker_revenue_tens = []
        time_per_dis_tens = []
        task_comp_windows_tens = []
        budget_tens = []
        num_workers_per_task_tens = []
        task_quality_req_tens = []
        worker_quality_req_tens = []
        jaccard_values_tens = []
        paths_tens = []
        current_partial_profit_tens = []

        # store the tensors of the next state
        next_location_matrix_tens = []
        next_worker_location_tens = []
        next_task_location_tens = []
        next_worker_adj_matrix_tens = []
        next_task_adj_matrix_tens = []
        next_worker_task_adj_matrix_tens = []
        next_distance_matrix_tens = []
        next_worker_dist_matrix_tens = []
        next_task_dist_matrix_tens = []
        next_worker_task_dist_matrix_tens = []
        next_worker_unstopped_tens = []
        next_task_incomplete_tens = []
        next_rest_capacity_tens = []
        next_current_time_tens = []
        next_worker_revenue_tens = []
        next_time_per_dis_tens = []
        next_task_comp_windows_tens = []
        next_budget_tens = []
        next_num_workers_per_task_tens = []
        next_task_quality_req_tens = []
        next_worker_quality_req_tens = []
        next_jaccard_values_tens = []
        next_paths_tens = []
        next_partial_profit_tens = []

        # store the tensors of the actions, done flags, rewards, and the next action spaces
        action_tens = []
        done_tens = []
        reward_tens = []
        action_space_tens = []
        next_action_space_tens = []

        # list here is used to verification, not used in training or testing
        data_no_batch = []

        for (location_matrix_tensor, distance_matrix_tensor, worker_location_tensor, task_location_tensor,
             worker_adj_matrix_tensor, task_adj_matrix_tensor, worker_task_adj_matrix_tensor,
             worker_dist_matrix_tensor, task_dist_matrix_tensor,
             worker_task_dist_matrix_tensor, worker_unstopped_tensor, task_incomplete_tensor,
             rest_capacity_tensor, current_time_tensor, worker_revenue_tensor, time_per_dis_tensor,
             task_comp_windows_tensor, budget_tensor, num_workers_per_task_tensor, task_quality_req_tensor,
             worker_quality_req_tensor, jaccard_values_tensor, paths_tensor, next_location_matrix_tensor,
             next_worker_location_tensor, next_task_location_tensor,
             next_worker_adj_matrix_tensor, next_task_adj_matrix_tensor, next_worker_task_adj_matrix_tensor,
             next_distance_matrix_tensor,
             next_worker_dist_matrix_tensor, next_task_dist_matrix_tensor,
             next_worker_task_dist_matrix_tensor, next_worker_unstopped_tensor, next_task_incomplete_tensor,
             next_rest_capacity_tensor, next_current_time_tensor, next_worker_revenue_tensor,
             next_time_per_dis_tensor,
             next_task_comp_windows_tensor, next_budget_tensor, next_num_workers_per_task_tensor,
             next_task_quality_req_tensor, next_worker_quality_req_tensor, next_jaccard_values_tensor,
             next_paths_tensor, action_tensor, done_tensor, action_space_tensor, next_action_space_tensor,
             reward_tensor,
             paths_batch, next_paths_batch, next_rest_capacity, next_time_batch, next_rest_capacity_batch,
             next_num_workers_per_task_batch, next_task_budge_batch, current_partial_profit_tensor,
             next_partial_profit_tensor, data_nos) in mini_batch:
            location_matrix_tens.append(location_matrix_tensor)
            worker_location_tens.append(worker_location_tensor)
            task_location_tens.append(task_location_tensor)
            worker_adj_matrix_tens.append(worker_adj_matrix_tensor)
            task_adj_matrix_tens.append(task_adj_matrix_tensor)
            worker_task_adj_matrix_tens.append(worker_task_adj_matrix_tensor)
            distance_matrix_tens.append(distance_matrix_tensor)
            worker_dist_matrix_tens.append(worker_dist_matrix_tensor)
            task_dist_matrix_tens.append(task_dist_matrix_tensor)
            worker_task_dist_matrix_tens.append(worker_task_dist_matrix_tensor)
            worker_unstopped_tens.append(worker_unstopped_tensor)
            task_incomplete_tens.append(task_incomplete_tensor)
            rest_capacity_tens.append(rest_capacity_tensor)
            current_time_tens.append(current_time_tensor)
            worker_revenue_tens.append(worker_revenue_tensor)
            time_per_dis_tens.append(time_per_dis_tensor)
            task_comp_windows_tens.append(task_comp_windows_tensor)
            budget_tens.append(budget_tensor)
            num_workers_per_task_tens.append(num_workers_per_task_tensor)
            task_quality_req_tens.append(task_quality_req_tensor)
            worker_quality_req_tens.append(worker_quality_req_tensor)
            jaccard_values_tens.append(jaccard_values_tensor)
            data_no_batch.append(data_nos)
            paths_tens.append(paths_tensor)
            current_partial_profit_tens.append(current_partial_profit_tensor)

            next_location_matrix_tens.append(next_location_matrix_tensor)
            next_worker_location_tens.append(next_worker_location_tensor)
            next_task_location_tens.append(next_task_location_tensor)
            next_worker_adj_matrix_tens.append(next_worker_adj_matrix_tensor)
            next_task_adj_matrix_tens.append(next_task_adj_matrix_tensor)
            next_worker_task_adj_matrix_tens.append(next_worker_task_adj_matrix_tensor)
            next_distance_matrix_tens.append(next_distance_matrix_tensor)
            next_worker_dist_matrix_tens.append(next_worker_dist_matrix_tensor)
            next_task_dist_matrix_tens.append(next_task_dist_matrix_tensor)
            next_worker_task_dist_matrix_tens.append(next_worker_task_dist_matrix_tensor)
            next_worker_unstopped_tens.append(next_worker_unstopped_tensor)
            next_task_incomplete_tens.append(next_task_incomplete_tensor)
            next_rest_capacity_tens.append(next_rest_capacity_tensor)
            next_current_time_tens.append(next_current_time_tensor)
            next_worker_revenue_tens.append(next_worker_revenue_tensor)
            next_time_per_dis_tens.append(next_time_per_dis_tensor)
            next_task_comp_windows_tens.append(next_task_comp_windows_tensor)
            next_budget_tens.append(next_budget_tensor)
            next_num_workers_per_task_tens.append(next_num_workers_per_task_tensor)
            next_task_quality_req_tens.append(next_task_quality_req_tensor)
            next_worker_quality_req_tens.append(next_worker_quality_req_tensor)
            next_jaccard_values_tens.append(next_jaccard_values_tensor)
            next_paths_tens.append(next_paths_tensor)
            next_partial_profit_tens.append(next_partial_profit_tensor)

            action_tens.append(action_tensor)
            done_tens.append(done_tensor)
            reward_tens.append(reward_tensor)
            action_space_tens.append(action_space_tensor)
            next_action_space_tens.append(next_action_space_tensor)

        # assemble the tensors to batches
        location_matrix_tensor = torch.cat(location_matrix_tens, dim=0)
        worker_location_tensor = torch.cat(worker_location_tens, dim=0)
        task_location_tensor = torch.cat(task_location_tens, dim=0)
        worker_adj_matrix_tensor = torch.cat(worker_adj_matrix_tens, dim=0)
        task_adj_matrix_tensor = torch.cat(task_adj_matrix_tens, dim=0)
        worker_task_adj_matrix_tensor = torch.cat(worker_task_adj_matrix_tens, dim=0)
        distance_matrix_tensor = torch.cat(distance_matrix_tens, dim=0)
        worker_dist_matrix_tensor = torch.cat(worker_dist_matrix_tens, dim=0)
        task_dist_matrix_tensor = torch.cat(task_dist_matrix_tens, dim=0)
        worker_task_dist_matrix_tensor = torch.cat(worker_task_dist_matrix_tens, dim=0)
        worker_unstopped_tensor = torch.cat(worker_unstopped_tens, dim=0)
        task_incomplete_tensor = torch.cat(task_incomplete_tens, dim=0)
        rest_capacity_tensor = torch.cat(rest_capacity_tens, dim=0)
        current_time_tensor = torch.cat(current_time_tens, dim=0)
        worker_revenue_tensor = torch.cat(worker_revenue_tens, dim=0)
        time_per_dis_tensor = torch.cat(time_per_dis_tens, dim=0)
        task_comp_windows_tensor = torch.cat(task_comp_windows_tens, dim=0)
        budget_tensor = torch.cat(budget_tens, dim=0)
        num_workers_per_task_tensor = torch.cat(num_workers_per_task_tens, dim=0)
        task_quality_req_tensor = torch.cat(task_quality_req_tens, dim=0)
        worker_quality_req_tensor = torch.cat(worker_quality_req_tens, dim=0)
        jaccard_values_tensor = torch.cat(jaccard_values_tens, dim=0)
        paths_tensor = torch.cat(paths_tens, dim=0)
        current_partial_profit_tensor = torch.cat(current_partial_profit_tens, dim=0)

        action_tens = torch.cat(action_tens, dim=0)
        done_tens = torch.cat(done_tens, dim=0)
        reward_tens = torch.cat(reward_tens, dim=0)
        action_space_tens = torch.cat(action_space_tens, dim=0)
        next_action_space_tens = torch.cat(next_action_space_tens, dim=0)

        next_location_matrix_tens = torch.cat(next_location_matrix_tens, dim=0)
        next_worker_location_tens = torch.cat(next_worker_location_tens, dim=0)
        next_task_location_tens = torch.cat(next_task_location_tens, dim=0)
        next_worker_adj_matrix_tens = torch.cat(next_worker_adj_matrix_tens, dim=0)
        next_task_adj_matrix_tens = torch.cat(next_task_adj_matrix_tens, dim=0)
        next_worker_task_adj_matrix_tens = torch.cat(next_worker_task_adj_matrix_tens, dim=0)
        next_distance_matrix_tens = torch.cat(next_distance_matrix_tens, dim=0)
        next_worker_dist_matrix_tens = torch.cat(next_worker_dist_matrix_tens, dim=0)
        next_task_dist_matrix_tens = torch.cat(next_task_dist_matrix_tens, dim=0)
        next_worker_task_dist_matrix_tens = torch.cat(next_worker_task_dist_matrix_tens, dim=0)
        next_worker_unstopped_tens = torch.cat(next_worker_unstopped_tens, dim=0)
        next_task_incomplete_tens = torch.cat(next_task_incomplete_tens, dim=0)
        next_rest_capacity_tens = torch.cat(next_rest_capacity_tens, dim=0)
        next_current_time_tens = torch.cat(next_current_time_tens, dim=0)
        next_worker_revenue_tens = torch.cat(next_worker_revenue_tens, dim=0)
        next_time_per_dis_tens = torch.cat(next_time_per_dis_tens, dim=0)
        next_task_comp_windows_tens = torch.cat(next_task_comp_windows_tens, dim=0)
        next_budget_tens = torch.cat(next_budget_tens, dim=0)
        next_num_workers_per_task_tens = torch.cat(next_num_workers_per_task_tens, dim=0)
        next_task_quality_req_tens = torch.cat(next_task_quality_req_tens, dim=0)
        next_worker_quality_req_tens = torch.cat(next_worker_quality_req_tens, dim=0)
        next_jaccard_values_tens = torch.cat(next_jaccard_values_tens, dim=0)
        next_paths_tens = torch.cat(next_paths_tens, dim=0)
        next_partial_profit_tens = torch.cat(next_partial_profit_tens, dim=0)

        return location_matrix_tensor, distance_matrix_tensor, worker_location_tensor, task_location_tensor, \
            worker_adj_matrix_tensor, task_adj_matrix_tensor, worker_task_adj_matrix_tensor, \
            worker_dist_matrix_tensor, task_dist_matrix_tensor, worker_task_dist_matrix_tensor, worker_unstopped_tensor, \
            task_incomplete_tensor, rest_capacity_tensor, current_time_tensor, worker_revenue_tensor, time_per_dis_tensor, \
            task_comp_windows_tensor, budget_tensor, num_workers_per_task_tensor, task_quality_req_tensor, \
            worker_quality_req_tensor, jaccard_values_tensor, paths_tensor, next_location_matrix_tens, \
            next_worker_location_tens, next_task_location_tens, next_worker_adj_matrix_tens, next_task_adj_matrix_tens, \
            next_worker_task_adj_matrix_tens, next_distance_matrix_tens, next_worker_dist_matrix_tens, next_task_dist_matrix_tens, \
            next_worker_task_dist_matrix_tens, next_worker_unstopped_tens, next_task_incomplete_tens, next_rest_capacity_tens, \
            next_current_time_tens, next_worker_revenue_tens, next_time_per_dis_tens, next_task_comp_windows_tens, next_budget_tens, \
            next_num_workers_per_task_tens, next_task_quality_req_tens, next_worker_quality_req_tens, next_jaccard_values_tens, \
            next_paths_tens, action_tens, done_tens, reward_tens, action_space_tens, next_action_space_tens, \
            current_partial_profit_tensor, next_partial_profit_tens
