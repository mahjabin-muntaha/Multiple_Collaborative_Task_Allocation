import os
import pickle
from copy import deepcopy
import numpy as np

from optimal_solution import OptimalSolution

"""
Codes in this file are a class that simulate the environment.
"""


class Environment:
    def __init__(self, dataset, file_name_best_result, file_name_best_actions, data_no=0):
        """
        Initialize task allocation environment
        :param dataset: the dataset of the environment
        :param file_name_best_result: the file name of the best results for problem instances in the training dataset
        :param file_name_best_actions: the file name of the best actions for problem instances in the training dataset
        :param data_no: the index/pointer of the problem instance in the dataset
        :return: None
        """
        self.dataset = dataset
        self.num_data_sample = len(self.dataset)
        self.data_sample = self.dataset[data_no]
        (self.location_matrix, self.worker_location, self.task_location, self.worker_adj_matrix, self.task_adj_matrix,
         self.worker_task_adj_matrix, self.distance_matrix, self.worker_capacity, self.worker_revenue,
         self.time_per_dis, self.task_budget,
         self.task_comp_windows, self.initial_paths, self.optimal_profit, self.optimal_path, self.acs_profit,
         self.acs_path, self.worker_list, self.task_list,
         self.initial_action_space, self.num_workers_per_task, self.jaccard_values, self.task_quality_req,
         self.worker_quality_req) = self.data_sample
        self.num_tasks = len(self.task_budget)
        self.num_workers = len(self.worker_capacity)

        if not os.path.exists(file_name_best_result):
            self.best_exploration = np.zeros(len(dataset)) - 1000
        else:
            self.best_exploration = np.load(file_name_best_result)
        self.best_action_traces = []

        # initialize the best action traces and best results for problem instances
        if not os.path.exists(file_name_best_actions):
            for i in range(len(dataset)):
                optimal_profit = dataset[i][13]  # FIXME: should not use constant values like these
                optimal_path = dataset[i][14]
                acs_profit = dataset[i][15]
                acs_path = dataset[i][16]
                if optimal_profit >= acs_profit:
                    good_result = optimal_profit
                    good_paths = optimal_path
                else:
                    good_result = acs_profit
                    good_paths = acs_path
                self.best_exploration[i] = good_result
                action_trace = [good_paths]
                self.best_action_traces.append(action_trace)
        else:
            with open(file_name_best_actions, 'rb') as f:
                self.best_action_traces = pickle.load(f)

    def reset(self, data_no):
        """
        Reset the current problem instance to the problem instance indexed by data_no
        :param data_no: the index/pointer of the problem instance in the dataset
        :return: None
        """
        self.data_sample = self.dataset[data_no]
        (self.location_matrix, self.worker_location, self.task_location, self.worker_adj_matrix, self.task_adj_matrix,
         self.worker_task_adj_matrix, self.distance_matrix, self.worker_capacity, self.worker_revenue,
         self.time_per_dis,
         self.task_budget, self.task_comp_windows, self.initial_paths, self.optimal_profit, self.optimal_path,
         self.acs_profit, self.acs_path, self.worker_list, self.task_list, self.initial_action_space,
         self.num_workers_per_task, self.jaccard_values, self.task_quality_req,
         self.worker_quality_req) = self.data_sample
        self.num_tasks = len(self.task_budget)
        self.num_workers = len(self.worker_capacity)

    def get_data_sample(self, data_no):
        """
        Get the problem instance indexed by data_no
        :param data_no: the index/pointer of the problem instance in the dataset
        :return: the problem instance
        """
        (
            location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix, worker_task_adj_matrix,
            distance_matrix, worker_capacity, worker_revenue, time_per_dis, task_budget,
            task_comp_windows, initial_paths, optimal_profit, optimal_path, acs_profit, acs_path,
            worker_list, task_list, initial_action_space, num_workers_per_task,
            jaccard_values, task_quality_req, worker_quality_req) = self.dataset[data_no]
        return (
            location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix, worker_task_adj_matrix,
            distance_matrix, worker_capacity, worker_revenue, time_per_dis, task_budget,
            task_comp_windows, initial_paths, optimal_profit, optimal_path, acs_profit, acs_path,
            worker_list, task_list, initial_action_space, num_workers_per_task,
            jaccard_values, task_quality_req, worker_quality_req)

    @staticmethod
    def get_distance_matrix(node_list_first, node_list_second, distance_matrix):
        temp_distance_matrix = np.zeros((len(node_list_first), len(node_list_second)))
        for i in range(len(node_list_first)):
            for j in range(len(node_list_second)):
                first_index = node_list_first[i]
                second_index = node_list_second[j]
                temp_distance_matrix[i][j] = distance_matrix[first_index][second_index]

        return temp_distance_matrix

    def get_greedy(self):
        """
        Get the greedy solution of the current problem instance
        :return: the greedy solution
        """
        return self.optimal_profit, self.optimal_path

    def get_acs(self):
        """
        Get the acs solution of the current problem instance
        :return: the acs solution
        """
        return self.acs_profit, self.acs_path

    def step(self, state_helper, action, current_profit):
        """
        Take an action and transit the current state to the next state
        :param state_helper: the helper of the current state
        :param action: the action to take
        :param current_profit: the current profit of the current state
        :return: reward, the helper of the next state, done flag
        """
        # transit the current state to the next state
        (paths, rest_capacity, current_time, num_workers_per_task, task_budget, action_space, worker_stopped,
         task_completed, current_partial_profit) = state_helper
        num_nodes = len(self.worker_list) + len(self.task_list)
        worker_index = action[0]
        task_index = action[1]
        next_paths = deepcopy(paths)
        next_rest_capacity = rest_capacity.copy()
        next_time = current_time.copy()
        next_num_worker_per_task = num_workers_per_task.copy()
        next_task_budget = task_budget.copy()
        next_action_space = action_space.copy()
        next_worker_stopped = worker_stopped.copy()
        next_task_completed = task_completed.copy()
        next_partial_profit = current_partial_profit.copy()
        next_state_helper = (next_paths, next_rest_capacity, next_time, next_num_worker_per_task,
                             next_task_budget, next_action_space, next_worker_stopped, next_task_completed,
                             next_partial_profit)
        if np.any(paths[worker_index, :] > -1):
            worker_last_position = np.argmax(paths[worker_index, :])
            worker_last_position = self.task_list[worker_last_position]
        else:
            worker_last_position = self.worker_list[worker_index]
        if task_index != self.num_tasks:
            partial_budget = self.task_budget[task_index] / self.num_workers_per_task[task_index]
            task = self.task_list[task_index]
            path_max_val = np.max(paths[worker_index, :])
            #  next paths were here

            partial_profit_temp = partial_budget

            temp_dis = self.distance_matrix[worker_last_position][task]  # NOTE: last_pos can be worker or task node
            cost_dis = temp_dis * self.worker_revenue[worker_index]
            cost_time = temp_dis * self.time_per_dis[worker_index]
            next_time[worker_index] += cost_time
            next_rest_capacity[worker_index] -= temp_dis
            next_num_worker_per_task[task_index] -= 1
            partial_profit_temp -= cost_dis
            next_task_budget[task_index] -= partial_budget

            # print("action_space", action_space)
            # print(f"action: [{worker_index}, {task_index}]")
            # print("paths", next_paths)
            # print("budgets", next_task_budget)

            # Identify the workers assigned to the task
            assigned_workers = np.where(next_paths[:, task_index] > -1)[0]
            average_jaccard = OptimalSolution.average_jaccard_for_task(group=assigned_workers,
                                                                       jaccard_values=self.jaccard_values)

            next_paths[worker_index][
                task_index] = path_max_val + 1  # NOTE: record task in paths even if incomplete finally

            raw_profit = current_partial_profit[task_index] / average_jaccard
            assigned_workers = assigned_workers.tolist()
            assigned_workers.append(worker_index)
            current_average_jaccard = OptimalSolution.average_jaccard_for_task(group=assigned_workers,
                                                                               jaccard_values=self.jaccard_values)
            earned_profit = (partial_profit_temp + raw_profit) * current_average_jaccard
            reward = earned_profit - current_partial_profit[task_index]
            next_partial_profit[task_index] = earned_profit
            if next_num_worker_per_task[task_index] == 0:
                next_action_space[:, task_index] = 1
                next_task_completed[task_index] = 1  # NOTE: all required samples completed

            # NOTE: feasible actions changed for current worker;
            #       if the current task is completed, it may cause infeasible actions to other workers;
            #       current action can only make existing feasible actions to be infeasible but not vice versa.
            # update action space
            next_action_space[worker_index][task_index] = 1

            for next_worker_index in range(len(self.worker_list)):
                if next_worker_index == worker_index:  # this workers
                    temp_task_ids = np.arange(len(self.task_list))
                else:
                    # the rest required workers of the current task is changed and affects feasibility of other workers
                    temp_task_ids = [task_index]
                for next_task_index in temp_task_ids:
                    next_task = self.task_list[next_task_index]
                    if np.any(next_paths[next_worker_index, :] >= 0):
                        if np.argmax(next_paths[next_worker_index, :]) == len(self.task_list):
                            continue
                        else:
                            worker_last_position = self.task_list[np.argmax(next_paths[next_worker_index, :])]
                    else:
                        worker_last_position = self.worker_list[next_worker_index]

                    temp_distance = self.distance_matrix[worker_last_position, next_task]  # overall node ids
                    temp_time = next_time[next_worker_index] + temp_distance * self.time_per_dis[next_worker_index]
                    # NOTE: temp_revenue from worker->task can be negative
                    if int(next_action_space[next_worker_index, next_task_index]) == 0:
                        if (temp_distance > next_rest_capacity[next_worker_index]
                                or temp_time > self.task_comp_windows[next_task_index]
                                or next_num_worker_per_task[next_task_index] <= 0):
                            next_action_space[next_worker_index, next_task_index] = 1

            # # refine next_action_space
            # for next_task_index in range(len(self.task_list)):
            #     temp_avail_workers = np.sum(next_action_space[:, next_task_index].astype(int) == 0)
            #     if temp_avail_workers == 0:
            #         next_action_space[:, next_task_index] = 1  # task cannot be fulfilled anyway

            next_profit = np.sum(next_partial_profit)
        else:
            # for stopping action, only current worker is stopped
            # print("Taken action:", "STOP")
            # print("paths", next_paths)

            next_worker_stopped[worker_index] = 1  # NOTE: stop_action selected for this worker
            next_action_space[worker_index, :] = 1
            earned_profit = 0
            # path_max_val = np.max(next_paths[worker_index, :])
            # next_paths[worker_index, task_index] = path_max_val + 1  # NOTE: record stop action in paths
            next_profit = np.sum(next_partial_profit)
            reward = 0  # FIXME

        if np.all(next_action_space[:, :-1]):  # NOTE: no feasible real_action
            done = 0
        else:
            done = 1

        return reward, done, next_profit, next_state_helper
