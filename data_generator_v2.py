import numpy as np
import torch
import time
from copy import deepcopy
from optimal_solution import OptimalSolution
from aco_solution import ACSSearch
from itertools import combinations
import os
import pickle
import shutil
import global_variables
from math import comb
import re
from util import Util

optimal_solution_var = OptimalSolution()
acssearch = ACSSearch()


class DataGenerator:
    def __init__(self, data, batch_size, sensing_area_dim, num_workers, num_tasks, task_budget_low, task_budget_high,
                 task_req_time_low, task_req_time_high, num_workers_per_task_low, num_workers_per_task_high,
                 worker_time_per_dis_low, worker_time_per_dis_high, worker_req_rev_low, worker_req_rev_high,
                 worker_capacity_low, worker_capacity_high,
                 quality_req_low, quality_req_high, iteration, num_data_sample, file_location):
        """
              Initialize the DataGenerator class.

              Parameters:
              data (type): Description of the data parameter
              batch_size (int): The size of the batch
              sensing_area_dim (int): The dimension of the sensing area
              num_workers (int): The number of workers
              num_tasks (int): The number of tasks
              task_budget_low (int): The lower limit of the task budget
              task_budget_high (int): The upper limit of the task budget
              task_req_time_low (int): The lower limit of the task required time
              task_req_time_high (int): The upper limit of the task required time
              task_reward_low (int): The lower limit of the task reward
              task_reward_high (int): The upper limit of the task reward
              worker_time_per_dis_low (int): The lower limit of the unit time per distance the worker can travel
              worker_time_per_dis_high (int): The upper limit of the unit time per distance the worker can travel
              worker_req_rev_low (int): The lower limit of the required revenue of the worker
              worker_req_rev_high (int): The upper limit of the required revenue of the worker
              quality_req_low (int): The lower limit of the quality requirement
              quality_req_high (int): The upper limit of the quality requirement

              Returns:
              None
              """

        np.random.seed(139)
        self.data = data
        self.batch_size = batch_size
        self.sensing_area_dim = sensing_area_dim
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.task_budget_low = task_budget_low
        self.task_budget_high = task_budget_high
        self.task_req_time_low = task_req_time_low
        self.task_req_time_high = task_req_time_high
        self.num_workers_per_task_low = num_workers_per_task_low
        self.num_workers_per_task_high = num_workers_per_task_high
        self.worker_time_per_dis_low = worker_time_per_dis_low
        self.worker_time_per_dis_high = worker_time_per_dis_high
        self.worker_req_rev_low = worker_req_rev_low
        self.worker_req_rev_high = worker_req_rev_high
        self.worker_capacity_low = worker_capacity_low
        self.worker_capacity_high = worker_capacity_high
        self.quality_req_low = quality_req_low
        self.quality_req_high = quality_req_high
        self.location_matrix = np.random.randint(low=0.0, high=self.sensing_area_dim,
                                                 size=(self.num_tasks + self.num_workers, 2))
        self.task_location_matrix = None
        self.worker_location_matrix = None
        self.iteration = iteration
        self.file_location = file_location
        self.beginning = 0
        self.num_data_sample = num_data_sample
        self.util = Util()

    def get_distance_matrix_all(self, location_matrix):
        """
        Calculate the distance matrix for the location matrix.

        This function calculates the distance matrix by first expanding the location matrix along two different dimensions.
        Then, it calculates the absolute difference between the expanded matrices and sums along the last dimension.
        The resulting tensor is then converted back to a numpy array and cast to integer type.

        Returns:
        numpy.ndarray: The distance matrix as a 2D numpy array of integers.
        """

        location_matrix = torch.from_numpy(location_matrix)
        node_matrix_diff = location_matrix.unsqueeze(dim=0)
        node_matrix_same = location_matrix.unsqueeze(dim=1)
        node_matrix_diff = node_matrix_diff.expand(location_matrix.shape[0], location_matrix.shape[0],
                                                   location_matrix.shape[1])
        node_matrix_same = node_matrix_same.expand(location_matrix.shape[0], location_matrix.shape[0],
                                                   location_matrix.shape[1])
        distance_matrix = torch.abs(node_matrix_same - node_matrix_diff)
        distance_matrix = torch.sum(distance_matrix, dim=-1)
        return distance_matrix.numpy().astype(int)

    def generate_worker_adjacency_matrix(self, num_nodes_1, num_nodes_2, graph_type):
        adj_matrix = np.zeros((num_nodes_1, num_nodes_2), dtype=int)

        if graph_type == 1:
            for i in range(num_nodes_1):
                for j in range(i + 1, num_nodes_2):
                    connection = np.random.choice([0, 1], p=[0.25, 0.75])
                    adj_matrix[i][j] = connection
                    adj_matrix[j][i] = connection  # Ensure the graph is undirected
        elif graph_type == 2:
            for i in range(num_nodes_1):
                for j in range(i + 1, num_nodes_2):
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
        else:
            for i in range(num_nodes_1):
                for j in range(num_nodes_2):
                    adj_matrix[i][j] = 1

        # Ensure there is at least one connection in the graph
        if np.sum(adj_matrix) == 0:
            i, j = np.random.choice(num_nodes_1, 2, replace=False)
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1  # Ensure the graph is undirected

        return adj_matrix

    def calculate_jaccard_coefficient(self, adj_matrix, min_jaccard_value=0.5, max_jaccard_value=1.0):
        """
        Calculate the Jaccard coefficient for an adjacency matrix and scale it between min_jaccard_value and max_jaccard_value.
        :param adj_matrix: adjacency matrix of the graph
        :param min_jaccard_value: minimum Jaccard value after scaling
        :param max_jaccard_value: maximum Jaccard value after scaling
        :return: Jaccard matrix with values scaled between min_jaccard_value and max_jaccard_value
        """
        n = len(adj_matrix)  # Number of nodes
        jaccard_matrix = np.zeros((n, n))  # Initialize the Jaccard matrix with zeros

        for i in range(n):
            for j in range(i + 1, n):
                neighbors_i = set(np.where(adj_matrix[i] == 1)[0])
                neighbors_j = set(np.where(adj_matrix[j] == 1)[0])

                intersection = len(neighbors_i.intersection(neighbors_j))
                union = len(neighbors_i.union(neighbors_j))

                if union == 0:  # Avoid division by zero
                    jaccard_index = 0
                else:
                    jaccard_index = intersection / union

                # Scale the Jaccard index to be between min_jaccard_value and max_jaccard_value
                scaled_jaccard_index = min_jaccard_value + (jaccard_index * (max_jaccard_value - min_jaccard_value))

                jaccard_matrix[i][j] = scaled_jaccard_index
                jaccard_matrix[j][i] = scaled_jaccard_index  # Mirror the index for the lower triangle

        np.fill_diagonal(jaccard_matrix, max_jaccard_value)  # Fill diagonal with max_jaccard_value for self-similarity
        return jaccard_matrix

    def set_graph_size(self, num_workers, num_tasks):
        """
        Reset the number of workers and the number of tasks
        :param num_workers: the new number of workers
        :param num_tasks: the new number of tasks
        :return: None
        """

        self.num_workers = num_worker
        self.num_tasks = num_task

    def set_beginning(self, beginning):
        """
        Set the beginning number of the data sample
        :param beginning: the beginning number
        :return: None
        """
        self.beginning = beginning


    def save_simulation(self, count, location_matrix, worker_location, task_location,
                        worker_adj_matrix, task_adj_matrix, worker_task_adj_matrix,
                        distance_matrix, worker_list,
                        task_list, task_budget,
                        task_comp_windows, worker_capacity, time_per_dis, worker_revenue,
                        num_workers_per_task, task_quality_req,
                        worker_quality_req, init_action_space, optimal_path, optimal_profit,
                        acs_path, acs_profit, optimal_time, acs_time, jaccard_values):

        files = os.listdir(self.file_location)
        # Use a regular expression to find the count in each file name
        counts = [int(re.search(r'graph_(\d+)', file).group(1)) for file in files if re.search(r'graph_(\d+)', file)]
        if not counts:
            file_count = 0
        else:
            file_count = max(counts)
        file_name = self.file_location + "graph_" + str(file_count + count + 1) + "/"
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        self.util.save_file(file_name, location_matrix.astype(float), "location_matrix", 1)
        self.util.save_file(file_name, worker_location.astype(float), "worker_location", 1)
        self.util.save_file(file_name, task_location.astype(float), "task_location", 1)
        self.util.save_file(file_name, worker_adj_matrix, "worker_adj_matrix", 1)
        self.util.save_file(file_name, task_adj_matrix, "task_adj_matrix", 1)
        self.util.save_file(file_name, worker_task_adj_matrix, "worker_task_adj_matrix", 1)
        self.util.save_file(file_name, distance_matrix.astype(float), "distance_matrix", 1)
        initial_paths = np.full((len(worker_list), (len(task_list) + 1)), -1)
        self.util.save_file(file_name, initial_paths, "initial_paths", 2)
        self.util.save_file(file_name, init_action_space, "initial_action_space", 1)
        self.util.save_file(file_name, task_budget.astype(float), "task_budget", 1)
        self.util.save_file(file_name, worker_list, "worker_list", 1)
        self.util.save_file(file_name, task_list, "task_list", 1)
        self.util.save_file(file_name, task_comp_windows.astype(float), "task_comp_windows", 1)
        self.util.save_file(file_name, worker_capacity.astype(float), "worker_capacity", 1)
        self.util.save_file(file_name, time_per_dis.astype(float), "time_per_dis", 1)
        self.util.save_file(file_name, worker_revenue.astype(float), "worker_revenue", 1)
        self.util.save_file(file_name, optimal_path, "optimal_path", 2)
        self.util.save_file(file_name, float(optimal_profit), "optimal_profit", 1)
        self.util.save_file(file_name, acs_path, "acs_path", 2)
        self.util.save_file(file_name, float(acs_profit), "acs_profit", 1)
        self.util.save_file(file_name, float(optimal_time), "optimal_time", 1)
        self.util.save_file(file_name, float(acs_time), "acs_time", 1)
        self.util.save_file(file_name, task_quality_req.astype(float), "task_quality_req", 1)
        self.util.save_file(file_name, worker_quality_req.astype(float), "worker_quality_req", 1)
        self.util.save_file(file_name, num_workers_per_task.astype(float), "num_workers_per_task", 1)
        self.util.save_file(file_name, jaccard_values, "jaccard_values", 1)

    def generate_simulation(self):
        i = 0
        while i <= self.beginning + self.num_data_sample:
            num_nodes = self.num_workers + self.num_tasks

            # Define worker_list and task_list as numpy arrays
            worker_list = np.arange(self.num_workers)
            task_list = np.arange(self.num_workers, num_nodes)

            # Initialize acs_performance
            acs_performance = []

            # Create masks for worker and task locations
            worker_mask = np.arange(num_nodes) < self.num_workers
            task_mask = ~worker_mask

            # Extract worker and task locations
            loc_workers = self.location_matrix[worker_mask, :]
            loc_tasks = self.location_matrix[task_mask, :]

            # Generate adjacency matrices
            worker_adj_matrix = self.generate_worker_adjacency_matrix(self.num_workers, self.num_workers, 1)
            task_adj_matrix = self.generate_worker_adjacency_matrix(self.num_tasks, self.num_tasks, 2)

            # Generate worker-task adjacency matrix based on quality requirements
            task_quality_req = np.random.randint(self.quality_req_low, self.quality_req_high + 1, self.num_tasks)
            worker_quality_req = np.random.randint(self.quality_req_low, self.quality_req_high + 1,
                                                   (self.num_workers, self.num_tasks))

            worker_task_adj_matrix = np.zeros((self.num_workers, self.num_tasks), dtype=int)
            for j in range(self.num_workers):
                for k in range(self.num_tasks):
                    if worker_quality_req[j][k] >= task_quality_req[k]:
                        worker_task_adj_matrix[j][k] = 1

            # Create full location matrices initialized with -1
            loc_workers_full = np.full(self.location_matrix.shape, -1)
            loc_tasks_full = np.full(self.location_matrix.shape, -1)

            # Populate full location matrices
            loc_workers_full[worker_list, :] = loc_workers
            loc_tasks_full[task_list, :] = loc_tasks

            jaccard_values = self.calculate_jaccard_coefficient(worker_adj_matrix)
            distance_matrix = self.get_distance_matrix_all(self.location_matrix)

            num_workers_per_task = np.random.randint(self.num_workers_per_task_low,
                                                     self.num_workers_per_task_high + 1,
                                                     self.num_tasks)

            task_budget_unit = np.random.randint(self.task_budget_low, self.task_budget_high + 1)
            task_budget = np.full(self.num_tasks, task_budget_unit * num_workers_per_task)
            task_comp_windows = np.random.randint(self.task_req_time_low, self.task_req_time_high,
                                                  size=self.num_tasks)
            worker_capacity = np.random.randint(self.worker_capacity_low, self.worker_capacity_high + 1,
                                                self.num_workers)
            time_per_dis = np.random.randint(self.worker_time_per_dis_low, self.worker_time_per_dis_high + 1,
                                             self.num_workers)
            worker_revenue = np.random.randint(self.worker_req_rev_low, self.worker_req_rev_high + 1,
                                               self.num_workers)

            eligible_workers = set()
            for worker_index in range(self.num_workers):
                for task_index in range(self.num_tasks):
                    worker = worker_list[worker_index]
                    task = task_list[task_index]
                    temp_distance = distance_matrix[worker][task]
                    temp_time = temp_distance * time_per_dis[worker_index]
                    temp_revenue = temp_distance * worker_revenue[worker_index]
                    if (temp_distance <= worker_capacity[worker_index] and temp_time <= task_comp_windows[
                        task_index] and
                            temp_revenue <= task_budget[task_index] and worker_quality_req[worker_index][
                                task_index] >= task_quality_req[task_index]):
                        eligible_workers.add(worker)

            eligible_workers = list(eligible_workers)
            start_time = time.time()

            max_workers_per_task = min(len(eligible_workers), global_variables.max_workers_per_task)
            total_columns = int(sum(comb(len(eligible_workers), k) for k in range(1, max_workers_per_task + 1)))
            all_combo_groups = np.zeros((self.num_workers, total_columns), dtype=int)
            column_index = 0

            worker_index_map = {worker: index for index, worker in enumerate(worker_list)}

            for group_size in range(1, max_workers_per_task + 1):
                for group in combinations(eligible_workers, group_size):
                    original_indices = [worker_index_map[worker] for worker in group]
                    all_combo_groups[original_indices, column_index] = 1
                    column_index += 1

            # obtain the initial action space
            init_action_space = np.zeros((self.num_workers, self.num_tasks + 1))
            for worker_index in range(self.num_workers):
                for task_index in range(self.num_tasks):
                    worker = worker_list[worker_index]
                    task = task_list[task_index]
                    temp_distance = distance_matrix[worker][task]
                    temp_time = temp_distance * time_per_dis[worker_index]
                    if not (temp_distance <= worker_capacity[worker_index] and temp_time <= task_comp_windows[
                        task_index] and
                            worker_quality_req[worker_index][task_index] >= task_quality_req[task_index]):
                        init_action_space[worker_index][task_index] = 1.0
                        worker_task_adj_matrix[worker_index][task_index] = 0

            for task_index, task in enumerate(task_list):
                actual_action_space = init_action_space[:, :-1]
                workers_available_list = np.where(np.any(actual_action_space == 0, axis=1))[0]
                workers_available = len(workers_available_list)
                if num_workers_per_task[task_index] > workers_available:
                    init_action_space[:, task_index] = 1.0

            # Calculate optimal solution
            optimal_profit, optimal_path = optimal_solution_var.optimal_solution(distance_matrix, worker_list,
                                                                                 task_list,
                                                                                 worker_capacity, task_comp_windows,
                                                                                 task_budget,
                                                                                 time_per_dis, worker_revenue,
                                                                                 num_workers_per_task,
                                                                                 all_combo_groups, eligible_workers,
                                                                                 worker_quality_req,
                                                                                 task_quality_req,
                                                                                 jaccard_values)

            end_time = time.time()
            optimal_time = end_time - start_time

            # Run ACS search
            start_time = time.time()
            acs_profit, acs_path = acssearch.acs_search(self.num_tasks, self.num_workers, distance_matrix,
                                                        worker_list,
                                                        task_list, worker_capacity, task_comp_windows, task_budget,
                                                        time_per_dis, worker_revenue, num_workers_per_task,
                                                        eligible_workers,
                                                        worker_quality_req, task_quality_req, jaccard_values,
                                                        self.iteration,
                                                        all_combo_groups)
            end_time = time.time()
            acs_time = end_time - start_time

            if optimal_profit != 0:
                acs_performance.append(acs_profit / optimal_profit)

            if optimal_profit != 0 or acs_profit != 0:
                file_name = self.file_location + "graph_" + str(i) + "/"
                if not os.path.exists(file_name):
                    os.makedirs(file_name)
                # Save the simulation
                self.save_simulation(i, self.location_matrix, loc_workers, loc_tasks, worker_adj_matrix,
                                     task_adj_matrix,
                                     worker_task_adj_matrix, distance_matrix, worker_list, task_list, task_budget,
                                     task_comp_windows, worker_capacity, time_per_dis, worker_revenue,
                                     num_workers_per_task, task_quality_req, worker_quality_req, init_action_space,
                                     optimal_path, optimal_profit, acs_path, acs_profit, optimal_time, acs_time,
                                     jaccard_values)
                i += 1
            else:
                continue

            if i > 1500:
                break


if __name__ == "__main__":
    training_nodes = 28
    training_num_workers = [12]
    training_num_data_sample = 500  # num of samples per setting, not total num of samples
    data_generator = DataGenerator(data=None, batch_size=1, sensing_area_dim=10, num_workers=12, num_tasks=16,
                                   task_budget_low=20, task_budget_high=60, task_req_time_low=60,
                                   task_req_time_high=80,
                                   num_workers_per_task_low=2, num_workers_per_task_high=4, worker_time_per_dis_low=2,
                                   worker_time_per_dis_high=8, worker_req_rev_low=2, worker_req_rev_high=10,
                                   worker_capacity_low=10, worker_capacity_high=20, quality_req_low=1,
                                   quality_req_high=10,
                                   iteration=20, num_data_sample=training_num_data_sample,
                                   file_location="./testing_dataset/training_dataset_28_new/")
    beginning = 0
    for num_worker in training_num_workers:
        num_task = training_nodes - num_worker
        data_generator.set_beginning(beginning)
        data_generator.set_graph_size(num_worker, num_task)
        data_generator.generate_simulation()
        beginning += training_num_data_sample
