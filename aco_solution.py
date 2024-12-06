import numpy as np
from copy import deepcopy
from optimal_solution import OptimalSolution
from itertools import combinations
import random

os = OptimalSolution()


class ACSSearch:

    def __init__(self):
        self.alpha = 1
        self.beta = 5
        self.rho = 0.05
        self.prob = 0.9
        self.pheromone = None
        self.initial = 0

    def calculate_value_step(self, all_combo_groups, task_number, value_vector,
                             num_workers_per_task, num_columns, last_index):
        value_step = np.zeros(num_columns)  # heuristic value
        for col_index in range(num_columns):
            column_sum = np.sum(all_combo_groups[:, col_index])
            if column_sum <= num_workers_per_task[task_number]:
                value_step[col_index] = (self.pheromone[task_number][last_index][col_index] ** self.alpha) * (
                        value_vector[col_index] ** self.beta)

        return value_step

    def calculate_value_vector(self, rest_capacity_ant, task_number, group, temp_distances, col_index, temp_times,
                               total_temp_profit,
                               value_vector, task_comp_windows, worker_quality_req, task_quality_req):
        if (all(temp_distance <= rest_capacity_ant[x] and temp_time <= task_comp_windows[task_number]
                for x, temp_distance, temp_time in
                zip(group, temp_distances, temp_times)) and
                all(worker_quality_req[x][task_number] >= task_quality_req[task_number] for x in group) and
                total_temp_profit > 0):  # constraints
            value_vector[col_index] = total_temp_profit  # positive value
        else:
            value_vector[col_index] = 0  # negative value to 0

        return value_vector

    def initialize_pheromone(self, initial, num_task, num_columns):
        self.pheromone = np.zeros((num_task, num_columns + 1, num_columns + 1))

        self.pheromone[self.pheromone == 0] = initial

    def update_pheromone_global(self, task_number, last_index, index, profit_best):
        self.pheromone[task_number][last_index][index] = (1 - self.rho) * \
                                                         self.pheromone[task_number][last_index][
                                                             index] + self.rho * profit_best

    def update_best_solution(self, total_profit_ant, sequence, paths_ant, profit_best, sequence_best, paths_best):
        if total_profit_ant > profit_best:
            profit_best = total_profit_ant
            sequence_best = sequence.copy()
            paths_best = deepcopy(paths_ant)
        return profit_best, sequence_best, paths_best

    def update_worker(self, x, group, task, task_number, budget_left, task_done_time_ant, current_time_ant,
                      rest_capacity_ant, distance_matrix, worker_revenue, time_per_dis, worker_list, paths, i,
                      partial_raw_profits_2):
        worker = worker_list[x]
        budget_left -= distance_matrix[worker][task] * worker_revenue[x]
        partial_raw_profits_2.append(budget_left)
        task_done_time = current_time_ant[x] + distance_matrix[worker][task] * time_per_dis[x]
        task_done_time_ant[task_number] = max(task_done_time_ant[task_number], task_done_time)
        current_time_ant[x] += distance_matrix[worker][task] * time_per_dis[x]
        rest_capacity_ant[x] -= distance_matrix[worker][task]
        paths_max_val = np.max(paths[x, :])
        paths[x][i] = paths_max_val + 1
        return budget_left, task_done_time_ant, current_time_ant, rest_capacity_ant, paths, partial_raw_profits_2

    def greedy_selection(self, value_step):
        value_select = np.max(value_step)
        index = np.argmax(value_step)
        return index, value_select

    def biased_random_selection(self, value_step, value_sum, num_workers_per_task, task_number, all_combo_groups):
        columns_with_sum_n = [col_index for col_index in range(all_combo_groups.shape[1])
                              if np.sum(all_combo_groups[:, col_index]) <= num_workers_per_task[task_number]]

        relative = [value_step[i] for i in columns_with_sum_n]
        relative = relative / value_sum

        # sample a new worker to index here
        index = np.random.choice(columns_with_sum_n, p=relative)
        value_select = value_step[index]
        return index, value_select

    def calculate_compatibility_score(self, group, jaccard_values):
        average_compatibility_score = 0
        compatibility_score = 0
        pair_count = 0
        for j in group:
            for k in group:
                if j != k:
                    compatibility_score += jaccard_values[j][k]
                    pair_count += 1
        if len(group) > 1:
            average_compatibility_score = compatibility_score / (pair_count / 2) if pair_count else 0
        return average_compatibility_score

    def initialize_paths_ant(self, all_combo_groups, eligible_workers):
        return {tuple([eligible_workers[index] for index in group_indices]): [] for group in all_combo_groups for
                group_indices in group}

    def acs_search(self, num_tasks, num_workers, distance_matrix, worker_list, task_list, worker_capacity,
                   task_comp_windows, task_budget, time_per_dis, worker_revenue, num_workers_per_task,
                   eligible_workers, worker_quality_req, task_quality_req, jaccard_values, iteration, all_combo_groups):

        profit_iteration = []
        paths_iteration = []
        sorted_tasks = np.argsort(task_comp_windows)  # sort tasks by ending time
        start = False  # signal of greedy start
        stop = 0  # signal to terminate searching
        num_ants = 10  # number of ants
        maximum_total_profit = 0  # overall maximum profit
        num_columns = all_combo_groups.shape[1]
        maximum_paths = np.full((len(worker_list), len(task_list)), -1)
        initial_pheromone = None
        best_profit = 0

        # start searching
        while stop < iteration:
            if not start:
                start = True  # end greedy start
                total_profit_greedy, paths_greedy = os.optimal_solution(
                    distance_matrix, worker_list, task_list, worker_capacity, task_comp_windows,
                    task_budget, time_per_dis, worker_revenue, num_workers_per_task, all_combo_groups,
                    eligible_workers, worker_quality_req, task_quality_req, jaccard_values)
                best_profit = total_profit_greedy
                paths_iteration.append(paths_greedy)
                profit_iteration.append(total_profit_greedy)  # record greedy result

                # initialize pheromone matrix
                initial_pheromone = total_profit_greedy / num_tasks
                self.initialize_pheromone(initial_pheromone, num_tasks, num_columns)

            else:
                best_sequence = np.zeros(num_tasks, dtype=int)  # worker sequence of best solution
                best_paths = np.full((len(worker_list), len(task_list)), -1)

                for ant in range(num_ants):
                    total_profit_ant = 0
                    task_assigned_ant = np.zeros(num_tasks, dtype=int)
                    task_done_time_ant = np.zeros(num_tasks, dtype=int)
                    paths_ant = np.full((len(worker_list), len(task_list)), -1)
                    rest_capacity_ant = worker_capacity.copy()
                    current_time_ant = np.zeros_like(worker_capacity)
                    sequence = np.zeros(num_tasks, dtype=int)  # sequence of current solution
                    value_vector = np.zeros(num_columns)  # value vector of workers for current task
                    last_index = num_columns

                    for task_number in sorted_tasks:
                        task = task_list[task_number]
                        partial_budget = task_budget[task_number] / num_workers_per_task[task_number]

                        for col_index in range(num_columns):
                            column_sum = np.sum(all_combo_groups[:, col_index])
                            if column_sum <= num_workers_per_task[task_number]:
                                group = np.where(all_combo_groups[:, col_index] == 1)[0]
                                temp_distances, temp_times = [], []
                                budget_left = partial_budget
                                partial_raw_profits = []

                                for worker_index in group:
                                    worker = worker_list[worker_index]
                                    if np.any(paths_ant[worker_index, :] > -1):
                                        last_position = np.argmax(paths_ant[worker_index, :])
                                        last_position = task_list[last_position]
                                    else:
                                        last_position = worker

                                    temp_distance = distance_matrix[last_position][task]
                                    temp_time = current_time_ant[worker_index] + temp_distance * time_per_dis[
                                        worker_index]
                                    budget_left -= temp_distance * worker_revenue[worker_index]
                                    temp_distances.append(temp_distance)
                                    temp_times.append(temp_time)
                                    partial_raw_profits.append(budget_left)

                                compatibility_score = os.average_jaccard_for_task(group, jaccard_values)
                                total_temp_profit = compatibility_score * sum(partial_raw_profits)

                                value_vector = self.calculate_value_vector(
                                    rest_capacity_ant, task_number, group, temp_distances, col_index,
                                    temp_times, total_temp_profit, value_vector, task_comp_windows,
                                    worker_quality_req, task_quality_req)

                        value_select = 0
                        best_group_index = num_columns
                        value_step = self.calculate_value_step(
                            all_combo_groups, task_number, value_vector, num_workers_per_task,
                            num_columns, last_index)

                        value_sum = np.sum(value_step)
                        if value_sum > 0:
                            prob_temp = np.random.rand()  # generate selection probability
                            if prob_temp < self.prob:  # greedy selection
                                best_group_index, value_select = self.greedy_selection(value_step)
                            else:  # biased random
                                best_group_index, value_select = self.biased_random_selection(
                                    value_step, value_sum, num_workers_per_task, task_number, all_combo_groups)

                        if value_select > 0 and best_group_index != num_columns:  # positive profit and not virtual worker
                            group = np.where(all_combo_groups[:, best_group_index] == 1)[0]
                            budget_left = partial_budget
                            partial_raw_profits_2 = []

                            for worker_index in group:
                                budget_left, task_done_time_ant, current_time_ant, rest_capacity_ant, paths_ant, \
                                    partial_raw_profits_2 = self.update_worker(
                                    worker_index, group, task, task_number, budget_left, task_done_time_ant,
                                    current_time_ant, rest_capacity_ant, distance_matrix, worker_revenue,
                                    time_per_dis, worker_list, paths_ant, task_number, partial_raw_profits_2)

                            compatibility_score = os.average_jaccard_for_task(group, jaccard_values)
                            task_assigned_ant[task_number] += 1  # update task assignment
                            total_temp_profit = compatibility_score * sum(partial_raw_profits_2)  # update total profit
                            total_profit_ant += total_temp_profit  # update total profit

                        # local update of pheromone
                        self.pheromone[task_number][last_index][best_group_index] = (
                                (1 - self.rho) * self.pheromone[task_number][last_index][best_group_index] +
                                self.rho * initial_pheromone)

                        # update sequence
                        sequence[task_number] = best_group_index
                        last_index = best_group_index

                    # update of best solution
                    best_profit, best_sequence, best_paths = self.update_best_solution(
                        total_profit_ant, sequence, paths_ant, best_profit, best_sequence, best_paths)

                # global update of pheromone
                last_index = num_columns
                for task_number in range(num_tasks):
                    index = best_sequence[task_number].astype(int)
                    self.update_pheromone_global(task_number, last_index, index, best_profit)
                    last_index = index

                profit_iteration.append(best_profit)  # record best profit
                paths_iteration.append(best_paths)

                # update overall maximum profit
                if best_profit > maximum_total_profit:
                    maximum_total_profit = best_profit
                    maximum_paths = deepcopy(best_paths)

                stop += 1  # go to next iteration

            print(f"{stop}: {profit_iteration[-1]}: {maximum_total_profit}")
        maximum_paths = np.c_[maximum_paths, np.full((maximum_paths.shape[0], 1), -1)]
        return maximum_total_profit, maximum_paths
