from itertools import combinations
import numpy as np


class OptimalSolution:

    @staticmethod
    def average_jaccard_for_task(group, jaccard_values):
        """
        Calculate the average Jaccard index for a group of workers.
        :param group: list of worker indices in the group
        :param jaccard_values: matrix of Jaccard values between all workers
        :return: average Jaccard index for the group
        """
        # Check if only one worker is in the group
        if len(group) == 1 or len(group) == 0:
            return 1  # If only one worker is in the group, the Jaccard value is 1

        # Retrieve the Jaccard values for each pair of workers in the group
        jaccard_values_for_group = [jaccard_values[i][j] for i in group for j in group if i < j]

        # Calculate the average of these Jaccard values
        if jaccard_values_for_group:
            average_jaccard = sum(jaccard_values_for_group) / len(jaccard_values_for_group)
        else:
            average_jaccard = 0  # If no valid pairs are in the group, the average Jaccard value is 0

        return average_jaccard

    def optimal_solution(self, distance_matrix, worker_list, task_list, worker_capacity, task_comp_windows,
                         task_budget, time_per_dis, money_per_dis, num_workers_per_task, all_combo_groups,
                         eligible_workers, worker_quality_req, task_quality_req, jaccard_values):
        """
        Find the optimal task assignment for workers to maximize profit.
        :param distance_matrix: matrix of distances between workers and tasks
        :param worker_list: list of workers
        :param task_list: list of tasks
        :param worker_capacity: array of remaining capacities for each worker
        :param task_comp_windows: array of time windows for task completion
        :param task_budget: array of budgets for each task
        :param time_per_dis: array of time per unit distance for each worker
        :param money_per_dis: array of money per unit distance for each worker
        :param num_workers_per_task: array of number of workers required for each task
        :param all_combo_groups: matrix of all possible worker group combinations
        :param eligible_workers: array of eligible workers for each task
        :param worker_quality_req: matrix of worker quality requirements per task
        :param task_quality_req: array of quality requirements for each task
        :param jaccard_values: matrix of Jaccard values between all workers
        :return: total profit and the paths taken by workers
        """
        num_columns = all_combo_groups.shape[1]
        paths = np.full((len(worker_list), len(task_list)), -1)

        total_profit = 0
        rest_capacity = worker_capacity.copy()
        current_time = np.zeros_like(worker_capacity)

        for i, task in enumerate(task_list):
            assigned_distance = 0
            assigned_time = 0
            partial_budget = task_budget[i] / num_workers_per_task[i]
            partial_profit = 0

            for k in range(num_workers_per_task[i]):
                max_profit_worker = 0
                best_worker = -1

                for j, worker in enumerate(worker_list):
                    if rest_capacity[j] <= 0 or worker_quality_req[j][i] < task_quality_req[i] or paths[j, i] > -1:
                        continue

                    if np.any(paths[j, :] > -1):
                        last_position = np.argmax(paths[j, :])
                        last_position = task_list[last_position]
                    else:
                        last_position = worker

                    temp_distance = distance_matrix[last_position][task]
                    temp_time = current_time[j] + temp_distance * time_per_dis[j]
                    current_partial_profit = partial_budget - temp_distance * money_per_dis[j]

                    if temp_distance <= rest_capacity[j] and temp_time <= task_comp_windows[
                        i] and current_partial_profit > max_profit_worker:
                        max_profit_worker = current_partial_profit
                        best_worker = j
                        assigned_distance = temp_distance
                        assigned_time = temp_time

                if best_worker != -1:
                    current_task_path = paths[:, i]
                    workers_assigned = np.where(current_task_path > -1)[0]
                    average_jaccard = self.average_jaccard_for_task(workers_assigned, jaccard_values)
                    raw_profit = partial_profit / average_jaccard
                    workers_assigned = workers_assigned.tolist()
                    workers_assigned.append(best_worker)
                    current_average_jaccard = self.average_jaccard_for_task(workers_assigned, jaccard_values)
                    total_profit_current = ((max_profit_worker + raw_profit) * current_average_jaccard)
                    rest_capacity[best_worker] -= assigned_distance
                    current_time[best_worker] += assigned_time
                    paths_max_val = np.max(paths[best_worker, :])
                    paths[best_worker][i] = paths_max_val + 1
                    partial_profit = total_profit_current
            total_profit += partial_profit

        print("total_profit:", total_profit)
        paths = np.c_[paths, np.full((paths.shape[0], 1), -1)]
        return total_profit, paths
