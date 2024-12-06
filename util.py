import os
import shutil
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import networkx as nx
from optimal_solution import OptimalSolution

matplotlib.use('TkAgg')


class Util:
    def __init__(self):
        self.file_paths = {
            "greedy_ratio_testing": "greedy_ratio_testing",
            "acs_ratio_testing": "acs_ratio_testing",
            "testing_ratio": "testing_ratio",
            "greedy_results_testing": "greedy_results_testing",
            "acs_results_testing": "acs_results_testing",
            "testing_results": "testing_results",
            "training_results": "training_results",
            "training_ratio": "training_ratio",
            "greedy_results_training": "greedy_results_training",
            "acs_results_training": "acs_results_training",
            "greedy_ratio_training": "greedy_ratio_training",
            "acs_ratio_training": "acs_ratio_training"
        }
        self.result_folder = "./experiment_result/gdrl_result/"

    def rename_directories(self, parent_dir):
        # Get a list of all directories
        dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

        # Sort directories by the number in their name
        dirs.sort(key=lambda x: int(re.search(r'graph_(\d+)', x).group(1)))

        # Rename directories
        for i, dir_name in enumerate(dirs, start=0):
            old_dir_path = os.path.join(parent_dir, dir_name)
            new_dir_name = "graph_" + str(i)
            new_dir_path = os.path.join(parent_dir, new_dir_name)

            # Rename directory
            shutil.move(old_dir_path, new_dir_path)

    def delete_empty_subdirectories(self, directory):
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                if not os.listdir(subdir_path):  # Check if directory is empty
                    os.rmdir(subdir_path)  # Delete the directory

    def save_file(self, file_name, content, name, type):
        """
        Save data to the disk
        :param file_name: the folder
        :param content: saved data
        :param name: the name of the file
        :param type: the type of data, 1 for numpy array, 2 for python object
        :return: none
        """
        if type == 1:
            f = open(file_name + name, "wb+")
            np.save(f, content)
            f.close()
        if type == 2:
            f = open(file_name + name, "wb+")
            pickle.dump(content, f)
            f.close()

    def save_learning_files(self, file_name, content, name, type):
        """
        Save data to the disk in multiple formats
        :param file_name: the folder
        :param content: saved data
        :param name: the name of the file
        :param type: the type of data, 1 for numpy array, 2 for python object
        :return: none
        """
        if type == 1:
            # Save as numpy array and Excel
            file_path_npy = file_name + name + ".npy"
            with open(file_path_npy, "wb+") as f:
                np.save(f, content)

            file_path_excel = file_name + name + ".xlsx"
            df = pd.DataFrame(content)
            df.to_excel(file_path_excel, index=False)

        elif type == 2:
            # Save as pickle and Excel
            file_path_pkl = file_name + name + ".pkl"
            with open(file_path_pkl, "wb+") as f:
                pickle.dump(content, f)

            file_path_excel = file_name + name + ".xlsx"
            if isinstance(content, dict) or isinstance(content, list):
                df = pd.DataFrame(content)
            else:
                raise ValueError("Unsupported pickle content type. Only dictionaries and lists are supported.")
            df.to_excel(file_path_excel, index=False)
        elif type == 3:
            file_path_npy = file_name + name + ".npy"
            with open(file_path_npy, "wb+") as f:
                np.save(f, content)

    def load_data(self, file_name, is_continue):
        file_path = self.result_folder + file_name
        if os.path.exists(file_path) and is_continue:
            data = np.load(file_path)
            return data.tolist()
        else:
            return []

    def clear_dirs(self, file_location):
        """
        Clear files in a floder
        :return:
        """
        for root, dirs, files in os.walk(file_location):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    def save_training_results(self, content, type):
        """
        Save training results
        :param content: the content to be saved
        :param file_name: the file name
        :return: none
        """
        (learning_ratio, greedy_ratio, acs_ratio, learning_results, greedy_results, acs_results,
         learning_path, greedy_path, acs_path) = content

        # Save the results
        if type == "training":
            self.save_learning_files(self.result_folder, learning_results, "training_results", 1)
            self.save_learning_files(self.result_folder, learning_ratio, "training_ratio", 1)
            self.save_learning_files(self.result_folder, greedy_results, "greedy_results_training", 1)
            self.save_learning_files(self.result_folder, acs_results, "acs_results_training", 1)
            self.save_learning_files(self.result_folder, greedy_ratio, "greedy_ratio_training", 1)
            self.save_learning_files(self.result_folder, acs_ratio, "acs_ratio_training", 1)
            self.save_learning_files(self.result_folder, greedy_path, "greedy_path_training", 3)
            self.save_learning_files(self.result_folder, acs_path, "acs_path_training", 3)
            self.save_learning_files(self.result_folder, learning_path, "learning_path_training", 3)
        else:
            self.save_learning_files(self.result_folder, learning_results, "testing_results", 1)
            self.save_learning_files(self.result_folder, learning_ratio, "testing_ratio", 1)
            self.save_learning_files(self.result_folder, greedy_results, "greedy_results_testing", 1)
            self.save_learning_files(self.result_folder, acs_results, "acs_results_testing", 1)
            self.save_learning_files(self.result_folder, greedy_ratio, "greedy_ratio_testing", 1)
            self.save_learning_files(self.result_folder, acs_ratio, "acs_ratio_testing", 1)
            self.save_learning_files(self.result_folder, greedy_path, "greedy_path_testing", 3)
            self.save_learning_files(self.result_folder, acs_path, "acs_path_testing", 3)
            self.save_learning_files(self.result_folder, learning_path, "learning_path_testing", 3)

    def load_and_compute_data(self, file_path):
        if os.path.exists(file_path):
            data = np.load(file_path)
            return np.mean(data, axis=1)
        return None

    def _save_text_file(self, file_suffix, data, result_folder_location):
        file_path = os.path.join(result_folder_location, file_suffix)
        np.savetxt(file_path, np.array([data]), fmt='%.6f', delimiter=' ')

    def save_experiment_results(self, training_results_per_episode, greedy_results_per_episode, acs_results_per_episode,
                                training_time_record, greedy_time_record, acs_time_record,
                                training_ratio_per_episode, acs_ratio_per_episode,
                                app_ratio_result, result_folder_location, is_experiment, is_hyper_para_tuning):

        if not is_experiment or is_hyper_para_tuning:
            return

        # Compute performance and running time metrics
        learning_abs_performance = sum(training_results_per_episode)
        greedy_abs_performance = sum(greedy_results_per_episode)
        acs_abs_performance = sum(acs_results_per_episode)
        learning_running_time = np.mean(training_time_record)

        # File locations
        file_locations = {
            "app_ratio": "_app_ratio.txt",
            "abs": "_abs.txt",
            "time": "_time.txt",
            "learning_app_record": "_learning_app_record.txt",
            "acs_app_record": "_acs_app_record.txt",
            "learning_abs_record": "_learning_abs_record.txt",
            "acs_abs_record": "_acs_abs_record.txt",
            "greedy_abs_record": "_greedy_abs_record.txt",
            "learning_time_record": "_learning_time_record.txt",
            "acs_time_record": "_acs_time_record.txt",
            "greedy_time_record": "_greedy_time_record.txt"
        }

        abs_result = [learning_abs_performance, greedy_abs_performance, acs_abs_performance]
        running_time_result = [learning_running_time, np.mean(greedy_time_record), np.mean(acs_time_record)]

        # Save results
        self._save_text_file(file_locations["app_ratio"], app_ratio_result, result_folder_location)
        self._save_text_file(file_locations["abs"], abs_result, result_folder_location)
        self._save_text_file(file_locations["time"], running_time_result, result_folder_location)
        self._save_text_file(file_locations["learning_app_record"], training_ratio_per_episode, result_folder_location)
        self._save_text_file(file_locations["acs_app_record"], acs_ratio_per_episode, result_folder_location)
        self._save_text_file(file_locations["learning_abs_record"], training_results_per_episode,
                             result_folder_location)
        self._save_text_file(file_locations["acs_abs_record"], acs_results_per_episode, result_folder_location)
        self._save_text_file(file_locations["greedy_abs_record"], greedy_results_per_episode, result_folder_location)
        self._save_text_file(file_locations["learning_time_record"], training_time_record, result_folder_location)
        self._save_text_file(file_locations["acs_time_record"], acs_time_record, result_folder_location)
        self._save_text_file(file_locations["greedy_time_record"], greedy_time_record, result_folder_location)

    def plot_profits(self, result_folder_location, info, show_result):
        results = {key: self.load_and_compute_data(result_folder_location + value) for key, value in
                   self.file_paths.items()}
        (greedy_ratio_testing, acs_ratio_testing, testing_ratio, greedy_results_testing, acs_results_testing,
         testing_results,
         training_results, training_ratio, greedy_results_training, acs_results_training, greedy_ratio_training,
         acs_ratio_training) = results.values()
        if os.path.exists(result_folder_location + "training_loss"):
            loss = np.load(result_folder_location + "training_loss")

        fig1 = plt.figure(figsize=(12, 8))
        fig2 = plt.figure()

        (epsilon_decay, learning_rate, n_step, gamma, batch_size, tau, game_number) = info

        # First subplot
        ax1 = fig1.add_subplot(2, 1, 1)  # 2 rows, 1 column, plot number 1
        ax1.plot(greedy_ratio_training, label='Greedy')
        ax1.plot(acs_ratio_training, label='ACS')
        ax1.plot(training_ratio, label='Training')
        ax1.legend()
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Profit')

        ax1.set_title('Profit comparison between Greedy, ACS and Final')
        fig1.text(0.1, 0.1,  # Adjust these values as needed to position the text box
                  'epsilon_decay: {}\nlearning_rate: {}\nn_step: {}\ngamma: {}\nbatch_size: {}\ntau: {}\ngame_number: {}'.format(
                      epsilon_decay, learning_rate, n_step, gamma, batch_size, tau, game_number),
                  horizontalalignment='center',
                  verticalalignment='center')

        # Second subplot
        if show_result:
            ax2 = fig1.add_subplot(2, 1, 2)  # 2 rows, 1 column, plot number 2
            ax2.plot(loss)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('loss')
            ax2.set_title('loss of each episode')

        ax3 = fig2.add_subplot(2, 1, 1)  # 2 rows, 1 column, plot number 3
        ax3.plot(greedy_ratio_testing, label='Greedy')
        ax3.plot(acs_ratio_testing, label='ACS')
        ax3.plot(testing_ratio, label='Testing')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Profit')
        ax3.set_title('Profit of testing')

        ax3.legend()
        fig2.text(0.1, 0.1,  # Adjust these values as needed to position the text box
                  'epsilon_decay: {}\nlearning_rate: {}\nn_step: {}\ngamma: {}\nbatch_size: {}\ntau: {}\ngame_number: {}'.format(
                      epsilon_decay, learning_rate, n_step, gamma, batch_size, tau, game_number),
                  horizontalalignment='center',
                  verticalalignment='center')

        fig1.savefig(f'./experiment_result/training_process/fig_rltrain_{game_number}.png', dpi=300)
        fig2.savefig(f'./experiment_result/training_process/fig_rltest_{game_number}.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    # util = Util()
    # # util.delete_empty_subdirectories("./testing_dataset/training_dataset_28_new/")
    # util.rename_directories("./check/dataset_4/")
    experiment_location = "./experiment_result/gdrl_result/"
    greedy_path = np.load(experiment_location + "greedy_path_testing.npy")
    print(greedy_path[26])
    acs_path = np.load(experiment_location + "acs_path_testing.npy")
    print(acs_path[26])
    learning_path = np.load(experiment_location + "learning_path_testing.npy")
    print(learning_path[26])
    acs_ratio = np.load(experiment_location + "acs_ratio_testing.npy")
    learning_ratio = np.load(experiment_location + "testing_ratio.npy")

    jaccard_values = np.load("./testing datasets/dataset_2/graph_18/" + "jaccard_values")
    num_worker_per_task = np.load("./testing datasets/dataset_2/graph_18/" + "num_workers_per_task")
    # worker_location = np.array([
    #     [4, 0],
    #     [4, 1],
    #     [5, 7],
    #     [1, 4],
    #     [4, 8],
    #     [3, 3],
    #     [8, 3],
    #     [5, 2],
    #     [6, 6],
    #     [8, 2],
    #     [5, 5],
    #     [0, 3]
    # ])
    #
    # # Tasks' locations (x, y)
    # task_location = np.array([
    #     [1, 0],
    #     [6, 9],
    #     [5, 8],
    #     [1, 9],
    #     [3, 9],
    #     [1, 5],
    #     [6, 1],
    #     [2, 3],
    #     [1, 1],
    #     [6, 3],
    #     [9, 2],
    #     [0, 7],
    #     [2, 6],
    #     [9, 5],
    #     [8, 9],
    #     [0, 0]
    # ])

    worker_location = np.load("./testing datasets/dataset_2/graph_21/" + "worker_location")
    task_location = np.load("./testing datasets/dataset_2/graph_21/" + "task_location")

    path_matrix_latest = greedy_path[18]
    # Index of the task we are interested in
    task_index = 3

    all_paths = np.stack([greedy_path, acs_path, learning_path], axis=0)

    # Step 1: Count the number of workers assigned to each task for each path
    worker_counts = np.sum(all_paths > -1, axis=2)  # This gives a count of workers for each sample/task in each path

    # Step 2: Check if the counts are the same across all three path types
    same_worker_count = np.all(worker_counts == worker_counts[0], axis=0)

    non_zero_worker_counts = worker_counts > 1  # Create a boolean array for non-zero worker counts
    valid_same_worker_count = same_worker_count & np.all(non_zero_worker_counts,
                                                         axis=0)  # Combine with the same count condition

    # Find the tasks where the worker count is the same and non-zero
    tasks_with_non_zero_worker_count = np.where(valid_same_worker_count)

    # Display the indices of the tasks (sample number, task number) where worker count is the same and non-zero
    print(tasks_with_non_zero_worker_count)

    # For each task in tasks_with_non_zero_worker_count, calculate the average Jaccard value for the group of workers
    results = []

    for sample_idx, task_idx in zip(*tasks_with_non_zero_worker_count):
        # For each method, get the workers assigned to this task (ignoring worker identity, only checking for assignment > -1)
        greedy_workers = np.where(greedy_path[sample_idx, :, task_idx] > -1)[0]
        acs_workers = np.where(acs_path[sample_idx, :, task_idx] > -1)[0]
        learning_workers = np.where(learning_path[sample_idx, :, task_idx] > -1)[0]

        jaccard_values = np.load(f"./testing datasets/dataset_2/graph_{sample_idx}/" + "jaccard_values")
        # Calculate the average Jaccard value for each method
        avg_jaccard_greedy = OptimalSolution().average_jaccard_for_task(greedy_workers, jaccard_values)
        avg_jaccard_acs = OptimalSolution().average_jaccard_for_task(acs_workers, jaccard_values)
        avg_jaccard_learning = OptimalSolution().average_jaccard_for_task(learning_workers, jaccard_values)

        results.append({
            'Sample': sample_idx,
            'Task': task_idx,
            'Greedy Avg Jaccard': avg_jaccard_greedy,
            'ACS Avg Jaccard': avg_jaccard_acs,
            'Learning Avg Jaccard': avg_jaccard_learning
        })

    # Convert results to a DataFrame for easier visualization
    jaccard_results_df = pd.DataFrame(results)

    # Display the DataFrame with results
    print(jaccard_results_df)

    # Plotting workers and tasks
    plt.figure(figsize=(10, 8))

    # Plotting workers with a different symbol (e.g., 'x')
    for i, loc in enumerate(worker_location):
        plt.scatter(loc[0], loc[1], c='black', marker='x', label=f'Worker' if i == 0 else "", s=100)
        plt.text(loc[0] + 0.2, loc[1] + 0.2, f'W{i + 1}', fontsize=14)

    # Plotting tasks with the original symbol (circle)
    for i, loc in enumerate(task_location):
        if loc[0] != 4 and loc[1] != 8:
            plt.scatter(loc[0], loc[1], c='red', marker='o', label=f'Task' if i == 0 else "", s=100)
            plt.text(loc[0] + 0.2, loc[1] + 0.2, f'T{i + 1}', fontsize=14)
        else:
            loc[0] = 5
            plt.scatter(loc[0], loc[1], c='red', marker='o', label=f'Task' if i == 0 else "", s=100)
            plt.text(loc[0] + 0.2, loc[1] + 0.2, f'T{i + 1}', fontsize=14)

    worker_group = []

    # Highlighting workers assigned to task
    for worker_index, worker_path in enumerate(path_matrix_latest):
        if worker_path[task_index] > -1:
            worker_location_temp = worker_location[worker_index]
            worker_group.append(worker_index)
            task_location_temp = task_location[task_index]
            plt.arrow(worker_location_temp[0], worker_location_temp[1], task_location_temp[0] - worker_location_temp[0],
                      task_location_temp[1] - worker_location_temp[1], head_width=0.1, head_length=0.1, fc='blue',
                      ec='blue')

    # Adding legend
    plt.rcParams.update({'font.size': 18})
    plt.legend()
    plt.xlabel('X Coordinate', fontsize=18)
    plt.ylabel('Y Coordinate', fontsize=18)
    plt.grid(True)
    plt.show()

    worker_adj_matrix = np.load(f"./testing datasets/dataset_2/graph_18/" + "worker_adj_matrix")
    print(worker_adj_matrix)
    print(np.load(f"./testing datasets/dataset_2/graph_18/" + "jaccard_values"))

    # G = nx.from_numpy_matrix(worker_adj_matrix)
    #
    # # Draw the graph
    # plt.figure(figsize=(8, 6))
    # nx.draw(G)
    # plt.title('Graph Generated from Adjacency Matrix')
    # plt.show()
