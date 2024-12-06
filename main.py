import datetime
import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

import DQN_Agent as agent
import environment
import global_variables
import gat_model_mod as model
from replay_memory import ReplayMemory
from util import Util

"""
Codes for training and testing the learning agent
"""


class LearningMethod:

    def __init__(self, is_continue, is_training_mode=True, is_experiment=False, validation_size=20, training_size=500,
                 testing_size=20, band_size=50,
                 num_episode=1, dataset_location="./training_dataset_28/", is_hyper_para_tuning=False,
                 is_stop_guide=True):
        """
        Control training modes and set some basic parameters
        :param is_training_mode: the flag that denotes whether it is a training mode
        :param is_experiment: the flag that denotes whether it is an experiment
        :param validation_size: the size of the validation dataset
        :param training_size: the size of the training dataset
        :param testing_size: the size of the testing dataset
        :param num_episode: the maximum training episodes
        :param dataset_location: the location of the dataset
        :param is_hyper_para_tuning: the flag that denotes whether it is a hyper-parameter tuning process
        :param is_stop_guide: the flag that denotes whether it is a stop-guide process (i.e., whether stopping action is available)
        (i.e., whether stopping action is considered in both training and testing phases)
        """
        self.is_training_mode = is_training_mode
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        self.band_size = band_size
        self.num_episode = num_episode
        self.is_experiment = is_experiment
        self.dataset_location = dataset_location
        self.num_task = -1
        self.num_worker = -1
        self.is_hyper_para_tuning = is_hyper_para_tuning
        self.small_sampled_games = [20, 64, 639, 730, 740, 867, 769, 93, 775, 433, 231, 745, 23, 1, 735, 711, 423, 101,
                                    392, 770]
        self.is_stop_guide = is_stop_guide
        self.util = Util()
        self.is_continue = is_continue

    @staticmethod
    def sum_reward(trace, t_update, n, T, gamma):
        """
        Sum over reward from t_update to min(t_update+n, T)
        :param trace: the trace of the game that records all necessary rewards and states
        :param t_update: the time instant that needs to be updated
        :param n: the number of states that needs to be looked ahead
        :param T: the terminate time instant of the game
        :return:  summed reward from t_update to min(t_update+n, T)
        """
        rewards = []
        for step in trace:
            rewards.append(step[0])
        n_step_return = 0
        for t in range(t_update + 1, min(t_update + n, T) + 1):
            n_step_return += rewards[t - 1] * gamma ** (
                    t - t_update - 1)  # NOTE: reward saved with from-state of state transition
        return n_step_return

    def load_graphs(self, data_size, folder_name):
        """
        Load data into memory
        :param data_size: the size of the dataset
        :param folder_name: the location of the folder
        :return: dataset
        """
        dataset = []

        # record the running time of the greedy and acs methods
        optimal_time_record = []
        acs_time_record = []

        for count in range(data_size):
            file_name = folder_name + "graph_" + str(count) + "/"
            task_budget = np.load(file_name + "task_budget") / global_variables.normalization_step
            distance_matrix = np.load(file_name + "distance_matrix")
            file = open(file_name + "optimal_path", "rb")
            optimal_path = pickle.load(file)
            optimal_profit = np.load(file_name + "optimal_profit")
            file = open(file_name + "acs_path", "rb")
            acs_path = pickle.load(file)
            acs_profit = np.load(file_name + "acs_profit")
            file = open(file_name + "initial_paths", "rb")
            initial_paths = pickle.load(file)
            location_matrix = np.load(file_name + "location_matrix")
            worker_location = np.load(file_name + "worker_location")
            task_location = np.load(file_name + "task_location")
            worker_adj_matrix = np.load(file_name + "worker_adj_matrix")
            task_adj_matrix = np.load(file_name + "task_adj_matrix")
            worker_task_adj_matrix = np.load(file_name + "worker_task_adj_matrix")
            worker_revenue = np.load(file_name + "worker_revenue") / global_variables.normalization_step
            time_per_dis = np.load(file_name + "time_per_dis")
            worker_list = np.load(file_name + "worker_list")
            task_list = np.load(file_name + "task_list")
            task_comp_windows = np.load(file_name + "task_comp_windows")
            num_workers_per_task = np.load(file_name + "num_workers_per_task")
            jaccard_values = np.load(file_name + "jaccard_values")
            task_quality_req = np.load(file_name + "task_quality_req")
            worker_quality_req = np.load(file_name + "worker_quality_req")
            worker_capacity = np.load(file_name + "worker_capacity")
            initial_action_space = np.load(file_name + "initial_action_space")
            optimal_time = np.load(file_name + "optimal_time")
            acs_time = np.load(file_name + "acs_time")
            data_sample = (
                location_matrix, worker_location, task_location, worker_adj_matrix, task_adj_matrix,
                worker_task_adj_matrix,
                distance_matrix, worker_capacity, worker_revenue,
                time_per_dis, task_budget,
                task_comp_windows, initial_paths, optimal_profit, optimal_path, acs_profit, acs_path,
                worker_list, task_list, initial_action_space, num_workers_per_task,
                jaccard_values, task_quality_req, worker_quality_req)
            if optimal_time >= 0:
                dataset.append(data_sample)
                optimal_time_record.append(optimal_time)
                acs_time_record.append(acs_time)
        # calculate the average running time of the greedy method and the acs method
        avg_greedy_time = sum(optimal_time_record) / len(optimal_time_record)
        avg_acs_time = sum(acs_time_record) / len(acs_time_record)
        return dataset, avg_greedy_time, avg_acs_time, optimal_time_record, acs_time_record

    def set_result_location(self, num_worker, num_task):
        """
        Set the location of the result folder
        :param num_worker: the number of workers of experiments
        :param num_task: the number of tasks of experiments
        """
        self.num_task = num_task
        self.num_worker = num_worker

    def get_num_taken_tasks(self, paths):
        """
        Get the number of taken tasks over paths
        :param paths: the paths planned by a method
        :return:
        """
        taken_tasks = 0
        non_zero_mask = paths != -1
        columns_with_non_zeros = np.any(non_zero_mask, axis=0)
        taken_tasks = np.sum(columns_with_non_zeros)
        return taken_tasks

    def run_dqn_agent(self):
        """
        Run the DDQN agent to solve the problem.
        :return: None
        """

        folder_name = "model_parameters"

        # load files
        # the best_performance.txt is the best performance of the agent in the training process over validation dataset
        file_name_best_performance = folder_name + "/best_performance.txt"
        # the net_parameters.pt is the parameters of the online network
        file_network_name = folder_name + "/net_parameters.pt"
        # the net_target_parameters.pt is the parameters of the target network
        file_network_target_name = folder_name + "/net_target_parameters.pt"
        # the net_parameters_best.pt is the parameters of the online network that achieves the best performance
        file_network_best = folder_name + "/net_parameters_best.pt"
        # the epsilon.pt is the epsilon value of the epsilon-greedy policy
        file_name_epsilon = folder_name + "/epsilon.pt"
        # the performance.txt is the performance of the agent in the training process over validation dataset
        if global_variables.is_optimal_stopping:
            file_name_performance = folder_name + "/stopping_performance.txt"
        else:
            file_name_performance = folder_name + "/no_stopping_performance.txt"

        # the best_result.npy is the best result of problem instances in the training dataset
        file_name_best_result = folder_name + "/best_result.npy"
        # the best_actions.pkl is the action trace of problem instances in the training dataset
        file_name_best_actions = folder_name + "/best_actions.pkl"

        # load the best performance
        if os.path.exists(file_name_best_performance):
            with open(file_name_best_performance, 'r') as f:
                learning_performance_best = float(f.read())
        else:
            learning_performance_best = -100000

        # load dataset
        if self.is_experiment:
            data_size = self.testing_size
        else:
            data_size = self.training_size
        dataset, greedy_running_time, acs_running_time, greedy_time_record, acs_time_record = self.load_graphs(
            data_size, self.dataset_location)

        # create the environment
        env = environment.Environment(dataset, file_name_best_result, file_name_best_actions)

        # create the online network
        embed_model = model.GATQN(hidden_dim=global_variables.hidden_dim,
                                  embed_dim=global_variables.embed_dim,
                                  worker_node_attribute_size=7,
                                  task_node_attribute_size=8,
                                  worker_to_worker_edge_size=2,
                                  worker_to_task_edge_size=4,  # NOTE: replace int order to two binary feats
                                  task_to_task_edge_size=1,
                                  embed_iter=3).to(global_variables.device)
        # create the target network
        embed_target_model = model.GATQN(hidden_dim=global_variables.hidden_dim,
                                         embed_dim=global_variables.embed_dim,
                                         worker_node_attribute_size=7,
                                         task_node_attribute_size=8,
                                         worker_to_worker_edge_size=2,
                                         worker_to_task_edge_size=4,  # NOTE: replace int order to two binary feats
                                         task_to_task_edge_size=1,
                                         embed_iter=3).to(global_variables.device)

        # if during the experiment, load the best parameters of the online network and the target network
        # otherwise, load the saved parameters of the online network and the target network
        if self.is_experiment:
            file_network_name = file_network_best
            file_network_target_name = file_network_best

        # load the parameters of the online network and the target network
        if os.path.exists(file_network_name) and not self.is_hyper_para_tuning:
            embed_model.load_state_dict(torch.load(file_network_name, map_location=global_variables.dev))
        if os.path.exists(file_network_target_name) and not self.is_hyper_para_tuning:
            embed_target_model.load_state_dict(torch.load(file_network_target_name, map_location=global_variables.dev))
        else:
            for target_param, param in zip(embed_target_model.parameters(), embed_model.parameters()):
                target_param.data.copy_(param)

        # load hyper-parameters
        n_step = global_variables.n_step
        gamma = global_variables.gamma
        learning_rate = global_variables.learning_rate
        if os.path.exists(file_name_epsilon) and not self.is_hyper_para_tuning:
            epsilon = torch.load(file_name_epsilon, map_location=global_variables.dev)
        else:
            epsilon = 1

        # create the agent
        replay_memory = ReplayMemory(global_variables.maximum_buffer)
        dqn_agent = agent.DQNAgent(models=[embed_model, embed_target_model], replay_memory=replay_memory)

        # if the agent is not in the training mode, then the epsilon is set to 0 nad exploration is not allowed
        if not self.is_training_mode:
            dqn_agent.epsilon = 0.0
        else:
            dqn_agent.epsilon = epsilon

        # get pointers of the training dataset
        data_pointers = np.arange(len(dataset))

        # randomly shuffle the pointers
        if not self.is_experiment:
            np.random.shuffle(data_pointers)

        # use global_variables.training_switch to control the training and validation process
        training_switch = global_variables.training_switch

        # record the performance of the different methods
        training_time_record = []

        training_ratio_all = self.util.load_data("training_ratio.npy", self.is_continue)
        greedy_ratio_train_all = self.util.load_data("greedy_ratio_training.npy", self.is_continue)
        acs_ratio_train_all = self.util.load_data("acs_ratio_training.npy", self.is_continue)
        training_results_all = self.util.load_data("training_results.npy", self.is_continue)
        greedy_results_train_all = self.util.load_data("greedy_results_training.npy", self.is_continue)
        acs_results_train_all = self.util.load_data("acs_results_training.npy", self.is_continue)
        testing_ratio_all = self.util.load_data("testing_ratio.npy", self.is_continue)
        greedy_ratio_test_all = self.util.load_data("greedy_ratio_testing.npy", self.is_continue)
        acs_ratio_test_all = self.util.load_data("acs_ratio_testing.npy", self.is_continue)
        testing_results_all = self.util.load_data("testing_results.npy", self.is_continue)
        greedy_results_test_all = self.util.load_data("greedy_results_testing.npy", self.is_continue)
        acs_results_test_all = self.util.load_data("acs_results_testing.npy", self.is_continue)
        training_loss_all = self.util.load_data("training_loss.npy", self.is_continue)
        training_path_all = self.util.load_data("training_path.npy", self.is_continue)
        greedy_path_train_all = self.util.load_data("greedy_path.npy", self.is_continue)
        acs_path_train_all = self.util.load_data("acs_path.npy", self.is_continue)
        testing_path_all = self.util.load_data("testing_path.npy", self.is_continue)
        greedy_path_test_all = self.util.load_data("greedy_path_testing.npy", self.is_continue)
        acs_path_test_all = self.util.load_data("acs_path_testing.npy", self.is_continue)

        # for each episode
        for ep in range(self.num_episode):
            training_results_per_episode = []
            greedy_results_per_episode = []
            acs_results_per_episode = []
            testing_result_per_episode = []
            training_ratio_per_episode = []
            greedy_ratio_per_episode = []
            acs_ratio_per_episode = []
            testing_ratio_per_episode = []
            training_loss_record = []

            if ep % training_switch != 0:
                # if current episode is the training episode, then the agent is in the training mode
                self.is_training_mode = True

                current_start_point = (self.validation_size - 1 + ep) % len(data_pointers)
                end_point = current_start_point + self.band_size

                if end_point <= len(data_pointers):
                    sampled_games = data_pointers[current_start_point:end_point]
                else:
                    sampled_games = np.concatenate(
                        (data_pointers[current_start_point:], data_pointers[:end_point - len(data_pointers)]))

                if (ep - 1) % training_switch == 0:
                    dqn_agent.epsilon = temp_epsilon

            else:
                self.is_training_mode = False
                if not self.is_experiment:
                    sampled_games = data_pointers[0:self.validation_size]
                else:
                    sampled_games = data_pointers[0:self.validation_size]
                temp_epsilon = dqn_agent.epsilon
                dqn_agent.epsilon = 0.0

            # if current process is for hyper-parameter tuning or verifying effectiveness of stopping action, then use the small sampled_games

            # put the agent into the environment
            dqn_agent.set_env(env)

            # scan each problem instance in the mini-training dataset or the validation dataset
            loss = None  # model not updated yet
            sample_number = 8
            if self.is_training_mode:
                n_samples = len(sampled_games)
            else:
                n_samples = len(sampled_games)

            for game_number in range(n_samples):  # repeat rounds in each episode and update epsilon at end
                data_sample_pointer = sampled_games[game_number]

                # print("game number", game_number, "real game number ", data_sample_pointer, "ep", ep,
                #       "*****************************************************************************************")
                # print("saving flag", global_variables.saving_flag)

                # save the parameters of the online network and the target network periodically
                if (
                        ep + 1) % global_variables.saving_period == 0 and self.is_training_mode and global_variables.saving_flag \
                        and not self.is_experiment and not self.is_hyper_para_tuning:
                    torch.save(embed_model.state_dict(), file_network_name)
                    torch.save(embed_target_model.state_dict(), file_network_target_name)
                    torch.save(dqn_agent.epsilon, file_name_epsilon)
                    np.save(file_name_best_result, env.best_exploration)
                    with open(file_name_best_actions, 'wb+') as f:
                        pickle.dump(env.best_action_traces, f)

                # reset the environment to the current problem instance
                env.reset(data_sample_pointer)

                # load results of baseline methods
                greedy_result, greedy_path = env.get_greedy()
                acs_result, acs_path = env.get_acs()

                paths = deepcopy(env.initial_paths)

                # initialize the state of the environment
                num_worker = env.num_workers
                num_task = env.num_tasks
                rest_capacity = env.worker_capacity.copy()
                action_space = env.initial_action_space.copy()
                num_workers_per_task = env.num_workers_per_task.copy()
                task_budget = env.task_budget.copy()
                current_time = np.zeros(num_worker)
                worker_stopped = np.zeros(num_worker)
                task_completed = np.zeros(num_task)
                current_partial_profit = np.zeros(num_task)
                state_helper = (paths, rest_capacity, current_time, num_workers_per_task, task_budget,
                                action_space, worker_stopped, task_completed, current_partial_profit)

                # initialize the Q-learning algorithm
                t = 0
                t_final = 100000
                game_trace = []
                # reward = 0
                action_trace = []
                # done = False
                current_profit = 0

                # is_random_episode is used to record whether the decisions for the current problem instance involve random exploration
                is_random_episode = False

                # starting_time is used to record the starting time of the current problem instance
                starting_time = time.time()

                while True:
                    if t < t_final:

                        # derive the action for the current state with epsilon-greedy policy
                        action, is_random_action = dqn_agent.derive_action(state_helper,
                                                                           data_sample_pointer,
                                                                           is_stop_guide=self.is_stop_guide,
                                                                           is_training_mode=self.is_training_mode)
                        is_random_episode = is_random_action or is_random_action

                        # transition to the next state
                        reward, done, next_profit, next_state_helper = env.step(state_helper, action, current_profit)

                        # record the step
                        game_step = (reward, state_helper, action, deepcopy(next_state_helper))
                        game_trace.append(game_step)
                        action_trace.append(action)

                        # if the current state is the terminal state, then set the t_final to the current time step plus 1
                        if done == 0:
                            t_final = t + 1
                            final_profit = next_profit

                            # ending_time is used to record the ending time of the current problem instance
                            ending_time = time.time()
                        state_helper = deepcopy(next_state_helper)
                        current_profit = next_profit

                    # t_update is the time step when the memory sample is available to be created
                    t_update = t - n_step + 1
                    if self.is_training_mode:
                        if t_update >= 0:
                            # get n-step return g
                            g = self.sum_reward(game_trace, t_update, n_step, t_final, gamma)
                            if (t_update + n_step - 1) < (t_final - 1):
                                done_t_update = 1  # NOTE: 1 -> not terminated
                                state_t_update_n_next = game_trace[t_update + n_step - 1][-1]
                            else:
                                done_t_update = 0  # NOTE: 0 -> terminated
                                state_t_update_n_next = game_trace[-1][-1]

                            # memorize the sample
                            mem_pos = dqn_agent.memorize(state=game_trace[t_update][1],
                                                         action=game_trace[t_update][2],
                                                         reward=g, next_state=state_t_update_n_next,
                                                         done=done_t_update, data_no=data_sample_pointer,
                                                         memory=dqn_agent.memory)

                            # memory replay if the memory size is larger than the batch size
                            memory_size = len(replay_memory.memory)
                            if memory_size >= dqn_agent.batch_size:
                                loss = dqn_agent.replay()
                                training_loss_record.append(loss)
                    # in testing mode, do NOT update model

                    if t_update >= t_final - 1:
                        # print("end of game **************************************************************************************")
                        real_profit = max(0, final_profit * global_variables.normalization_step)

                        # record the best result and best action trace of the current problem instance
                        if real_profit > env.best_exploration[data_sample_pointer] or len(
                                env.best_action_traces[data_sample_pointer]) == 0:
                            env.best_exploration[data_sample_pointer] = real_profit
                            env.best_action_traces[data_sample_pointer] = deepcopy(action_trace)
                            assert len(action_trace) > 0

                        # print test results
                        # if ep % training_switch == 0:
                        # print(f"************* game={game_number}/{data_sample_pointer}, "
                        #       f"ep={ep}, train={self.is_training_mode} *************")
                        # print(f"Best={env.best_exploration[data_sample_pointer]:.3f}, Greedy={greedy_result:.3f}, "
                        #       f"ACS={acs_result:.3f}, DRL={real_profit:.3f}")
                        # print(
                        #     f"DRL/greedy={real_profit / greedy_result:.3f}, DRL/ACS={real_profit / acs_result:.3f}")

                        # record the running time
                        training_time_record.append(ending_time - starting_time)

                        # record the profit ratios and absolute profits of the current problem instance
                        greedy_results_per_episode.append(greedy_result)
                        acs_results_per_episode.append(acs_result)
                        greedy_ratio_per_episode.append(greedy_result / greedy_result)
                        acs_ratio_per_episode.append(acs_result / greedy_result)

                        # record the paths of the current problem instance
                        (paths, _, _, _, _, _, _, _, _) = state_helper

                        if not self.is_training_mode:
                            testing_result_per_episode.append(real_profit)
                            testing_ratio_per_episode.append(real_profit / greedy_result)
                            greedy_path_test_all.append(greedy_path)
                            acs_path_test_all.append(acs_path)
                            testing_path_all.append(paths)

                        else:
                            training_results_per_episode.append(real_profit)
                            training_ratio_per_episode.append(real_profit / greedy_result)
                            training_path_all.append(paths)
                            greedy_path_train_all.append(greedy_path)
                            acs_path_train_all.append(acs_path)

                        # obtain the number of taken tasks
                        if real_profit != 0:
                            learning_taken_task = self.get_num_taken_tasks(next_state_helper[0])
                        else:
                            learning_taken_task = 0
                        acs_taken_task = self.get_num_taken_tasks(acs_path)
                        greedy_taken_task = self.get_num_taken_tasks(greedy_path)

                        break
                    t += 1

            # update the epsilon
            if dqn_agent.epsilon > dqn_agent.epsilon_min and self.is_training_mode:
                dqn_agent.epsilon *= dqn_agent.epsilon_decay
                # if loss is not None:
                #     print(f"Epsilon={dqn_agent.epsilon:.3f}, Learn_rate={learning_rate:.5f}, "
                #           f"Train_loss={loss:.4f}, Update_steps={dqn_agent.update_steps}")  # show latest loss

            # record avg results of each mini-batch of games
            print(f"-------------------------- end of ep {ep} -----------------------------")
            (paths, rest_capacity, current_time, num_workers_per_task, task_budget,
             action_space, worker_stopped, task_completed, current_partial_profit) = state_helper

            # save results at the end of each episode
            if self.is_training_mode:
                greedy_ratio_train_all.append(greedy_ratio_per_episode)
                acs_ratio_train_all.append(acs_ratio_per_episode)
                training_ratio_all.append(training_ratio_per_episode)
                greedy_results_train_all.append(greedy_results_per_episode)
                acs_results_train_all.append(acs_results_per_episode)
                training_results_all.append(training_results_per_episode)

                greedy_ratio_train_all_np = np.array(greedy_ratio_train_all)
                acs_ratio_train_all_np = np.array(acs_ratio_train_all)
                training_ratio_all_np = np.array(training_ratio_all)
                greedy_results_train_all_np = np.array(greedy_results_train_all)
                acs_results_train_all_np = np.array(acs_results_train_all)
                training_results_all_np = np.array(training_results_all)

                greedy_path_train_all_np = np.array(greedy_path_train_all)
                acs_path_train_all_np = np.array(acs_path_train_all)
                training_path_all_np = np.array(training_path_all)

                learning_performance = sum(training_ratio_per_episode) / len(training_ratio_per_episode)
                content = (training_ratio_all_np, greedy_ratio_train_all_np, acs_ratio_train_all_np,
                           training_results_all_np, greedy_results_train_all_np, acs_results_train_all_np,
                           training_path_all_np, greedy_path_train_all_np, acs_path_train_all_np)

                self.util.save_training_results(content, type="training")

                print(
                    f"avg_greedy={np.mean(greedy_ratio_per_episode[-n_samples:])}/{np.mean(greedy_ratio_per_episode[-n_samples:]):.3f}, "
                    f"avg_acs={np.mean(acs_ratio_per_episode[-n_samples:]):.3f}/{np.mean(acs_ratio_per_episode[-n_samples:]):.3f}, "
                    f"avg_learn={np.mean(training_ratio_per_episode[-n_samples:]):.3f}/{np.mean(training_ratio_per_episode[-n_samples:]):.3f}")

            else:
                greedy_ratio_test_all.append(greedy_ratio_per_episode)
                acs_ratio_test_all.append(acs_ratio_per_episode)
                testing_ratio_all.append(testing_ratio_per_episode)
                greedy_results_test_all.append(greedy_results_per_episode)
                acs_results_test_all.append(acs_results_per_episode)
                testing_results_all.append(testing_result_per_episode)

                greedy_ratio_test_all_np = np.array(greedy_ratio_test_all)
                acs_ratio_test_all_np = np.array(acs_ratio_test_all)
                testing_ratio_all_np = np.array(testing_ratio_all)
                greedy_results_test_all_np = np.array(greedy_results_test_all)
                acs_results_test_all_np = np.array(acs_results_test_all)
                testing_results_all_np = np.array(testing_results_all)

                greedy_path_test_all_np = np.array(greedy_path_test_all)
                acs_path_test_all_np = np.array(acs_path_test_all)
                testing_path_all_np = np.array(testing_path_all)

                learning_performance = sum(testing_ratio_per_episode) / len(testing_ratio_per_episode)
                content = (testing_ratio_all_np, greedy_ratio_test_all_np, acs_ratio_test_all_np,
                           testing_results_all_np, greedy_results_test_all_np, acs_results_test_all_np,
                           testing_path_all_np, greedy_path_test_all_np, acs_path_test_all_np)

                self.util.save_training_results(content, type="testing")
                print(
                    f"avg_greedy={np.mean(greedy_ratio_per_episode[-n_samples:])}/{np.mean(greedy_ratio_per_episode[-n_samples:]):.3f}, "
                    f"avg_acs={np.mean(acs_ratio_per_episode[-n_samples:]):.3f}/{np.mean(acs_ratio_per_episode[-n_samples:]):.3f}, "
                    f"avg_learn={np.mean(testing_ratio_per_episode[-n_samples:]):.3f}/{np.mean(testing_ratio_per_episode[-n_samples:]):.3f}")

            # print the average profit ratios of the current episode
            greedy_performance = sum(greedy_ratio_per_episode) / len(greedy_ratio_per_episode)
            acs_performance = sum(acs_ratio_per_episode) / len(acs_ratio_per_episode)

            app_ratio_result = [learning_performance, greedy_performance, acs_performance]
            if not self.is_training_mode and global_variables.saving_flag and not self.is_experiment and not self.is_hyper_para_tuning:
                # save current performance to minotor the learning process when the current episode is not
                # the training episode and the saving flag is true and the current run is not an experiment
                current_sys_time = datetime.datetime.now()
                with open(file_name_performance, 'a+') as f:
                    f.write(str(current_sys_time) + "----" + str(app_ratio_result) + '\n')
                if learning_performance_best < learning_performance:
                    torch.save(embed_model.state_dict(), file_network_best)
                    with open(file_name_best_performance, 'w+') as f:
                        f.write(str(learning_performance))
                    learning_performance_best = learning_performance

            # save experiment results
            result_folder_location = "./experiment_result/gdrl_result/"

            self.util.save_experiment_results(
                training_results_per_episode, greedy_results_per_episode, acs_results_per_episode,
                training_time_record, greedy_time_record, acs_time_record,
                training_ratio_per_episode, acs_ratio_per_episode,
                app_ratio_result, result_folder_location, self.is_experiment, self.is_hyper_para_tuning
            )

            training_loss_all = training_loss_all + training_loss_record
            training_loss_all_df = pd.DataFrame(training_loss_all)

            # save the performance of different groups of hyper-parameters
            if self.is_hyper_para_tuning and not self.is_experiment:
                file_name_training_process = "./experiment_result/training_process/" + "lr_" + str(
                    global_variables.learning_rate) + \
                                             "_batch_size_" + str(global_variables.batch_size) + "_" + str(
                    self.is_stop_guide) + "n-step_" + str(global_variables.n_step) + "_"
                # with open(file_name_training_process + "training_loss.txt", 'w+') as f:
                #     f.write(str(training_loss_all) + '\n')
                training_loss_all_df.to_excel(file_name_training_process + "training_loss.xlsx", index=False)
                if self.is_training_mode:
                    # with open(file_name_training_process + "training_performance.txt", 'w+') as f:
                    #     f.write(str(np.mean(training_ratio_all_np, axis=1)) + '\n')

                    training_ratio_all_df = pd.DataFrame(training_ratio_all_np)
                    training_ratio_all_df.to_excel(file_name_training_process + "training_performance.xlsx",
                                                   index=False)
                else:
                    # with open(file_name_training_process + "testing_performance.txt", 'w+') as f:
                    #     f.write(str(np.mean(testing_ratio_all_np, axis=1)) + '\n')
                    testing_ratio_all_df = pd.DataFrame(testing_ratio_all_np)
                    testing_ratio_all_df.to_excel(file_name_training_process + "testing_performance.xlsx", index=False)

        # training_loss_np = np.array(training_loss_record)
        # Util().save_learning_files(result_folder_location, training_loss_np, "training_loss", 1)

        # plot the results
        info = (global_variables.epsilon_decay, learning_rate, n_step, gamma, global_variables.batch_size,
                global_variables.tau, sample_number)

        # Util().plot_profits(result_folder_location, info, show_result=True)


if __name__ == "__main__":

    # set random seeds
    np.random.seed(1000)
    random.seed(100)
    torch.manual_seed(100)

    # is_experiment is used to indicate whether the current run is an experiment
    is_experiment = True

    # is_hyper_para_tuning is used to indicate whether the current run is for hyper-parameter tuning
    is_hyper_para_tuning = False

    if not is_experiment:
        if global_variables.is_optimal_stopping:
            learning_method = LearningMethod(is_continue=False,
                                             is_training_mode=True,
                                             is_experiment=is_experiment,
                                             validation_size=20,
                                             training_size=65,
                                             band_size=20,
                                             num_episode=1000,
                                             is_stop_guide=False,  # during training, we do not allow automatic stop
                                             is_hyper_para_tuning=is_hyper_para_tuning,
                                             dataset_location="./training_dataset_28/")
            learning_method.run_dqn_agent()
        else:
            learning_method = LearningMethod(is_continue=False,
                                             is_training_mode=True,
                                             is_experiment=is_experiment,
                                             validation_size=20,
                                             training_size=30,
                                             band_size=50,
                                             num_episode=1000000,
                                             is_stop_guide=True,
                                             is_hyper_para_tuning=is_hyper_para_tuning,
                                             dataset_location="./training_dataset_28/")
            learning_method.run_dqn_agent()

    else:
        folder_location = "./testing datasets/dataset_2/"
        learning_method = LearningMethod(is_continue=False,
                                         is_training_mode=False,
                                         is_experiment=is_experiment,
                                         testing_size=45,
                                         validation_size=45,
                                         training_size=500,
                                         band_size=50,
                                         num_episode=1,
                                         is_stop_guide=True,  # during testing, we allow automatic stop
                                         dataset_location=folder_location)
        learning_method.run_dqn_agent()
