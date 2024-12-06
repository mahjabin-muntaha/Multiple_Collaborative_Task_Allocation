import torch

# control where the program runs (cpu or gpu)
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

lrelu_neg_slope = 0.2
mix_activate_tanh = True
eps_cnst = 1e-10  # deal with 0 when needed
max_value_scale = 50  # scale down rewards

# scale the input and output of the network
normalization_step = 1

# network model parameters
embed_dim = 64
hidden_dim = 64

# set reward of the "STOP" action
final_reward = 0

is_optimal_stopping = True

# set the minimum q value
minimum_value = -1000000

# you can vary parameters from here
# learning rate of the network

learning_rate =1e-3

# the number of bootstrapping steps
n_step = 1

# how the learning agent value future rewards
gamma = 1

# the size of the memory buffer for memory replay method
maximum_buffer = 20000

# the minimum of the epsilon in the epsilon-greedy method
epsilon_min = 0.1  # the decaying rate of the epsilon epsilon=epsilon*epsilon_decay
epsilon_decay = 0.999

# training batch size
batch_size = 32

# network updating rate for double DQN
tau = 0.01

# Every training_switch episode, the agent takes action without randomness, i.e., testing current performance
training_switch = 1

# Data generator parameters
sensing_length = 30
ending_time_low = 20
ending_time_up = 40
budget_low = 6
budget_up = 10
distance_low = 90
distance_up = 100
time_per_dis_up = 5
money_per_dis_up = 5

# saving_flag is used to control whether to save the model parameters
saving_flag = True
saving_period = 10

# experiment types
LCT = 0  # large-scale, varying num of tasks
LCW = 1  # large-scale, varying num of workers
SCT = 2  # small-scale, varying num of tasks
SCW = 3  # small-scale, varying num of workers

# experiment parameters
LCT_para = {
    'num_task': [20, 30, 40, 50, 60],
    'num_worker': [10]
}
LCW_para = {
    'num_task': [50],
    'num_worker': [4, 6, 8, 10, 12, 14, 16, 18, 20]
}
SCT_para = {
    'num_task': [30, 31, 32, 33, 34, 35, 36],
    'num_worker': [7]
}
SCW_para = {
    'num_task': [35],
    'num_worker': [5, 6, 7, 8, 9, 10]
}

max_workers_per_task = 5
