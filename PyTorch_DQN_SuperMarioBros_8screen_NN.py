#########################################################
#########################################################
# Import packages
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('SuperMarioBros-1-1-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

#########################################################
#########################################################
# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#########################################################
#########################################################
# Q-network
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(16*13*4, 200)
        self.layer2 = nn.Linear(200, 100)
        self.layer3 = nn.Linear(100, 30)
        self.layer4 = nn.Linear(30, 4)

    def forward(self, x):
        x = x.view(-1, 16*13*4)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


#########################################################
#########################################################
# Input extraction
def get_screen(screen1, screen2, screen3, screen4):
    screen1 = np.resize(screen1, (16*13, 1))
    screen2 = np.resize(screen2, (16*13, 1))
    screen3 = np.resize(screen3, (16*13, 1))
    screen4 = np.resize(screen3, (16*13, 1))
    screen = np.concatenate((screen1, screen2, screen3, screen4), axis=0)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 4
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    output = screen.to(device)
    return output


#########################################################
#########################################################
# Training: Hyperparameters and utilities
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 400000
TARGET_UPDATE = 200

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.0001)
memory = ReplayMemory(100000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

episode_durations = []
episode_distances = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    if len(durations_t) % 100 == 0:
        plt.savefig('Duration_' + str(len(durations_t)) + '.png')

def plot_distances():
    plt.figure(3)
    plt.clf()
    distances_t = torch.tensor(episode_distances, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.plot(distances_t.numpy())
    # Take 100 episode averages and plot them too
    if len(distances_t) >= 100:
        means = distances_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.00001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    if len(distances_t) % 100 == 0:
        plt.savefig('Distance_' + str(len(distances_t)) + '.png')


#########################################################
#########################################################
# Training: Training loop
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


#########################################################
#########################################################
# Training
screen_reset = env.reset()
num_episodes = 30000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    screen1 = screen_reset
    screen2 = screen1
    screen3 = screen1
    screen4 = screen1
    screen5 = screen1
    screen6 = screen1
    screen7 = screen1
    screen8 = screen1
    state = get_screen(screen2, screen4, screen6, screen8)
    # state = current_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        A = action.item()
        if A == 0:
            Action = [0,0,0,0,0,0]
        elif A == 1:
            Action = [0,0,0,0,1,0]
        elif A == 2:
            Action = [0,0,0,1,0,0]
        elif A == 3:
            Action = [0,0,0,1,1,0]

        screen1 = screen3
        screen2 = screen4
        screen3 = screen5
        screen4 = screen6
        screen5 = screen7
        screen6 = screen8
        screen7, reward1, Done, info = env.step(Action)
        if not Done:
            screen8, reward2, Done, info = env.step(Action)
            if Done:
                reward2 = -2
        else:
            screen8 = screen7
            reward1 = -1
            reward2 = -1

        reward = reward1 + reward2 - 0.2
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not Done:
            next_state = get_screen(screen2, screen4, screen6, screen8)
            # next_state = current_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if Done:
            episode_durations.append(t + 1)
            episode_distances.append(info['distance'])
            plot_durations()
            plot_distances()

            print('*******')
            print('Epoch: ' + str(i_episode))
            print('Action: ' + str(A))
            print('Steps: ' + str(steps_done))
            print('Reward: ' + str(reward))
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % 100 == 0:
        # torch.save(policy_net.state_dict(), 'policy_net_' + str(i_episode) + '.pth')
        torch.save(target_net.state_dict(), 'target_net_' + str(i_episode) + '.pth')

print('Complete')
env.close()
plt.ioff()
plt.show()