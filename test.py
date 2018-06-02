'''
Test how our Q-network performs
Author: Guanya Shi
California Institute of Technology
gshi@caltech.edu
'''

#########################################################
#########################################################
# Import packages
import gym
import math
import time
import random
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

parser = argparse.ArgumentParser(description='DQN_test')
parser.add_argument('--model', type=str, default='target_net_32000.pth')
parser.add_argument('--CNN', type=bool, default=False)
args = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('SuperMarioBros-1-1-v0').unwrapped

'''
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
'''

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

class DQN_CNN(nn.Module):

    def __init__(self):
        super(DQN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 4, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(8)
        self.layer1 = nn.Linear(11*8*8, 100)
        self.layer2 = nn.Linear(100, 30)
        self.layer3 = nn.Linear(30, 4)

    def forward(self, x):
        # x = x.view(-1, 16*13*4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.layer1(x.view(x.size(0), -1)))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


#########################################################
#########################################################
# Input extraction
def get_screen_NN(screen1, screen2, screen3, screen4):
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

def get_screen_CNN(screen1, screen2, screen3, screen4):
    screen1 = torch.from_numpy(screen1)
    screen2 = torch.from_numpy(screen2)
    screen3 = torch.from_numpy(screen3)
    screen4 = torch.from_numpy(screen4)
    screen = torch.zeros(1,4,13,16)
    screen[:,0,:,:] = screen1
    screen[:,1,:,:] = screen2 
    screen[:,2,:,:] = screen3 
    screen[:,3,:,:] = screen4 
    output = screen.to(device)
    return output

def get_screen(screen1, screen2, screen3, screen4):
    if args.CNN:
        return get_screen_CNN(screen1, screen2, screen3, screen4)
    else:
        return get_screen_NN(screen1, screen2, screen3, screen4)

if args.CNN:
    test_net = DQN_CNN().to(device)
else:
    test_net = DQN().to(device)

test_net.load_state_dict(torch.load(args.model))

def select_action(state):
    sample = random.random()
    eps_threshold = 0.05
    if sample > eps_threshold:
        with torch.no_grad():
            return test_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


#########################################################
#########################################################
# Test
episode_durations = []
episode_distances = []
episode_ratios = []

screen_reset = env.reset()
num_episodes = 100
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
        else:
            screen8 = screen7
            reward2 = reward1

        reward = reward1 + reward2
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not Done:
            next_state = get_screen(screen2, screen4, screen6, screen8)
            # next_state = current_screen
        else:
            next_state = None

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        if Done:
            episode_durations.append(t + 1)
            episode_distances.append(info['distance'])
            episode_ratios.append((info['distance'])/(t+1))
            print('*******')
            print('Epoch: ' + str(i_episode))
            print('Duration: ' + str(t + 1))
            print('Distance: ' + str(info['distance']))
            print('Ratio: ' + str((info['distance'])/(t+1)))
            break

print('Complete')

plt.figure(1)
plt.title('Durations')
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.plot(episode_durations)
print(np.mean(episode_durations))
plt.savefig('Durations.png')

plt.figure(4)
plt.title('Distances')
plt.xlabel('Episode')
plt.ylabel('Distance')
plt.plot(episode_distances)
print(np.mean(episode_distances))
plt.savefig('Distances.png')

plt.figure(5)
plt.title('Ratios')
plt.xlabel('Episode')
plt.ylabel('Ratio')
plt.plot(episode_ratios)
print(np.mean(episode_ratios))
plt.savefig('Ratios.png')
plt.show()

env.close()