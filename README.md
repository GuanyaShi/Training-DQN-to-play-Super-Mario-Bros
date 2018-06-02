# Training-DQN-to-play-Super-Mario-Bros.
We present a deep learning model to successfully learn control policies from high-dimensional input data using reinforcement learning. The model is based on the idea of Deep Q-Network (DQN), with convolutional neural network trained by Q-learning algorithm, whose input is tile representation of the screen and output is a value estimation function. Also, replay buffer, target network and double Q-learning are applied to lower data dependency and approximate real gradiant descent. We applied our model to Super Mario Bros., and get some preliminary results.

To run this code, you need to add the following environment into OpenAI Gym:
https://github.com/ppaquette/gym-super-mario
