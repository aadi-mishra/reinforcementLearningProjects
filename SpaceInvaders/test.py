import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam 
from environment import build_environment

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from train import build_model, build_agent
from environment import build_environment

if __name__ == '__main__':

	env, states, actions = build_environment()

	# Create instance of our model  
	model = build_model(states, actions)
	print(model.summary())

	dqn = build_agent(model, actions)
	dqn.compile(Adam(lr=1e-4), metrics=['mae'])

	dqn.load_weights('dqn_weights.h5f')

	scores = dqn.test(env, nb_episodes =10, visualize = True)

	print(np.mean(scores.history['episode_reward']))
