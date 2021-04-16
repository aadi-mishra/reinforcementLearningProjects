import gym 
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Import keras RL dependencies
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from dqn_train import build_model, build_agent
from environment import build_environment

if __name__ == '__main__':

	env, states, actions = build_environment()
	
	# Create instance of our model  
	model = build_model(states, actions)
	print(model.summary())
	
	dqn = build_agent(model, actions)
	dqn.compile(Adam(lr=1e-3), metrics=['mae'])
	
	dqn.load_weights('dqn_weights_v1.h5f')
	
	_ = dqn.test(env, nb_episodes =10, visualize = True)
