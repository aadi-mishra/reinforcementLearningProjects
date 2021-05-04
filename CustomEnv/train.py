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

from environment import build_environment

def build_model(states, actions):

	model = Sequential()
	model.add(Dense(24, activation='relu', input_shape=states))
	model.add(Dense(24, activation='relu'))
	model.add(Dense(actions, activation='linear'))
	
	return model
	
def build_agent(model, actions):
	policy = BoltzmannQPolicy()
	memory = SequentialMemory(limit=50000, window_length=1)
	dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=15, target_model_update=1e-2)
	
	return dqn


if __name__ == '__main__':
	env = build_environment()
	
	states = env.observation_space.shape
	actions = env.action_space.n

	# Create instance of our model  
	model = build_model(states, actions)
	print(model.summary())
	
	dqn = build_agent(model, actions)
	dqn.compile(Adam(lr=1e-3), metrics=['mae'])
	dqn.fit(env,nb_steps=50000, visualize = False, verbose=1)
	
	dqn.save_weights('customEnv_weights_v1.h5f', overwrite=True)
	

