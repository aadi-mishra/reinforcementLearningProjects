import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam 
from environment import build_environment

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def build_model(states, actions):
	height, width, channels = states[0], states[1], states[2]
	
	model = Sequential()
	model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels))) 
	model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
	model.add(Convolution2D(64, (3,3), activation='relu'))
	model.add(Flatten())
	model.add(Dense(512, activation = 'relu'))
	model.add(Dense(256, activation = 'relu'))
	model.add(Dense(actions, activation = 'relu'))
	
	return model
	

def build_agent(model, actions):
	
	policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
	memory = SequentialMemory(limit=1000, window_length=3)
	dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network = True, dueling_type='avg', nb_actions = actions,  nb_steps_warmup = 1000)
	
	return dqn
	

if __name__ == '__main__':
	env, states, actions = build_environment()
	model = build_model(states, actions)
	#print(model.summary())
	agent = build_agent(model, actions)
	agent.compile(Adam(lr=1e-4))
	agent.fit(env, nb_steps=10000, visualize=False, verbose=2)
	agent.save_weights('atari_weights_v1.h5f', overwrite=True)
	
	
