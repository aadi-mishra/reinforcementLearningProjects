import gym 
import random
from shower_env import ShowerEnv

def build_environment():

	env = ShowerEnv()

	
	return env
	

if __name__ == '__main__':

	# Let's create a random environment
	env = build_environment()

	episodes = 10
	for episode in range(1, episodes +1):
		state = env.reset()
		done = False
		score = 0
		
		while not done:
			action = env.action_space.sample()
			n_state, reward, done, info = env.step(action)
			score += reward
		print("Episode: {}  Score: {}".format(episode, score))


