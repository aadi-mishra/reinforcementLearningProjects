import gym
import random

def build_environment():
	env = gym.make('SpaceInvaders-v0')
	height, width, channels = env.observation_space.shape
	actions = env.action_space.n 
	
	states = [height, width, channels] # Shape of the image output
	
	return  env, states, actions


if __name__ == '__main__':
	env, states, actions = build_environment()
	height = states[0]
	width = states[0]
	channels = states[0]
	
	# Print what the actions mean
	print(env.unwrapped.get_action_meanings())
	
	episodes = 5
	for episode in range(1,episodes+1):
		state = env.reset()
		done = False
		score = 0
		
		while not done:
			env.render()
			action = random.choice([0,1,2,3,4,5])
			n_state, reward, done, info = env.step(action)
			score += reward
	
		print("Episode: {} Score: {}".format(episode, score))
	env.close()
	
	
