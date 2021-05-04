from gym import Env # Placeholder class that allows us to build cust env on top
from gym.spaces import Discrete, Box # Allows what actions we can take and state of the env
import numpy as np
import random

'''
Building a reinforcement learning to adjust the temperature automatically
to get it in optimal temperature range 37-39 degrees
Episode Length : 60s (Shower Length)
Actions : Turn Down, leave, Turn Up

We Build a model later that keeps the optimal range as long as possible
We also model some noise that emulates temperature fluctuation

'''

#Class ShowerEnv inherits from Env class
class ShowerEnv(Env): 
	
	def __init__(self):
		# Actions we can take down, leave, up ->> 0,1,2
		self.action_space = Discrete(3)
		# Temperature range
		self.observation_space = Box(low=np.array([0]), high=np.array([100]))
		# Set starting temperature 
		self.state = 38 + random.randint(-3,3)
		# Set Shower length
		self.shower_length = 60
		
	def step(self, action):
		# Apply action
		# 0 -1 = -1 temp down
		# 1  0 =  0 temp leave
		# 2  1 =  1 temp up
		self.state += action - 1
		# Reduce shower length by 1 sec
		self.shower_length -= 1
		
		#Calculate Reward
		if self.state >= 37 and self.state <=39:
			reward = 1
		else:
			reward = -1
		
		if self.shower_length <=0:
			done = True
		else:
			done = False
		
		#Apply Temperature noise
		self.state = random.randint(-1,1)
		
		info = {}
		
		return self.state, reward, done, info
		
	
	
	
	def render(self):
		# Implement visualization
		pass
		
	def reset(self):
		self.state = 38 + random.randint(-3,3)
		self.shower_length = 60
		
		return self.state


