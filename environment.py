import numpy as np
from preprocess import preprocess

def build_enrironment(high_beta_stock_index,low_beta_stock_index, smooth_index):
	"""
	input:
		chose a portfolio of one high_beta stock and one low_beta stock
		high_beta_stock_index: index from 0 to 9
		low_beta_stock_index: index from 10 to 19

	price_array:
		numpy array of (20, len(date_set)), close value of 20 stocks in each valid days

	state:
		total portfolio value, suppose we have 1000000 dollars
		discrete allocations of two stocks, [0/10,1/9, 2/8, 3/7, 4/6, 5/5, 6/4, 7/3, 8/2, 9/1,10/0], rounded to the nearest
		previous 2 history prices change, 0: not change too much(varies smaller than smooth_index), 1: rise 2: get down
		not added yet: left-over cash

		how to encode state: 
		history = 3-2 concate 2-1  (5-4 means the prices changes from 5 times ago to 4 times ago, low_beta first, high_beta next)
		state = total_value + state_low_beta_percent*10*1000 + history
		
	action:
		buy(if the numeric value if smaller then zero, it turns to sell) [-3,-2,-1,0,1,2,3] amount of low_beta stocks and buy corresponding high beta stock
		not added yet: transaction cost

	reward:
		the close value of today minus the close value of yesterday

	env:
		env[state] is a dictionary
		env[state][action] is reward (next state is fixed knowing the history and low_beta_percentage, when running the test time series, next state will generate with respect to the test data)
		env[state]['cnt'] is the times each state appears

	"""
	price, date_list = preprocess()
	env = {}
	for i in range(4, len(price[0])):
		print i
		h1=0
		l1=0
		h2=0
		l2=0
		h3=0
		l3=0
		if price[high_beta_stock_index,i-2] - price[high_beta_stock_index,i-3] > smooth_index:
			h1 = 1
		elif price[high_beta_stock_index,i-2] - price[high_beta_stock_index,i-3]  < -smooth_index:
			h1 = 2
		if price[low_beta_stock_index,i-2] - price[low_beta_stock_index,i-3] > smooth_index:
			l1 = 1
		elif price[low_beta_stock_index,i-2] - price[low_beta_stock_index,i-3]  < -smooth_index:
			l1 = 2
		if price[high_beta_stock_index,i-1] - price[high_beta_stock_index,i-2] > smooth_index:
			h2 = 1
		elif price[high_beta_stock_index,i-1] - price[high_beta_stock_index,i-2]  < -smooth_index:
			h2 = 2
		if price[low_beta_stock_index,i-1] - price[low_beta_stock_index,i-2] > smooth_index:
			l2 = 1
		elif price[low_beta_stock_index,i-1] - price[low_beta_stock_index,i-2]  < -smooth_index:
			l2 = 2
		if price[high_beta_stock_index,i] - price[high_beta_stock_index,i-1] > smooth_index:
			h3 = 1
		elif price[high_beta_stock_index,i] - price[high_beta_stock_index,i-1]  < -smooth_index:
			h3 = 2
		if price[low_beta_stock_index,i] - price[low_beta_stock_index,i-1] > smooth_index:
			l3 = 1
		elif price[low_beta_stock_index,i] - price[low_beta_stock_index,i-1]  < -smooth_index:
			l3 = 2
		history = h1*1000 + l1*100 + h2*10 + l2
		total_value = 1000000
		state_index_array = [total_value+x*10000 + history for x in [0,1,2,3,4,5,6,7,8,9,10]]
		action_array = [-3,-2,-1,0,1,2,3]
		for state_index in state_index_array:
			low_beta_percent = (state_index - total_value)//10000
			if state_index in env:
				### in this case we should use cnt to calculate the average reward
				cnt = env[state_index]['cnt']
				for action in range(-min(3, low_beta_percent), min(10-low_beta_percent, 3)):
					new_low_beta_percent = low_beta_percent + action
					reward = total_value * (new_low_beta_percent * 0.1 * price[low_beta_stock_index,i]/price[low_beta_stock_index,i-1] + (10-new_low_beta_percent) * 0.1 * price[high_beta_stock_index,i]/price[high_beta_stock_index,i-1]) - total_value	
					env[state_index][action] = (env[state_index][action] * cnt + reward)/(cnt+1)
					env[state_index]['cnt'] += 1
			else:
				env[state_index] = {}
				env[state_index]['cnt'] = 1
				for action in range(-min(3, low_beta_percent), min(10-low_beta_percent, 3)):
					new_low_beta_percent = low_beta_percent + action
					reward = total_value * (new_low_beta_percent * 0.1 * price[low_beta_stock_index,i]/price[low_beta_stock_index,i-1] + (10-new_low_beta_percent) * 0.1 * price[high_beta_stock_index,i]/price[high_beta_stock_index,i-1]) - total_value
					
					env[state_index][action] = reward
	return env

env = build_enrironment(1,11,0.7)
print env
print env.keys() ## some keys may not exist since we may never come across that scenario

					








