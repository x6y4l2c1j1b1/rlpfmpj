from preprocess import preprocess
import numpy as np

def epsilon_greedy(e, q):
    if np.random.uniform(0,1) < e:
        return np.random.choice(len(q))
    else:
        m = np.max(q)
        indexes = [i for i in range(len(q)) if q[i] == m]
        return np.random.choice(indexes)

def calHStamp(index, timestamp, price, smooth):
	ans = 0
	delta = price[index, timestamp] - price[index, timestamp - 1]
	if delta > smooth:
		ans = 1
	elif delta < - smooth:
		ans = 2
	return ans 

def calH(loIndex, hiIndex, timestamp, price, smooth):
	h1 = calHStamp(hiIndex, timestamp - 2, price, smooth)
	l1 = calHStamp(loIndex, timestamp - 2, price, smooth)

	h2 = calHStamp(hiIndex, timestamp - 1, price, smooth)
	l2 = calHStamp(loIndex, timestamp - 1, price, smooth)

	return h1 * 1000 + l1 * 100 + h2 * 10 + l2	


def QLearning(gamma, lr, e, smooth):
	total_value = 1000000

	QTable = np.zeros((102222, 7))#total value fixed, now need to encode

	action_array = [-3,-2,-1,0,1,2,3]#convenient to look up
	lowP_array   = [0,1,2,3,4,5,6,7,8,9,10]#the proportion of low stock

	price, date_list = preprocess()

	pairs = [(i, j) for i in range(0, 10) for j in range(10, 20)]#all pairs

	for pair in pairs:#let the agent run through all low - high combination

		loIndex = pair[0]
		hiIndex = pair[1]
		print pair
		t = 4
		#the initial low - high ratio is 1 : 1
		lowP  = 5 
		state = lowP * 10000 + calH(loIndex, hiIndex, t, price, smooth)

		while(t < len(price[0])):
			action = epsilon_greedy(e, QTable[state, :])#restriction should be added here

			newlowP = lowP + action_array[action]

			state_next = newlowP * 10000 + calH(loIndex, hiIndex, t + 1, price, smooth)
			if newlowP < 0:
				t += 1
				continue
			reward = total_value * (newlowP * 0.1 * price[loIndex,t]/price[loIndex,t-1] + (10 - newlowP) * 0.1 * price[hiIndex,t]/price[hiIndex,t-1]) - total_value
			#print "reward: " + str(reward)
			QTable[state, action] = QTable[state, action] + lr * (reward + gamma * np.max(QTable[state_next, :]) - QTable[state, action])#restriction
			#print "q: " + str(state) + "a: "  + str(action) + " " + str(QTable[state, action])
			state  = state_next

			t += 1

	return QTable

Q = QLearning(gamma=0.95, lr=0.1, e=0.1, smooth=0.7)
print np.sum(Q)
