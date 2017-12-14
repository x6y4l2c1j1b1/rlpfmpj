# rlpfmpj

### to do list     
#### 1. Try new method
1.1 policy gradient        
1.2 replace nn with lstm

#### 2. Visualization and Evaluation
2.1 plot relevant graphs to visualize the trends and comparison between train, test and random        
2.2 use sharp ratio as reward          
2.3 add more layers and modify the number of neurons in each layers          
2.4 modify the states and add more history


### Current result:
* Training process:
2000 episode: 1,640,000 w-l: 1837-163
* Test:
100 episode: 1,000,000~ w-l: 92-8
* Random:
100 episode:

### some thoughts about the current result:
1. change the reward from the absolute value to relative value
2. applying prioritised replay
