import glob
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import math

class Env(object):
    
    def __init__(self):
        high_beta_path =  r'Data/rawdata/high_beta'
        low_beta_path  =  r'Data/rawdata/low_beta'
        high_beta_file_names = glob.glob(high_beta_path+"/*.csv")
        low_beta_file_names  = glob.glob(low_beta_path+"/*.csv")
        
        self.hi_data = []
        self.low_data = []
        self.Hi_Li = "" #store merged data contain hi and lo data
        self.actions = [-0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25] #action list
        self.index_hi = -1  #high_beta stock index
        self.index_lo  = -1 #low_beta stock index
        
        self.state = "" #current state 
        self.t = 0      #current timestamp
        self.lc = 0     #current leftover cash
        self.tv = 0     #current total value
        for hi in range(len(high_beta_file_names)):
            self.hi_data.append(pd.read_csv(high_beta_file_names[hi]))
        for lo in range(len(low_beta_file_names)):
            self.low_data.append(pd.read_csv(low_beta_file_names[lo]))

    
    def reset(self, hi, lo):#create env object, using high_beta stock(index: hi) and low_beta stock(index: lo)
        self.index_hi = hi
        self.index_lo = lo
        
        Hi_Data = self.hi_data[hi]
        Li_Data = self.low_data[lo]
        
        Hi_Li = pd.merge(Hi_Data,Li_Data,on="Date")
        self.Hi_Li = Hi_Li
        self.t = 2
        
        #shareHi = 1000
        #shareLo = 1000
        
        #tv = 1000 * self.Hi_Li.loc[2]["Close_x"] + 1000 * self.Hi_Li.loc[2]["Close_y"]
        tv = 1000000
        self.tv = tv #save to global 
        
        chp = self.Hi_Li.loc[2]["Close_x"]
        clp = self.Hi_Li.loc[2]["Close_y"]
        
        shareHi = int(tv / (2 * chp)) #the maximum share of high beta stock half million can buy
        shareLo = int(tv / (2 * clp)) #the maximum share of low beta stock half million can buy
        
        lc = tv - shareHi * chp - shareLo * clp
        
        self.lc = lc #save to global 
        
        #generate state and save to global 
        self.state = [ shareHi, shareLo, Hi_Li.loc[2]["Close_x"], Hi_Li.loc[2]["Close_y"] ] 

    def take_action(self, a):
        
        action = self.actions[a] #action 
        
        ctv = self.tv       #current total value
        cstate = self.state #current state
        ct  = self.t        #current timestamp
        chp = self.Hi_Li.loc[ct]["Close_x"] #current price of high beta stock 
        clp = self.Hi_Li.loc[ct]["Close_y"] #current price of low beta stock 
        Hi_Li = self.Hi_Li
        lc  = self.lc           #current leftover cash 
        
        shareHi = cstate[0] #current share of high beta stock 
        shareLo = cstate[1] #current share of low beta stock 
        
        if action > 0:#sell low buy high:
            
            need_to_sell_mon = ctv * action
            
            share_to_sell    = min(int(need_to_sell_mon / clp), shareLo) 
            shareLo -= share_to_sell #update low share
            
            m = share_to_sell * clp  #money get by selling low
            
            m += lc                  #money could be used to buy high
            
            share_to_buy = int(m / chp)
            shareHi += share_to_buy  #update hi share 
            
            lc = m - share_to_buy * chp
        
        elif action < 0:#sell high buy low:
            
            action = -action
            
            need_to_sell_mon = ctv * action
            
            share_to_sell    = min(int(need_to_sell_mon / chp), shareHi)
            shareHi -= share_to_sell #update high share
            
            m = share_to_sell * chp  #money get by selling high
            
            m += lc                  #money could be used to buy low
            
            share_to_buy = int(m / clp)
            shareLo += share_to_buy  #update hi share 
            
            lc = m - share_to_buy * clp
        
        
        new_chp = Hi_Li.loc[ct + 1]["Close_x"]
        #print "new_chp " + str(new_chp)
        new_clp = Hi_Li.loc[ct + 1]["Close_y"]
        #print "new_clp " + str(new_clp)
        #print "lc: " + str(lc)
        new_tv  = new_chp * shareHi + new_clp * shareLo + lc 
        #print "new_TV: " + str(new_tv)
        reward  = new_tv - ctv 
        
        new_state = [shareHi, shareLo, new_chp, new_clp] 
        
        self.state =  new_state #update current state 
        self.t  = ct + 1        #update current timestamp
        self.lc = lc            #current leftover cash
        self.tv = new_tv        #current total value
        
        return new_state, reward, self.t == (Hi_Li.shape[0] - 1)