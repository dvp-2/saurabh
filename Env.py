# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

Time_matrix = np.load('TM.npy')


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(i,j) for i in range(m) for j in range(m) if i!=j] + [(0,0)]
        self.state_space = [(i,j,k) for i in range(m) for j in range(t) for k in range(d)]  # 0th index is for number of locations, 1st is for number of hours and 2nd is for number of days
        self.state_init = [1,0,0]  # choosing initial pos = 1 arbitrarily
        self.TIME = 0
        # Start the first round
        self.reset()


    # Encoding state (or state-action) for NN input

    def state_encod_arch2(self, state):
        """convert the state into a vector so that it can be fed to the NN.
        This method converts a given state into a vector format.
        Hint: The vector is of size m + t + d."""
        pos, hr, day = state
        state_encod = [0]*(m+t+d)
        state_encod[pos] = 1
        state_encod[m+hr] = 1
        state_encod[m+t+day] = 1

        return state_encod


    # Use this function if you are using architecture-2

    # def state_encod_arch2(self, state, action):
    #
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format.
    #     Hint: The vector is of size m + t + d + m + m."""
    #     curr_pos, hr, day = state
    #     pick_pos, drop_pos = action
    #     state_encod = [0]*(m+t+d+m+m)
    #     state_encod[curr_pos] = 1
    #     state_encod[m+hr] = 1
    #     state_encod[m+t+day] = 1
    #     state_encod[m+t+day+pick_pos] = 1
    #     state_encod[m+t+day+pick_pos+drop_pos] = 1
    #
    #
    #     return state_encod


    ## Getting number of requests

    def get_requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        # all_actions = [(i,j) for i in range(m) for j in range(m) if i!=j]
        location = state[0]

        if location == 0: requests = np.random.poisson(2)
        elif location == 1: requests = np.random.poisson(12)
        elif location == 2: requests = np.random.poisson(4)
        elif location == 3: requests = np.random.poisson(7)
        else: requests = np.random.poisson(8)
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(len(self.action_space)-1), requests) # (0,0) is not considered as customer request
        possible_actions_index.append(20)
        actions = [self.action_space[i] for i in possible_actions_index]

        
        

        return possible_actions_index,actions   



    def reward_func(self, state, id_action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        curr_pos, hr, day = state
        action = self.action_space[id_action]

        if action == (0,0):
            reward = -C
        else:
            pick_pos, dest_pos = action
            # print('reward_func', pick_pos,dest_pos,hr,day)
            time_pq = Time_matrix[pick_pos][dest_pos][hr][day]
            time_ip = Time_matrix[curr_pos][pick_pos][hr][day]
            reward = R *(time_pq) - C * (time_ip+time_pq) 

        return reward




    def next_state_func(self, state, id_action, Time_matrix):
        """Takes state and action as input and returns next state"""
        curr_pos,hr,day = state
        action = self.action_space[id_action]

        if action == (0,0):
            dest_pos = curr_pos
            # print('(0,0)',curr_pos,dest_pos,hr,day)
            time = Time_matrix[curr_pos][dest_pos][hr][day]
            new_curr_pos, new_hr, new_day = curr_pos, int(time+hr), day
        else:
            pick_pos, drop_pos = action
            # print('nsf1',curr_pos,pick_pos,hr,day)
            time = Time_matrix[curr_pos][pick_pos][hr][day]
            new_hr_ = int((hr+time) % t)
            new_day_ = int((day + (hr+time)//t) % d)
            # print('nsf2',pick_pos,drop_pos,hr,day)
            time += Time_matrix[pick_pos][drop_pos][new_hr_][new_day_]
            new_hr = int((new_hr_+time)%t)
            new_day = int((new_day_ + (new_hr_+time)//t) % d)
            new_curr_pos = drop_pos

        next_state = [new_curr_pos, new_hr, new_day]
        self.TIME+=1
        return next_state



    def reset(self):
        self.TIME = 0
        return self.action_space, self.state_space, self.state_init
