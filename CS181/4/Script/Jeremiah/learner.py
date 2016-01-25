import numpy as np
import numpy.random as npr
import os
import sys

from SwingyMonkey import SwingyMonkey

data_dir = "../../../Practical 4 Data/"

class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.estlen = 1e3
        self.speedList = []
        self.graviList = []
        self.impulList = []

        self.state_grid = []
        self.state_num = []

        # set discount/learning rate and epsilon
        self.disct = 0.9
        self.epsbase = 0.001

        # initiate Q matrix
        self.grid_x_len = 50
        self.grid_x_rgn = [-100, 450]
        self.grid_p_len = 50
        self.grid_p_rgn = [-300, 300]
        self.grid_v_len = 20
        self.grid_v_rgn = [-40, 40]


        self.grid_x_num = \
            round(float(self.grid_x_rgn[1] - self.grid_x_rgn[0])/self.grid_x_len) + 2
        self.grid_p_num = \
            round(float(self.grid_p_rgn[1] - self.grid_p_rgn[0])/self.grid_p_len) + 2
        self.grid_v_num = \
            round(float(self.grid_v_rgn[1] - self.grid_v_rgn[0])/self.grid_v_len) + 2

        self.Q = np.zeros([self.grid_x_num,
                           self.grid_p_num, self.grid_v_num, 2])

        # initiate learning time matrix
        self.learnTime = \
            np.zeros([self.grid_x_num, self.grid_p_num, self.grid_v_num, 2])


    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.state_grid = []
        self.state_num = []


    def state_index(self, state):
        ''''''
        '''
        1.  Calculate state parameters
        '''
        # state 1:  relative vertical position
        monkeyPos   = float(state["monkey"]["top"] + state["monkey"]["bot"])/2
        treePos     = float(state["tree"]["top"] + state["tree"]["bot"])/2
        state_p     = monkeyPos - treePos

        # state 2: horizontal position relative to "tree cave"
        state_x     = float(state["tree"]["dist"])

        # state 3: vertical velocity
        state_v     = float(state["monkey"]["vel"])

        '''
        2.  Calculate state index
        '''
        idx_x = int((state_x - self.grid_x_rgn[0])/self.grid_x_len) + 1
        idx_p = int((state_p - self.grid_p_rgn[0])/self.grid_p_len) + 1
        idx_v = int((state_v - self.grid_v_rgn[0])/self.grid_v_len) + 1

        if state_x < self.grid_x_rgn[0]:
            idx_x = 0
        elif state_x >= self.grid_x_rgn[1]:
            idx_x = int(self.grid_x_num - 1)

        if state_p < self.grid_p_rgn[0]:
            idx_p = 0
        elif state_p >= self.grid_p_rgn[1]:
            idx_p = int(self.grid_p_num - 1)

        if state_v < self.grid_v_rgn[0]:
            idx_v = 0
        elif state_v >= self.grid_v_rgn[1]:
            idx_v = int(self.grid_v_num - 1)

        return idx_x, idx_p, idx_v, [state_x, state_p, state_v]

    def result_callback(self):
        state = self.last_state
        screen_height = 400
        monkey_bot = state["monkey"]["bot"]
        monkey_top = state["monkey"]["top"]
        trunk_top = state["tree"]["top"]
        trunk_bot = state["tree"]["bot"]
        trunk_dist = state["tree"]["dist"]
        # Fail on hitting top or bottom.
        if monkey_bot > screen_height:
            res = ["Top Edge", trunk_dist]
        elif monkey_top < 0:
            res = ["Bot Edge", trunk_dist]
        elif monkey_top >= trunk_top:
            res = ["Top Tree", monkey_top - trunk_top]
        elif monkey_bot <= trunk_bot:
            res = ["Bot Tree", trunk_bot - monkey_bot]
        return res

    def action_callback(self, state):
        ''''''
        # random action if first time
        if self.last_state is None:
            new_action = (npr.rand() < 0.5)
            idx_x_new, idx_p_new, idx_v_new, stat_num_new = \
                self.state_index(state)

        else:
            """
            '''
            # 0. Running mean estimates of Gravity, Speed and Impulse
            '''
            #trim running list if it is too long
            if len(self.speedList) > self.estlen:
                self.speedList.pop(0)
            if len(self.graviList) > self.estlen:
                self.graviList.pop(0)
            if len(self.impulList) > self.estlen:
                self.impulList.pop(0)

            # estimate speed using distance to tree.
            # (Using old estimate during transition period)
            if state["tree"]["dist"] > 0 and self.last_state["tree"]["dist"] > 0:
                self.speedList.append(
                    self.last_state["tree"]["dist"] - state["tree"]["dist"]
                )

            # estimate gravity using velocity
            # (Using old estimate if previous action is 1)
            if self.last_action is False:
                self.graviList.append(
                    state["monkey"]["vel"] - self.last_state["monkey"]["vel"]
                )
            else:
                self.impulList.append(state["monkey"]["vel"])

            speed = np.mean(self.speedList)
            gravity = np.mean(self.graviList)
            impulse = np.mean(self.impulList)
            '''
            print str(speed) + "\t" + str(gravity) + "\t" + str(impulse) +\
                    "\t" + str(state["tree"]["dist"]) +\
                  "\t" + str(state["tree"]["dist"]/speed)
            '''
            """

            '''
            # 2. Identify state, determine learning rate, then update Q matrix
            '''
            idx_x_old, idx_p_old, idx_v_old, stat_num_old = \
                self.state_index(self.last_state)
            idx_x_new, idx_p_new, idx_v_new, stat_num_new = \
                self.state_index(state)

            '''
            # identify learning rate
            a_old = int(self.last_action)
            k_old = float(self.learnTime[idx_x_old, idx_p_old, idx_v_old, a_old])
            if k_old > 0:
                learn = 1/k_old
            else:
                #learn = 1
                ValueError("k_old should be > 0!")

            # update state and learn time
            Q_old = self.Q[idx_x_old, idx_p_old, idx_v_old, a_old]
            R_new = self.last_reward
            Q_max = np.max(self.Q[idx_x_new, idx_p_new, idx_v_new, :])
            Q_new = Q_old + learn * (R_new + self.disct * Q_max - Q_old)

            self.Q[idx_x_old, idx_p_old, idx_v_old, a_old] = Q_new
            '''

            max_Q = np.max(self.Q[idx_x_new,idx_p_new,idx_v_new, :])
            ALPHA = 1/self.learnTime[idx_x_old,idx_p_old,idx_v_old, self.last_action]
            self.Q[idx_x_old,idx_p_old,idx_v_old, self.last_action] += \
            ALPHA*(self.last_reward+
                   self.disct * max_Q -
                   self.Q[idx_x_old,idx_p_old,idx_v_old, self.last_action])

            '''
            # 3. select optimal policy
            '''
            new_action = 1 if \
                self.Q[idx_x_new, idx_p_new, idx_v_new][1] > \
                self.Q[idx_x_new, idx_p_new, idx_v_new][0] else 0
            k = self.learnTime[idx_x_new, idx_p_new, idx_v_new, new_action]
            # epsilon-greedy
            if k > 0:
                eps = self.epsbase/k
            else:
                eps = self.epsbase
            if (npr.rand() < eps):
                new_action = (npr.rand() < 0.5)

            '''
            # epsilon greedy
            opt_action_num = \
                np.argmax(self.Q[idx_x_new, idx_p_new, idx_v_new, :])
            opt_action = bool(opt_action_num)

            k = self.learnTime[idx_x_new, idx_p_new, idx_v_new, opt_action_num]
            if k == 0 or (npr.rand() < self.epsbase/k):
                #random action if haven't learned this state before
                new_action = (npr.rand() < 0.5)
            else:
                new_action = opt_action
            '''

            '''
            if new_action is True:
                print "(" + str(int(idx_x_new)) + "\t" + \
                      str(int(idx_p_new)) + "\t" + str(int(idx_v_new)) + ")" + "\t" + \
                      str(k) + \
                      "\t" + "Action:" + str(new_action) + " \t" + \
                      str(round(self.Q[idx_x_new, idx_p_new,
                                       idx_v_new, new_action], 3)) + \
                      " vs. " + \
                      str(round(self.Q[idx_x_new, idx_p_new,
                                       idx_v_new, 1 - new_action], 3))
            '''
        new_action_num = int(new_action)
        self.learnTime[idx_x_new, idx_p_new, idx_v_new, new_action_num] += 1
        self.last_action = new_action
        self.last_state  = state

        self.state_grid.append([idx_x_new, idx_p_new, idx_v_new])
        self.state_num.append(stat_num_new)


        """
        '''
        4. Random action Backup
        '''
        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        """

        return self.last_action

    def reward_callback(self, reward):
        '''
        This gets called so you can see what reward you get.
        '''

        self.last_reward = reward


# formal learning step
reset = True

if reset is True:
    # reset learning
    iters = 10000
    learner = Learner()
    reward = []
    score = []
    Qnorm = []

    state_grid = []
    state_num = []

    result = []
    score_cur = 0
    ii = 0

    #for ii in xrange(iters):

    # learner.Q = np.load("Qmat_manual.npy")
    # learner.learnTime = np.load("Lmat_manual.npy")

#while score_cur < 5000:
while ii < 1e5:
    ii += 1
    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=0,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass
    reward.append(learner.last_reward)
    score_cur = swing.get_state()["score"]
    veloc_cur = swing.get_state()["monkey"]["vel"]
    result_cur = learner.result_callback()
    qnorm = np.linalg.norm(learner.Q)
    score.append(score_cur)
    state_grid.append(learner.state_grid)
    state_num.append(learner.state_num)

    result.append(result_cur)
    Qnorm.append(qnorm)

    State = np.sum(learner.Q!=0)
    totalState = np.sum(learner.Q > -np.inf)

    if ii>0 and ii % 50 == 0:
        np.save(data_dir + "Qmat_backup.npy", learner.Q)
        np.save(data_dir + "Lmat_backup.npy", learner.learnTime)
        np.save(data_dir + "chain_backup.npy", score)
        np.save(data_dir + "state_grid_backup.npy", state_grid)
        np.save(data_dir + "state_num_backup.npy", state_num)

    '''
    print "################### Score = " + \
          str(swing.get_state()["score"]) + " ########################"
    '''
    minii = np.max([0, ii-1000])
    #minii = 0
    if ii % 10 == 0:
        print "Iter " + str(ii) + ": Score: " + str(score_cur) + \
              ", Mean: " + str(round(np.mean(score[minii:(ii-1)]), 3)) +\
              ",\t(" + result_cur[0] + ":\tDist:" + str(result_cur[1]) +\
              ", Vel:" + str(veloc_cur) +\
              ")\t" + str(State) + "/" + str(totalState)
    # Reset the state of the learner.
    learner.reset()


np.save(data_dir + "/data/last/Qmat_manual.npy", learner.Q)
np.save(data_dir + "/data/last/Lmat_manual.npy", learner.learnTime)
np.save(data_dir + "/data/last/chain_manual.npy", score)

np.save(data_dir + "/data/last/state_grid_manual.npy", state_grid)
np.save(data_dir + "/data/last/state_num_manual.npy", state_num)

