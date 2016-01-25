import Queue
import numpy as np
import numpy.random as npr
import sys

from SwingyMonkey import SwingyMonkey

class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.estlen = 1e3
        self.speedList = []
        self.graviList = []
        self.impulList = []

        # set discount/learning rate and learning time
        self.disct = 0.9
        self.learn = 0.5

        # initiate Q matrix
        self.grid_x_len = 50
        self.grid_x_rgn = [-5, 10]
#        self.grid_y_len = 50
#        self.grid_y_rgn = [0, 1]
        self.grid_p_len = 50
        self.grid_p_rgn = [-0.2, 1.2]
        self.grid_v_len = 6
        self.grid_v_rgn = [-24, 24]

        self.grid_x_num = \
            round(float(self.grid_x_rgn[1] - self.grid_x_rgn[0])/self.grid_x_len) + 2
#        self.grid_y_num = \
#            round(float(self.grid_y_rgn[1] - self.grid_y_rgn[0])/self.grid_y_len)
        self.grid_p_num = \
            round(float(self.grid_p_rgn[1] - self.grid_p_rgn[0])/self.grid_p_len) + 2
        self.grid_v_num = \
            round(float(self.grid_v_rgn[1] - self.grid_v_rgn[0])/self.grid_v_len) + 2

        self.Q = np.zeros([self.grid_x_num, self.grid_y_num,
                           self.grid_p_num, self.grid_v_num, 2])

        # initiate learning time matrix
        self.learnTime = \
            np.zeros([self.grid_x_num, self.grid_y_num,
                      self.grid_p_num, self.grid_v_num])


    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def state_index(self, state, speed):
        ''''''
        '''
        1.  Calculate state parameters
        '''
#        # state 1:  vertical position, monkey
        monkeyPos   = float(state["monkey"]["top"] + state["monkey"]["bot"])/2
#        state_y     = float(monkeyPos)

        # state 2:  vertical position, tree
        treePos   = float(state["tree"]["top"] + state["tree"]["bot"])/2
        state_p     = monkeyPos - treePos

        # state 3: horizontal position relative to "tree cave"
        state_x     = float(state["tree"]["dist"])

        # state 4: vertical velocity
        state_v     = float(state["monkey"]["vel"])

        '''
        2.  Calculate state index
        '''
        idx_x = int(float(state_x - self.grid_x_rgn[0])/self.grid_x_len)
        idx_y = int(float(state_y - self.grid_y_rgn[0])/self.grid_y_len)
        idx_p = int(float(state_p - self.grid_p_rgn[0])/self.grid_p_len)
        idx_v = int(float(state_v - self.grid_v_rgn[0])/self.grid_v_len)

        if state_x < self.grid_x_rgn[0]:
            idx_x = self.grid_x_num - 2
        elif state_x >= self.grid_x_rgn[1]:
            idx_x = self.grid_x_num - 1

        if state_p < self.grid_p_rgn[0]:
            idx_p = int(self.grid_p_num - 2)
        elif state_p >= self.grid_p_rgn[1]:
            idx_p = int(self.grid_p_num - 1)

        if state_v < self.grid_v_rgn[0]:
            idx_v = int(self.grid_v_num - 1)
        elif state_v >= self.grid_v_rgn[1]:
            idx_v = int(self.grid_v_num - 2)

        return idx_x, idx_y, idx_p, idx_v


    def action_callback(self, state):
        ''''''
        if self.last_state is None:
            new_action = npr.rand() < 0.1
        else:
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

            '''
            # 2. Identify state, then update Q matrix and learning time
            '''
            if speed is None:
                speed = 25
            idx_x_old, idx_y_old, idx_p_old, idx_v_old = \
                self.state_index(self.last_state, speed)
            idx_x_new, idx_y_new, idx_p_new, idx_v_new = \
                self.state_index(state, speed)

            '''
            print str(idx_x) + ":" + str(state_x) + "\t" + \
                  str(idx_p) + ":" + str(state_p)
            '''
            # update state and learn time
            a_old = int(self.last_action)
            Q_old = self.Q[idx_x_old, idx_y_old, idx_p_old, idx_v_old, a_old]
            R_new = self.last_reward
            Q_max = np.max(self.Q[idx_x_new, idx_y_new, idx_p_new, idx_v_new])
            Q_new = Q_old + self.learn * (R_new + self.disct * Q_max - Q_old)

            self.Q[idx_x_old, idx_y_old, idx_p_old, idx_v_old, a_old] = Q_new

            self.learnTime[idx_x_old, idx_y_old, idx_p_old, idx_v_old] += 1

            '''
            # 3. select optimal policy
            '''
            # epsilon greedy
            k = float(self.learnTime[idx_x_new, idx_y_new, idx_p_new, idx_v_new])
            if k == 0:
                #random action if haven't learned this state before
                new_action = (npr.rand() < 0.1)
            else:
                decision_epsilon = \
                    np.argmax(npr.multinomial(1, [1 - 1.0 / (2 * k), 1.0 / (2 * k)]))
                decision_optimal = \
                    np.argmax(self.Q[idx_x_new, idx_y_new, idx_p_new, idx_v_new])
                new_action_num = np.abs(decision_epsilon - decision_optimal)
                new_action = bool(new_action_num)

            print "(" + str(int(idx_x_new)) + "\t" + str(int(idx_y_new)) + "\t" + \
                  str(int(idx_p_new)) + "\t" + str(int(idx_v_new)) + ")" + "\t" + \
                    str(k) +\
                    "\t" + "Action:" + str(new_action) + " \t" +\
                    str(round(self.Q[idx_x_new, idx_y_new, idx_p_new,
                                     idx_v_new, new_action],   3)) +\
                    " vs. " +\
                    str(round(self.Q[idx_x_new, idx_y_new, idx_p_new,
                                     idx_v_new, 1-new_action], 3))

        self.last_action = new_action
        self.last_state  = state

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
iters = 10000
learner = Learner()
reward = []
score = []
score_cur = 0
ii = 0

#for ii in xrange(iters):

while score_cur < 100:
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
    score.append(swing.get_state()["score"])

    print "################### Score = " + \
          str(swing.get_state()["score"]) + " ########################"
    # Reset the state of the learner.
    learner.reset()

