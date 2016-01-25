import Queue
import os
import numpy as np
import numpy.random as npr
import sys

os.chdir("/Users/Nika/Desktop/Class/Machine Learning/CS-181-Practical-4/Script/Jing")

from threevar import Learner

# formal learning step
reset = False

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
while ii < 1e6:
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
        np.save("Qmat_backup.npy", learner.Q)
        np.save("Lmat_backup.npy", learner.learnTime)
        np.save("chain_backup.npy", score)

    '''
    print "################### Score = " + \
          str(swing.get_state()["score"]) + " ########################"
    '''
    minii = np.max([0, ii-500])
    #minii = 0
    print "Iter " + str(ii) + ": Score: " + str(score_cur) + \
          ", Mean: " + str(round(np.mean(score[minii:(ii-1)]), 3)) +\
          ",\t(" + result_cur[0] + ":\tDist:" + str(result_cur[1]) +\
          ", Vel:" + str(veloc_cur) +\
          ")\t" + str(State) + "/" + str(totalState)
    # Reset the state of the learner.
    learner.reset()

    np.save("Qmat_manual.npy", learner.Q)
    np.save("Lmat_manual.npy", learner.learnTime)
    np.save("chain_manual.npy", score)
