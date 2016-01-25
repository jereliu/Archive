import numpy as np
from scipy import stats

import itertools as it
from ggplot import *
import pandas as pd

import pandas.rpy.common as com
import rpy2.robjects as robj
from rpy2.robjects import r
import rpy2.robjects.lib.ggplot2 as gg2


data_dir = "../../../Practical 4 Data/"
sett_dir = "/data/grid/e 0.001 avg 225 state 6636/"

'''
# Analysis
'''
state_num = \
    np.load(data_dir + sett_dir + "state_num_manual.npy")
polcy_Q = \
    np.load(data_dir + sett_dir + "Qmat_manual.npy")

'''
# 1. Generate dataset
'''

### space state travelled
state_num_np = np.array([np.array(item) for item in state_num])
state_num_np = np.vstack(state_num_np)
state_num_len = state_num_np.shape[0]

state_dth_np = \
    np.concatenate(
        np.array([np.append(np.zeros(len(item)-1), 1) for item in state_num])
    ).reshape((state_num_len, 1))

np.save(data_dir + sett_dir + "state_num_np.npy", state_num_np)

state_num_pd = pd.DataFrame(state_num_np,
                            columns = ["x", "p", "v"])
state_num_pd2 = state_num_pd.tail(100000)
state_num_pd2.index = range(0, 100000)


### space state when death
state_death_np = np.array([item[-1] for item in state_num])
state_death_pd = pd.DataFrame(state_death_np,
                              columns = ["x", "p", "v"])

### "mean" state strategy used for each state
def relDiff(pair):
    a = pair[0]
    b = pair[1]
    reldiff = float(abs(b-a))/(abs(a) + abs(b))
    return reldiff


decsn_Q = np.argmax(polcy_Q, axis = 3)
decsn_Q_mean = np.mean(decsn_Q, axis = 2)

Qmean_pd = pd.DataFrame(decsn_Q_mean)

# (compute center of each state:)
x_center = \
    np.append(np.append([-100], np.array(range(-100, 350, 25)) + 12.5), [350])
p_center = \
    np.append(np.append([-200], np.array(range(-200, 200, 10)) + 5), [200])

Qmean_pd.columns = p_center
Qmean_pd.index = x_center

# (convert to long format)
def expand_grid(*args, **kwargs):
    columns = []
    lst = []
    if args:
        columns += xrange(len(args))
        lst += args
    if kwargs:
        columns += kwargs.iterkeys()
        lst += kwargs.itervalues()
    return pd.DataFrame(list(it.product(*lst)), columns=columns)

Qmean_pd_long = expand_grid(x_center, p_center)
Qmean_pd_long["Qmean"] = np.array(Qmean_pd.stack())
Qmean_pd_long.columns = ["x", "p", "Qmean"]


# (convert to r object)
Qmean_r_long = {'x':  robj.FloatVector(Qmean_pd_long['x']),
                'p': robj.FloatVector(Qmean_pd_long['p']),
                'Qmean': robj.FloatVector(Qmean_pd_long['Qmean'])}
Qmean_r_long = robj.DataFrame(Qmean_r_long)

Qmean_r_long2 = com.convert_to_r_dataframe(Qmean_pd_long)
r.assign("Qmean_r_long", Qmean_r_long2)
r("save(Qmean_r_long, file ='./plot/R/Qmean_r_long.rdata') ")


'''
# 1. Generate plot
'''

# space state traveled
scatter2 = ggplot(state_num_pd2, aes(x = "x", y = "p")) + \
          geom_point(alpha = 0.1) + \
          geom_point(state_death_pd, aes(x = "x", y = "p"),
                     colour = "red", alpha = 0.1) +\
          geom_hline(yintercept = range(-200, 201, 10),
                     colour="darkred") +\
          geom_vline(xintercept = range(-100, 401, 25),
                     colour="darkred") +\
          xlab("Horizontal Difference") + ylab("Vertical Difference")


ggsave(scatter2, "./plot/s_scatter2.jpg")

# velocity state travelled
v_hist = ggplot(state_num_pd, aes(x = "v")) + \
          geom_bar() + \
          geom_vline(xintercept = range(-40, 21, 5),
                     colour="darkred") + \
          ggtitle("Velocity State Traveled")

ggsave(v_hist, "./plot/v_hist.jpg")

# velocity state when died
v_hist_d = ggplot(state_death_pd, aes(x = "v")) + \
           geom_bar() + \
           geom_vline(xintercept = range(-40, 21, 5),
                      colour="darkred") + \
           ggtitle("Velocity State when Death")


ggsave(v_hist_d, "./plot/v_death_hist.jpg")


'''
# "mean" state decided for each state
# (do this in r)
decn_mean = ggplot(Qmean_pd_long, aes(x = "x", y = "p")) + \
            geom_point(aes(color = "Qmean")) + \
            scale_colour_gradient(low = "white", high = "red")

ggsave(decn_mean, "./plot/meanDecision.jpg")
'''





'''
# 2. Generate Quantile-based Partition
'''

vel_perc = np.percentile(state_num_pd["v"], q = range(5, 96, 10))

bin_edges =\
    stats.mstats.mquantiles(
    state_num_np,
    prob = np.array(range(5, 96, 10)).astype("float")/100,
    axis = 0)


'''
# 2.1. Generate plot with alternative grid
'''


scatter2 = ggplot(state_num_pd2, aes(x = "x", y = "p")) + \
          geom_point(alpha = 0.1) + \
          geom_point(state_death_pd, aes(x = "x", y = "p"),
                     colour = "red", alpha = 0.1) +\
          geom_hline(yintercept = bin_edges.T[1],
                     colour="darkred") +\
          geom_vline(xintercept = bin_edges.T[0],
                     colour="darkred") +\
          xlab("Horizontal Difference") + ylab("Vertical Difference")


ggsave(scatter2, "./plot/s_scatter_grid.jpg")

# velocity state travelled
v_hist = ggplot(state_num_pd, aes(x = "v")) + \
          geom_bar() + \
          geom_vline(xintercept = bin_edges.T[2],
                     colour="darkred") + \
          ggtitle("Velocity State Traveled")

ggsave(v_hist, "./plot/v_hist_grid.jpg")
