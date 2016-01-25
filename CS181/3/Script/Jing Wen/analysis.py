import os
import sys

import numpy as np
import scipy as sp
import scipy.stats as stat
import csv

import pandas as pd
from ggplot import *

os.chdir("/Users/Nika/Desktop/Class/Machine Learning/CS181-Practical-3/Script/Jing Wen")
# Predict via the median number of plays.

'''
#### #### #### #### #### #### ####
#### 1. Data Read in          ####
#### #### #### #### #### #### ####
'''

data_path = "../../../Data/"
train_file = data_path + 'train.csv'
user_file = data_path + 'profiles.csv'
arts_file = data_path + 'artists.csv'
test_file  = data_path + 'test.csv'
soln_file  = data_path + 'global_median.csv'



# Load the training data.
train_data = {}

with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = int(row[2])

        if not user in train_data:
            train_data[user] = {}

        train_data[user][artist] = plays

# Load the training data.
train_data_art = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = int(row[2])

        if not artist in train_data_art:
            train_data_art[artist] = {}

        train_data_art[artist][user] = plays


# Load the test data.
test_data = {}
with open(test_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[1]
        artist = row[2]

        if not user in test_data:
            test_data[user] = {}

        test_data[user][artist] = np.nan

# Load the user data.
user_data0 = []
with open(user_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        gender = row[1]
        if row[2] == '':
            age = np.nan
        else: age  = int(row[2])
        country = row[3]

        user_data0.append([user, gender, age, country])

user_data = np.vstack(user_data0)

# Load the artist data.
arts_data = {}
with open(arts_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        id   = row[0]
        name = row[1]

        arts_data[id] = [name]

arts_id = arts_data.keys()

'''
#### #### #### #### #### #### #### #### ####
#### 1.1 Prepare for Formal Analysis    ####
#### #### #### #### #### #### #### #### ####
'''

# dictionary between id and numeric code for use r& artist
user_dict_idx = dict(zip(range(len(train_data)), train_data.keys()))
arts_dict_idx = dict(zip(range(len(arts_data)), arts_data.keys()))

user_dict_nam = dict(zip(train_data.keys(), range(len(train_data))))
arts_dict_nam = dict(zip(arts_data.keys(), range(len(arts_data))))

#art data, unnormalized version
train_data1 = train_data_art
train_data_art = {}

for artName, arts in train_data1.iteritems():
    userID = [user_dict_nam[key] for key in arts.keys()]
    artName_new = arts_dict_nam[artName]
    art_new = dict(zip(userID, arts.values()))
    train_data_art[artName_new] = art_new


# rename user & artist by their numeric order, then
# normalize count to per-user frequency
train_data0 = train_data
train_data_user = {}

for userName, user in train_data0.iteritems():
    artID = [arts_dict_nam[key] for key in user.keys()]
    userName_new = user_dict_nam[userName]
    userVal_new = np.array(user.values()).astype("float")
    user_new = dict(zip(artID, userVal_new/np.sum(userVal_new)))
    train_data_user[userName_new] = user_new

train_data1 = train_data_art
train_data_art = {}

#swap order of data so its oriented toward artist
for artName, arts in train_data1.iteritems():
    userID = [user_dict_nam[key] for key in arts.keys()]
    artName_new = arts_dict_nam[artName]
    art_new = {}
    for id in userID:
        art_new[id] = train_data_user[id][artName_new]
    train_data_art[artName_new] = art_new


#extract importance (listening count)
user_imp_count = [sum(user.values())
                  for userName, user in train_data.iteritems()]
user_imp = user_imp_count/np.std(user_imp_count)

dat = train_data_art
weight = np.array(user_imp)


#art data, alternative normalization version
train_data1 = train_data_art
train_data_art = {}

for artName, arts in train_data1.iteritems():
    userID = [user_dict_nam[key] for key in arts.keys()]
    artName_new = arts_dict_nam[artName]
    art_new = dict(zip(userID, arts.values()/np.std(user_imp_count)))
    train_data_art[artName_new] = art_new



'''
#### #### #### #### #### #### ####
#### 2. Exploratory Analysis  ####
#### #### #### #### #### #### ####
'''

# 0. make a sparse matrix ##################
usart_dens= np.zeros((user_data.shape[0], arts_data.shape[0]))
arts_id = arts_data.T[0]

i = 0

for user, artists in train_data.iteritems():
    usart_dens[i] = \
        map(lambda id: artists[id] if id in artists.keys() else 0, arts_id)
    i += 1
    if i % 5000 == 0:
        print str(i) + " obs complete!"

# userat_dens = [[int(id in user_data.keys())
#                     for id in arts_id]
#                         for user, user_data in train_data.iteritems()]

# np.save("../../Data/user-artist.npy", usart_dens)
# usrat = np.load("../../Data/user-artist.npy")
# usrat_co = sp.sparse.coo_matrix(usrat)
# np.save("../../Data/user-artist matrix.npy", usrat_co)
usrat = np.load("../../Data/user-artist matrix.npy")


# basic statistics #######################
stat.itemfreq(user_data.T[2]) # check age
stat.itemfreq(user_data.T[3]) # check country

# total number of plays
plays_array  = []
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

plays_array = np.array(plays_array)
plays_quant = np.percentile(plays_array,
                            q = range(0, 100, 1))

freq_pd = pd.DataFrame(plays_array, columns=['Number of Plays'])
hist = ggplot(aes(x='Number of Plays'), data=freq_pd) + geom_histogram()
ggsave("hist_all.pdf", hist)

# high frequency user
user_medians = []
user_means = []
user_sds = []

for user, user_data in train_data_user.iteritems():
    user_play = np.array(user_data.values())
    user_medians.append(np.median(user_play))
    user_means.append(np.mean(user_play))
    user_sds.append(np.std(user_play))

user_means = np.array(user_means)
user_medians = np.array(user_medians)
user_sds = np.array(user_sds)

user_means_quant = np.percentile(user_means, q = range(0, 100, 1))
user_means_large = np.where(user_means > 10000)[0]
[train_data_user[idx] for idx in user_means_large]

np.percentile(user_medians, q = range(0, 100, 1))


# common users/artists between train and prediction data
user_tran = train_data.keys()
user_test = test_data.keys()
arts_tran = train_data_art.keys()

user_freq = [len(train_data[user]) for user in user_tran]

user_freq_pd = pd.DataFrame(user_freq, columns=['Number of Artist'])
hist = ggplot(aes(x='Number of Artist'), data=user_freq_pd) + geom_histogram()
ggsave("hist_user.pdf", hist)

arts_freq = [len(train_data_art[artist]) for artist in arts_tran]

arts_freq_pd = pd.DataFrame(arts_freq, columns=['Number of Audience'])
hist = ggplot(aes(x='Number of Audience'), data=arts_freq_pd) + geom_histogram()
ggsave("hist_arts.pdf", hist)



'''
#### #### #### #### #### #### ####
#### 3. Model Building        ####
#### #### #### #### #### #### ####
'''
# Compute overall mean and user/artist bias (initial values)
mean_user_dict = {}
count_art = {}
sum_all = 0
num_all = 0

sys.stdout.write("\t * Initial count preparing..")
for userName in dat.keys():
    user = dat[userName]
    mean_user_dict[userName] = np.mean(user.values())
    for artName in user.keys():
        if not artName in count_art:
            count_art[artName] = [user[artName]]
        else:
            count_art[artName].append(user[artName])
    sum_all += np.sum(user.values())
    num_all += len(user.values())
sys.stdout.write("Done!\n")

mean_all = float(sum_all) / num_all
mean_user = np.array(mean_user_dict.values())
mean_arts = np.array([np.mean(art_value)
                      for art_value in count_art.values()])

# initiate u: use overall sample mean
sys.stdout.write("\t * Initial parameters preparing..")
u_init = mean_all

# initiate bias: use sample bias
b_u_val = mean_user - mean_all
b_u_key = mean_user_dict.keys()

b_i_val = mean_arts - mean_all
b_i_key = count_art.keys()

bu_init = {b_u_key[i]: b_u_val[i] for i in range(len(b_u_key))}
bi_init = {b_i_key[i]: b_i_val[i] for i in range(len(b_i_key))}

sys.stdout.write("Done!\n")


# Compute the global median.
plays_array = []
for user, user_data in train_data.iteritems():
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
global_median = np.median(np.array(plays_array))
print "global median:", global_median


# Compute the global median and per-user median.
plays_array  = []
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

    user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))



'''
#### #### #### #### #### #### ####
#### 4. Prediction            ####
#### #### #### #### #### #### ####
'''

# Load the train data, assess in-sample fit
train_data_array = []
train_data_out = {}


plays_array = []
urmed_array = []
for user, user_data in train_data.iteritems():
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        urmed_array.append(user_medians[user][artist])
global_median = np.median(np.array(plays_array))


with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]

        id_user = user_dict_nam[user]
        id_arts = arts_dict_nam[artist]

        if not user in train_data_out:
            train_data_out[user] = {}
        pred = \
            user_imp_count[id_user] * \
            (U + B_i[id_arts] + B_u[id_user] + \
                np.inner(P_u[id_user], Q_i[id_arts]))
        pred = pred[0]

        train_data_out[user][artist] = pred
        train_data_array.append(pred)

train_data_array = np.array(train_data_array)
MAE = np.sum(np.abs(plays_array - train_data_array))/N_user

MAE = np.sum(np.abs(plays_array - global_median))/N_user


# Load the test data.
test_data_array = []
test_data_out = {}

with open(test_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[1]
        artist = row[2]

        id_user = user_dict_nam[user]
        id_arts = arts_dict_nam[artist]

        if not user in test_data_out:
            test_data_out[user] = {}
        pred = \
            user_imp_count[id_user] * \
            (U + B_i[id_arts] + B_u[id_user] + \
                np.inner(P_u[id_user], Q_i[id_arts]))
        pred = pred[0]

        test_data_out[user][artist] = pred
        test_data_array.append(pred)


'''
#### #### #### #### #### #### ####
#### 5. Output                ####
#### #### #### #### #### #### ####
'''

# Write out test solutions.
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]

            soln_csv.writerow([id, global_median])