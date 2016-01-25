import sys
import numpy as np
import scipy as sp
import time

from joblib import Parallel, delayed

def naive_model(dat, weight = None,
                max_iter = 100, tol = 0.001,
                step = 0.0001, lmda = 0.001, factorDim = 50,
                alpha = 0.5,
                u_init = None, bu_init = None, bi_init = None,
                p_init = None, q_init = None):
    """
    Estimate the item-item model on Koren and Bell (Page 176)
    :param dat:         Data, a dictionary of users
    :param max_iter:    maximum iteration
    :param step:        step size in Gradient Descent
    :param lmda:        penalize variable for square norm of latent variable
    :param factorDim:   dimension of latent factors
    :return:
        u:              overall mean
        b_u:            user-specific bias
        b_i:            item-specific bias
        q:              the item-specific latent factor
        y:              the item-specific user factor, implicit prfrnc
    """

    '''
    #### #### #### #### #### #### #### ####
    #### 0. Variable Initialization    ####
    #### #### #### #### #### #### #### ####
    '''
    sys.stdout.write("0. Data preparation initiated:\n")

    if u_init == None or bu_init == None or bi_init == None:
        # create mean listening frequency for all users/artists
        mean_art_dict = {}
        count_usr = {}

        sys.stdout.write("\t * Initial count preparing..")
        # calculate mean user frequency per artist, and
        # create a artist-specific dictionary contain only user frequency
        for artsName in dat.keys():
            user = dat[artsName]
            mean_art_dict[artsName] = np.median(user.values())
            for usrName in user.keys():
                if not usrName in count_usr:
                    count_usr[usrName] = [user[usrName]]
                else:
                    count_usr[usrName].append(user[usrName])
        sys.stdout.write("Done!\n")

        mean_arts = np.zeros(2000)#np.array(mean_art_dict.values())
        mean_user = np.array([np.median(usr_value)
                            for usr_value in count_usr.values()])


        # initiate u: use overall sample mean
        sys.stdout.write("\t * Initial parameters preparing..")
        plays_array = []
        for user, user_data in dat.iteritems():
            for artist, plays in user_data.iteritems():
                plays_array.append(plays)
        U = np.median(np.array(plays_array))

        sys.stdout.write("Done!\n")
    else:
        U = u_init
        B_u = bu_init
        B_i = bi_init

    # initiate latent factors: set to random digit b/w (0, 1)
    N_user = B_u.shape[0]
    N_arts = B_i.shape[0]

    if weight == None:
        weight = np.ones(N_user).reshape((N_user, 1))
    else:
        weight = np.array(weight).reshape((N_user, 1))

    # initiate bias: use sample bias
    B_u = np.array(mean_user - U)
    B_i = np.zeros(N_arts)#np.array(mean_arts - U)

    B_u = B_u.reshape((N_user, 1))
    B_i = B_i.reshape((N_arts, 1))
    P_u = np.zeros((N_user, factorDim))#(np.random.sample((N_user, factorDim)) - 0.5)/1000
    Q_i = np.zeros((N_arts, factorDim))#(np.random.sample((N_arts, factorDim)) - 0.5)/1000


    '''
    #### #### #### #### #### #### #### ####
    #### 1. Loops                      ####
    #### #### #### #### #### #### #### ####
    '''
    history = []

    sys.stdout.write("1. Optimization initiated:\n")
    report_freq = N_arts/5

    i = 0
    TotalError_prev = 0

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # iteration loop
    while True:
        t0 = time.time()
        sys.stdout.write("\t Iteration " + str(i+1) + ":")
        j = 0
        TotalError = 0

        # artist loop
        for id_arts in range(N_arts): # for each artist (N_arts)

            n_users = len(dat[id_arts])
            id_user = dat[id_arts].keys()

            r_i = np.array(dat[id_arts].values()).reshape((n_users, 1))
            r_i_hat = U + B_i[id_arts] + B_u[id_user] + \
                        np.inner(P_u[id_user],
                                 Q_i[id_arts]).reshape((n_users, 1))
            e_i = weight[id_user] * (r_i - r_i_hat)
            #TotalError.append(np.median(np.abs(e_i)))
            TotalError += np.sum(np.abs(e_i))

            #update B_u, B_i, Q_i, P_u
            B_u[id_user] += step * (e_i - lmda * B_u[id_user])

            e_Pu = e_i * P_u[id_user]
            for user in range(n_users):
                B_i[id_arts] += step * \
                                (e_i[user] - lmda * n_users * B_i[id_arts])
                Q_i[id_arts] += step * \
                                (e_Pu[user] - lmda * Q_i[id_arts])

            P_u[id_user] += step * (e_i * Q_i[id_arts] - lmda * P_u[id_user])

            #report
            j += 1

            #sys.stdout.write("Done!\n")
            if j % report_freq == 0:
                sys.stdout.write(".." + str(float(j)*100/ N_arts) + "%")
            #    sys.stdout.write(str(j) + ": " +
            #                     str(round(np.mean(TotalError), 4)) + "\n")

        '''
        TotalError += lmda * \
                      (np.sum(B_i**2) + np.sum(B_u**2) +
                       np.sum(Q_i**2) + np.sum(P_u**2)
                       )
        '''
        TotalError = round(float(TotalError)*np.std(user_imp_count)/N_user, 5)
        history.append(TotalError)
        t1 = time.time()
        sys.stdout.write("\n\t\t\t\t\t Done! (" + str(round((t1 - t0)/60, 3)) +
                         " min, MAE=" + str(round(TotalError, 5)) + ") \n")
        if np.abs(TotalError_prev - TotalError) < tol:
            break
        TotalError_prev = TotalError
        i += 1
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


    np.save("naiv_U.npy", U)
    np.save("naiv_Bu.npy", B_u)
    np.save("naiv_Bi.npy", B_i)
    np.save("naiv_Pu.npy", P_u)
    np.save("naiv_Qi.npy", Q_i)
    np.save("naiv_hist.npy", history)


    return B_u, B_i, P_u, Q_i, U







def item_model(dat, max_iter = 50, tol = 0.001,
               step = 0.002, lmda = 0.04, factorDim = 10,
               alpha = 0.5,
               u_init = None, bu_init = None, bi_init = None,
               q_init = None, x_init = None, y_init = None):
    """
    Estimate the item-item model on Koren and Bell (Page 176)
    :param dat:         Data, a dictionary of users
    :param max_iter:    maximum iteration
    :param step:        step size in Gradient Descent
    :param lmda:        penalize variable for square norm of latent variable
    :param factorDim:   dimension of latent factors
    :return:
        u:              overall mean
        b_u:            user-specific bias
        b_i:            item-specific bias
        q:              the item-specific latent factor
        x:              the item-specific user factor, explicit rating
        y:              the item-specific user factor, implicit prfrnc
    """

    '''
    #### #### #### #### #### #### #### ####
    #### 0. Variable Initialization    ####
    #### #### #### #### #### #### #### ####
    '''
    sys.stdout.write("0. Data preparation initiated:\n")


    if u_init == None or bu_init == None or bi_init == None:
        # create mean listening frequency for all users/artists
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

        mean_all = float(sum_all)/num_all
        mean_user = np.array(mean_user_dict.values())
        mean_arts = np.array([np.mean(art_value)
                            for art_value in count_art.values()])

        # initiate u: use overall sample mean
        sys.stdout.write("\t * Initial parameters preparing..")
        u = mean_all

        # initiate bias: use sample bias
        b_u_val = np.array(mean_user - mean_all)
        b_u_key = np.array(mean_user_dict.keys())

        b_i_val = np.array(mean_arts - mean_all)
        b_i_key = np.array(count_art.keys())

        b_u = {b_u_key[i]: b_u_val[i] for i in range(len(b_u_key))}
        b_i = {b_i_key[i]: b_i_val[i] for i in range(len(b_i_key))}

        sys.stdout.write("Done!\n")
    else:
        u = u_init
        b_u = bu_init
        b_i = bi_init

    # initiate latent factors: set to random digit b/w (0, 1)
    q = np.random.sample((len(b_i), factorDim))
    x = np.random.sample((len(b_i), factorDim))
    y = np.random.sample((len(b_i), factorDim))


    '''
    #### #### #### #### #### #### #### ####
    #### 1. Loops                      ####
    #### #### #### #### #### #### #### ####
    '''
    history = []

    sys.stdout.write("1. Optimization initiated:\n")
    N = len(b_u_key)
    report_freq = len(b_u_key)/20

    # iteration loop
    for i in range(max_iter):
        t0 = time.time()
        sys.stdout.write("\t Iteration " + str(i+1) + ": \n")
        j = 0
        TotalError = 0

        # user loop
        for userName, user in dat.iteritems(): # for each user:
            user_size = len(user)
            user_arts_idx = np.array(
                [np.where(b_i_key == key)[0][0] for key in user.keys()])

            # for user u, calculate p_u
            r_u = np.array(user.values())
            b_uj = np.array([b_i[arts] for arts in user.keys()]) + \
                            b_u[userName] + u

            p_u_x = np.inner((r_u - b_uj).reshape((1, user_size)),
                         x[user_arts_idx].T).T
            p_u_y = np.inner(np.ones((1, user_size)),
                             y[user_arts_idx].T).T
            p_u = np.power(user_size, -0.5)*(p_u_x + p_u_y)

            # update steps
            xy_step = np.zeros((factorDim, 1))
            b_u_val = np.array(b_u.values())
            b_i_val = np.array(b_i.values())

            # item loop, update b_u, b_i, q_i
            for artName, r_ui in user.iteritems():
                # obtain q_i
                artIdx = np.where(b_i_key == artName)[0][0]
                q_i = q[artIdx].reshape((factorDim, 1))

                # obtain error, then accumulate x,y step
                r_ui_h = u + b_u[userName] + b_i[artName] + \
                         np.inner(q_i.T, p_u.T)[0]
                e_ui = r_ui - r_ui_h
                xy_step += e_ui * q_i
                TotalError += abs(e_ui)

                # update b_u, b_i, q_i
                q_i = q_i + step * (e_ui * p_u - lmda * q_i)
                b_u_val = b_u_val + step * (e_ui - lmda * b_u_val)
                b_i_val = b_i_val + step * (e_ui - lmda * b_i_val)

            b_u = {b_u_key[i]: b_u_val[i] for i in range(len(b_u_key))}
            b_i = {b_i_key[i]: b_i_val[i] for i in range(len(b_i_key))}

            # update x and y
            for artName, r_ui in user.iteritems():
                artIdx = np.where(b_i_key == artName)[0][0]
                b_ui = u + b_u[userName] + b_i[artName]
                x[artIdx] = x[artIdx] + step * \
                        (np.power(user_size, -0.5) * (r_ui - b_ui) * xy_step.T -
                         lmda * x[artIdx]
                        )
                y[artIdx] = y[artIdx] + step * \
                        (np.power(user_size, -0.5) * xy_step.T -
                         lmda * y[artIdx]
                        )

            #report
            j+= 1
            if j % report_freq == 0:
                sys.stdout.write("\t\t\t\t" + str(float(j+1)*100/ N) + "%\n")

        TotalError = TotalError/N
        history.append(TotalError)
        t1 = time.time()
        sys.stdout.write("\t\t\t\t Done! (" + str((t1 - t0)/60) + " min, MAE=" +
                         str(TotalError) + ") \n")



























































    return u, b_u, b_i, q, x, y