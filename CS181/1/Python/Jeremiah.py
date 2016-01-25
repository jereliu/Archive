## basic packages
import os
import time
import csv
import gzip
import numpy as np
from scipy import sparse
import scipy.stats as sp_st
import sys
## ML packages
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn import cross_validation as cv
from sklearn import svm
from sklearn import ensemble as es
from sklearn import cluster as cl
## visualization packages
from ggplot import *
#import pylab as pl
#from rdkit import Chem

os.chdir("/Users/Jeremiah/GitHub/CS-181-Practical-1/Data")

######################################
######## 1. Data Read-In     #########
######################################
train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'submission.csv'

# Load the training file.
train_data = []
with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')

    # Skip the header row.
    next(train_csv, None)

    # Load the data.
    for row in train_csv:
        smiles   = row[0]
        features = np.array([float(x) for x in row[1:257]])
        gap      = float(row[257])

        train_data.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })
                            

# Compute the mean of the gaps in the training data.
gaps0 = np.array([datum['gap'] for datum in train_data])
feature0 = np.array([datum['features'] for datum in train_data])
smile0 = np.array([datum['smiles'] for datum in train_data])


#function to evaluate performance
def AME (pred, obs):
    out = np.sqrt(sum((pred - obs)**2)/len(obs))
    return out

#response outlier removal
olr_idx = np.where(gaps0<0)
feature = np.delete(feature0, olr_idx, 0)
gaps = np.delete(gaps0, olr_idx, 0)
n = len(gaps)

sgaps = sparse.csr_matrix(gaps).T #create sparse matrix
sfeature = sparse.csr_matrix(feature) #create sparse matrix
n = len(gaps)

#########################################################
######## 1. Feature Importance Exploration ##############
#########################################################
#why the X is sparse: Entry sum:
featureSum = np.array([sum(feature0.T[i]) for i in range(feature0.T.shape[0])])
featureSum_sort = sorted(featureSum, reverse=True)
featureSum_freq = sp_st.itemfreq(featureSum_sort)

for row in featureSum_freq:
    print "P(feature) = %.3f \t Feature Freq: %.0f " % (row[0]/len(gaps0), row[1])

featureNonzeroId = np.array([(item/len(gaps0)>0 and item/len(gaps0) < 1) for 
                    item in featureSum])
featureImp0 = feature0.T[featureNonzeroId].T
featureImp0_bin = np.array([''.join(map(str, map(int, row)))
 for row in featureImp0])
featureImp0_freq = sp_st.itemfreq(featureImp0_bin)
featureImp0_freq = sorted(featureImp0_freq, key = lambda freq: int(freq[1]), 
                            reverse = True)

for row in featureImp0_freq[0:99]:
    print "Combn = %s \t Freq = %f " % (row[0], float(row[1])/len(gaps0))

featureImp0_cumfreq = np.cumsum(np.array([float(freq[1]) for 
                        freq in featureImp0_freq]))/len(gaps0)
    
#convert to pandas dataframe  
featureImp0_freq_row = np.array([int(freq[0], 2) for freq in featureImp0_freq])
featureImp0_freq_num = np.array([int(freq[1]) for freq in featureImp0_freq])

rowFreq_DF = pd.DataFrame({"Category": featureImp0_freq_row, 
"Freq": featureImp0_freq_num})
p = ggplot(aes(x= range(400), weight = "Freq"), data = rowFreq_DF[0:400])
p + geom_bar() + ggtitle("Frequency of Feature Combinations") + \
labs("Combinations (Binary Representation)", "Freq")

#response outlier detection : histogram
Y_CI = [np.percentile(gaps0, q) for q in [2.5, 97.5]]

gaps_DF = pd.DataFrame({"gaps": gaps0})
p = ggplot(aes(x='gaps'), data=gaps_DF)
p + geom_histogram() + ggtitle("Histogram of Gaps") + labs("Gaps", "Freq")

#
olr_idx = np.where(gaps0<0)
feature = np.delete(feature0, olr_idx, 0)
gaps = np.delete(gaps0, olr_idx, 0)
n = len(gaps)

sgaps = sparse.csr_matrix(gaps).T #create sparse matrix
sfeature = sparse.csr_matrix(feature) #create sparse matrix
n = len(gaps)

gaps_DF = pd.DataFrame({"gaps": gaps})
p = ggplot(aes(x='gaps'), data=gaps_DF)
p + geom_histogram() + ggtitle("Histogram of Gaps") + labs("Gaps", "Freq")

#SVM-based outlier detection  
dat = np.column_stack((feature, gaps))
dat1 = [dat[i] for i in range(len(gaps))]
sdat = sparse.csr_matrix(dat1)

t0 = time.time()
olSVM = svm.OneClassSVM(verbose = True)
olSVM_Fit = olSVM.fit(sdat)
t1 = time.time()
totalTime = (t1-t0)/60

print("Outlier SVM finished in " + str(round(totalTime, 3)) + " miniutes")

######################################
######## 2.1. Ridge Regression #######
######################################
#CV-based tuning parameter selection
rgError_cv = []

for i in range(101):
    kf = cv.KFold(n, n_folds=10)
    rg_cv = Ridge(alpha = i)
    t0 = time.time() #Timing        
    rgErrori_cv = [] #AME estimate for ith tunning par 
    for train_index, test_index in kf:
        #create CV dataset and fit
        X_train, X_test = feature[train_index], feature[test_index]
        y_train, y_test = gaps[train_index], gaps[test_index]
        rgFit_cv = rg_cv.fit(X_train, y_train)
        #prediction
        rfPred_cv = rgFit_cv.predict(X_test)
        rgErrori_cv.append(AME(rfPred_cv, y_test))
    #### Time code, report
    t1 = time.time()
    totalTime = (t1-t0)/60
    rgError_cv.append(np.mean(rgErrori_cv))
    print(str(i) + ": Iteration finished in " + str(round(totalTime, 3)) + 
    " miniutes, AME = " + str(np.mean(rgErrori_cv)))
    sys.stdout.flush()

######################################
######## 2.2. Random Forest ############
######################################
rfSize = 100

####    Random Forest: Total sample run, naive

t0 = time.time()
rf = es.RandomForestRegressor(n_estimators = rfSize, n_jobs = -1, 
                              max_features = 20, max_depth = 20)
rfFit_0 = rf.fit(feature, gaps)
rfPred = rfFit_0.predict(feature)
rfPredError = AME(rfPred, gaps)
t1 = time.time()
totalTime = (t1-t0)/60

print("randForest finished in " + str(round(totalTime, 3)) + 
    " miniutes, AME = " + str(rfPredError)
    )

rfAbias_DF = pd.DataFrame({"bias":abs(rfPred - gaps), "gaps": gaps})

p = ggplot(aes(x='gaps', y = "bias"), data=rfAbias_DF)
p + geom_point(alpha = 0.1) + \
ggtitle("Observed Gaps vs Prediction Bias") + labs("Gaps", "Bias")
       
####    Feature selection using radom Forest

rfFeatureImp = rfFit_0.feature_importances_
rfFeature = rfFit_0.transform(feature, threshold = "0.001*mean" )

#calculate index of selected covariates
rfFeature_col = rfFeature.T
feature_col = feature.T
rfFeature_idx = [ np.where((feature_col == rfFeature_col[i]).all(axis=1)) 
for i in range(rfFeature_col.shape[0])]
rfFeature_idx = np.unique(np.concatenate(rfFeature_idx, axis = 1))



####    5-fold CV for outcome prediction error calculation

#1. Tree size: the more the merrier
#2. m = max_features considered when splitting. 
#    use randFor selected to reduce overfitting

#select RF size
for rfSize in [10, 20, 50, 100, 500, 1000, 2000]:
    rfError_cv = []
    kf = cv.KFold(n, n_folds=10)
    
    rf_cv = es.RandomForestRegressor(n_estimators = rfSize, n_jobs = -1,
                                    max_features = 20, max_depth = 28)
    t0 = time.time()
    for train_index, test_index in kf:
        #Timing
        t0 = time.time()
        #create CV dataset and fit
        X_train, X_test = rfFeature[train_index], rfFeature[test_index]
        y_train, y_test = gaps[train_index], gaps[test_index]    
        rfFit_cv = rf_cv.fit(X_train, y_train)
        #prediction
        rfPred_cv = rfFit_cv.predict(X_test)
        rfErrori_cv = AME(rfPred_cv, y_test)
        rfError_cv.append(rfErrori_cv)
        #### Time code, report
        t1 = time.time()
        totalTime = (t1-t0)/60
        print("Size " + str(rfSize) + ": Iteration finished in " + 
        str(round(totalTime, 3)) + " miniutes")
    
    print("Size " + str(rfSize) + ": AME = " + str(np.mean(rfError_cv)))
    print("###########################################################")
    sys.stdout.flush()


for rfSize in [10, 20, 50, 100, 500, 1000, 2000]:
    rfError_cv = []
    kf = cv.KFold(n, n_folds=10)
    
    rf_cv = es.RandomForestRegressor(n_estimators = rfSize, n_jobs = -1,
                                    max_features = 20, max_depth = 20)
    t0 = time.time()
    for train_index, test_index in kf:
        #Timing
        t0 = time.time()
        #create CV dataset and fit
        X_train, X_test = rfFeature[train_index], rfFeature[test_index]
        y_train, y_test = gaps[train_index], gaps[test_index]    
        rfFit_cv = rf_cv.fit(X_train, y_train)
        #prediction
        rfPred_cv = rfFit_cv.predict(X_test)
        rfErrori_cv = AME(rfPred_cv, y_test)
        rfError_cv.append(rfErrori_cv)
        #### Time code, report
        t1 = time.time()
        totalTime = (t1-t0)/60
        print("Size " + str(rfSize) + ": Iteration finished in " + 
        str(round(totalTime, 3)) + " miniutes")
    
    print("Size " + str(rfSize) + ": AME = " + str(np.mean(rfError_cv)))
    print("###########################################################")
    sys.stdout.flush()





####    Random Forest: Total sample run, with selected parameter

rfSize = 1000
t0 = time.time()
rf = es.RandomForestRegressor(n_estimators = rfSize, n_jobs = -1, 
                              max_features = 20, max_depth = 20)
rfFit = rf.fit(rfFeature, gaps)
rfPred = rfFit.predict(rfFeature)
rfPredError = AME(rfPred, gaps)
t1 = time.time()
totalTime = (t1-t0)/60

print("randForest finished in " + str(round(totalTime, 3)) + 
    " miniutes, AME = " + str(rfPredError)
    )
    
rfAbias_DF = pd.DataFrame({"bias":abs(rfPred - gaps), "gaps": gaps})

p = ggplot(aes(x='gaps', y = "bias"), data=rfAbias_DF)
p + geom_point(alpha = 0.05) + \
ggtitle("Observed Gaps vs Prediction Bias") + labs("Gaps", "Bias")


######################################
######## 3. Output ######################
######################################
# Load the test file.
test_data = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')

    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    for row in test_csv:
        id       = row[0]
        smiles   = row[1]
        features = np.array([float(x) for x in row[2:258]])

        test_data.append({ 'id':       id,
                           'smiles':   smiles,
                           'features': features })
                           
feature_test = np.array([datum['features'] for datum in test_data])
sfeature_test = sparse.csr_matrix(feature_test)

Id_test = np.array([datum['id'] for datum in test_data])

#produce prediction
rfFeature_test = (feature_test.T[rfFeature_idx]).T
mean_gap = rfFit.predict(rfFeature_test)

# Write a prediction file.
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    for i in range(len(mean_gap)):
        pred_csv.writerow([Id_test[i], mean_gap[i]])