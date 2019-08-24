import glob
import pickle
import scipy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pandas.plotting import scatter_matrix

#from collections import Counter
import scipy.stats as stats
from statistics import mean

## Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Model selection and evaluation
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve, auc, brier_score_loss, classification_report, cohen_kappa_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


from matplotlib import rcParams
rcParams['font.family'] = 'serif'


def heatmap(x, y, size):

    cm = confusion_matrix(x, y)
    df_cm = pd.DataFrame(cm, range(1, size), range(1, size))

    fig, ax = plt.subplots()

    sns.heatmap(df_cm, cmap='YlGnBu', annot=False, cbar=False)

    ax.set_ylabel('True label', fontdict={'fontsize': '16', 'family' : 'serif'})    
    ax.set_xlabel('Predicted label', fontdict={'fontsize': '16', 'family' : 'serif'})

    plt.title('Living Environment Deprivation', fontdict={'fontsize': '20', 'fontweight' : '3', 'family' : 'serif'})
    plt.show()

#DATA
# OUTCOMES
pickle_in = open("../ONSPD_AUG_2017_LONDON_W_METADATA_NEW_IMGID_HC_LSOA_LABELS.p","rb")
metadata = pickle.load(pickle_in)

#print (list(metadata.columns))


# OBJECTS
obj_df_img = pd.read_csv('../all_objects_updated.csv')


outcome = 'dep-liv-env-decile-london-lsoa'
#mean-income-decile-london-lsoa
#dep-health-decile-london-lsoa
#dep-liv-env-decile-london-lsoa

### indexing
X = obj_df_img
y = metadata[[outcome, 'lsoa11', 'img_id']]
y.set_index('img_id')


Z = (pd.merge(X, y, on='img_id', how='inner'))

counts = Z.groupby(['lsoa11']).size().to_frame('count')
#counts['lsoa11'] = counts.index

Z = pd.merge(Z, counts, on='lsoa11')
print (Z.head(), Z.shape)

counts_dict = dict(zip(Z['lsoa11'], Z['count']))
mean_inc_dict = dict(zip(Z['lsoa11'], Z[outcome]))

#  img_id  airplane  apple  backpack  ...  zebra  mean-income-decile-london-lsoa     lsoa11  count
#0       1         0      0         0  ...      0                               8  E01000677     25
#1      24         0      0         0  ...      0                               8  E01000677     25

#shape: (116828, 83)


objects = Z.groupby('lsoa11').sum()
objects = objects.iloc[:,1:80]
objects['counts'] = objects.index
objects['counts'].update(pd.Series(counts_dict))

objects['mean_income'] = objects.index
objects['mean_income'].update(pd.Series(mean_inc_dict))

final = objects.iloc[:,0:79].div(objects.counts, axis=0)

print (final.head())

variables = ['car']
#variables = ['car', 'person', 'truck', 'potted plant', 'bus', 'bench', 'bicycle', 'motorcycle', 'traffic light', 'chair']

X = final[variables]
y = objects[['mean_income']].astype('float')

##filter out top and low scores
#top_bottom = [1, 5, 10]
#y = y[y.mean_income.isin(top_bottom)]
#y_ind = y.index
#X = X.loc[y_ind]

#print (X.shape, y.shape) #834


## CLASSIFIER
mae_l = []
tau_l = []
kappa_l = []
pearsons_l = []


skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=None)
for train_index, test_index in skf.split(X, y.values.ravel()):
     print("TRAIN:", len(train_index), "TEST:", len(test_index))
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

     rfc = SVC(kernel='linear', C=1000, gamma=0.001)

     rfc.fit(X_train, y_train.values.ravel())

     y_true, y_pred = y_test.astype('Float64'), rfc.predict(X_test).astype('Float64')

     y_true.astype("Float64")
     y_pred.astype("Float64")

     #evaluation
     print ('MAE:', mean_absolute_error(y_true, y_pred))
     mae_l.append(mean_absolute_error(y_true, y_pred))


     tau, p_value_tau = stats.kendalltau(y_true, y_pred)
     print ('TAU:', tau)
     tau_l.append(tau)

     y_true = y_true[outcome].values

     pearson, p_value_pearsons = stats.pearsonr(y_true, y_pred)
     print ('Pearsons:', pearson)
     pearsons_l.append(pearson)

     print ('Kappa:', cohen_kappa_score(y_true, y_pred))
     kappa_l.append(cohen_kappa_score(y_true, y_pred))

print ('*********')
print (mean(mae_l), mean(tau_l), mean(pearsons_l), mean(kappa_l))

heatmap(x=y_true, y=y_pred, size=11)