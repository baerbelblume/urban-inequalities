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

objects['living environment deprivation decile'] = objects.index
objects['living environment deprivation decile'].update(pd.Series(mean_inc_dict))

final = objects.iloc[:,0:79].div(objects.counts, axis=0)

print (final.head())

variables = ['person']
#variables = ['car', 'person', 'truck', 'potted plant', 'bus', 'bench', 'bicycle', 'motorcycle', 'traffic light', 'chair']

X = final[variables]
y = objects[['living environment deprivation decile']].astype('float')

print (X[1:5], y[1:5])

sns.regplot(y=X['person'], x=y['living environment deprivation decile'], color='darkseagreen')
plt.show()

