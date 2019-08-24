import pickle
import scipy as sc
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from geoplot import polyplot
import matplotlib.colors as colors

from matplotlib import rcParams
rcParams['font.family'] = 'serif'


#DATA
# OUTCOMES
#metadata = pd.read_csv('../BARBARA_ONSPD_AUG_2017_LONDON_W_METADATA_NEW_IMGID_HC_LSOA_LABELS.csv')

pickle_in = open("../ONSPD_AUG_2017_LONDON_W_METADATA_NEW_IMGID_HC_LSOA_LABELS.p","rb")
metadata = pickle.load(pickle_in)

#print (list(metadata.columns))


# OBJECTS
obj_df_img = pd.read_csv('../all_objects_updated.csv')

#create dict to map lsoa to img_id
id_dict = dict(zip(metadata['img_id'], metadata['lsoa11']))

### POINT BASED ANALYSIS
#obj_df_img = pd.crosstab(obj_df.img_id, obj_df.object)
#obj_df_img = obj_df_img.groupby(['img_id']).sum()


outcome = 'mean-income-decile-london-lsoa'
#mean-income-decile-london-lsoa
#dep-health-decile-london-lsoa
#dep-liv-env-decile-london-lsoa



sns.distplot(metadata[outcome], color='darkseagreen')
plt.title('Mean Income', fontdict={'fontsize': '20', 'fontweight' : '3', 'family' : 'serif'})
plt.show()

'''
### indexing
X = obj_df_img
y = metadata[['mean-income-decile-london-lsoa', 'lsoa11', 'img_id']]
y.set_index('img_id')

Z = (pd.merge(X, y, on='img_id', how='inner'))

counts = Z.groupby(['lsoa11']).size().to_frame('count')
#counts['lsoa11'] = counts.index

Z = pd.merge(Z, counts, on='lsoa11')

counts_dict = dict(zip(Z['lsoa11'], Z['count']))
mean_inc_dict = dict(zip(Z['lsoa11'], Z['mean-income-decile-london-lsoa']))

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


variables = ['car', 'person', 'truck', 'potted plant', 'bus', 'bench', 'bicycle', 'motorcycle', 'traffic light', 'chair']




# total number of objects per object
obj_sum = pd.DataFrame(objects.iloc[:,0:79].sum(), columns=['object'])


#print (pd.DataFrame(objects['counts']).median())

final = objects.iloc[:,0:79].div(objects.counts, axis=0)
final['sum_10'] = final[variables].sum(axis=1)
final['sum'] = final.sum(axis=1)

print (final.shape)

#print (final['person'].median(), final['person'].describe())
print (final['potted plant'].mean(), final['potted plant'].describe)


objects['counts'].plot.hist(bins=100, color='darkseagreen')
plt.xlim(0, 90)
plt.xlabel('Totoal number of panorama images per LSOA')
plt.show()


#plot objects per lsoa



fig, ax = plt.subplots()


obj_sum_10 = obj_sum.sort_values(by='object', ascending=False)[0:20]
rest = obj_sum.sort_values(by='object', ascending=False)[9:]

obj_sum = obj_sum.iloc[:,0:79].sort_values(by='object', ascending=False)[0:15]

obj_sum.plot.bar(ax=ax, color='darkseagreen')

ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.ylabel('Object count')


fig.tight_layout()
plt.show()


### map per lsoa

VARIABLE = 'sum_10'

final = objects.iloc[:,0:79].div(objects.counts, axis=0)


final['sum_10'] = final[variables].sum(axis=1)
final['sum'] = final.sum(axis=1)

'''

