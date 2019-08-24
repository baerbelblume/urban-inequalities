import glob
import pickle
import scipy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


#DATA

# OUTCOMES
metadata = pd.read_csv('../../data/all_ids.csv')
obj_df_img = pd.read_csv('../preprocess/all_objects_updated.csv')

#create dict to map lsoa to img_id
id_dict = dict(zip(metadata['img_id'], metadata['lsoa11']))

### POINT BASED ANALYSIS
#obj_df_img = pd.crosstab(obj_df.img_id, obj_df.object)
#obj_df_img = obj_df_img.groupby(['img_id']).sum()
obj_df_img['obj_sum'] = obj_df_img.sum(axis=1)


### indexing
X = obj_df_img
y = metadata[['mean-income-decile-lsoa', 'gsv_copyright', 'lsoa11', 'img_id']] #.set_index('img_id') #, ['lsoa11']] #think of potential log-transformation
Z = (pd.merge(X, y, on='img_id', how='outer'))

Z = Z[Z['gsv_copyright'] == '© Google, Inc.']


# index and variable
# set Variables
index_ = 'img_id'
variable = 'X'

variables = ['car', 'person', 'truck', 'potted plant', 'bus', 'bench', 'bicycle', 'motorcycle', 'traffic light', 'train']

outcome = 'mean-income-decile-london-lsoa'

Z.set_index(index_, inplace=True)
#Z = Z.groupby(Z.index)[variable].sum() #.reset_index()

Zz = pd.DataFrame(Z.groupby(['lsoa11']).size().reset_index(name='counts'))

dict_z = pd.Series(Zz.counts.values, index=Zz.lsoa11).to_dict()

Z['counts'] = Z['lsoa11']
Z.set_index('lsoa11')
Z['counts'].replace(dict_z, inplace=True)

Z.loc[:,variables] = Z.loc[:, variables].div(Z['counts'], axis=0)
Z['sum_variables'] = Z.loc[:, variables].sum(axis=1)


print (Z)


#reindex
#Z.set_index('lsoa11', inplace=True)

#Z = Z.groupby(Z.index)['car'].sum().reset_index()



#Z.set_index('lsoa11', inplace=True)

## counts of objects
#counts = pd.DataFrame(X[,:79].sum(), columns=['count'])
#counts.plot(kind='bar')

###MAPPING

# set the filepath and load in a shapefile
fp = "../../data/statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp"
map_df = gpd.read_file(fp)


# join the geodataframe with the cleaned up csv dataframe
map_df.rename(columns={'LSOA11CD' : 'lsoa11'}, inplace=True)

merged = pd.merge(map_df, Z, on='lsoa11')
#not_included = map_df[(~map_df.lsoa11.isin(Z.lsoa11))]

merged.set_index('lsoa11')


# set a variable that will call whatever column we want to visualise on the map
variable = 'car'#'mean-income-decile-london-lsoa'

# set the range for the choropleth
vmin, vmax = Z['car'].min(), Z['car'].max()

print(vmin, vmax) #cars: 0, 104

# create figure and axes for Matplotlib
ax = plt.axes()

Z_20 = merged.sort_values(by='car', ascending=False)[:20]
Z_top = merged.sort_values(by='car', ascending=False)[20:]

# create map
#zero_values.plot(column=variable, cmap='RdBu', linewidth=0.1)
sc = Z_top.plot(column=variable, cmap='RdBu', linewidth=0.1, ax=ax)
sc2 = Z_20.plot(column=variable, color='darkblue', linewidth=0.1, ax=ax)

# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=vmin, vmax=vmax)) #coolwarm
sm.set_array(ax)

# add the colorbar to the figure
cbar = plt.colorbar(sm)
plt.title("Number of cars per LSOA", fontdict={'fontsize': '24', 'fontweight' : '3'})
#fig.savefig(Density of cars, dpi=300)

#print (Z)

#Z_top = Z.sort_values(by='car', ascending=False)[0:20]

#print (Z_top)
#sns.barplot(x='car', y='lsoa11', data=Z_top)

plt.show()
