import pickle
import scipy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from geoplot import polyplot
import matplotlib.colors as colors



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


outcome = 'dep-health-decile-london-lsoa'

#mean-income-decile-london-lsoa
#dep-liv-env-decile-london-lsoa
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


VARIABLE = outcome

# set the filepath and load in a shapefile
fp = "../statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp"
map_df = gpd.read_file(fp)


boroughs = "../statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"
borough = gpd.read_file(boroughs)


map_df = map_df.rename(columns={'LSOA11CD':'lsoa11'})

merged = map_df.merge(Z, how='inner', on='lsoa11')

# set the range for the choropleth
vmin, vmax = merged[VARIABLE].min(), merged[VARIABLE].max()

#print(vmin, vmax) #cars: 0, 104
#print (merged.head())
ax = plt.axes()

base = merged.plot(column=VARIABLE, cmap='YlGnBu', ax=ax)

borough.geometry.boundary.plot(color=None, edgecolor='black',linewidth = 0.5, ax=ax, alpha=0.8)
plt.axis('off')


# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=colors.Normalize(vmin=vmin, vmax=vmax)) #coolwarm


sm.set_array(ax)
cb = plt.colorbar(sm)

for l in cb.ax.yaxis.get_ticklabels():
    l.set_family("serif")


# add the colorbar to the figure
plt.title("Health Deprivation", fontdict={'fontsize': '24', 'fontweight' : '3', 'family' : 'serif'})


plt.show()