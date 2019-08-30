import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pickle_in = open("../data/ONSPD_AUG_2017_LONDON_W_METADATA_NEW_IMGID_HC_LSOA_LABELS.p","rb")
df = pickle.load(pickle_in)

#['index', 'pcd', 'pcd2', 'pcds', 'dointr', 'doterm', 'oscty', 'oslaua', 'osward', 'usertype', 'oseast1m', 'osnrth1m', 'osgrdind', 'oshlthau', 'hro', 'ctry', 'gor', 'streg', 'pcon', 'eer', 'teclec', 'ttwa', 'pct', 'nuts', 'psed', 'cened', 'edind', 'oshaprev', 'lea', 'oldha', 'wardc91', 'wardo91', 'ward98', 'statsward', 'oa01', 'casward', 'park', 'lsoa01', 'msoa01', 'ur01ind', 'oac01', 'oldpct', 'oa11', 'lsoa11', 'msoa11', 'parish', 'wz11', 'ccg', 'bua11', 'buasd11', 'ru11ind', 'oac11', 'lat', #'long', 'lep1', 'lep2', 'pfa', 'imd', 'gsv_lat', 'gsv_lng', 'gsv_pano_id', 'gsv_metastatus', 'gsv_copyright', 'gsv_date', 'status', 'img_id', 'pano_id', 'Population', 'Household', 'lsoa-age-mean', 'lsoa-age-median', 'perc-lsoa-generalhealth-bad-or-verybad', 'decile-lsoa-generalhealth-bad-or-verybad', 'perc-lsoa-occupancy-rating--1orless', 'decile-lsoa-occupancy-rating--1orless', 'perc-lsoa-hrp-unemployment-percent', 'decile-lsoa-hrp-unemployment', 'perc-lsoa-qual-below-level2', 'decile-lsoa-
# below-level2', 'dep-income-rank-uk-lsoa', 'dep-employment-rank-uk-lsoa', 'dep-education-rank-uk-lsoa', 'dep-health-rank-uk-lsoa', 'dep-crime-rank-uk-lsoa', 'dep-housing-rank-uk-lsoa', 'dep-liv-env-rank-uk-lsoa', 'dep-income-decile-london-lsoa', 'dep-employment-decile-london-lsoa', 'dep-education-decile-london-lsoa', 'dep-health-decile-london-lsoa', 'dep-crime-decile-london-lsoa', 'dep-housing-decile-london-lsoa', 'dep-liv-env-decile-london-lsoa', 'mean-income-12-13-lsoa', 'mean-income-decile-#london-lsoa', 'median-income-12-13-lsoa', 'median-income-decile-london-lsoa']

df = df.groupby(['img_id']).sum())

#print (df.head(50))

outcomes = df[['index', 'lat', 'long', 'dep-income-decile-london-lsoa', 'mean-income-12-13-lsoa',
    'mean-income-decile-london-lsoa', 'median-income-12-13-lsoa', 'median-income-decile-london-lsoa',
    'img_id', 'pano_id', 'Population', 'Household']]


