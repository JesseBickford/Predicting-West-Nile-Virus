import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
%matplotlib inline


spray_merged = pd.read_csv('spray_0.75_merged.csv')

#What's in the data? Okay, knowing which ones were sprayed and not is also important....

spray_merged.info()

#Adding a variable to describe count of sprays for a trap area. Two usually means
#it was sprayed once per year.
spray_merged['spray_ind'] = 0

spray_list = []

for i in spray_merged.index:
    spray_list.append(np.sum(spray_merged.iloc[i, 12:21]))

spray_merged['spray_ind'] = spray_list

#Years in data set...
spray_merged.Year.unique()

#Okay, let's first scale the weather data.
scale_cols = ['AvgSpeed', 'Cool', 'Depart', 'Depth', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'Station', 'StnPressure', 'Sunrise',
       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']
scale_cols2 = ['NumMosquitos', 'AvgSpeed', 'Cool', 'Depart', 'Depth', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'Station', 'StnPressure', 'Sunrise',
       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']


non_scale = ['Date', 'Address', 'Block', 'Street', 'Trap',
       'AddressNumberAndStreet', 'Latitude', 'Longitude',
       'AddressAccuracy', 'spray_2011-08-29',
       'spray_2011-09-07', 'spray_2013-07-17', 'spray_2013-07-25',
       'spray_2013-08-08', 'spray_2013-08-15', 'spray_2013-08-22',
       'spray_2013-08-29', 'spray_2013-09-05', 'Species_CULEX ERRATICUS',
       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS']

non_scale_use = ['Latitude', 'Longitude',
       'AddressAccuracy', 'spray_2011-08-29',
       'spray_2011-09-07', 'spray_2013-07-17', 'spray_2013-07-25',
       'spray_2013-08-08', 'spray_2013-08-15', 'spray_2013-08-22',
       'spray_2013-08-29', 'spray_2013-09-05', 'Species_CULEX ERRATICUS',
       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS']

target_wnv = spray_merged['WnvPresent']
target_nm = spray_merged['NumMosquitos']

'''Standard Scaler for weather data'''

scaler = StandardScaler()

X_scaled_array = scaler.fit_transform(spray_merged[scale_cols])
X_scaled = pd.DataFrame(X_scaled_array, columns = scale_cols, index = spray_merged.index)


'''Stnd Scaler for weather and num_mosquitos'''
scaler2 = StandardScaler()
X_sc_withnm = scaler2.fit_transform(spray_merged[scale_cols2])

'''Merge the scaled data and ind vars...keeping num mosquitos out of this...'''
X_merged_no_nm = pd.merge(spray_merged[non_scale_use], X_scaled, left_index = True, right_index = True)

X_merged_no_nm.info()

'''Let's start our PCA process.'''
X_covmat = np.cov(X_merged_no_nm.T)

eig_vals, eig_vecs = np.linalg.eig(X_covmat)

print eig_vals
print '----------------------'
print eig_vecs

eigen_pairs = [[eig_vals[i], eig_vecs[:,i]] for i in range(len(eig_vals))]

eigenpairs = pd.DataFrame(eigen_pairs, columns = ['eigenvalue', 'eigenvector'])

eigenpairs.sort_values('eigenvalue', ascending = False)

#Total explained variance

#sum all the eigenvalues together.
totaleig_val = eigenpairs.eigenvalue.sum()
print "Total Eigenvalue Sum is:", totaleig_val
print '-------------------'
indiv_var = [eigenpairs.eigenvalue[i]/totaleig_val*100 for i in range(len(eigenpairs))]
cum_exp_var = np.cumsum(indiv_var)

print 'Cumulative Variance Explained as we include principal components:', cum_exp_var
print "There are %i eigenvalues." % len(eigenpairs.eigenvalue)

#Plotting Explained Variance
plt.figure(figsize=(9,7))

component_number = range(1,37)

plt.plot(component_number, cum_exp_var, lw=7)

plt.axhline(y=0, linewidth=5, color='grey', ls='dashed')
plt.axhline(y=100, linewidth=3, color='grey', ls='dashed')
plt.axhline(y=95, linewidth = 3, color = 'green', ls = 'dashed')
plt.axhline(y=90, linewidth = 3, color = 'purple', ls = 'dashed')

ax = plt.gca()
ax.set_xlim([1,36])
ax.set_ylim([-5,105])

ax.set_ylabel('cumulative variance explained', fontsize=16)
ax.set_xlabel('component', fontsize=16)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax.set_title('component vs cumulative variance explained\n', fontsize=20)

'''At component 12, we seem to explain most if not all of our variance. 99.56% to be exact.
At component 7, it looks like we explain 95% of our variance. At component 6, we explain 90% of variance.'''

print "Cumulative variance explained at Component 10:", cum_exp_var[9]
print "Cumulative variance explained at Component 8:", cum_exp_var[7]
print "Cumulative variance explained at Component 5:", cum_exp_var[4]

'''PCA - creating the Principal Components from all numerical features and species inds.
Since 8 components explain 93% of variance, using that.'''

wnv_pca = PCA(n_components = 8)
X_PCs = wnv_pca.fit_transform(X_merged_no_nm)

#Creating Df of PCs.

prin_comps = pd.DataFrame(X_PCs, columns = ['PC' + str(i) for i in range(1,9)])
#['PC_' + str(i) for i in range(1,6)]
prin_comps

#Merging with wnv target...

wnv_PCs = pd.merge(spray_merged[['WnvPresent', 'Trap']], prin_comps, left_index = True, right_index = True)

wnv_PCs.info()

#Now let's see how our eight PCs related back to our original features...

prin_comps_features = pd.merge(prin_comps, X_merged_no_nm, left_index = True, right_index = True)

corr_prin_comps = prin_comps_features.corr().drop(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'], axis = 0)

cols_drop = ['Latitude', 'Longitude', 'AddressAccuracy', 'spray_2011-08-29',
       'spray_2011-09-07', 'spray_2013-07-17', 'spray_2013-07-25',
       'spray_2013-08-08', 'spray_2013-08-15', 'spray_2013-08-22',
       'spray_2013-08-29', 'spray_2013-09-05', 'Species_CULEX ERRATICUS',
       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS', 'AvgSpeed',
       'Cool', 'Depart', 'Depth', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'Station', 'StnPressure', 'Sunrise',
       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']

corr_prin_comps.drop(cols_drop, axis = 1, inplace = True)
corr_prin_comps

'''Note that PC1 essentially is the weather data, and it explains 40% of the variance.
PC2 is mainly pressure and sunrise/sunset time, and some strength from Pipiens.
PC3 is wind essentially, and location.
'''

wnv_pca.components_
