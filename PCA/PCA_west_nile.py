import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt
%matplotlib inline

#Using Brian's file from 3 pm...
spray_merged = pd.read_csv('../west_nile/input/spray_0.75_merged.csv')

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
scale_cols = ['AvgSpeed', 'Cool', 'Depart', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'StnPressure', 'Sunrise',
       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']

#Incorporating NumMosquitos for one scaling b/c 1) different scale from weather data; and 2) we built
#a model to impute NumMosquitos for the test data, so taking its insights from here is useful.
scale_cols2 = ['NumMosquitos', 'AvgSpeed', 'Cool', 'Depart', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'StnPressure', 'Sunrise',
       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']


non_scale = ['Date', 'Address', 'Block', 'Street', 'Trap',
       'AddressNumberAndStreet', 'Latitude', 'Longitude',
       'AddressAccuracy', 'spray_2011-08-29',
       'spray_2011-09-07', 'spray_2013-07-17', 'spray_2013-07-25',
       'spray_2013-08-08', 'spray_2013-08-15', 'spray_2013-08-22',
       'spray_2013-08-29', 'spray_2013-09-05', 'Species_CULEX ERRATICUS',
       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS', 'spray_ind']

non_scale_use = ['Latitude', 'Longitude',
       'AddressAccuracy', 'spray_2011-08-29',
       'spray_2011-09-07', 'spray_2013-07-17', 'spray_2013-07-25',
       'spray_2013-08-08', 'spray_2013-08-15', 'spray_2013-08-22',
       'spray_2013-08-29', 'spray_2013-09-05', 'Species_CULEX ERRATICUS',
       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS']

non_scale_nospray = ['Latitude', 'Longitude',
       'AddressAccuracy', 'Species_CULEX ERRATICUS',
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
X_sc_withnm_array = scaler2.fit_transform(spray_merged[scale_cols2])
X_sc_withnm = pd.DataFrame(X_sc_withnm_array, columns = scale_cols2, index = spray_merged.index)

'''Version 1: Merge the scaled data and ind vars...keeping num mosquitos out of this...'''
X_merged_no_nm = pd.merge(spray_merged[non_scale_use], X_scaled, left_index = True, right_index = True)

X_merged_no_nm.info()

''' Version 2: Merging scaled data with NumMosquitos and ind vars...'''
X_merged_nm = pd.merge(spray_merged[non_scale_use], X_sc_withnm, left_index = True, right_index = True)

X_merged_nm.info()

'''Version 3: Merging scaled data with NumMosquitos, ind vars, but no spray site data.'''
X_merged_m_nospray = pd.merge(spray_merged[non_scale_nospray], X_sc_withnm, left_index = True, right_index = True)

X_merged_m_nospray.info()
'''Let's start our PCA process.'''


#Separate file - looking to see what PCs are like without spraying...
#no_spray_X = X_merged_no_nm[X_merged_no_nm['spray_ind'] == 0]

#cols_no_spray = ['Latitude', 'Longitude', 'AddressAccuracy', 'Species_CULEX ERRATICUS',
#       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
#       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
#       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS',
#       'AvgSpeed', 'Cool', 'Depart', 'DewPoint', 'Heat', 'ResultDir',
#       'ResultSpeed', 'SeaLevel', 'StnPressure', 'Sunrise', 'Sunset',
#       'Tavg', 'Tmax', 'Tmin', 'WetBulb']

#X_no_spray = no_spray_X[cols_no_spray]



'''Version 1: Original PCA with spray data but no mosquito count'''
X_covmat = np.cov(X_merged_no_nm.T)

eig_vals, eig_vecs = np.linalg.eig(X_covmat)

print eig_vals
print '----------------------'
print eig_vecs

eigen_pairs = [[eig_vals[i], eig_vecs[:,i]] for i in range(len(eig_vals))]

eigenpairs = pd.DataFrame(eigen_pairs, columns = ['eigenvalue', 'eigenvector'])

eigenpairs.sort_values('eigenvalue', ascending = False)

#'''Eigenvalues for PCA without spray data'''
#X_nospray_covmat = np.cov(X_no_spray.T)
#eig_vals2, eig_vecs2 = np.linalg.eig(X_nospray_covmat)
#eigen_pairs2 = [[eig_vals2[i], eig_vecs2[:,i]] for i in range(len(eig_vals2))]

#eigenpairs2 = pd.DataFrame(eigen_pairs2, columns = ['eigenvalue', 'eigenvector'])

#eigenpairs2.sort_values('eigenvalue', ascending = False)


'''PCA with spray data and NumMosquitos in features scaled.'''
X_cov_m = np.cov(X_merged_nm.T)

eig_vals_m, eig_vecs_m = np.linalg.eig(X_cov_m)

print eig_vals_m
print '----------------------'
print eig_vecs_m

eigen_pairs_m = [[eig_vals_m[i], eig_vecs_m[:,i]] for i in range(len(eig_vals_m))]

eigenpairs_m = pd.DataFrame(eigen_pairs_m, columns = ['eigenvalue', 'eigenvector'])

eigenpairs_m.sort_values('eigenvalue', ascending = False)

'''Version 3: PCA with NumMosquitos in features scaled, no spray data.'''
X_cov_m_nospray = np.cov(X_merged_m_nospray.T)

eig_vals_m_nospray, eig_vecs_m_nospray = np.linalg.eig(X_cov_m_nospray)

print eig_vals_m_nospray
print '----------------------'
print eig_vecs_m_nospray

eigen_pairs_m_nospray = [[eig_vals_m_nospray[i], eig_vecs_m_nospray[:,i]] for i in range(len(eig_vals_m_nospray))]

eigenpairs_m_nospray = pd.DataFrame(eigen_pairs_m_nospray, columns = ['eigenvalue', 'eigenvector'])

eigenpairs_m_nospray.sort_values('eigenvalue', ascending = False)



'''
#Version 1: Total explained variance for PCA with spray data'''

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

component_number = range(1,35)

plt.plot(component_number, cum_exp_var, lw=7)

plt.axhline(y=0, linewidth=5, color='grey', ls='dashed')
plt.axhline(y=100, linewidth=3, color='grey', ls='dashed')
plt.axhline(y=95, linewidth = 3, color = 'green', ls = 'dashed')
plt.axhline(y=90, linewidth = 3, color = 'purple', ls = 'dashed')

ax = plt.gca()
ax.set_xlim([1,34])
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



'''Version 3: Total Explained Variance for PCA w/o spray data but with NumMosquitos'''

#sum all the eigenvalues together.
totaleig_val_m_nospray = eigenpairs_m_nospray.eigenvalue.sum()
print "Total Eigenvalue Sum is:", totaleig_val_m_nospray
print '-------------------'
indiv_var_m_nospray = [eigenpairs_m_nospray.eigenvalue[i]/totaleig_val_m_nospray*100 for i in range(len(eigenpairs_m_nospray))]
cum_exp_var_m_nospray = np.cumsum(indiv_var_m_nospray)

print 'Cumulative Variance Explained as we include principal components:', cum_exp_var_m_nospray
print "There are %i eigenvalues." % len(eigenpairs_m_nospray.eigenvalue)

#Plotting Explained Variance
plt.figure(figsize=(9,7))

component_number = range(1,27)

plt.plot(component_number, cum_exp_var_m_nospray, lw=7)

plt.axhline(y=0, linewidth=5, color='grey', ls='dashed')
plt.axhline(y=100, linewidth=3, color='grey', ls='dashed')
plt.axhline(y=95, linewidth = 3, color = 'green', ls = 'dashed')
plt.axhline(y=90, linewidth = 3, color = 'purple', ls = 'dashed')

ax = plt.gca()
ax.set_xlim([1,26])
ax.set_ylim([-5,105])

ax.set_ylabel('cumulative variance explained', fontsize=16)
ax.set_xlabel('component', fontsize=16)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax.set_title('component vs cumulative variance explained\n', fontsize=20)

#'''At component 12, we seem to explain most if not all of our variance. 99.56% to be exact.
#At component 7, it looks like we explain 95% of our variance. At component 6, we explain 90% of variance.'''

print "Cumulative variance explained at Component 10:", cum_exp_var_m_nospray[9]
print "Cumulative variance explained at Component 8:", cum_exp_var_m_nospray[7]
print "Cumulative variance explained at Component 5:", cum_exp_var_m_nospray[4]

'''
Version 2:
Remember: we want to know the ideal number of PCs.
Total explained variance for PCA with spray data and Mosquito Count'''

#sum all the eigenvalues together.
totaleig_val_m = eigenpairs_m.eigenvalue.sum()
print "Total Eigenvalue Sum is:", totaleig_val_m
print '-------------------'
indiv_var_m = [eigenpairs_m.eigenvalue[i]/totaleig_val_m*100 for i in range(len(eigenpairs_m))]
cum_exp_var_m = np.cumsum(indiv_var_m)

print 'Cumulative Variance Explained as we include principal components:', cum_exp_var_m
print "There are %i eigenvalues." % len(eigenpairs_m.eigenvalue)

#Plotting Explained Variance
plt.figure(figsize=(9,7))

component_number = range(1,36)

plt.plot(component_number, cum_exp_var_m, lw=7)

plt.axhline(y=0, linewidth=5, color='grey', ls='dashed')
plt.axhline(y=100, linewidth=3, color='grey', ls='dashed')
plt.axhline(y=95, linewidth = 3, color = 'green', ls = 'dashed')
plt.axhline(y=90, linewidth = 3, color = 'purple', ls = 'dashed')
plt.axvline(x=10, linewidth=3, color='black')
plt.axvline(x=8, linewidth = 3, color = 'red')
plt.axvline(x=5, linewidth = 3, color = 'black')



ax = plt.gca()
ax.set_xlim([1,35])
ax.set_ylim([-5,105])

ax.set_ylabel('Cumulative Variance Explained', fontsize=16)
ax.set_xlabel('Component', fontsize=16)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(12)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(12)

ax.set_title('Component vs Cumulative variance Explained\n', fontsize=20)
plt.savefig('PCAChicago.png')


print "Cumulative variance explained at Component 10:", cum_exp_var_m[9]
print "Cumulative variance explained at Component 8:", cum_exp_var_m[7]
print "Cumulative variance explained at Component 5:", cum_exp_var_m[4]

'''Could use 8 components b/c 92% of variance, or 10 b/c 95%. Going to use 8'''




'''
Version 1:
PCA - creating the Principal Components from all numerical features and species inds.
Since 8 components explain 93% of variance, using that.'''

wnv_pca = PCA(n_components = 8)
X_PCs = wnv_pca.fit_transform(X_merged_no_nm)

#Creating Df of PCs.

prin_comps = pd.DataFrame(X_PCs, columns = ['PC' + str(i) for i in range(1,9)])
#['PC_' + str(i) for i in range(1,6)]
prin_comps

#Merging with wnv target...

wnv_PCs = pd.merge(spray_merged[['Date', 'WnvPresent', 'NumMosquitos','Trap']], prin_comps, left_index = True, right_index = True)

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
       'Cool', 'Depart', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'StnPressure', 'Sunrise',
       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']

corr_prin_comps.drop(cols_drop, axis = 1, inplace = True)
corr_prin_comps

'''Note that PC1 essentially is the weather data, and it explains 40% of the variance.
PC2 is mainly pressure and sunrise/sunset time, and some strength from Pipiens.
PC3 is wind essentially, and location.
'''

wnv_pca.components_

#Exports PCs with the spray info incorporated.

wnv_PCs.to_csv('train_data_PCA_spraydata.csv', sep = ',', index = True, index_label = 'Index')

#'''PCA for the Training Data for sites we know were not sprayed.'''
#wnv_pca_nospray = PCA(n_components = 8)
#X_PCs_nospray = wnv_pca.fit_transform(X_no_spray)


#Creating Df of PCs.

#prin_comps = pd.DataFrame(X_PCs_nospray, columns = ['PC' + str(i) for i in range(1,9)])
#['PC_' + str(i) for i in range(1,6)]
#prin_comps

#Merging with wnv target...

#wnv_PCs_nospray = pd.merge(spray_merged[['WnvPresent', 'NumMosquitos','Trap']][spray_merged['spray_ind'] == 0], prin_comps, left_index = True, right_index = True)

#wnv_PCs_nospray.info()

#Now let's see how our eight PCs related back to our original features...

#prin_comps_features = pd.merge(prin_comps, X_no_spray, left_index = True, right_index = True)

#corr_prin_comps = prin_comps_features.corr().drop(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'], axis = 0)

#cols_drop = ['Latitude', 'Longitude', 'AddressAccuracy', 'Species_CULEX ERRATICUS',
#       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
#       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
#       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS', 'AvgSpeed',
#       'Cool', 'Depart', 'DewPoint', 'Heat', 'ResultDir',
#       'ResultSpeed', 'SeaLevel', 'StnPressure', 'Sunrise',
#       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']

#corr_prin_comps.drop(cols_drop, axis = 1, inplace = True)
#corr_prin_comps

#'''Testing the PCs on predicting number of mosquitos...'''

#target_nm = wnv_PCs_nospray['NumMosquitos']

#X_nm = wnv_PCs_nospray[['PC1', 'PC2', 'PC3', 'PC4', 'PC5','PC6','PC7','PC8']]


'''Version 2: PCA for Spray Data that includes NumMosquitos -
creating the Principal Components from all numerical features and species inds.
Since 8 components explain 90% of variance, using that.'''

wnv_pca_m = PCA(n_components = 8)
X_PCs_m = wnv_pca_m.fit_transform(X_merged_nm)

#Creating Df of PCs.

prin_comps_m = pd.DataFrame(X_PCs_m, columns = ['PC' + str(i) for i in range(1,9)])
#['PC_' + str(i) for i in range(1,6)]
prin_comps_m

#Merging with wnv target...

wnv_PCs_m = pd.merge(spray_merged[['Date', 'WnvPresent', 'NumMosquitos','Trap', 'spray_ind']], prin_comps_m, left_index = True, right_index = True)

wnv_PCs_m.info()

#Now let's see how our eight PCs related back to our original features...

prin_comps_features_m = pd.merge(prin_comps_m, X_merged_nm, left_index = True, right_index = True)

corr_prin_comps_m = prin_comps_features_m.corr().drop(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'], axis = 0)

cols_drop_m = ['Latitude', 'Longitude', 'AddressAccuracy', 'spray_2011-08-29',
       'spray_2011-09-07', 'spray_2013-07-17', 'spray_2013-07-25',
       'spray_2013-08-08', 'spray_2013-08-15', 'spray_2013-08-22',
       'spray_2013-08-29', 'spray_2013-09-05', 'NumMosquitos', 'Species_CULEX ERRATICUS',
       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS', 'AvgSpeed',
       'Cool', 'Depart', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'StnPressure', 'Sunrise',
       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']

corr_prin_comps_m.drop(cols_drop_m, axis = 1, inplace = True)
corr_prin_comps_m

'''Note that PC1 essentially is the weather data and pressure, and it explains 40% of the variance. 37% var.
PC2 is mainly pressure and sunrise/sunset time, wind data (greater the wind, neg corr with PC2.) 12% of var.
PC3 is address accuracy primarily.
PC5 is NumMosquitos and wind, though all below 50% corr.
PC6 is the NumMosquitos (-.65 corr, so increase in nummosquitos assoc with decrease in this PC)

'''

wnv_pca.components_

#Exports PCs with the spray info incorporated.

wnv_PCs_m.to_csv('train_PCA_spray_0.75_NumMosquitos.csv', sep = ',', index = True, index_label = 'Index')

'''Version 3:
PCA on Training Data with NumMosquitos but no spray data'''

wnv_pca_m_nospray = PCA(n_components = 8)
X_PCs_m_nospray = wnv_pca_m_nospray.fit_transform(X_merged_m_nospray)

#Creating Df of PCs.

prin_comps_m_nospray = pd.DataFrame(X_PCs_m_nospray, columns = ['PC' + str(i) for i in range(1,9)])
#['PC_' + str(i) for i in range(1,6)]
prin_comps_m_nospray

#Merging with wnv target...

wnv_PCs_m_nospray = pd.merge(spray_merged[['Date', 'WnvPresent', 'NumMosquitos','Trap']], prin_comps_m_nospray, left_index = True, right_index = True)

wnv_PCs_m_nospray.info()

#Now let's see how our eight PCs related back to our original features...

prin_comps_features_m_nospray = pd.merge(prin_comps_m_nospray, X_merged_m_nospray, left_index = True, right_index = True)

corr_prin_comps_m_nospray = prin_comps_features_m_nospray.corr().drop(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'], axis = 0)

cols_drop_m_nospray = ['Latitude', 'Longitude', 'AddressAccuracy', 'NumMosquitos', 'Species_CULEX ERRATICUS',
       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS', 'AvgSpeed',
       'Cool', 'Depart', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'StnPressure', 'Sunrise',
       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']

corr_prin_comps_m_nospray.drop(cols_drop_m_nospray, axis = 1, inplace = True)
corr_prin_comps_m_nospray

'''Note that PC1 essentially is the weather data and pressure, and it explains 40% of the variance. 37% var.
PC2 is mainly pressure and sunrise/sunset time, wind data (greater the wind, neg corr with PC2.) 12% of var.
PC3 is address accuracy primarily.
PC5 is NumMosquitos and wind, though all below 50% corr.
PC6 is the NumMosquitos (-.65 corr, so increase in nummosquitos assoc with decrease in this PC)

'''



#Exports PCs with the spray info incorporated.

wnv_PCs_m_nospray.to_csv('train_PCA_NumMosquitos_nospray.csv', sep = ',', index = True, index_label = 'Index')


#Now that we have the PCs...


'''Model Comparison Massive Script'''
#data = pd.read_csv('pathname.csv')

#X will be PCs. y will be WnvPresent
X =  X_PCs
y =  wnv_PCs['WnvPresent']



'''The code below runs through several different classification models
and adds their scores to a list. Once completed, it graphs each score for
comparison.
'''

#Import various scores from sklearn and other dependencies
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

#Set the test, train, split parameters and define the evaluation model for repeating model application
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    a = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba[:,1])
    print cm
    print cr
    print roc
    return a, roc

all_models = {}
all_models_roc = {}
#Run K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
a, roc  = evaluate_model(KNeighborsClassifier())

#Run GridSearch cross validation on KNN
from sklearn.grid_search import GridSearchCV

params = {'n_neighbors': range(2,60)}

gsknn = GridSearchCV(KNeighborsClassifier(),
                     params, scoring = 'roc_auc',n_jobs=-1,
                     cv=KFold(len(y), n_folds=3, shuffle=True))
gsknn.fit(X, y)
gsknn.best_params_
gsknn.best_score_
a, roc = evaluate_model(gsknn.best_estimator_)
all_models['knn'] = {'model': gsknn.best_estimator_,
                     'score': a}
all_models_roc['knn'] = {'model': gsknn.best_estimator_,
                     'score': roc}

#Run a bagging classifier with KNN
from sklearn.ensemble import BaggingClassifier
baggingknn = BaggingClassifier(KNeighborsClassifier())
a, roc = evaluate_model(baggingknn)

#Run a bagging grid search cross validation with knn
bagging_params = {'n_estimators': [10, 20],
                  'max_samples': [0.7, 1.0],
                  'max_features': [0.7, 1.0],
                  'bootstrap_features': [True, False]}


gsbaggingknn = GridSearchCV(baggingknn,
                            bagging_params, n_jobs=-1,
                            cv=KFold(len(y), n_folds=3, shuffle=True))

gsbaggingknn.fit(X, y)
gsbaggingknn.best_params_
a, roc = evaluate_model(gsbaggingknn.best_estimator_)
all_models['gsbaggingknn'] = {'model': gsbaggingknn.best_estimator_,
                              'score': a}
all_models_roc['gsbaggingknn'] = {'model': gsbaggingknn.best_estimator_,
                              'score': roc}

#Run a logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
all_models['lr'] = {'model': lr,
                    'score': evaluate_model(lr)}
params = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
          'penalty': ['l1', 'l2']}

#Run a GridSearch cross validation with logistic regression
gslr = GridSearchCV(lr,
                    params, n_jobs=-1,
                    cv=KFold(len(y), n_folds=3, shuffle=True))

gslr.fit(X, y)

print gslr.best_params_
print gslr.best_score_

a, roc = evaluate_model(gslr.best_estimator_)

all_models['gslr'] = {'model': gslr.best_estimator_,
                             'score': a}
all_models_roc['gslr'] = {'model': gslr.best_estimator_,
                             'score': roc}
#Run a bagging grid search cross validation with logistic regression
gsbagginglr = GridSearchCV(BaggingClassifier(gslr.best_estimator_),
                           bagging_params, n_jobs=-1,
                           cv=KFold(len(y), n_folds=3, shuffle=True))

gsbagginglr.fit(X, y)

print gsbagginglr.best_params_
print gsbagginglr.best_score_

all_models['gsbagginglr'] = {'model': gsbagginglr.best_estimator_,
                             'score': evaluate_model(gsbagginglr.best_estimator_)}


#Run a Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
evaluate_model(dt)
all_models['dt'] = {'model': dt,
                    'score': evaluate_model(dt)[0]}
params = {'criterion': ['gini', 'entropy'],
          'splitter': ['best', 'random'],
          'max_depth': [None, 5, 10],
          'min_samples_split': [2, 5],
          'min_samples_leaf': [1, 2, 3]}

#Run a grid search with decision tree
gsdt = GridSearchCV(dt,
                    params, n_jobs=-1,
                    cv=KFold(len(y), n_folds=3, shuffle=True))

gsdt.fit(X, y)
print gsdt.best_params_
print gsdt.best_score_
a, roc = evaluate_model(gsdt.best_estimator_)
all_models['gsdt'] = {'model': gsdt.best_estimator_,
                      'score': a}
all_models_roc['gsdt'] = {'model': gsdt.best_estimator_,
                      'score': roc}
#Run a bagging grid search cross validation on a decision tree
gsbaggingdt = GridSearchCV(BaggingClassifier(gsdt.best_estimator_),
                           bagging_params, n_jobs=-1,
                           cv=KFold(len(y), n_folds=3, shuffle=True))

gsbaggingdt.fit(X, y)

print gsbaggingdt.best_params_
print gsbaggingdt.best_score_

a, roc = evaluate_model(gsbaggingdt.best_estimator_)

all_models['gsbaggingdt'] = {'model': gsbaggingdt.best_estimator_,
                             'score': a}
all_models_roc['gsbaggingdt'] = {'model': gsbaggingdt.best_estimator_,
                             'score': roc}
#Run a support vector machine classifier
from sklearn.svm import SVC

svm = SVC()

def evaluate_model_noroc(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    a = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print cm
    print cr

    return a

all_models['svm'] = {'model': svm,
                     'score': evaluate_model_noroc(svm)}


#Run a random forest and extra trees classifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

rf = RandomForestClassifier()

a, roc = evaluate_model(rf)
all_models['rf'] = {'model': rf,
                    'score': a}
all_models_roc['rf'] = {'model': rf,
                    'score': roc}


et = ExtraTreesClassifier()
a, roc = evaluate_model(et)

all_models['et'] = {'model': et,
                    'score': a}
all_models_roc['et'] = {'model': et,
                    'score': roc}

#Run grid search cross validation with random forests
params = {'n_estimators':[3, 5, 10, 50],
          'criterion': ['gini', 'entropy'],
          'max_depth': [None, 3, 5],
          'min_samples_split': [2,5],
          'class_weight':[None, 'balanced']}


gsrf = GridSearchCV(RandomForestClassifier(n_jobs=-1),
                    params, n_jobs=-1,
                    cv=KFold(len(y), n_folds=3, shuffle=True))

gsrf.fit(X, y)
print gsrf.best_params_
print gsrf.best_score_

a, roc = evaluate_model(gsrf.best_estimator_)

all_models['gsrf'] = {'model': gsrf.best_estimator_,
                      'score': a}
all_models['gsrf'] = {'model': gsrf.best_estimator_,
                      'score': roc}


#Run grid search cross validation with extra trees
gset = GridSearchCV(RandomForestClassifier(n_jobs=-1),
                    params, n_jobs=-1,
                    cv=KFold(len(y), n_folds=3, shuffle=True))

gset.fit(X, y)
print gset.best_params_
print gset.best_score_

a, roc = evaluate_model(gset.best_estimator_)

all_models['gset'] = {'model': gset.best_estimator_,
                      'score': a}
all_models_roc['gset'] = {'model': gset.best_estimator_,
                      'score': roc}

#AdaBoost

adaboost = AdaBoostClassifier()
a, roc = evaluate_model(adaboost)

all_models['adaboostdt'] = {'model': adaboost,
                      'score': a}
all_models_roc['adaboostdt'] = {'model': adaboost,
                      'score': roc}


#Plot the scores for comparison
all_models

scores = pd.DataFrame(all_models)

scores_transpose = scores.T
scores_transpose.score.plot(kind='bar')
plt.ylim(0.6, 1.1)

scores_roc = pd.DataFrame(all_models_roc).transpose()
scores_roc.score.plot(kind = 'bar')

'''Best models are baggingknn and dt classifier'''
from sklearn.cross_validation import cross_val_score, StratifiedKFold

def retest(model):
    scores = cross_val_score(model, X, y,
                             cv=StratifiedKFold(y, shuffle=True),
                             n_jobs=-1)
    m = scores.mean()
    s = scores.std()

    return m, s

for k, v in all_models.iteritems():
    cvres = retest(v['model'])
    print k,
    all_models[k]['cvres'] = cvres


cvscores = pd.DataFrame([(k, v['cvres'][0], v['cvres'][1] ) for k, v in all_models.iteritems()],
                        columns=['model', 'score', 'error']).set_index('model').sort_values('score', ascending=False)



fig, ax = plt.subplots()
rects1 = ax.bar(range(len(cvscores)), cvscores.score,
                yerr=cvscores.error,
                tick_label=cvscores.index)

ax.set_ylabel('Scores')
plt.xticks(rotation=70)
plt.ylim(0.6, 1.1)


'''Applying PCA to the test data.
First file is the test_imputed_avg, where NumMosquitos is imputed and for the new
mosquitos species in the data, the avg count is what is used for that.
'''

test_imp_avg = pd.read_csv('../west_nile/input/test_imputed_avg.csv')

test_imp_avg.head()

#Scaled weather data needs to be merged to the df, and Mosquitos needs to be scaled.
#Let's scale the weather_mean data right now...

weather_mean.info()
weather_stdscale = scaler.fit_transform(weather_mean[scale_cols])

#Join scaled weather data back to weather mean index.
weather_scale = pd.DataFrame(weather_stdscale, columns = scale_cols, index = weather_mean.index)
weather_scale.info()


#Scale NumMosq in test data.
scaler_mosq = StandardScaler()

NumMosqscaled = scaler_mosq.fit_transform(test_imp_avg['NumMosq'])
mosq_df = pd.DataFrame(NumMosqscaled, columns = ['NumMosq'], index = test_imp_avg.index)

#Have to dummify the species since those were in my PCs...

dummies_test_imp_avg = pd.get_dummies(test_imp_avg.Species)

dummies_test_imp_avg.drop('UNSPECIFIED CULEX', axis = 1, inplace = True)

#Merge dummies to test, then to NumMosq scaled, then to weather scaled. Then extract
#columns used in training PCA, and apply PCA to them.

test_imp_avg_dummies = pd.merge(test_imp_avg, dummies_test_imp_avg, left_index = True, right_index = True)
test_avg_mosq = pd.merge(test_imp_avg_dummies, mosq_df, left_index = True, right_index = True)

#Have to convert Date in test data to Datetime first before merge.
test_avg_mosq['Date'] = pd.to_datetime(test_avg_mosq['Date'], infer_datetime_format = True)
test_avg_mosq_weather = pd.merge(test_avg_mosq, weather_scale, how = 'left', left_on = 'Date', right_index = True)



PCA_features = ['Latitude',
 'Longitude',
 'AddressAccuracy',
 'CULEX ERRATICUS',
 'CULEX PIPIENS',
 'CULEX PIPIENS/RESTUANS',
 'CULEX RESTUANS',
 'CULEX SALINARIUS',
 'CULEX TARSALIS',
 'CULEX TERRITANS','NumMosq_y',
 'AvgSpeed',
 'Cool',
 'Depart',
 'DewPoint',
 'Heat',
 'ResultDir',
 'ResultSpeed',
 'SeaLevel',
 'StnPressure',
 'Sunrise',
 'Sunset',
 'Tavg',
 'Tmax',
 'Tmin',
 'WetBulb']

#Extracting features from test_imp_avg with scaling for PCA.
wnv_pca_m_nospray
wnv_test_avg_mosq_pca = wnv_pca_m_nospray.fit_transform(test_avg_mosq_weather[PCA_features])

wnv_test_avg_mosq_df = pd.DataFrame(wnv_test_avg_mosq_pca, columns = ['PC' + str(i) for i in range(1,9)], index = test_avg_mosq_weather.index)
id_df = pd.DataFrame(test_avg_mosq_weather['Id'], columns = ['Id'], index = test_avg_mosq_weather.index)

wnv_test_avg_imp_PCs = pd.merge(id_df, wnv_test_avg_mosq_df, left_index = True, right_index = True)
wnv_test_avg_imp_PCs.to_csv('test_avg_imp_PCs.csv', sep = ',', index = True, index_label = 'Index')


'''Loading test_zero_imp which has no spray data, applying PCA, exporting to CSV.'''


test_imp_zero = pd.read_csv('../west_nile/input/test_imputed_zero.csv')

test_imp_zero.head()

#weather_scale.info()

#Scale NumMosq in test data.
scaler_mosq_zero = StandardScaler()

NumMosqscaled = scaler_mosq_zero.fit_transform(test_imp_zero['NumMosq'])
mosq_df = pd.DataFrame(NumMosqscaled, columns = ['NumMosq'], index = test_imp_zero.index)

#Have to dummify the species since those were in my PCs...

dummies_test_imp_zero = pd.get_dummies(test_imp_zero.Species)

dummies_test_imp_zero.drop('UNSPECIFIED CULEX', axis = 1, inplace = True)

#Merge dummies to test, then to NumMosq scaled, then to weather scaled. Then extract
#columns used in training PCA, and apply PCA to them.

test_imp_zero_dummies = pd.merge(test_imp_zero, dummies_test_imp_zero, left_index = True, right_index = True)
test_zero_mosq = pd.merge(test_imp_zero_dummies, mosq_df, left_index = True, right_index = True)

#Have to convert Date in test data to Datetime first before merge.
test_zero_mosq['Date'] = pd.to_datetime(test_zero_mosq['Date'], infer_datetime_format = True)
test_zero_mosq_weather = pd.merge(test_zero_mosq, weather_scale, how = 'left', left_on = 'Date', right_index = True)



PCA_features = ['Latitude',
 'Longitude',
 'AddressAccuracy',
 'CULEX ERRATICUS',
 'CULEX PIPIENS',
 'CULEX PIPIENS/RESTUANS',
 'CULEX RESTUANS',
 'CULEX SALINARIUS',
 'CULEX TARSALIS',
 'CULEX TERRITANS','NumMosq_y',
 'AvgSpeed',
 'Cool',
 'Depart',
 'DewPoint',
 'Heat',
 'ResultDir',
 'ResultSpeed',
 'SeaLevel',
 'StnPressure',
 'Sunrise',
 'Sunset',
 'Tavg',
 'Tmax',
 'Tmin',
 'WetBulb']

#Extracting features from test_imp_zero with scaling for PCA.
wnv_pca_m_nospray
wnv_test_zero_mosq_pca = wnv_pca_m_nospray.fit_transform(test_zero_mosq_weather[PCA_features])

wnv_test_zero_mosq_df = pd.DataFrame(wnv_test_zero_mosq_pca, columns = ['PC' + str(i) for i in range(1,9)], index = test_zero_mosq_weather.index)
id_df = pd.DataFrame(test_zero_mosq_weather['Id'], columns = ['Id'], index = test_zero_mosq_weather.index)

wnv_test_zero_imp_PCs = pd.merge(id_df, wnv_test_zero_mosq_df, left_index = True, right_index = True)
wnv_test_zero_imp_PCs.to_csv('test_zero_imp_PCs.csv', sep = ',', index = True, index_label = 'Index')

'''Loading test data with spray dates from 2013 in it (assuming these were also sprayed in test years, essentially). Avg NumMosq imputation.'''

test_imp_avg_spray = pd.read_csv('../west_nile/input/test_imputed_avg_spray.csv')

test_imp_avg_spray.info()


#Scale NumMosq in test data.
scaler_mosq_spray = StandardScaler()

NumMosqscaled = scaler_mosq_spray.fit_transform(test_imp_avg_spray['NumMosq'])
mosq_df = pd.DataFrame(NumMosqscaled, columns = ['NumMosq'], index = test_imp_avg_spray.index)



#Merge dummies to test, then to NumMosq scaled, then to weather scaled. Then extract
#columns used in training PCA, and apply PCA to them.

test_avg_mosq_spray = pd.merge(test_imp_avg_spray, mosq_df, left_index = True, right_index = True)

#Have to convert Date in test data to Datetime first before merge.
test_avg_mosq_spray['Date'] = pd.to_datetime(test_avg_mosq_spray['Date'], infer_datetime_format = True)
test_avg_mosq_weather_spray = pd.merge(test_avg_mosq_spray, weather_scale, how = 'left', left_on = 'Date', right_index = True)



PCA_features_spray = ['Latitude',
 'Longitude',
 'AddressAccuracy',
 'spray_2011-08-29', 'spray_2011-09-07',
 'spray_2013-07-17', 'spray_2013-07-25', 'spray_2013-08-08',
 'spray_2013-08-15', 'spray_2013-08-22', 'spray_2013-08-29',
 'spray_2013-09-05',
 'Species_CULEX ERRATICUS',
 'Species_CULEX PIPIENS',
 'Species_CULEX PIPIENS/RESTUANS',
 'Species_CULEX RESTUANS',
 'Species_CULEX SALINARIUS',
 'Species_CULEX TARSALIS',
 'Species_CULEX TERRITANS','NumMosq_y',
'AvgSpeed_y', 'Cool_y', 'Depart_y', 'DewPoint_y',
'Heat_y', 'ResultDir_y', 'ResultSpeed_y', 'SeaLevel_y',
'StnPressure_y', 'Sunrise_y', 'Sunset_y', 'Tavg_y', 'Tmax_y',
'Tmin_y', 'WetBulb_y']

#Extracting features from test_imp_avg with scaling for PCA.
wnv_pca_m
wnv_test_avg_mosq_spray_pca = wnv_pca_m.fit_transform(test_avg_mosq_weather_spray[PCA_features_spray])

wnv_test_avg_mosq_spray_df = pd.DataFrame(wnv_test_avg_mosq_spray_pca, columns = ['PC' + str(i) for i in range(1,9)], index = test_avg_mosq_weather_spray.index)
#id_df = pd.DataFrame(test_avg_mosq_weather_spray['Id'], columns = ['Id'], index = test_avg_mosq_weather_spray.index)

#wnv_test_avg_imp_PCs = pd.merge(id_df, wnv_test_avg_mosq_df, left_index = True, right_index = True)
wnv_test_avg_mosq_spray_df.to_csv('test_avg_imp_spray_PCs.csv', sep = ',', index = True, index_label = 'Index')


'''Test data with zero for new species, getting PCs'''


test_imp_zero_spray = pd.read_csv('../west_nile/input/test_imputed_zero_spray.csv')

test_imp_zero_spray.info()

#weather_scale.info()

#Scale NumMosq in test data.
scaler_mosq_zero_spray = StandardScaler()

NumMosqscaled = scaler_mosq_zero_spray.fit_transform(test_imp_zero_spray['NumMosq'])
mosq_df = pd.DataFrame(NumMosqscaled, columns = ['NumMosq'], index = test_imp_zero_spray.index)

#Have to dummify the species since those were in my PCs...

#Merge dummies to test, then to NumMosq scaled, then to weather scaled. Then extract
#columns used in training PCA, and apply PCA to them.


test_zero_mosq_spray = pd.merge(test_imp_zero_spray, mosq_df, left_index = True, right_index = True)

#Have to convert Date in test data to Datetime first before merge.
test_zero_mosq_spray['Date'] = pd.to_datetime(test_zero_mosq_spray['Date'], infer_datetime_format = True)
test_zero_mosq_spray_weather = pd.merge(test_zero_mosq_spray, weather_scale, how = 'left', left_on = 'Date', right_index = True)


#Extracting features from test_imp_zero with scaling for PCA.
wnv_pca_m
wnv_test_zero_mosq_pca_spray = wnv_pca_m.fit_transform(test_zero_mosq_spray_weather[PCA_features_spray])

wnv_test_zero_mosq_df_spray = pd.DataFrame(wnv_test_zero_mosq_pca_spray, columns = ['PC' + str(i) for i in range(1,9)], index = test_zero_mosq_spray_weather.index)
#id_df = pd.DataFrame(test_zero_mosq_weather['Id'], columns = ['Id'], index = test_zero_mosq_weather.index)
wnv_test_zero_mosq_df_spray.info()
#wnv_test_zero_imp_PCs = pd.merge(id_df, wnv_test_zero_mosq_df, left_index = True, right_index = True)
wnv_test_zero_mosq_df_spray.to_csv('test_zero_imp_spray_PCs.csv', sep = ',', index = True, index_label = 'Index')
