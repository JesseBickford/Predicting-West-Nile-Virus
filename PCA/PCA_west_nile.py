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


spray_merged = pd.read_csv('../west_nile/spray_0.75_merged.csv')

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
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS', 'spray_ind']

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

#Separate file - looking to see what PCs are like without spraying...
no_spray_X = X_merged_no_nm[X_merged_no_nm['spray_ind'] == 0]

cols_no_spray = ['Latitude', 'Longitude', 'AddressAccuracy', 'Species_CULEX ERRATICUS',
       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS',
       'AvgSpeed', 'Cool', 'Depart', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'StnPressure', 'Sunrise', 'Sunset',
       'Tavg', 'Tmax', 'Tmin', 'WetBulb']

X_no_spray = no_spray_X[cols_no_spray]

'''Original PCA with spray data'''
X_covmat = np.cov(X_merged_no_nm.T)

eig_vals, eig_vecs = np.linalg.eig(X_covmat)

print eig_vals
print '----------------------'
print eig_vecs

eigen_pairs = [[eig_vals[i], eig_vecs[:,i]] for i in range(len(eig_vals))]

eigenpairs = pd.DataFrame(eigen_pairs, columns = ['eigenvalue', 'eigenvector'])

eigenpairs.sort_values('eigenvalue', ascending = False)

'''Eigenvalues for PCA without spray data'''
X_nospray_covmat = np.cov(X_no_spray.T)
eig_vals2, eig_vecs2 = np.linalg.eig(X_nospray_covmat)
eigen_pairs2 = [[eig_vals2[i], eig_vecs2[:,i]] for i in range(len(eig_vals2))]

eigenpairs2 = pd.DataFrame(eigen_pairs2, columns = ['eigenvalue', 'eigenvector'])

eigenpairs2.sort_values('eigenvalue', ascending = False)

'''
#Total explained variance for PCA with spray data'''

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



'''Total Explained Variance for PCA w/o spray data'''

#sum all the eigenvalues together.
totaleig_val2 = eigenpairs2.eigenvalue.sum()
print "Total Eigenvalue Sum is:", totaleig_val2
print '-------------------'
indiv_var2 = [eigenpairs2.eigenvalue[i]/totaleig_val*100 for i in range(len(eigenpairs2))]
cum_exp_var2 = np.cumsum(indiv_var2)

print 'Cumulative Variance Explained as we include principal components:', cum_exp_var2
print "There are %i eigenvalues." % len(eigenpairs2.eigenvalue)

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

print "Cumulative variance explained at Component 10:", cum_exp_var2[9]
print "Cumulative variance explained at Component 8:", cum_exp_var2[7]
print "Cumulative variance explained at Component 5:", cum_exp_var2[4]



'''PCA - creating the Principal Components from all numerical features and species inds.
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

'''PCA for the Training Data for sites we know were not sprayed.'''
wnv_pca_nospray = PCA(n_components = 8)
X_PCs_nospray = wnv_pca.fit_transform(X_no_spray)


#Creating Df of PCs.

prin_comps = pd.DataFrame(X_PCs_nospray, columns = ['PC' + str(i) for i in range(1,9)])
#['PC_' + str(i) for i in range(1,6)]
prin_comps

#Merging with wnv target...

wnv_PCs_nospray = pd.merge(spray_merged[['WnvPresent', 'NumMosquitos','Trap']][spray_merged['spray_ind'] == 0], prin_comps, left_index = True, right_index = True)

wnv_PCs_nospray.info()

#Now let's see how our eight PCs related back to our original features...

prin_comps_features = pd.merge(prin_comps, X_no_spray, left_index = True, right_index = True)

corr_prin_comps = prin_comps_features.corr().drop(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'], axis = 0)

cols_drop = ['Latitude', 'Longitude', 'AddressAccuracy', 'Species_CULEX ERRATICUS',
       'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
       'Species_CULEX RESTUANS', 'Species_CULEX SALINARIUS',
       'Species_CULEX TARSALIS', 'Species_CULEX TERRITANS', 'AvgSpeed',
       'Cool', 'Depart', 'DewPoint', 'Heat', 'ResultDir',
       'ResultSpeed', 'SeaLevel', 'StnPressure', 'Sunrise',
       'Sunset', 'Tavg', 'Tmax', 'Tmin', 'WetBulb']

corr_prin_comps.drop(cols_drop, axis = 1, inplace = True)
corr_prin_comps

'''Testing the PCs on predicting number of mosquitos...'''

target_nm = wnv_PCs_nospray['NumMosquitos']

X_nm = wnv_PCs_nospray[['PC1', 'PC2', 'PC3', 'PC4', 'PC5','PC6','PC7','PC8']]

#Now that we have the PCs...

#Running a quick logistic regression...

from sklearn.linear_model import LassoCV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
#lm = LinearRegression()
#lasso = LassoCV(random_state = 31, n_jobs = -1, verbose = True)

X_train, X_test, y_train, y_test = train_test_split(X_PCs, wnv_PCs['WnvPresent'], stratify = wnv_PCs['WnvPresent'], test_size = .25, random_state = 31)

#Logistic Regression

logreg = LogisticRegression(random_state = 31)
params = {'C': [.1, .5, 1, 10, 100, 500], 'penalty': ['l1', 'l2']}

grid_logreg = GridSearchCV(logreg, param_grid = params, scoring = 'roc_auc', cv = 5, verbose = True, n_jobs = -1)


grid_logreg.fit(X_train, y_train)
grid_logreg.best_estimator_
grid_logreg.best_score_

y_pred_logreg = grid_logreg.predict_proba(X_test)[:, 1]
print "ROC AUC Score with PCs for WNV is:" , roc_auc_score(y_test, y_pred_logreg)

print "R2 score for Linear Regression is:", r2_score(y_test, y_pred_lm)

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

test = pd.read_csv('assets/test.csv')

test.info()

weather_mean = pd.read_csv('weather_mean.csv')



test_dummies = pd.merge(test, dummies, right_index = True, left_index = True)
test_dummies.info()

#merge weather data with test_dummies
test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format = True)
weather_mean.Date = pd.to_datetime(weather_mean['Date'], infer_datetime_format = True)

weather_mean.set_index('Date', inplace = True)

test_weather = pd.merge(test, weather_mean, how = 'left', left_on = 'Date', right_index = True)

test_dummies.info()
weather_mean.info()

dummies = pd.get_dummies(test_weather.Species)
dummies.info()
#drop the species not in train data.



test_weather.drop('Species', axis = 1, inplace = True)
dummies.drop('UNSPECIFIED CULEX', axis = 1, inplace = True)

#join dummies back
test_weather_dummies = pd.merge(test_weather, dummies, left_index = True, right_index = True)
test_weather_dummies.info()

#Scale variables we scaled above...
scaler_test = StandardScaler()


test_scaled_array = scaler_test.fit_transform(test_weather_dummies[scale_cols])
test_scaled = pd.DataFrame(test_scaled_array, columns = scale_cols, index = test_weather_dummies.index)


non_scale_use_test = ['Latitude', 'Longitude',
       'AddressAccuracy', 'CULEX ERRATICUS',
       'CULEX PIPIENS', 'CULEX PIPIENS/RESTUANS',
       'CULEX RESTUANS', 'CULEX SALINARIUS',
       'CULEX TARSALIS', 'CULEX TERRITANS']


'''Merge the scaled data and ind vars...keeping num mosquitos out of this...'''
test_merged_no_nm = pd.merge(test_weather_dummies[non_scale_use_test], test_scaled, left_index = True, right_index = True)

wnv_PCs.info()
