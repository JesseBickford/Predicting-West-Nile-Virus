import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics
from sklearn.cross_validation import KFold


def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='mse', optimizer="adadelta")
    return model

data = pd.read_csv('/Users/Brian/spray_0.75_merged.csv')

scale_cols = ['NumMosquitos','AvgSpeed','DewPoint',
            'ResultDir','ResultSpeed','SeaLevel','StnPressure','Tavg',
            'Tmax','Tmin','WetBulb']

nonscale_cols = ['Species_CULEX ERRATICUS','Species_CULEX PIPIENS',
                'Species_CULEX PIPIENS/RESTUANS','Species_CULEX RESTUANS',
                'Species_CULEX SALINARIUS','Species_CULEX TARSALIS',
                'Species_CULEX TERRITANS']

# Split our data into the columns that need scaled and those that don't
X_scale = data[scale_cols]
X_scale = X_scale.as_matrix()
X_nonscale = data[nonscale_cols]
X_nonscale = X_nonscale.as_matrix()
# Scale our data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_scale)
X_scaled = scaler.transform(X_scale)
# Join the two back together
X = np.concatenate((X_scaled,X_nonscale),axis=1)
y = y.as_matrix()
# Convert to DF so we can use feature selection
X = pd.DataFrame(X)
# Select the best features
from sklearn.feature_selection import SelectKBest,f_classif
def k_best(X,y,k):
    select = SelectKBest(f_classif, k=k)
    selected_data = select.fit_transform(X,y)
    selected_cols = X.columns[select.get_support()]
    X_selected = pd.DataFrame(selected_data, columns=selected_cols)
    return X_selected
X = k_best(X,y,20)
# Convert X back to a matrix for use in the ANN
X = X.as_matrix()

# Set up our ANN parameters
Y = np_utils.to_categorical(y)
input_dim = X.shape[1]
output_dim = 2

# Split our data into folds and set up neccessary variables
nb_folds = 4
kfolds = KFold(len(y), nb_folds,shuffle=True,random_state=87)
av_roc = 0.
f = 0
# Train our ANN and predict probabilities
for train, valid in kfolds:
    print('---'*20)
    print('Fold', f)
    print('---'*20)
    f += 1
    X_train = X[train]
    X_valid = X[valid]
    Y_train = Y[train]
    Y_valid = Y[valid]
    y_valid = y[valid]

    print("Building model...")
    model = build_model(input_dim, output_dim)

    print("Training model...")

    model.fit(X_train, Y_train, nb_epoch=200, batch_size=16,
                validation_data=(X_valid, Y_valid), verbose=0)
    valid_preds = model.predict_proba(X_valid, verbose=0)
    valid_preds = valid_preds[:, 1]
    roc = metrics.roc_auc_score(y_valid, valid_preds)
    print("AUC:", roc)
    av_roc += roc
# Check our AUC score
print('Average AUC:', av_roc/nb_folds)

# Now we want to predict our test set
# Train on the entire training set
model = build_model(input_dim, output_dim)
model.fit(X, Y, nb_epoch=200, batch_size=16, verbose=0)
# Read in our test set
test = pd.read_csv('/Users/Brian/test_imputed_zero_spray.csv')
scale_cols = ['NumMosq','AvgSpeed','DewPoint',
            'ResultDir','ResultSpeed','SeaLevel','StnPressure','Tavg',
            'Tmax','Tmin','WetBulb']
nonscale_cols = ['Species_CULEX ERRATICUS','Species_CULEX PIPIENS',
                'Species_CULEX PIPIENS/RESTUANS','Species_CULEX RESTUANS',
                'Species_CULEX SALINARIUS','Species_CULEX TARSALIS',
                'Species_CULEX TERRITANS']
# Scale our testing data
X_t_scale = test[scale_cols]
X_t_nonscale = test[nonscale_cols]
X_t_scaled = scaler.transform(X_t_scale)
X_test = np.concatenate((X_t_scaled,X_t_nonscale),axis=1)
# Predict our test data
preds = model.predict_proba(X_test, verbose=0)

# Export our predictions into the required submission format
preds_df = pd.DataFrame(preds,index=range(1,116294))
preds_df.drop(0,inplace=True,axis=1)
preds_df.columns = ['WnvPresent']
preds_df.to_csv('preds_imputued_zero1.csv',index_label='id')
