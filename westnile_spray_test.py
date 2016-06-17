import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

'''This script adds the spray dummy columns to the Test set from the values
    we computed on the Train set'''

# Read in our datasets
train = pd.read_csv('/Users/Brian/spray_0.75_clean.csv')
test = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/assets/test.csv')

# Get a list of the traps in our training set
traps = train.Trap.unique()
# Create a blank DF for the traps
master_df = pd.DataFrame(columns=train.columns.values)
# Pull the spray columns for each trap from our training set
for trap in traps:
    trap_df = train[train.Trap == trap]
    row = trap_df.iloc[0,:]
    master_df = master_df.append(row,ignore_index=True)

# Define our column names
master_df = master_df[['Trap','spray_2011-08-29','spray_2011-09-07',
                    'spray_2013-07-17', 'spray_2013-07-25','spray_2013-08-08',
                    'spray_2013-08-15', 'spray_2013-08-22','spray_2013-08-29',
                    'spray_2013-09-05']]
spray_cols = ['spray_2011-08-29','spray_2011-09-07', 'spray_2013-07-17',
            'spray_2013-07-25','spray_2013-08-08', 'spray_2013-08-15',
            'spray_2013-08-22','spray_2013-08-29', 'spray_2013-09-05']
# Add the spray columns to our testing set
for col in spray_cols:
    test[col] = np.nan

# Fill in the spray columns for the traps that appear in our training set
for index, row in test.iterrows():
    test_trap = row.Trap
    temp = master_df[master_df['Trap'] == test_trap]
    # Check to see if the trap was in the training set
    if len(temp) == 1:
        spray_vals = list(temp.values[0])
        del spray_vals[0]
        i = 0
        # Assign the values
        for col in spray_cols:
            test.set_value(index,col,spray_vals[i])

# Find the traps that weren't in our training set
f = test[test['spray_2011-08-29'].isnull()]['Trap'].unique()

# Create a new DF for these missing traps
unknown_spray1 = pd.DataFrame(f, columns=['Trap'])
unknown_spray1['Latitude'] = np.nan
unknown_spray1['Longitude'] = np.nan
unknown_spray2 = pd.DataFrame(columns=spray_cols)
unknown_spray = unknown_spray1.join(unknown_spray2,how='outer')
# Get the latitude and longitude for the missing traps
for index,row in unknown_spray.iterrows():
    trap = row.Trap
    temp = test[test['Trap'] == trap]
    unknown_spray.set_value(index,'Latitude',temp.Latitude.unique()[0])
    unknown_spray.set_value(index,'Longitude',temp.Longitude.unique()[0])
# Read in our Spray data
spray = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/assets/spray.csv')

# This function calculates the distance between two lat/long points on the earth
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956 # Radius of earth in miles
    return c * r

# Compute the distances for missing traps
for dist in [.75]:
    print 'Starting distance',str(dist)
    # Look at each spraying date individually
    for date in spray.Date.unique():
        spray_date = spray[spray.Date == date]
        date_col = 'spray_' + date
        # Look at each row in the train dataframe individually
        for index1, t_row in unknown_spray.iterrows():
            print index1
            # Get the latitude and longitude for this trap
            lon1 = t_row.Longitude
            lat1 = t_row.Latitude
            # For every spray that occured on the specified date, check if
            # the trap is within specified distance of the spray
            for index2, s_row in spray_date.iterrows():
                # Get the latitude and longitude for this spray
                lon2 = s_row.Longitude
                lat2 = s_row.Latitude
                # Compute the distance between the trap and the spray
                distance = haversine(lon1,lat1,lon2,lat2)
                # Check if the distance is within our radius
                if distance <= dist:
                    # If it is, we change the value of the cell to show that
                    # there was a spray within the specified radius of this trap
                    # on the specified date
                    unknown_spray.set_value(index1,date_col,1)
                    # As soon as we find a spray withing the radius, we can
                    # break out of this loop and move on to the next trap
                    break
# Replace any NaNs with 0
unknown_spray.fillna(0,inplace=True)

# Fill in the missing traps in our test set with the distances we just computed
for index, row in test.iterrows():
    test_trap = row.Trap
    temp = unknown_spray[unknown_spray['Trap'] == test_trap]
    if len(temp) == 1:
        spray_vals = list(temp.values[0])
        del spray_vals[0]
        i = 0
        for col in spray_cols:
            test.set_value(index,col,spray_vals[i])

# Make sure there are no missing values
print test.isnull().sum()

# Read in our weather csv
weather = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/weather_mean.csv')


# Merge our test set (now with spray columns) with the weather data and
# make dummy variables for the Species column

# Drop the columns created from our join above
test.drop(['Id'],axis=1,inplace=True)
# Get dummy variables for the Species column
test = pd.get_dummies(test,columns=['Species'])
# Merge the train DF with the weather DF on the Date columns
merged = test.merge(weather,on='Date',how='outer')
# Drop the rows where we have weather data but no traps were checked
merged.dropna(axis=0,how='any',inplace=True)
# Drop the UNSPECIFIED CULEX column, because it doesn't appear in our training
# data. A 0 in every other Species_ columns represents a 1 in UNSPECIFIED
merged.drop('Species_UNSPECIFIED CULEX',axis=1,inplace=True)

test_avg = pd.read_csv('/Users/Brian/test_imputed_avg.csv')
test_zero = pd.read_csv('/Users/Brian/test_imputed_zero.csv')

test_avg_spray = merged
test_avg_spray['NumMosq'] = test_avg['NumMosq']

test_zero_spray = merged
test_zero_spray['NumMosq'] = test_zero['NumMosq']

# Save our test DataFrames to csv
test_avg_spray.to_csv('test_imputed_avg_spray.csv')
test_zero_spray.to_csv('test_imputed_zero_spray.csv')
