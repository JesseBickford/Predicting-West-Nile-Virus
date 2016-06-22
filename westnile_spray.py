import pandas as pd
from math import radians, cos, sin, asin, sqrt

''' This script computes the distance between all traps and spray locations
 and creates dummy variables to signal if the trap was within a specified
 distance of a spray on each date '''

spray = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/assets/spray.csv')

train = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/assets/train.csv')

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

'''Warning: Each iteration through this loop will take approx 3 hours'''
# We want to go through each trap in the train dataset and compute (for each day
# that spraying occured) if the trap is within a certain distance of spraying
# We aren't sure which radius we want to use for our distance, so we test a
# couple different values
for dist in [.25,.75]:
    print 'Starting distance',str(dist)
    # Create a DataFrame for this distance
    this_train = train
    # Look at each spraying date individually
    for date in spray.Date.unique():
        print str(date)+' for dist '+str(dist)
        # Create a new column that will signal if each trap was within the
        # specified distance of a spraying location on that date.
        # Initialize the column to have every element = 0
        date_col = 'spray_' + date
        this_train[date_col] = 0
        # Filter our spray database to only show the sprays that occured on
        # the specified date
        spray_date = spray[spray.Date == date]
        # Look at each row in the train dataframe individually
        for index1, t_row in this_train.iterrows():
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
                    this_train.set_value(index1,date_col,1)
                    # As soon as we find a spray withing the radius, we can
                    # break out of this loop and move on to the next trap
                    break
    # Save the DataFrame created for this distance to a csv
    filename = 'spray_'+str(dist)+'.csv'
    this_train.to_csv(filename)
    print 'Done with distance',str(dist)

# Read in the csv files we just created
# (Start here with the original csv files to save time)
spray_half = pd.read_csv('/Users/Brian/spray_0.5.csv')
spray_one = pd.read_csv('/Users/Brian/spray_1.csv')
spray_three = pd.read_csv('/Users/Brian/spray_3.csv')
spray_quarter = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/spray_0.25.csv')
spray_3quarter = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/spray_0.75.csv')

# The spray that occured on 8/15/13 started in the evening and continued past
# midnight into 8/16/13. Since this is part of the same spray, having two
# columns for one spray will cause issues with our models.
# We want to set the element in the 8/15/13 spray equal to 1 if there was a
# spray within the distance on either 8/15 or 8/16

# Iterate through our 5 DataFrames
for df in [spray_half,spray_one,spray_three,spray_quarter,spray_3quarter]:
    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        # We only need to update the 8/15 column if it isn't already = 1
        if row['spray_2013-08-15'] == 0:
            # We only update the 8/15 column if 8/16 = 1
            if row['spray_2013-08-16'] == 1:
                # Update the value in the 8/15/13 column
                df.set_value(index,'spray_2013-08-15',1)
    # Drop the 8/16/2013 column from the DataFrame
    df.drop('spray_2013-08-16',axis=1,inplace=True)

# Save our updated DataFrames to csv files
spray_half.to_csv('spray_0.5_clean.csv')
spray_one.to_csv('spray_1_clean.csv')
spray_three.to_csv('spray_3_clean.csv')
spray_quarter.to_csv('spray_0.25_clean.csv')
spray_3quarter.to_csv('spray_0.75_clean.csv')
