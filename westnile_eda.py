import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

''' This script performs some exploratory data analysis and creates some graphs
 to help visualize the data we are going to work with '''

# Read in our dataset
df = pd.read_csv('/Users/Brian/Downloads/west_nile/input/train.csv')
df
# Look at the head and tail
print df.head()
print df.tail()

# Look at the summary statistics
print df.describe()

# Check how many rows we have in our DataFrame
print 'Number of rows: %d' % len(df)

# Look at our dtypes
print df.dtypes
# We see that the only column dtype we need to fix is Date
df.Date = pd.to_datetime(df.Date)

# Check for any missing values
print df.isnull().sum()
# There are no missing values to deal with

# The data set description says that each entry is capped at 50 mosquitos, and
# the rest are split into a separate line.
# Lets see how many of these lines we have
print 'Number of lines with max allowable mosquitos: %d' % len(df[df.NumMosquitos == 50])

# Look at the distributions of numeric columns
sns.distplot(df.AddressAccuracy)
plt.show()
print 'AddressAccuracy value counts:\n%s' % str(df.AddressAccuracy.value_counts())

sns.distplot(df.NumMosquitos)
plt.show()

# Check the distribution of our target variable
print 'WnvPresent value counts:\n%s' % str(df.WnvPresent.value_counts())

# Check how many unique values there are for Species
print df.Species.unique()
