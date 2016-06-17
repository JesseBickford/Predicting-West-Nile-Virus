import pandas as pd

# Read in our train and test sets
train = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/assets/train.csv')
test = pd.read_csv('/Users/Brian/Predicting-West-Nile-Virus/assets/test.csv')

# Create a new column to bring all the subtraps to the main trap number
train['TrapS'] = train['Trap'].apply(lambda s: s[0:4])
test['TrapS'] = test['Trap'].apply(lambda s: s[0:4])

# Fix the issue where number of species per trap per day is capped at 50
pivot1 = pd.pivot_table(train,index=['TrapS','Species','Date'],values='NumMosquitos',aggfunc=np.sum)
pivot1 = pd.DataFrame(pivot1)
# Flatten out our DF so we can pivot again
flat = pd.DataFrame(pivot1.to_records()) # multiindex become columns and new index is integers only
# Compute the average number of mosquitos per species per trap per day
pivot2 = pd.pivot_table(flat,index=['TrapS','Species'],values='NumMosquitos',aggfunc=np.mean)
pivot2 = pd.DataFrame(pivot2)
# Flatten out our DF so we can access it by column indexes
trap_spec_avg = pd.DataFrame(pivot2.to_records()) # multiindex become columns and new index is integers only
# Compute the average number of mosquitos per species per day
spec_avg = pd.pivot_table(flat,index='Species',values='NumMosquitos',aggfunc=np.mean)
spec_avg = pd.DataFrame(spec_avg)
# Flatten out our DF so we can access it by column indexes\
spec_avg = pd.DataFrame(spec_avg.to_records())

# Compute the average number of mosquitos the daily species sums
avg_mosq = np.mean(flat.NumMosquitos)

# Create a new column for predicted number of mosqitoes
test['NumMosq'] = np.nan
i=0
for index, row in test.iterrows():
    print i
    i += 1
    trap = row.Trap
    # Trap T234 is the only trap in Test that isnt in Train
    # We will replace it with the closest trap, T005
    if trap == 'T234':
        trap == 'T005'
    species = row.Species
    temp = trap_spec_avg[(trap_spec_avg['TrapS'] == trap) & (trap_spec_avg['Species'] == species)]
    # Check to see if there was a match
    if len(temp) == 1:
        num = temp.NumMosquitos.values[0]
        test.set_value(index,'NumMosq',num)
    else:
        # There is no combination of this trap and species for the Train set,
        # so we replace it with the avgerage number of mosqitoes found for this
        # species instead
        temp = spec_avg[spec_avg['Species'] == species]
        # Check to see if there was a match
        if len(temp) == 1:
            num = temp.NumMosquitos.values[0]
            test.set_value(index,'NumMosq',num)
        else:
            # Else the species is 'UNSPECIFIED CULEX'
            # Only leave one of these options uncommented
            test.set_value(index,'NumMosq',avg_mosq)
            #test.set_value(index,'NumMosq',0)

# Save to csv depending on what we replaced the 'UNSPECIFIED CULEX' with
test.to_csv('test_imputed_avg.csv')
test.to_csv('test_imputed_zero.csv')
