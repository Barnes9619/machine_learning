import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Pull in CSVs
all_hexagons = pd.read_csv("all_hexagons_50c_grouped_withCounties.csv")
records = pd.read_csv("MOD_DF_0723.csv")

all_hexagons.set_index('subwatershed', inplace=True)

# Join data frames to make master file. Join key = 'subwatershed'
joined = records.join(all_hexagons, on='subwatershed')

# # Drop columns in 'drop_list'
# # drop_list = []
# joined.drop(
#     drop_list,
#     axis=1,
#     inplace=True
# )

# Drop 'bad data'
joined = joined[joined['subwatershed'] != 41900000200]
# Drop rows with blanks
joined = joined.dropna()

# Save the joined data to csv
joined.to_csv('joined_data.csv')

# Normalize the rows to 0-1, excluding the watershed identifier
scaler = MinMaxScaler(feature_range=(0, 1))
column_list = list(joined.columns)
column_list.remove('subwatershed')

joined[column_list] = scaler.fit_transform(joined[column_list])

scaler_by = scaler.scale_[44]
scaler_min = scaler.min_[44]

print("Note: claims_..._coverage_avg values were scaled by multiplying by {:.12f} and adding {:.10f}".format(
    scaler_by,
    scaler_min)
)

i_rows = len(joined)
print('Initial row count for data: ' + str(i_rows))
usable_data = joined[(np.abs(stats.zscore(joined)) < 3).all(axis=1)]
u_rows = len(usable_data)
print("- Identified and removed {} outliers.".format(str(i_rows-u_rows)))
print('Updated row count for all data: ' + str(u_rows))

usable_data.to_csv('final_data_scaled.csv')

# Use 70% of the data for training, 15% for testing, 15% for validation.
# Subset data for training, testing, validation
training_df = joined.sample(frac=.7, random_state=663168)
testing_df = joined.loc[~joined.index.isin(training_df.index)]
validation_df = testing_df.sample(frac=.5, random_state=663168)
testing_df = testing_df.loc[~testing_df.index.isin(validation_df.index)]

# Save subsets
training_df.to_csv('training_data.csv')
testing_df.to_csv('testing_data.csv')
validation_df.to_csv('validation_data.csv')

# Print number of input nodes for network
# Number of columns minus 2 (subwatershed, Unnamed: 0)
node_count = int(len(list(training_df.columns)) - 2)
print('Number of input nodes: ' + str(node_count))
