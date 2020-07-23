import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Pull in CSVs
all_hexagons = pd.read_csv("all_hexagons_50c_grouped_withCounties.csv")
records = pd.read_csv("june29_1589_records.csv")

all_hexagons.set_index('subwatershed', inplace=True)

# Join data frames to make master file. Join key = 'subwatershed'
joined = records.join(all_hexagons, on='subwatershed')

# Drop columns not needed.
# Comment out a line to leave in for processing.
drop_list = [
    'Unnamed: 0',       # Always remove
    # 'area',
    'perimeter',
    'circulatory_ratio',
    # 'relief',
    # 'avg_slope',
    'watershed_length',
    'elongation_ratio',
    # 'drainage_density',
    'shape_factor',
    # 'relief_ratio',
    # 'ruggedness',
    # 'aae_area',
    # 'buildings_aae_count',
    'x_area',
    'buildings_x_count',
    # 'water_bodies_area',
    'dams_count',
    'bridges_count',
    'streets_km',
    'railroads_km',
    # 'population',
    # 'population_density',
    # 'avg_median_income',
    # 'housing_density',
    'population_change',
    'dependent_population_pct',
    'dist_to_stream_avg..m.',
    'dist_to_stream_stdev..m.',
    # 'lu_22_area',
    # 'lu_23_area',
    # 'lu_24_area',
    # 'lu_41_area',
    # 'lu_82_area',
    # 'avg_impervious_percent',
    # 'orb100yr06h',
    # 'orb100yr12h',
    # 'orb100yr24h',
    # 'orb25yr06h',
    # 'orb25yr12h',
    # 'orb25yr24h',
    # 'orb2yr06h',
    # 'orb2yr12h',
    # 'orb2yr24h',
    # 'orb50yr06h',
    # 'orb50yr12h',
    # 'orb50yr24h',
    # 'orb100yr06ha_am',
    # 'orb100yr12ha_am',
    # 'orb100yr24ha_am',
    # 'orb25yr06ha_am',
    # 'orb25yr12ha_am',
    # 'orb25yr24ha_am',
    # 'orb2yr06ha_am',
    # 'orb2yr12ha_am',
    # 'orb2yr24ha_am',
    # 'orb50yr06ha_am',
    # 'orb50yr12ha_am',
    # 'orb50yr24ha_am',
    'File'          # Always remove
]

# Drop columns in 'drop_list'
joined.drop(
    drop_list,
    axis=1,
    inplace=True
)

# Drop all rows that have no claims info
joined['policy_total_building_coverage_avg'].fillna('none', inplace=True)
joined = joined[joined['policy_total_building_coverage_avg'] != 'none']

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
# for i, item in enumerate(column_list):
#     print(i, item)
joined[column_list] = scaler.fit_transform(joined[column_list])
print("Note: policy_..._coverage_avg values were scaled by multiplying by {:.12f} and adding {:.10f}".format(
    scaler.scale_[43],
    scaler.min_[43])
)
print("Note: claims_..._coverage_avg values were scaled by multiplying by {:.12f} and adding {:.10f}".format(
    scaler.scale_[44],
    scaler.min_[44])
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
training_df = usable_data.sample(frac=.7, random_state=663168)
testing_df = usable_data.loc[~usable_data.index.isin(training_df.index)]
validation_df = testing_df.sample(frac=.5, random_state=663168)
testing_df = testing_df.loc[~testing_df.index.isin(validation_df.index)]

# Save subsets
training_df.to_csv('training_data.csv')
testing_df.to_csv('testing_data.csv')
validation_df.to_csv('validation_data.csv')

# Print number of input nodes for network
# Number of columns minus 3 (index, subwatershed, Unnamed: 0)
column_count = int(len(list(training_df.columns))) - 3
print('Number of input nodes: ' + str(column_count))