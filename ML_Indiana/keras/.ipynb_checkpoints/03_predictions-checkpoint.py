import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model("Runs/0.012-val_loss_2/trained_model.h5")

validation_df = pd.read_csv("validation_data.csv")
validation_df.drop(['Unnamed: 0', 'subwatershed'], axis=1, inplace=True)

# Predicting 'policy_total_building_coverage_avg' & 'claims_total_building_insurance_coverage_avg'
prediction_columns = [
    'policy_total_building_coverage_avg',
    'claims_total_building_insurance_coverage_avg'
]

x_validate = validation_df.drop(prediction_columns, axis=1).values
y_validate = validation_df[prediction_columns].values
predictions = model.predict(x_validate)

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
# Note: policy_total_building_coverage_avg values were scaled by multiplying by 0.000007467342
# Note: claims_total_building_insurance_coverage_avg values were scaled by multiplying by 0.000006224341
for prediction in predictions:
    prediction[0] = prediction[0] / 0.000007467342
    prediction[1] = prediction[1] / 0.000006224341
print(predictions)

for value in y_validate:
    value[0] = value[0] / 0.000007467342
    value[1] = value[1] / 0.000006224341
print(y_validate)

print(type(predictions))
print(type(y_validate))

# policy_avg = 'policy_total_building_coverage_avg'
# claims_avg = 'claims_total_building_insurance_coverage_avg'
#
# check_results = validation_df[prediction_columns].copy()
# check_results[policy_avg] = check_results[policy_avg] / 0.000007467342
# check_results[claims_avg] = check_results[claims_avg] / 0.000006224341
#
#
#
# check_results.to_csv('predicted_values.csv')