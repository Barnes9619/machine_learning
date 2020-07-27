import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Upload training data:
train = h2o.upload_file(
    'training_data.csv',
    header=1,
    sep=',',
    skipped_columns=[0, 1]
)

# Upload testing data:
test = h2o.upload_file(
    'testing_data.csv',
    header=1,
    sep=',',
    skipped_columns=[0, 1]
)

x = train.columns
y = 'claims_total_building_insurance_coverage_avg'
x.remove(y)


aml = H2OAutoML(max_models=40, seed=1)
aml.train(x=x, y=y, training_frame=train)

lb = aml.leaderboard
print(lb.head(rows=lb.nrows))

perf = aml.leader.model_performance(test)
print(perf)
