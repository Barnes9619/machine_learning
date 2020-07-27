import pandas as pd
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from matplotlib import pyplot
import os
import glob
import shutil

# Load the training data
training_df = pd.read_csv('training_data.csv')
training_df.drop(['Unnamed: 0', 'subwatershed'], axis=1, inplace=True)

# Load the separate test data set
testing_df = pd.read_csv('testing_data.csv')
testing_df.drop(['Unnamed: 0', 'subwatershed'], axis=1, inplace=True)

# Load the validation data
validation_df = pd.read_csv("validation_data.csv")
validation_df.drop(['Unnamed: 0', 'subwatershed'], axis=1, inplace=True)

# Predicting 'policy_total_building_coverage_avg' & 'claims_total_building_insurance_coverage_avg'
prediction_column = 'claims_total_building_insurance_coverage_avg'

# Format training data
x_train = training_df.drop(prediction_column, axis=1).values
y_train = training_df[prediction_column].values

# Format the testing data
x_test = testing_df.drop(prediction_column, axis=1).values
y_test = testing_df[prediction_column].values

# Format the validation data
x_validate = validation_df.drop(prediction_column, axis=1).values
y_validate = validation_df[prediction_column].values


# Define the model
def base_model(input_nodes=72):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(86, input_dim=input_nodes, activation='relu', name='Initial'))
    model.add(tf.keras.layers.Dropout(.2, name='Dropout_1'))
    model.add(tf.keras.layers.Dense(172, activation='relu', name='Dense_2'))
    model.add(tf.keras.layers.Dropout(.2, name='Dropout_2'))
    model.add(tf.keras.layers.Dense(86, activation='relu', name='Dense_3'))
    model.add(tf.keras.layers.Dropout(.2, name='Dropout_3'))
    model.add(tf.keras.layers.Dense(1, activation='linear', name='Output'))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


epochs = 30

# Log with TensorBoard
logger = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    write_graph=True,
    histogram_freq=3
)

my_model = KerasRegressor(
    build_fn=base_model,
    input_nodes=72,
    epochs=epochs
)


# Train the model
history = my_model.fit(
    x_train,
    y_train,
    validation_data=(x_validate, y_validate),
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[logger]
)

perm = PermutationImportance(my_model, random_state=1).fit(x_train, y_train)
eli5.show_weights(perm, feature_names=x_train.columns.tolist())

train_mse = my_model.evaluate(x_train, y_train, verbose=0)
test_mse = my_model.evaluate(x_test, y_test, verbose=0)
print('Mean squared error (MSE) values:')
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

run_dir = 'Runs/%.3f-val_loss_1' % test_mse

try:
    os.mkdir(run_dir)
except OSError:
    os.chdir('Runs/')
    runs = sorted([f for f in glob.glob('{0:.3f}-val_loss_*'.format(test_mse))])
    last_run = runs[-1]
    last_instance = int(last_run[-1])
    new_instance = last_instance + 1
    run_dir = run_dir[:-1] + str(new_instance)
    os.chdir('../')
    os.mkdir(run_dir)

# Move the logs to the appropriate dir
# Visualize through terminal:
# C:\Users\Ebarnes\Desktop\Projects\ML_Indiana_test> tensorboard --logdir=Runs\0.012-val_loss_1\logs
shutil.move('logs', run_dir)

# Plot loss during training
pyplot.title('Loss / Mean Squared Error, {} epochs'.format(str(epochs)))
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig((run_dir + '/plot.png'))
pyplot.show()

# Save training, validation and testing data
training_df.to_csv((run_dir + '/training_data.csv'))
validation_df.to_csv((run_dir + '/validation_data.csv'))
testing_df.to_csv((run_dir + '/testing_data.csv'))

column_list = list(training_df.columns)
column_list = ',\n'.join(column_list)

# Document this run
readme = open((run_dir + '/README.txt'), 'w+')
readme.write(('Mean squared error (MSE) values: Training: {0:.5f}, Testing: {1:.5f} \n'.format(train_mse, test_mse)))
readme.write(('Epochs: {} \n'.format(epochs)))
readme.write(('Layer 1: Dense: {} nodes, {} input nodes \n'.format(dense_nodes_1, input_nodes)))
readme.write(('Layer 2: Dropout: {}\n'.format(dropout_1)))
readme.write(('Layer 3: Dense: {} nodes\n'.format(dense_nodes_2)))
readme.write(('Layer 4: Dropout: {}\n'.format(dropout_2)))
readme.write(('Layer 5: Dense: {} nodes\n'.format(dense_nodes_3)))
readme.write(('Layer 6: Dropout: {}\n'.format(dropout_3)))
readme.write('Layer 7: Dense: 2 nodes (final)\n')
readme.write(('Included columns: \n{}'.format(column_list)))
readme.close()

# Save the model to disk
model.save((run_dir + '/trained_model.h5'))
print('Model saved to disk.')
