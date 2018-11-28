import time

import numpy as np
import pandas as pd
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold

seed = 7
np.random.seed(seed)
start_time = time.time()
print('Start running..................')

datadir= './data/'
predictions = "./predictions/"
logs = './logs/'

training_data = pd.read_csv(datadir + 'numerai_training_data.csv', header=0)
tournament_data = pd.read_csv(datadir +'numerai_tournament_data.csv', header=0)

validation_data = tournament_data[tournament_data.data_type=='validation']
complete_training_data = pd.concat([training_data, validation_data])

features = [f for f in list(complete_training_data) if "feature" in f]
X = complete_training_data[features]
Y = complete_training_data["target"]

def create_model(neurons=200, dropout=0.2):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(50,), kernel_initializer='glorot_uniform', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=8, batch_size=128, verbose=0)

neurons = [7,10, 14]
dropout = [0.01, 0.26, 0.37]
param_grid = dict(neurons=neurons, dropout=dropout)

gkf = GroupKFold(n_splits=10)
kfold_split = gkf.split(X, Y, groups=complete_training_data.era)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold_split, scoring='neg_log_loss',n_jobs=1, verbose=3)
grid_result = grid.fit(X.values, Y.values)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

grid.best_estimator_.model.save('./my_model_2017-11-07_IV.h5')
grid.best_estimator_.model.summary()

def check_consistency(model, valid_data):
    eras = valid_data.era.unique()
    count = 0
    count_consistent = 0
    for era in eras:
        count += 1
        current_valid_data = valid_data[validation_data.era == era]
        features = [f for f in list(complete_training_data) if "feature" in f]
        X_valid = current_valid_data[features]
        Y_valid = current_valid_data["target"]
        loss = model.evaluate(X_valid.values, Y_valid.values, batch_size=128, verbose=0)[0]
        if (loss < -np.log(.5)):
            consistent = True
            count_consistent += 1
        else:
            consistent = False
        print("{}: loss - {} consistent: {}".format(era, loss, consistent))
    consistency = count_consistent / count
    print("Consistency: {}".format(consistency))
    return consistency

consistency = check_consistency(grid.best_estimator_.model, validation_data)

if consistency > 0.58:
    x_prediction = tournament_data[features]
    t_id = tournament_data["id"]
    y_prediction = grid.best_estimator_.model.predict_proba(x_prediction.values, batch_size=128)
    ll = grid_result.best_score_
    results = np.reshape(y_prediction,-1)
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(t_id).join(results_df)
    csv_path = predictions + './dl_predictions_{}_{}.csv'.format(int(time.time()), str(ll)[2:])
    joined.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))
    txt_path = logs + "stats_deep_learning.txt"
    with open(txt_path, "a") as g:
        g.write('\n' + 'deep learning')
        g.write('\n' + 'logloss:' + str(ll))
        g.write('\n' + 'consistency:' + str(consistency))
        g.close()
    print('Saved stats: {}'.format(txt_path))

print('Fished: {}s'.format(time.time() - start_time))
