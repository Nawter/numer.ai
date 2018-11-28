# In this version of the algorithm we have to test three approaches changing the values of
# n_estimators, and using autoencoders.
# The first approach using these data  give us n_estimators = [20, 30, 50, 100] a logloss of 0.692662, and confidence level of 91.67%
# The second approach using these data give us n_estimators = [17, 31, 51, 101] a logloss of 0.692666 and confidence level of 91.7%

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

seed = 7
np.random.seed(seed)
start_time = time.time()
print('Start running..................')

datadir= './data/'
predictions = "./predictions/"
logs = './logs/'

train = pd.read_csv(datadir  + "numerai_training_data.csv",header=0)
tour = pd.read_csv(datadir  + "numerai_tournament_data.csv",header=0)
valid = tour.loc[tour.data_type=='validation']
feature_cols = [f for f in train.columns if "feature" in f]
# Configuration params for random forest algorithm, bags how many times the loop run on the data.
bags=10
# The number of estimators is the number of the trees in the random forest
model = RandomForestClassifier(n_jobs=-1,max_depth=4,n_estimators=16)
bagged_prediction = np.zeros(valid.shape[0])
bagged_prediction_tour = np.zeros(tour.shape[0])
n_estimators = [17, 31, 51, 101, 203] # 5 different estimators

# valid, train, test
for n in range(0,bags):
    print("number of bag:", n)
    for n_est in n_estimators:
    # range([start], stop[, step])
        print("number of tree:", n_est)
        model.set_params(n_estimators=n_est)
        model.fit(train[feature_cols].values,train.target.values)
        preds = model.predict_proba(valid[feature_cols])[:,1]
        preds_tour = model.predict_proba(tour[feature_cols])[:,1]
        bagged_prediction += preds
        bagged_prediction_tour += preds_tour

print("finishesd loop")
bagged_prediction/=bags*len(n_estimators)
bagged_prediction_tour/=bags*len(n_estimators) # this is to upload it to the competetion
print(bagged_prediction)
print("Logloss: {}".format(log_loss(valid.target,bagged_prediction)))

valid['pred'] = bagged_prediction

eras = valid.era.unique()
good_eras = 0
for era in eras:
    tmp = valid[ valid.era == era ]
    ll = log_loss( tmp.target, tmp.pred)
    is_good = ll < 0.693
    if is_good:
        good_eras += 1
    print("{} {} {:.2%} {}".format( era, len( tmp ), ll, is_good))

consistency = good_eras / float( len( eras ))
print("\nconsistency: {:.1%} ({}/{})".format( consistency, good_eras, len( eras )))

if consistency > 0.58:
    df_pred_tournament = pd.DataFrame({
        'id': tour['id'],
        'probability': bagged_prediction_tour
        })
    ll=log_loss(valid.target, bagged_prediction)
    csv_path = predictions +  './rf_v2_predictions_{}_{}.csv'.format(int(time.time()), str(ll)[2:])
    df_pred_tournament.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))
    txt_path = logs + "stats_rf_v2.txt"
    with open(txt_path, "a") as g:
        g.write('\n' + 'random forest v2')
        g.write('\n' + 'logloss:' + str(ll))
        g.write('\n' + 'consistency:' + str(consistency))
        g.close()
    print('Saved stats: {}'.format(txt_path))

print('Fished: {}s'.format(time.time() - start_time))


