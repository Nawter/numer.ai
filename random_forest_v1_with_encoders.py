import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

seed = 7
np.random.seed(seed)
start_time = time.time()
print('Start running..................')

datadir= './data/'
predictions = "./predictions/"
logs = './logs/'
autoencoder = np.load('./autoencoders/denoising.npz')

train = pd.read_csv(datadir  + "numerai_training_data.csv",header=0)
tour = pd.read_csv(datadir  + "numerai_tournament_data.csv",header=0)
valid = tour.loc[tour.data_type=='validation']
feature_cols = [f for f in train.columns if "feature" in f]
bags=10
seed=11
z_train=autoencoder['z_train']
z_valid=autoencoder['z_valid']
z_test=autoencoder['z_test']
z_live=autoencoder['z_live']
z_tour= np.concatenate((z_valid, z_test,z_live), axis=0)

model = RandomForestClassifier(n_jobs=-1,max_depth=3,min_samples_leaf=6,n_estimators=16)
bagged_prediction = np.zeros(valid.shape[0])
bagged_prediction_tour = np.zeros(tour.shape[0])
n_estimators = [20, 30, 50, 100] # 4 different estimators

for n in range(0,bags):
    for n_est in n_estimators:
    # range([start], stop[, step])
        model.set_params(n_estimators=n_est)
        model.fit(z_train,train.target.values)
        preds = model.predict_proba(z_valid)[:,1]
        preds_tour = model.predict_proba(z_tour)[:,1]
        bagged_prediction += preds
        bagged_prediction_tour += preds_tour

bagged_prediction/=bags*len(n_estimators)
bagged_prediction_tour/=bags*len(n_estimators)

from sklearn.metrics import log_loss
print("Logloss: {}".format(log_loss(valid.target,bagged_prediction)))

valid['pf']=bagged_prediction
eras = valid.era.unique()
good_eras = 0
for era in eras:
    tmp = valid[ valid.era == era ]
    ll = log_loss( tmp.target, tmp.pf)
    is_good = ll < 0.693

    if is_good:
        good_eras += 1

    print( "{} {} {:.2%} {}".format( era, len( tmp ), ll, is_good ))

consistency = good_eras / float( len( eras ))
print( "\nconsistency: {:.1%} ({}/{})".format( consistency, good_eras, len( eras )))
if consistency > 0.58:
    df_pred_tournament = pd.DataFrame({
        'id': tour['id'],
        'probability': bagged_prediction_tour
    })
    ll = log_loss(valid.target, bagged_prediction)
    csv_path = predictions + './rf_with_encoders_predictions_{}_{}.csv'.format(int(time.time()), str(ll)[2:])
    df_pred_tournament.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))
    txt_path = logs + "stats_rf_v1_enc.txt"
    with open(txt_path, "a") as g:
        g.write('\n' + 'random forest v1 with encoders')
        g.write('\n' + 'logloss:' + str(ll))
        g.write('\n' + 'consistency:' + str(consistency))
        g.close()
    print('Saved stats: {}'.format(txt_path))

print('Fished: {}s'.format(time.time() - start_time))