import time

import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import (RBFSampler)
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

seed = 7
np.random.seed(seed)
start_time = time.time()
print('Start running..................')

datadir= './data/'
predictions = "./predictions/"
logs = './logs/'

df_train = pd.read_csv(datadir + "numerai_training_data.csv")
df_tournament = pd.read_csv(datadir + "numerai_tournament_data.csv")
df_valid = df_tournament[df_tournament['data_type'].isin(['validation'])]
feature_cols = [f for f in df_train.columns if "feature" in f]
target_col = df_train.columns[-1]

x_tr,x_te,y_tr,y_te = train_test_split(df_train, df_train.target,test_size =0.5)
clf = pipeline.Pipeline([("feature_map", RBFSampler(n_components=300,gamma=0.4,random_state=31)),
('Calibratedsvc', CalibratedClassifierCV(LinearSVC(C=1e-2, random_state=31), method='sigmoid'))])
clf.fit(df_train[feature_cols].values,df_train[target_col])
prob_pos = clf.predict_proba(df_valid[feature_cols].values)[:, 1]
df_valid['pf'] = prob_pos
eras = df_valid.era.unique()
good_eras = 0
for era in eras:

    tmp = df_valid[ df_valid.era == era ]
    ll = log_loss( tmp.target, tmp.pf)
    is_good = ll < 0.693

    if is_good:
        good_eras += 1

    print( "{} {} {:.2%} {}".format( era, len( tmp ), ll, is_good ))

consistency = good_eras / float( len( eras ))
print( "\nconsistency: {:.1%} ({}/{})".format( consistency, good_eras, len( eras )))


if consistency > 0.58:
    if hasattr(clf, "predict_proba"):
        p_tournament = clf.predict_proba(df_tournament[feature_cols])[:, 1]
    else:  # use decision function
        p_tournament = clf.decision_function(d)
        p_tournament = (p_tournament - p_tournament.min()) / (p_tournament.max() - p_tournament.min())

    df_pred_tournament = pd.DataFrame({
        'id': df_tournament['id'],
        'probability': p_tournament
        })
    ll=log_loss(df_valid.target, prob_pos)
    csv_path = predictions +  './non_linear_predictions_{}_{}.csv'.format(int(time.time()), str(ll)[2:])
    df_pred_tournament.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))
    txt_path = logs + "stats_non_linear.txt"
    with open(txt_path, "a") as g:
        g.write('\n' + 'non_linear')
        g.write('\n' + 'logloss:' + str(ll))
        g.write('\n' + 'consistency:' + str(consistency))
        g.close()
    print('Saved stats: {}'.format(txt_path))

print('Fished: {}s'.format(time.time() - start_time))