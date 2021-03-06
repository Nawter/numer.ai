import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score

from utils import read_input, choose_approach

data_dir = './data/'
predictions = "./predictions/"
logs = './logs/'

approach = read_input(sys)

seed = 7
np.random.seed(seed)
start_time = time.time()
print('Start running..................')

df_train = pd.read_csv(data_dir + "numerai_training_data.csv", header=0)
df_tour = pd.read_csv(data_dir + "numerai_tournament_data.csv", header=0)

feature_cols = [f for f in df_train.columns if "feature" in f]
target_col = df_train.columns[-1]

X = df_train[feature_cols].values
y = df_train[target_col].values

validation_ind = np.where(df_tour.data_type == 'validation')[0]
df_valid = df_tour.iloc[validation_ind]

X_valid = df_valid[feature_cols].values
y_valid = df_valid.target.values

X_tour = df_tour[feature_cols].values

sc, label = choose_approach(approach, False)

scores = cross_val_score(sc, X, y, cv=10, scoring='neg_log_loss', n_jobs=1, verbose=3)
print("Log loss: %0.5f (+/- %0.2f) [%s]"
      % (scores.mean(), scores.std(), label))
sc.fit(X, y)
df_valid = df_tour[df_tour['data_type'].isin(['validation'])]
prob_pos = sc.predict_proba(df_valid[feature_cols].values)[:, 1]
df_valid['pf'] = prob_pos
eras = df_valid.era.unique()
good_eras = 0
for era in eras:
    tmp = df_valid[df_valid.era == era]
    ll = log_loss(tmp.target, tmp.pf)
    is_good = ll < 0.693
    if is_good:
        good_eras += 1
    print("{} {} {:.2%} {}".format(era, len(tmp), ll, is_good))

consistency = good_eras / float(len(eras))
print("\nconsistency: {:.1%} ({}/{})".format(consistency, good_eras, len(eras)))

if consistency > 0.58:
    if hasattr(sc, "predict_proba"):
        p_tournament = sc.predict_proba(X_tour)[:, 1]
    else:  # use decision function
        p_tournament = sc.decision_function(df_tour[feature_cols])
        p_tournament = (p_tournament - p_tournament.min()) / (p_tournament.max() - p_tournament.min())

    df_pred_tournament = pd.DataFrame({
        'id': df_tour['id'],
        'probability': p_tournament
    })
    ll = log_loss(df_valid.target, prob_pos)
    csv_path = predictions + './stacking_real_' + approach +'_predictions_{}_{}.csv'.format(int(time.time()), str(ll)[2:])
    df_pred_tournament.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))
    txt_path = logs + approach + "_real_stats.txt"
    with open(txt_path, "a") as g:
        g.write('\n' + 'logloss:' + str(ll))
        g.write('\n' + 'consistency:' + str(consistency))
        g.close()
    print('Saved stats: {}'.format(txt_path))

print('Fished: {}s'.format(time.time() - start_time))
