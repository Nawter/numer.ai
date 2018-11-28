# This method give you 0.6924224, adn 75% of confidence.
import fcntl
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

# random search
# Best: 0.509408 using {'subsample': 0.7, 'reg_alpha': 0.005, 'n_estimators': 100, 'min_child_weight': 5, 'max_depth': 3, 'learning_rate': 0.05, 'gamma': 0.4, 'colsample_bytree': 0.9}
# grid search
#
#opt = str(sys.argv[1])
seed = 7
np.random.seed(seed)
start_time = time.time()
print('Start running..................')

datadir= './data/'
predictions = "./predictions/"
logs = './logs/'

df_train = pd.read_csv(datadir + "numerai_training_data.csv", header=0)
df_tour = pd.read_csv(datadir + "numerai_tournament_data.csv", header=0)
validation_ind = np.where(df_tour.data_type == 'validation')[0]
df_valid = df_tour.iloc[validation_ind]

complete_training_data = pd.concat([df_train, df_valid])

feature_cols = [f for f in complete_training_data.columns if "feature" in f]
X_train = complete_training_data[feature_cols].values
y_train = complete_training_data.target.values

X_tour = df_tour[feature_cols].values
X_valid = df_valid[feature_cols].values
y_valid = df_valid.target.values

xgtrain = xgb.DMatrix(X_train, y_train)
model = xgb.XGBClassifier(missing = 9999999999,
                          subsample = 0.7,
                          reg_alpha = 0.005,
                          n_estimators = 100,
                          nthread = -1,
                          min_child_weight = 5,
                          max_depth = 3,
                          learning_rate = 0.05,
                          gamma = 0.4,
                          colsample_bytree = 0.9,
                          seed=1301)

xgb_param = model.get_xgb_params()

# do cross validation
print('Start cross validation')
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=5000, nfold=11, metrics=['logloss'],
                  early_stopping_rounds=50, stratified=True, seed=1301)
print('Best number of trees = {}'.format(cvresult.shape[0]))
model.set_params(n_estimators=cvresult.shape[0])

eval_set = [(X_train, y_train), (X_valid, y_valid)]
train_set = [(X_train, y_train)]
parameters = {
    'n_estimators': [100, 250, 500],
    'learning_rate': [0.05, 0.1, 0.3],
    'max_depth': [3,6, 9, 12],
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    'min_child_weight':range(1,6,2),
    'gamma':[i/10.0 for i in range(0,5)],
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
}
model.fit(X_train, y_train, early_stopping_rounds=40, eval_metric="logloss", eval_set=eval_set, verbose=True)
prob_pos = model.predict_proba(X_valid)
print(prob_pos)
df_valid['pf'] = prob_pos[:, 1]

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
    if hasattr(model, "predict_proba"):
        p_tournament = model.predict_proba(X_tour)[:, 1]
    else:  # use decision function
        p_tournament = model.decision_function(df_tour[feature_cols])
        p_tournament = (p_tournament - p_tournament.min()) / (p_tournament.max() - p_tournament.min())

    df_pred_tournament = pd.DataFrame({
        'id': df_tour['id'],
        'probability': p_tournament
    })
    ll = log_loss(df_valid.target, prob_pos)
    csv_path = predictions + './xgboost_' + '_predictions_{}_{}.csv'.format(int(time.time()), str(ll)[2:])
    df_pred_tournament.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))
    txt_path = logs + "stats_xgboost.txt"
    with open(txt_path, "a") as g:
        g.write('\n' + 'xgboost' )
        g.write('\n' + 'logloss:' + str(ll))
        g.write('\n' + 'consistency:' + str(consistency))
        g.close()
    print('Saved stats: {}'.format(txt_path))

print('Fished: {}s'.format(time.time() - start_time))
