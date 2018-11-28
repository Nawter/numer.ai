import sys
import time
import numpy as np
import pandas as pd
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

from classifiers import lr, rf, gaussNB, gb, neural_net, xrf, gbt, ada, svc
from utils import read_input

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

params_meta_lr = {'logisticregression__C': [0.1, 10.0],
                  'randomforestclassifier__n_estimators': [10, 30],
                  'meta-logisticregression__C': [0.1, 10.0]}

params_meta_xgb = {'logisticregression__C': [0.1, 10.0],
                   'randomforestclassifier__n_estimators': [10, 30],
                   'meta-xgbclassifier__max_depth': [3, 12],
                   'meta-xgbclassifier__subsample': [i / 10.0 for i in range(6, 8)],
                   'meta-xgbclassifier__min_child_weight': range(1, 3),
                   'meta-xgbclassifier__gamma': [i / 10.0 for i in range(0, 2)],
                   'meta-xgbclassifier__reg_alpha': [0, 0.05]}

params_meta_gbt = {'kerasclassifier__neurons': [7,14],
                   'kerasclassifier__dropout': [0.01,0.37],
                   'randomforestclassifier__n_estimators': [10, 30],
                   'extratreesclassifier__n_estimators': [100, 500],
                   'xgbclassifier__max_depth': [3, 12],
                   'meta-gradientboostingclassifier__learning_rate': [1.0,0.1],
                   'meta-gradientboostingclassifier__subsample': [0.5, 1.0],
                   'meta-gradientboostingclassifier__max_features': [0, 2],
                   }

params_meta_rf = {'randomforestclassifier__n_estimators': [10, 30],
                  'extratreesclassifier__n_estimators': [100, 500],
                  'adaboostclassifier__n_estimators': [51, 101],
                  'svc__C': [0.025, 0.075],
                  'meta-randomforestclassifier__n_estimators': [10, 30]}

if approach == "first":
    sc = StackingCVClassifier(classifiers=[lr, rf, gaussNB],
                              meta_classifier=lr)
    label = "StackingCVClassifier-lr"
    params = params_meta_lr
elif approach == "second":
    sc = StackingCVClassifier(classifiers=[lr, rf],
                              meta_classifier=gb)
    label = "StackingCVClassifier-xgb"
    params = params_meta_xgb
elif approach == "third":
    sc = StackingCVClassifier(classifiers=[gb, neural_net, rf, xrf, gbt],
                              meta_classifier=gbt, verbose=2)
    label = "StackingCVClassifier-gbt"
    params = params_meta_gbt
else:
    sc = StackingCVClassifier(classifiers=[svc, rf, xrf, ada, gbt],
                              meta_classifier=rf, verbose=2)
    label = "StackingCVClassifier-rf"
    params = params_meta_rf

# Check hyperparameters
# print(sc.get_params().keys())
random = RandomizedSearchCV(estimator=sc,
                            param_distributions=params,
                            cv=10,
                            refit=True,
                            verbose=3)
random.fit(X, y)
cv_keys = ('mean_test_score', 'std_test_score', 'params')
for r, _ in enumerate(random.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (random.cv_results_[cv_keys[0]][r],
             random.cv_results_[cv_keys[1]][r] / 2.0,
             random.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % random.best_params_)
print('Accuracy: %.2f' % random.best_score_)
scores = cross_val_score(sc, X, y, cv=10, scoring='neg_log_loss', n_jobs=1)
print("Log loss: %0.5f (+/- %0.2f) [%s]"
      % (scores.mean(), scores.std(), 'Stacking CV Classifier'))

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
    csv_path = predictions + './stacking_random_' + approach +'_predictions_{}_{}.csv'.format(int(time.time()), str(ll)[2:])
    df_pred_tournament.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))
    txt_path = logs + approach + "_random_stats.txt"
    with open(txt_path, "a") as g:
        g.write('\n' + 'logloss:' + str(ll))
        g.write('\n' + 'consistency:' + str(consistency))
        g.close()
    print('Saved stats: {}'.format(txt_path))

print('Fished: {}s'.format(time.time() - start_time))