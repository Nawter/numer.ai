import time

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

seed = 7
np.random.seed(seed)
start_time = time.time()
print('Start running..................')

# ToDo understand the difference between df_train, df_tournament, df_valid, data_train, data_test
datadir= './data/'
predictions = "./predictions/"
logs = './logs/'

df_train = pd.read_csv(datadir + "numerai_training_data.csv")
df_tournament = pd.read_csv(datadir + "numerai_tournament_data.csv")
df_valid = df_tournament[df_tournament['data_type'].isin(['validation'])]
feature_cols = [f for f in df_train.columns if "feature" in f]
target_col = df_train.columns[-1]

data_train = df_train[feature_cols].values
data_test = df_valid[feature_cols].values
df_train.era = df_train.era.factorize()[0]

X_train, X_test, y_train, y_test = train_test_split(df_train, df_train.target, test_size=0.2)

# Probability Calibration with isotonic and sigmoid method
gkfcv = GroupKFold(n_splits=10)
skfcv = StratifiedKFold(n_splits=10, random_state=27, shuffle=True)
# LinearSVC using probability calibration with sigmoid method
Lsvm = svm.LinearSVC(C=1e-2)
sigmoid_calibrated_Lsvm_cv = CalibratedClassifierCV(svm.LinearSVC(C=1e-2), cv=10, method='sigmoid')
sigmoid_calibrated_Lsvm_gkfcv = CalibratedClassifierCV(svm.LinearSVC(C=1e-2),
                                                       cv=gkfcv.split(data_train, df_train.target, groups=df_train.era),
                                                       method='sigmoid')
sigmoid_calibrated_Lsvm_skfcv = CalibratedClassifierCV(svm.LinearSVC(C=1e-2),
                                                       cv=skfcv.split(data_train, df_train.target, groups=df_train.era),
                                                       method='sigmoid')
# LinearSVC using probability calibration with isotonic method
isotonic_calibrated_Lsvm_gkfcv = CalibratedClassifierCV(svm.LinearSVC(C=1e-2),
                                                        cv=gkfcv.split(data_train, df_train.target,
                                                                       groups=df_train.era), method='isotonic')
isotonic_calibrated_Lsvm_cv = CalibratedClassifierCV(svm.LinearSVC(C=1e-2), cv=10, method='isotonic')
isotonic_calibrated_Lsvm_skfcv = CalibratedClassifierCV(svm.LinearSVC(C=1e-2),
                                                        cv=skfcv.split(data_train, df_train.target,
                                                                       groups=df_train.era), method='isotonic')

classifiers = [Lsvm, sigmoid_calibrated_Lsvm_gkfcv, sigmoid_calibrated_Lsvm_cv, sigmoid_calibrated_Lsvm_skfcv,
               isotonic_calibrated_Lsvm_gkfcv, isotonic_calibrated_Lsvm_cv, isotonic_calibrated_Lsvm_skfcv]
classifiers_names = ["Lsvm", "sigmoid_calibrated_Lsvm_gkfcv", "sigmoid_calibrated_Lsvm_cv",
                     "sigmoid_calibrated_Lsvm_skfcv", "isotonic_calibrated_Lsvm_gkfcv", "isotonic_calibrated_Lsvm_cv",
                     "isotonic_calibrated_Lsvm_skfcv"]

# ToDo print automatically the best measure
i = 0
for clf in classifiers:
    clf.fit(data_train, df_train.target)
    y_pred = clf.predict(data_test)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(data_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(data_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    print("Classifier:", classifiers_names[i])
    print("\tLogLoss: %1.6f" % log_loss(df_valid.target, prob_pos))
    i = i + 1

clf = sigmoid_calibrated_Lsvm_cv.fit(data_train,df_train.target)
prob_pos = sigmoid_calibrated_Lsvm_cv.predict_proba(data_test)[:, 1]

print(log_loss(df_valid.target, prob_pos))

# Try the model on different errors, which is the consistency
df_valid['pred'] = prob_pos
eras = df_valid.era.unique()
good_eras = 0
for era in eras:

    tmp = df_valid[df_valid.era == era]
    ll = log_loss(tmp.target, tmp.pred)
    is_good = ll < 0.693    # -log( 0.5 )

    if is_good:
        good_eras += 1

    print("{} {} {:.2%} {}".format(era, len(tmp), ll, is_good))

consistency = good_eras / float(len(eras))
print("\nconsistency: {:.1%} ({}/{})".format(consistency, good_eras, len(eras)))

if consistency > 0.58:
    if hasattr(clf, "predict_proba"):
          p_tournament = clf.predict_proba(df_tournament[feature_cols])[:, 1]
    else:  # use decision function
        p_tournament = clf.decision_function(data_test)
        p_tournament = (p_tournament - p_tournament.min()) / (p_tournament.max() - p_tournament.min())

    df_pred_tournament = pd.DataFrame({
       'id': df_tournament['id'],
       'probability': p_tournament
       })
    ll=log_loss(df_valid.target, prob_pos)
    csv_path = predictions + './linear_predictions_{}_{}.csv'.format(int(time.time()), str(ll)[2:])
    df_pred_tournament.to_csv(csv_path, columns=('id', 'probability'), index=None)
    print('Saved: {}'.format(csv_path))
    txt_path = logs + "stats_linear.txt"
    with open(txt_path, "a") as g:
        g.write('\n' + 'linear')
        g.write('\n' + 'logloss:' + str(ll))
        g.write('\n' + 'consistency:' +  str(consistency))
        g.close()
    print('Saved stats: {}'.format(txt_path))

print('Fished: {}s'.format(time.time() - start_time))