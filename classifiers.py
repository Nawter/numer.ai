import xgboost as xgb
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation

def create_model(neurons=200, dropout=0.2):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(50,), kernel_initializer='glorot_uniform', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'])
    return model


lr = LogisticRegression(C=1e-2)
svc = SVC(kernel="linear", C=0.025)
gaussNB = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=1)
gb = xgb.XGBClassifier(missing = 9999999999,
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
neural_net = KerasClassifier(build_fn=create_model, epochs=8, batch_size=128, verbose=0)
xrf = ExtraTreesClassifier(n_estimators=10, max_features="log2",criterion="entropy")
rf = RandomForestClassifier(n_estimators=30, random_state=1)
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
gbt = GradientBoostingClassifier(**params)
ada = AdaBoostClassifier()