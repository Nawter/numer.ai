from classifiers import lr, rf, gb, neural_net, xrf, gbt, svc, ada
from mlxtend.classifier import StackingCVClassifier

def read_input(input):
    approach = None
    if len(input.argv) > 1:
        approach = str(input.argv[1])
    else:
        print("why is empty sweetheart!!!!")
        exit()
    return approach

def choose_approach(approach,probability):
    global sc, label
    if approach == "second":
        sc = StackingCVClassifier(classifiers=[lr, rf], use_probas=probability,
                                  meta_classifier=gb,verbose=2)
        label = "StackingCVClassifier-xgb"
    elif approach == "third":
        sc = StackingCVClassifier(classifiers=[gb, neural_net, rf, xrf, gbt], use_probas=probability,
                                  meta_classifier=gbt,verbose=2)
        label = "StackingCVClassifier-gbt"
    else:
        sc = StackingCVClassifier(classifiers=[svc, rf, xrf, ada, gbt], use_probas=probability,
                                  meta_classifier=rf,verbose=2)
        label = "StackingCVClassifier-rf"
    return sc,label

