#################### MODELLING (with validation) ####################
import numpy as np

from evaluation import evaluation_report

from sklearn.ensemble import BaggingClassifier
BAGGING_MODEL_LABLE = "Bagging"
'''
estimator: The base estimator to fit on random subsets of the dataset.
                If None, then the base estimator is a DecisionTreeClassifier.
n_estimators: The number of base estimators in the ensemble.
n_jobs: The number of jobs to run in parallel for both fit and predict.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors. 
'''
DEFAULT_RANDOMFOREST_MODEL_PARAS = {"n_estimators": 100, "n_jobs": -1}
def ensemble_bagging(X, Y, Xtest, Ytest, model_paras=DEFAULT_RANDOMFOREST_MODEL_PARAS, save_to=None):
    # train
    clf = BaggingClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model: Ensemble / Bagging", save_to)
    
    return clf

from sklearn.ensemble import RandomForestClassifier
RANDOMFOREST_MODEL_LABLE = "Forests of randomized trees"
'''
n_estimators: The number of trees in the forest. (指定森林中树的颗数，越多越好，只是不要超过内存)
                default: 100
max_depth: The maximum depth of the tree. 
            If None, then nodes are expanded until all leaves are pure or 
            until all leaves contain less than min_samples_split samples.
n_jobs: The number of jobs to run in parallel. (指定并行使用的进程数)
        -1 means using all processors.
'''
DEFAULT_RANDOMFOREST_MODEL_PARAS = {'n_estimators': 100, 'n_jobs': -1,}
def ensemble_randomforest(X, Y, Xtest, Ytest, model_paras=DEFAULT_RANDOMFOREST_MODEL_PARAS, save_to=None):
    # train
    print(f"[BEGIN] Training using parameters {model_paras}")
    clf = RandomForestClassifier(**model_paras).fit(X, Y)
    print(f"[END] Training using parameters {model_paras}")
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Random Forest", save_to)
    
    return clf

from sklearn.ensemble import AdaBoostClassifier
ADABOOST_MODEL_LABLE = "AdaBoost"
'''
estimator: object. The base estimator from which the boosted ensemble is built.
                    Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes.
                    If None, then the base estimator is DecisionTreeClassifier initialized with max_depth=1.

                    New in version 1.2: base_estimator was renamed to estimator.
n_estimators: int. The maximum number of estimators at which boosting is terminated.
                    In case of perfect fit, the learning procedure is stopped early. Values must be in the range [1, inf).

learning_rate: float, default=1.0
                Weight applied to each classifier at each boosting iteration.
                A higher learning rate increases the contribution of each classifier.
                There is a trade-off between the learning_rate and n_estimators parameters.
                Values must be in the range (0.0, inf).
'''
DEFAULT_ADABOOST_MODEL_PARAS = {'n_estimators': 300, 'learning_rate': 0.8}
def ensemble_adaboost(X, Y, Xtest, Ytest,
                      model_paras=DEFAULT_ADABOOST_MODEL_PARAS, save_to=None):
    # train
    clf = AdaBoostClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Adaboost", save_to)
    
    return clf

from sklearn.calibration import CalibratedClassifierCV
CALIBRATED_ADABOOST_MODEL_LABLE = "AdaBoost Calibrated"
DEFAULT_CALIBRATED_ADABOOST_MODEL_PARAS = {'n_estimators': 300, 'learning_rate': 0.8}
def ensemble_calibrated_adaboost(X, Y, Xtest, Ytest,
                      model_paras=DEFAULT_CALIBRATED_ADABOOST_MODEL_PARAS, save_to=None):
    # train
    clf = CalibratedClassifierCV(AdaBoostClassifier(**model_paras)).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Adaboost Calibrated", save_to)
    
    return clf

from sklearn.ensemble import GradientBoostingClassifier
GRADIENT_TREE_BOOSTING_MODEL_LABLE = "Gradient Tree Boosting"
DEFAULT_GRADIENTBOOSTING_MODEL_PARAS = {'n_estimators': 300, 'learning_rate': 0.8}
def ensemble_gradienttreeboosting(X, Y, Xtest, Ytest,
                                  model_paras=DEFAULT_GRADIENTBOOSTING_MODEL_PARAS, save_to=None):
    # train
    clf = GradientBoostingClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Gradient Tree Boosting", save_to)
    return clf

from sklearn.linear_model import SGDClassifier
SVM_MODEL_LABLE = "Support Vector Machine"
# note that here modify the default loss to "modified_huber" because "probability estimates are not available for loss='hinge'" 
# and ‘modified_huber’ is another smooth loss that brings tolerance to outliers as well as probability estimates.
DEFAULT_SVM_MODEL_PARAS = {"max_iter": 1000, "n_jobs": -1, "loss": "modified_huber"}
def svm_svc(X, Y, Xtest, Ytest,
            model_paras=DEFAULT_SVM_MODEL_PARAS, save_to=None):
    # train
    clf = SGDClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Support Vector Machine", save_to)
    return clf

from sklearn.gaussian_process import GaussianProcessClassifier
GAUSSIAN_PROCESS_MODEL_LABEL = "Gaussian Process"
DEFAULT_GAUSSIAN_PROCESS_MODEL_PARAS = {"max_iter_predict": 100}
def gaussian_process(X, Y, Xtest, Ytest,
            model_paras=DEFAULT_GAUSSIAN_PROCESS_MODEL_PARAS, save_to=None):
    # train
    clf = GaussianProcessClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Gaussian Process", save_to)
    return clf

from sklearn.neural_network import MLPClassifier
NEURAL_NETWORK_MODEL_LABEL = "Neural Network"
DEFAULT_NEURAL_NETWORK_MODEL_PARAS = {
    "hidden_layer_sizes": (100,),
    "activation": 'relu',
    "solver": 'adam',
    "alpha": 0.0001,
    "learning_rate": 'constant',
    "max_iter": 200,
}
def neural_network(X, Y, Xtest, Ytest,
            model_paras=DEFAULT_NEURAL_NETWORK_MODEL_PARAS, save_to=None):
    print(f"{NEURAL_NETWORK_MODEL_LABEL} Parameters: {model_paras}")
    # train
    clf = MLPClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Neural Network", save_to)
    return clf

DEFAULT_NEURAL_NETWORK_MODEL_WITH_LR_PARAS = DEFAULT_NEURAL_NETWORK_MODEL_PARAS
DEFAULT_NEURAL_NETWORK_MODEL_WITH_LR_PARAS["activation"] = 'logistic'
DEFAULT_NEURAL_NETWORK_MODEL_WITH_LR_PARAS['alpha'] = 0.01
# TOFIX:  ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
DEFAULT_NEURAL_NETWORK_MODEL_WITH_LR_PARAS["max_iter"] = 500
def NN_wrapper(X, Y, Xtest, Ytest, save_to=None):
        model_paras = DEFAULT_NEURAL_NETWORK_MODEL_PARAS
        #model_paras['hidden_layer_sizes'] = (100,50)
        model_paras["activation"] = 'logistic'
        model_paras['alpha'] = 0.01
        #model_paras["max_iter"] = 500
        return neural_network(X, Y, Xtest, Ytest, model_paras=model_paras, save_to=save_to)

# TOTEST: stacking model
from sklearn.ensemble import StackingClassifier as Stacker
from sklearn.linear_model import LogisticRegression
ENSEMBLE_STACK_MODEL_LABEL = "Ensemble Stacking"
# final_estimatorestimator: default=None. Aclassifier which will be used to combine the base estimators. The default classifier is a LogisticRegression.
DEFAULT_ENSEMBLE_STACK_MODEL_PARAS = {
    "estimators": [
        ("RandomForest", RandomForestClassifier(**DEFAULT_RANDOMFOREST_MODEL_PARAS)),
        ("AdaBoostCalibrated", CalibratedClassifierCV(AdaBoostClassifier(**DEFAULT_CALIBRATED_ADABOOST_MODEL_PARAS))),
        ("NeuralNetworksAndLR", MLPClassifier(**DEFAULT_NEURAL_NETWORK_MODEL_WITH_LR_PARAS)),
    ],
    # TOFIX: LineSearchWarning: The line search algorithm did not converge
    # ref: https://stats.stackexchange.com/questions/184017/how-to-fix-non-convergence-in-logisticregressioncv
    "final_estimator": LogisticRegression(max_iter=1000, solver='sag'),
    "n_jobs": -1,
    "stack_method": "predict_proba",
}
def ensemble_stack(X, Y, Xtest, Ytest,
            model_paras=DEFAULT_ENSEMBLE_STACK_MODEL_PARAS, save_to=None):
    print("Stacker parameters: ", model_paras)
    # train
    clf = Stacker(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Stacking", save_to)
    return clf

def ensemble_stack_wrapper_with_cvset(X, Y, Xtest, Ytest, save_to=None):
    model_paras = DEFAULT_ENSEMBLE_STACK_MODEL_PARAS
    model_paras['cv'] = 20
    return ensemble_stack(X, Y, Xtest, Ytest, model_paras=model_paras, save_to=save_to)

# blending model
# ref: https://www.geeksforgeeks.org/ensemble-methods-in-python/
# ref: https://towardsdatascience.com/ensemble-learning-stacking-blending-voting-b37737c4f483
# ref: https://machinelearningmastery.com/blending-ensemble-machine-learning-with-python/
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
ENSEMBLE_BLEND_MODEL_LABEL = "Ensemble Blending"
DEFAULT_ENSEMBLE_BLEND_MODEL_PARAS = {
    "estimators": [
        ("RandomForest", RandomForestClassifier(**DEFAULT_RANDOMFOREST_MODEL_PARAS)),
        ("AdaBoostCalibrated", CalibratedClassifierCV(AdaBoostClassifier(**DEFAULT_CALIBRATED_ADABOOST_MODEL_PARAS))),
        ("NeuralNetworksAndLR", MLPClassifier(**DEFAULT_NEURAL_NETWORK_MODEL_WITH_LR_PARAS)),
    ],
    "final_estimator": LogisticRegression(max_iter=1000, solver='sag'),
    "validate_percentage": 0.2,
    "blend_method": "predict_proba",
}

class BlendingClassifier:
    def __init__(self, estimators, final_estimator, blend_method="predict_proba", validate_percentage=0.2):
        self.estimators_ = estimators
        self.blender = final_estimator
        self.validate_percentage = validate_percentage
        self.feature_names_in_ = None
        self.blend_method = blend_method


    # Fit the estimators.
    def fit(self, X, y):
        Xtrain, Xval, Ytrain, Yval = train_test_split(X, y, test_size=self.validate_percentage)
        ## fit all models on the training set and predict on hold out set
        meta_X = []
        for name, model in self.estimators_:
            # fit in training set
            model = model.fit(Xtrain, Ytrain)
            if self.feature_names_in_ is None:
                self.feature_names_in_ = model.feature_names_in_
            # predict on hold out set
            yhat = None
            if self.blend_method == 'predict_proba':
                yhat = model.predict_proba(Xval)
            elif self.blend_method == 'predict':
                yhat = model.predict(Xval)
                # reshape predictions into a matrix with one column
                print(f"Blend base estimator: yhat.shape={yhat.shape}, yhat.len={len(yhat)}")
                yhat = yhat.reshape(-1, 1)
                print(f"Blend base estimator after reshape: yhat.shape={yhat.shape}, yhat.len={len(yhat)}")
            # store predictions as input for blending
            meta_X.append(yhat)

        ## create 2d array from predictions, each set is an input feature
        meta_X = np.hstack(meta_X)
        ## fit on predictions from base models
        self.blender = self.blender.fit(meta_X, Yval)
        return self

    def predict_proba(self, X):
        # make predictions with base models
        meta_X = list()
        for name, model in self.estimators_:
            # predict with base model
            yhat = None
            if self.blend_method == 'predict_proba':
                yhat = model.predict_proba(X)
            elif self.blend_method == 'predict':
                yhat = model.predict(X)
                # reshape predictions into a matrix with one column
                print(f"Blend base estimator: yhat.shape={yhat.shape}, yhat.len={len(yhat)}")
                yhat = yhat.reshape(-1, 1)
                print(f"Blend base estimator after reshape: yhat.shape={yhat.shape}, yhat.len={len(yhat)}")
            # store prediction
            meta_X.append(yhat)
        # create 2d array from predictions, each set is an input feature
        meta_X = np.hstack(meta_X)
        # predict
        return self.blender.predict_proba(meta_X)

def ensemble_blend(X, Y, Xtest, Ytest,
            model_paras=DEFAULT_ENSEMBLE_BLEND_MODEL_PARAS, save_to=None):
    print(f"{ENSEMBLE_BLEND_MODEL_LABEL} parameters: {model_paras}")
    # fit ensemble
    clf = BlendingClassifier(**model_paras).fit(X, Y)

    # test
    Ytest_pred = clf.predict_proba(Xtest)

    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Blending", save_to)

    return clf

# simple average model
ENSEMBLE_SA_MODEL_LABEL = "Ensemble Simple Averaging"
DEFAULT_ENSEMBLE_SA_MODEL_PARAS = {
    "estimators": [
        ("RandomForest", RandomForestClassifier(**DEFAULT_RANDOMFOREST_MODEL_PARAS)),
        ("AdaBoostCalibrated", CalibratedClassifierCV(AdaBoostClassifier(**DEFAULT_CALIBRATED_ADABOOST_MODEL_PARAS))),
        ("NeuralNetworksAndLR", MLPClassifier(**DEFAULT_NEURAL_NETWORK_MODEL_WITH_LR_PARAS)),
    ],
}

class SAClassifier:
    def __init__(self, estimators):
        self.estimators_ = {}
        for name, model in estimators:
            self.estimators_[name] = model
        self.feature_names_in_ = None

    # Fit the estimators.
    def fit(self, X, y):
        ## fit all models on the training set
        for name in self.estimators_:
            if isinstance(self.estimators_[name], str):
                self.estimators_[name] = load_model(self.estimators_[name])
            else:
                # fit in training set
                self.estimators_[name] = self.estimators_[name].fit(X, y)
            if self.feature_names_in_ is None:
                self.feature_names_in_ = self.estimators_[name].feature_names_in_
        return self

    def predict_proba(self, X):
        # make predictions with base models
        meta_Y = None
        for name, model in self.estimators_.items():
            # predict with base model
            yhat = model.predict_proba(X)
            # store prediction
            meta_Y = meta_Y+yhat if meta_Y is not None else yhat

        # predict
        print(f"shape of meta_Y: {meta_Y.shape}")
        return meta_Y / len(self.estimators_)

def ensemble_simple_average(X, Y, Xtest, Ytest,
            model_paras=DEFAULT_ENSEMBLE_SA_MODEL_PARAS, save_to=None):
    print(f"{ENSEMBLE_SA_MODEL_LABEL} parameters: {model_paras}")
    # fit ensemble
    clf = SAClassifier(**model_paras).fit(X, Y)

    # test
    Ytest_pred = clf.predict_proba(Xtest)

    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Simple Averaging", save_to)

    return clf

# Weighted Average Probabilities (Soft Voting)
# ref: https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
# ref: https://machinelearningmastery.com/weighted-average-ensemble-with-python/
from sklearn.ensemble import VotingClassifier
ENSEMBLE_SOFT_VOTING_MODEL_LABEL = "Ensemble Soft Voting"
DEFAULT_ENSEMBLE_SOFT_VOTING_MODEL_PARAS = {
    "estimators": [
        ("RandomForest", RandomForestClassifier(**DEFAULT_RANDOMFOREST_MODEL_PARAS)),
        ("AdaBoostCalibrated", CalibratedClassifierCV(AdaBoostClassifier(**DEFAULT_CALIBRATED_ADABOOST_MODEL_PARAS))),
        ("NeuralNetworksAndLR", MLPClassifier(**DEFAULT_NEURAL_NETWORK_MODEL_WITH_LR_PARAS)),
    ],
    "voting": "soft",
}
def ensemble_soft_voting(X, Y, Xtest, Ytest,
            model_paras=DEFAULT_ENSEMBLE_SOFT_VOTING_MODEL_PARAS, save_to=None):
    print(f"{ENSEMBLE_SOFT_VOTING_MODEL_LABEL} parameters: {model_paras}")
    # train
    clf = VotingClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Weighted Average Probabilities (Soft Voting)", save_to)
    return clf


ALGORITHMS = {
    BAGGING_MODEL_LABLE: ensemble_bagging,
    RANDOMFOREST_MODEL_LABLE: ensemble_randomforest,
    ADABOOST_MODEL_LABLE: ensemble_adaboost,
    GRADIENT_TREE_BOOSTING_MODEL_LABLE: ensemble_gradienttreeboosting,
}