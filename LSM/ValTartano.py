from processing import *
from config import base_dir
import os
from datetime import datetime

def ValTartano():
    vt_dir = base_dir + r"Val Tartano/"
    factor_dir = vt_dir+"1.factors"
    trainset_path = vt_dir+"/2.samples/ValTartano_Training_Points.csv"
    testset_path = vt_dir+"/2.samples/ValTartano_Testing_Points.csv"
    result_path = vt_dir+"3.results/ensemble/"

    def preprocess(train_xs, train_y, test_xs, test_y, target_xs):
        # 暂时处理：去掉dusaf_13
        target_xs = target_xs.drop(labels="dusaf_13", axis=1)
        return train_xs, train_y, test_xs, test_y, target_xs
        
    algorithms = {
        #CALIBRATED_ADABOOST_MODEL_LABLE: ensemble_calibrated_adaboost, 
        # ADABOOST_MODEL_LABLE: ensemble_adaboost,
        #NEURAL_NETWORK_MODEL_LABEL: NN_wrapper,
        ENSEMBLE_STACK_MODEL_LABEL: ensemble_stack,
        ENSEMBLE_BLEND_MODEL_LABEL: ensemble_blend,
        ENSEMBLE_SOFT_VOTING_MODEL_LABEL: ensemble_soft_voting,
    }

    #LSM('Val Tartano', factor_dir, trainset_path, testset_path, result_path, None)
    LSM('Val Tartano', factor_dir, trainset_path, testset_path, result_path, algorithms=algorithms)

def ValTartano_SVM_NN():
    vt_dir = base_dir + r"Val Tartano/"
    factor_dir = vt_dir+"1.factors"
    trainset_path = vt_dir+"/2.samples/ValTartano_Training_Points.csv"
    testset_path = vt_dir+"/2.samples/ValTartano_Testing_Points.csv"
    result_path = vt_dir+"3.results.svm.nn"
    preprocess = None

    LSM('Upper Valtellina', factor_dir, trainset_path, testset_path, result_path, preprocess, algorithms={SVM_MODEL_LABLE:svm_svc, NEURAL_NETWORK_MODEL_LABEL: neural_network})

def check_factors():
    vt_dir = base_dir + r"Val Tartano/"
    factor_dir = vt_dir+"1.factors"
    #load_rasters(factor_dir)
    get_factors_meta(factor_dir)

def plot_evaluation():
    vt_dir = base_dir + r"Val Tartano/"
    testset_path = vt_dir+"/2.samples/ValTartano_Testing_Points.csv"
    clfs = {
        BAGGING_MODEL_LABLE: {"path": vt_dir+"3.results/Val Tartano_Bagging.pkl", "color": 'tab:orange'},
        RANDOMFOREST_MODEL_LABLE: {"path": vt_dir+"3.results/Val Tartano_Fortests of randomized trees.pkl", "color": 'tab:green'},
        CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vt_dir+"3.results/Val Tartano_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
        GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vt_dir+"3.results/Val Tartano_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
        NEURAL_NETWORK_MODEL_LABEL: {"path": vt_dir+"3.results/NNLogistic/Val Tartano_Neural Network.pkl", "color": 'tab:pink'},
    }

    plot_LSM_evaluation(testset_path, clfs)

if __name__ == '__main__':
    #ValTartano()
    #ValTartano_SVM_NN()
    #check_factors()
    plot_evaluation()