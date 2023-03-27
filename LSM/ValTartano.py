from processing import *
from config import base_dir
import os
from datetime import datetime
from models import *

def ValTartano():
    vt_dir = base_dir + r"Val Tartano/"
    factor_dir = vt_dir+"1.factors"
    trainset_path = vt_dir+"/2.samples/ValTartano_Training_Points.csv"
    testset_path = vt_dir+"/2.samples/ValTartano_Testing_Points.csv"
    result_path = vt_dir+"3.results/"
    
    def ensemble_SA_wrapper(X, Y, Xtest, Ytest, save_to=None):
        model_paras = DEFAULT_ENSEMBLE_SA_MODEL_PARAS
        model_paras['estimators'] = [
            ("RandomForest", vt_dir+"3.results/Val Tartano_Fortests of randomized trees.pkl"),
            ("AdaBoostCalibrated", vt_dir+"3.results/Val Tartano_AdaBoost Calibrated.pkl"),
            ("NeuralNetworksAndLR", vt_dir+"3.results/NNlogistic/Val Tartano_Neural Network.pkl"),
        ]
        return ensemble_simple_average(X, Y, Xtest, Ytest, model_paras=model_paras, save_to=save_to)
        
    algorithms = {
        CALIBRATED_ADABOOST_MODEL_LABLE: ensemble_calibrated_adaboost, 
        # ADABOOST_MODEL_LABLE: ensemble_adaboost,
        #NEURAL_NETWORK_MODEL_LABEL: NN_wrapper,
        #ENSEMBLE_STACK_MODEL_LABEL: ensemble_stack_wrapper_with_cvset,#ensemble_stack,
        #ENSEMBLE_BLEND_MODEL_LABEL: ensemble_blend,
        #ENSEMBLE_SOFT_VOTING_MODEL_LABEL: ensemble_soft_voting,
        #ENSEMBLE_SA_MODEL_LABEL: ensemble_SA_wrapper,
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

def evaluation():
    testset_path = vt_dir+"/2.samples/ValTartano_Testing_Points.csv"
    save_info = [('basic', vt_dir+"3.results/"), ("ensemble", vt_dir+"3.results/ensemble/")]
    plot_evaluation_with_testset(testset_path, vt_clfs, save_info)

if __name__ == '__main__':
    ValTartano()
    #ValTartano_SVM_NN()
    #check_factors()
    #evaluation()