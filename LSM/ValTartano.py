from processing import *
from config import base_dir
import os
from datetime import datetime

def ValTartano():
    vt_dir = base_dir + r"Val Tartano/"
    factor_dir = vt_dir+"1.factors"
    trainset_path = vt_dir+"/2.samples/ValTartano_Training_Points.csv"
    testset_path = vt_dir+"/2.samples/ValTartano_Testing_Points.csv"
    result_path = vt_dir+"3.results"

    if not os.path.exists(os.path.join(result_path, "Report.md")):
        with open(os.path.join(result_path, "Report.md"), "w") as f:
            f.write(datetime.now())

    def preprocess(train_xs, train_y, test_xs, test_y, target_xs):
        # 暂时处理：去掉dusaf_13
        target_xs = target_xs.drop(labels="dusaf_13", axis=1)
        return train_xs, train_y, test_xs, test_y, target_xs
        
    algorithms = {
        #CALIBRATED_ADABOOST_MODEL_LABLE: ensemble_calibrated_adaboost, 
        # ADABOOST_MODEL_LABLE: ensemble_adaboost
        
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

if __name__ == '__main__':
    ValTartano()
    #ValTartano_SVM_NN()
    #check_factors()