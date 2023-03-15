from processing import *
from config import base_dir
import numpy as np
import rasterio
import os
from matplotlib import pyplot as plt

from models import *

def Valchiavenna_v1_without():
    vc_dir = base_dir + r"ValChiavenna/"
    factor_dir = vc_dir+"1.factors"
    trainset_path = vc_dir+"/2.samples/1st_without/UpperValtellina_ValTartano_training_points.csv"
    testset_path = vc_dir+"/2.samples/1st_without/Valchiavenna_LSM_testing_points.csv"
    result_path = vc_dir+"3.results/1st_without/ensemble/tune"

    def ensemble_SA_wrapper(X, Y, Xtest, Ytest, save_to=None):
        model_paras = DEFAULT_ENSEMBLE_SA_MODEL_PARAS
        model_paras['estimators'] = [
            ("RandomForest", vc_dir+"3.results/1st_without/Valchiavenna_Forests of randomized trees.pkl"),
            ("AdaBoostCalibrated", vc_dir+"3.results/1st_without/Valchiavenna_AdaBoost Calibrated.pkl"),
            ("NeuralNetworksAndLR", vc_dir+"3.results/1st_without/NNlogistic/Valchiavenna_Neural Network.pkl"),
        ]
        return ensemble_simple_average(X, Y, Xtest, Ytest, model_paras=model_paras, save_to=save_to)
    
    #algorithms={SVM_MODEL_LABLE:svm_svc, NEURAL_NETWORK_MODEL_LABEL: neural_network}
    algorithms={
        #CALIBRATED_ADABOOST_MODEL_LABLE: ensemble_calibrated_adaboost,
        #NEURAL_NETWORK_MODEL_LABEL: NN_wrapper,
        ENSEMBLE_STACK_MODEL_LABEL: ensemble_stack_wrapper_with_cvset, #ensemble_stack,
        #ENSEMBLE_BLEND_MODEL_LABEL: ensemble_blend,
        #ENSEMBLE_SOFT_VOTING_MODEL_LABEL: ensemble_soft_voting,
        #ENSEMBLE_SA_MODEL_LABEL: ensemble_SA_wrapper,
    }
        
    LSM('Valchiavenna', factor_dir, trainset_path, testset_path, result_path, preprocess=None, algorithms=algorithms)

def Valchiavenna_v2_with():
    vc_dir = base_dir + r"ValChiavenna/"
    factor_dir = vc_dir+"1.factors"
    trainset_path = vc_dir+"/2.samples/2nd_with/UpperValtellina_ValTartano_Valchiavenna_LSM_training_points.csv"
    testset_path = vc_dir+"/2.samples/2nd_with/Valchiavenna_LSM_testing_points.csv"
    result_path = vc_dir+"3.results/2nd_with/ensemble/"

    #algorithms={SVM_MODEL_LABLE:svm_svc, NEURAL_NETWORK_MODEL_LABEL: neural_network}
    algorithms={
        #CALIBRATED_ADABOOST_MODEL_LABLE: ensemble_calibrated_adaboost,
        #NEURAL_NETWORK_MODEL_LABEL: NN_wrapper,
        ENSEMBLE_STACK_MODEL_LABEL: ensemble_stack,
        ENSEMBLE_BLEND_MODEL_LABEL: ensemble_blend,
        ENSEMBLE_SOFT_VOTING_MODEL_LABEL: ensemble_soft_voting,
    }
        
    clfs = LSM('Valchiavenna', factor_dir, trainset_path, testset_path, result_path, preprocess=None, algorithms=algorithms)

def Valchiavenna_v3_onlyVC():
    vc_dir = base_dir + r"ValChiavenna/"
    factor_dir = vc_dir+"1.factors"
    trainset_path = vc_dir+"/2.samples/3rd_onlyVC/ValChiavenna_LSM_training_points.csv"
    testset_path = vc_dir+"/2.samples/3rd_onlyVC/Valchiavenna_LSM_testing_points.csv"
    result_path = vc_dir+"3.results/3rd_onlyVC/ensemble/"

    #algorithms={SVM_MODEL_LABLE:svm_svc, NEURAL_NETWORK_MODEL_LABEL: neural_network}
    algorithms={
        #CALIBRATED_ADABOOST_MODEL_LABLE: ensemble_calibrated_adaboost,
        #NEURAL_NETWORK_MODEL_LABEL: NN_wrapper,
        ENSEMBLE_STACK_MODEL_LABEL: ensemble_stack,
        ENSEMBLE_BLEND_MODEL_LABEL: ensemble_blend,
        ENSEMBLE_SOFT_VOTING_MODEL_LABEL: ensemble_soft_voting,
    }
        
    LSM('Valchiavenna', factor_dir, trainset_path, testset_path, result_path, preprocess=None, algorithms=algorithms)


def check_factors():
    vc_dir = base_dir + r"ValChiavenna/"
    factor_dir = vc_dir+"1.factors"
    #load_rasters(factor_dir)
    get_factors_meta(factor_dir)

def Valchiavenna_v1_without_result_evaluation():
    vc_dir = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/ValChiavenna/"
    model_label = "Random Forests using R"
    result_path = vc_dir + r"evaluation/v1_WithoutTrainingPoints/"
    testing_data_path = vc_dir + "processing/v1_WithoutTrainingPoints/testing_data_with_LSM.csv"

    result_evaluation(testing_data_path, result_path, model_label)
    # plot
    with rasterio.open(f'{vc_dir}/processing/v1_WithoutTrainingPoints/result/Valchiavenna_without_map.img') as ds:
        outmap = ds.read(1)
        outmap = np.where(outmap<0,np.nan,outmap)
        plt.figure(figsize=(10,10))
        plt.imshow(outmap,cmap='RdYlGn_r',vmin=0,vmax= 1)
        plt.title(f'Probability of Landslide - {model_label}')
        plt.colorbar()
        plt.savefig(os.path.join(result_path, f"LSM_{_modelname2filename(model_label)}"))
    

def Valchiavenna_v2_with_result_evaluation():
    vc_dir = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/ValChiavenna/"
    model_label = "Random Forests using R"
    result_path = vc_dir + r"evaluation/v2_WithTrainingPoints/"
    testing_data_path = vc_dir + "processing/v2_WithTrainingPoints/testing_data_with_LSM.csv"

    #result_evaluation(testing_data_path, result_path, model_label)

    with rasterio.open(f'{vc_dir}/processing/v2_WithTrainingPoints/result/Valchiavenna_with_map.img') as ds:
        outmap = ds.read(1)
        outmap = np.where(outmap<0,np.nan,outmap)
        plt.figure(figsize=(10,10))
        plt.imshow(outmap,cmap='RdYlGn_r',vmin=0,vmax= 1)
        plt.title(f'Probability of Landslide - {model_label}')
        plt.colorbar()
        plt.savefig(os.path.join(result_path, f"LSM_{_modelname2filename(model_label)}"))

def plot_evaluation():
    vc_dir = base_dir + r"ValChiavenna/"

    info = {
        "1_without": {
            "testset_path": vc_dir+"/2.samples/1st_without/Valchiavenna_LSM_testing_points.csv",
            "result_path": {"basic": vc_dir+"3.results/1st_without/", "ensemble": vc_dir+"3.results/1st_without/ensemble"},
            "clfs": {
                "basic": {
                    BAGGING_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_Bagging.pkl", "color": 'tab:orange'},
                    RANDOMFOREST_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_Forests of randomized trees.pkl", "color": 'tab:green'},
                    CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
                    GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vc_dir+"3.results/1st_without/Valchiavenna_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
                    NEURAL_NETWORK_MODEL_LABEL: {"path": vc_dir+"3.results/1st_without/NNLogistic/Valchiavenna_Neural Network.pkl", "color": 'tab:pink'},
                },
                "ensemble": {
                    ENSEMBLE_STACK_MODEL_LABEL: {"path": vc_dir+"3.results/1st_without/ensemble/Valchiavenna_Ensemble Stacking.pkl", "color": 'tab:orange'},
                    ENSEMBLE_BLEND_MODEL_LABEL: {"path": vc_dir+"3.results/1st_without/ensemble/Valchiavenna_Ensemble Blending.pkl", "color": 'tab:green'},
                    ENSEMBLE_SOFT_VOTING_MODEL_LABEL: {"path": vc_dir+"3.results/1st_without/ensemble/Valchiavenna_Ensemble Soft Voting.pkl", "color": 'tab:brown'},
                },
            }
        },
        "2_with": {
            "testset_path":vc_dir+"/2.samples/2nd_with/Valchiavenna_LSM_testing_points.csv",
            "result_path": {"basic": vc_dir+"3.results/2nd_with/", "ensemble": vc_dir+"3.results/2nd_with/ensemble"},
            "clfs": {
                "basic": {
                    BAGGING_MODEL_LABLE: {"path": vc_dir+"3.results/2nd_with/Valchiavenna_Bagging.pkl", "color": 'tab:orange'},
                    RANDOMFOREST_MODEL_LABLE: {"path": vc_dir+"3.results/2nd_with/Valchiavenna_Forests of randomized trees.pkl", "color": 'tab:green'},
                    CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/2nd_with/Valchiavenna_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
                    GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vc_dir+"3.results/2nd_with/Valchiavenna_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
                    NEURAL_NETWORK_MODEL_LABEL: {"path": vc_dir+"3.results/2nd_with/NNLogistic/Valchiavenna_Neural Network.pkl", "color": 'tab:pink'},
                },
                "ensemble": {
                    ENSEMBLE_STACK_MODEL_LABEL: {"path": vc_dir+"3.results/2nd_with/ensemble/Valchiavenna_Ensemble Stacking.pkl", "color": 'tab:orange'},
                    ENSEMBLE_BLEND_MODEL_LABEL: {"path": vc_dir+"3.results/2nd_with/ensemble/Valchiavenna_Ensemble Blending.pkl", "color": 'tab:green'},
                    ENSEMBLE_SOFT_VOTING_MODEL_LABEL: {"path": vc_dir+"3.results/2nd_with/ensemble/Valchiavenna_Ensemble Soft Voting.pkl", "color": 'tab:brown'},
                },
            }
        },
        "3_onlywith": {
            "testset_path":vc_dir+"/2.samples/3rd_onlyVC/Valchiavenna_LSM_testing_points.csv",
            "result_path": {"basic": vc_dir+"3.results/3rd_onlyVC/", "ensemble": vc_dir+"3.results/3rd_onlyVC/ensemble"},
            "clfs": {
                "basic": {
                    BAGGING_MODEL_LABLE: {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_Bagging.pkl", "color": 'tab:orange'},
                    RANDOMFOREST_MODEL_LABLE: {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_Forests of randomized trees.pkl", "color": 'tab:green'},
                    CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
                    GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
                    NEURAL_NETWORK_MODEL_LABEL: {"path": vc_dir+"3.results/3rd_onlyVC/NNLogistic/Valchiavenna_Neural Network.pkl", "color": 'tab:pink'},
                },
                "ensemble": {
                    ENSEMBLE_STACK_MODEL_LABEL: {"path": vc_dir+"3.results/3rd_onlyVC/ensemble/Valchiavenna_Ensemble Stacking.pkl", "color": 'tab:orange'},
                    ENSEMBLE_BLEND_MODEL_LABEL: {"path": vc_dir+"3.results/3rd_onlyVC/ensemble/Valchiavenna_Ensemble Blending.pkl", "color": 'tab:green'},
                    ENSEMBLE_SOFT_VOTING_MODEL_LABEL: {"path": vc_dir+"3.results/3rd_onlyVC/ensemble/Valchiavenna_Ensemble Soft Voting.pkl", "color": 'tab:brown'},
                },
            }
        },
    }
 
    for key in info:
        print(f"Handling {key}")
        testset_path = info[key]["testset_path"]
        result_path = info[key]["result_path"]["ensemble"]
        clfs = info[key]['clfs']["ensemble"]
        plot_LSM_evaluation(testset_path, clfs, result_path)
        reports = {}
        for model_lable in clfs:
            reports[model_lable] = evaluation_with_testset(testset_path, clfs[model_lable]["path"], model_lable, save_to=result_path)
        import json
        with open(os.path.join(result_path, "report.json"), "w") as f:
            json.dump(reports, f)

if __name__ == '__main__':
    Valchiavenna_v1_without()
    #Valchiavenna_v2_with()
    #Valchiavenna_v3_onlyVC()
    #check_factors()
    #Valchiavenna_v1_without_result_evaluation()
    #Valchiavenna_v2_with_result_evaluation()
    #plot_evaluation()
