from processing import *
from config import base_dir
import numpy as np
import rasterio
import os
from matplotlib import pyplot as plt

def Valchiavenna_v1_without():
    vc_dir = base_dir + r"ValChiavenna/"
    factor_dir = vc_dir+"1.factors"
    trainset_path = vc_dir+"/2.samples/1st_without/UpperValtellina_ValTartano_training_points.csv"
    testset_path = vc_dir+"/2.samples/1st_without/Valchiavenna_LSM_testing_points.csv"
    result_path = vc_dir+"3.results/1st_without/ensemble/"
    #algorithms={SVM_MODEL_LABLE:svm_svc, NEURAL_NETWORK_MODEL_LABEL: neural_network}
    algorithms={
        #CALIBRATED_ADABOOST_MODEL_LABLE: ensemble_calibrated_adaboost,
        #NEURAL_NETWORK_MODEL_LABEL: NN_wrapper,
        ENSEMBLE_STACK_MODEL_LABEL: ensemble_stack,
        ENSEMBLE_BLEND_MODEL_LABEL: ensemble_blend,
        ENSEMBLE_SOFT_VOTING_MODEL_LABEL: ensemble_soft_voting,
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
            "clfs": {
                BAGGING_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_Bagging.pkl", "color": 'tab:orange'},
                RANDOMFOREST_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_Forests of randomized trees.pkl", "color": 'tab:green'},
                CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
                GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vc_dir+"3.results/1st_without/Valchiavenna_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
                NEURAL_NETWORK_MODEL_LABEL: {"path": vc_dir+"3.results/1st_without/NNLogistic/Valchiavenna_Neural Network.pkl", "color": 'tab:pink'},
            }
        },
        "2_with": {
            "testset_path":vc_dir+"/2.samples/2nd_with/Valchiavenna_LSM_testing_points.csv",
            "clfs": {
                BAGGING_MODEL_LABLE: {"path": vc_dir+"3.results/2nd_with/Valchiavenna_Bagging.pkl", "color": 'tab:orange'},
                RANDOMFOREST_MODEL_LABLE: {"path": vc_dir+"3.results/2nd_with/Valchiavenna_Forests of randomized trees.pkl", "color": 'tab:green'},
                CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/2nd_with/Valchiavenna_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
                GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vc_dir+"3.results/2nd_with/Valchiavenna_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
                NEURAL_NETWORK_MODEL_LABEL: {"path": vc_dir+"3.results/2nd_with/NNLogistic/Valchiavenna_Neural Network.pkl", "color": 'tab:pink'},
            }
        },
        "3_onlywith": {
            "testset_path":vc_dir+"/2.samples/3rd_onlyVC/Valchiavenna_LSM_testing_points.csv",
            "clfs": {
                BAGGING_MODEL_LABLE: {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_Bagging.pkl", "color": 'tab:orange'},
                RANDOMFOREST_MODEL_LABLE: {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_Forests of randomized trees.pkl", "color": 'tab:green'},
                CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
                GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
                NEURAL_NETWORK_MODEL_LABEL: {"path": vc_dir+"3.results/3rd_onlyVC/NNLogistic/Valchiavenna_Neural Network.pkl", "color": 'tab:pink'},
            }
        },
    }
 
    for key in info:
        print(f"Handling {key}")
        plot_LSM_evaluation(info[key]["testset_path"], info[key]["clfs"])

if __name__ == '__main__':
    #Valchiavenna_v1_without()
    #Valchiavenna_v2_with()
    #Valchiavenna_v3_onlyVC()
    #check_factors()
    #Valchiavenna_v1_without_result_evaluation()
    #Valchiavenna_v2_with_result_evaluation()
    plot_evaluation()
