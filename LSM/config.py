import numpy as np

#base_dir = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/"
#base_dir = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/"
base_dir = r"/Users/elexu/Library/CloudStorage/OneDrive-PolitecnicodiMilano/thesis/practice/"

rFactors = ['dtm', 'east', 'ndvi', 'north', 'faults',
        'rivers','roads','dusaf',
        'plan','profile','twi']

categorical_factors = ['dusaf','faults','rivers','roads']
# categorical_factors = ['dusaf','faults','rivers','roads', 'plan', 'profile'] # add to check if plan/profile should be used as categorical factor
continuous_factors = [x for x in rFactors if x not in categorical_factors]
#print(f"{len(rFactors)} = {len(categorical_factors)} + {len(continuous_factors)}")

NaN = -9999

#JOBLIB_TEMP_FOLDER = r'/Users/elexu/tmp'
JOBLIB_TEMP_FOLDER = r'/Volumes/Another/tmp'
# TOAVOID: save memory
# float16: -65536 - 65536
MODEL_DATA_COLUMN_TYPES = {}
for f in continuous_factors:
       MODEL_DATA_COLUMN_TYPES[f] =  np.float32

# when deal with float->int, 'RuntimeWarning: invalid value encountered in cast'
'''
>>> arr = np.array([np.nan, 1, 2, ])
>>> arr.dtype
dtype('float64')
>>> arr1 = arr.astype('uint8')
<stdin>:1: RuntimeWarning: invalid value encountered in cast
>>> arr1
array([0, 1, 2], dtype=uint8)
>>> arr2 = arr.astype('float16')
>>> arr2
array([nan,  1.,  2.], dtype=float16)
'''
DTYPE_MAPPING = {'float64': np.float32}


##### for processing
# PROCESS_BATCH_SIZE = 67127581 # first try
PROCESS_BATCH_SIZE = 67127581
PREDICT_BATCH_SIZE = 10**6


############## model infos ##############
## basic
BAGGING_MODEL_LABLE = "Bagging"
RANDOMFOREST_MODEL_LABLE = "Forests of randomized trees"
ADABOOST_MODEL_LABLE = "AdaBoost"
CALIBRATED_ADABOOST_MODEL_LABLE = "AdaBoost Calibrated"
GRADIENT_TREE_BOOSTING_MODEL_LABLE = "Gradient Tree Boosting"
NEURAL_NETWORK_MODEL_LABEL = "Neural Network"
## ensembles
ENSEMBLE_STACK_MODEL_LABEL = "Ensemble Stacking"
ENSEMBLE_BLEND_MODEL_LABEL = "Ensemble Blending"
ENSEMBLE_SOFT_VOTING_MODEL_LABEL = "Ensemble Soft Voting"
ENSEMBLE_SA_MODEL_LABEL = "Ensemble Simple Averaging"

## combine with areas
vt_dir = base_dir + r"Val Tartano/"
vt_clfs = {
        "basic": {
            BAGGING_MODEL_LABLE: {"path": vt_dir+"3.results/Val Tartano_Bagging.pkl", "color": 'tab:orange'},
            RANDOMFOREST_MODEL_LABLE: {"path": vt_dir+"3.results/Val Tartano_Fortests of randomized trees.pkl", "color": 'tab:green'},
            ADABOOST_MODEL_LABLE: {"path": vt_dir+"3.results/Val Tartano_AdaBoost.pkl", "color": 'tab:red'},
            CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vt_dir+"3.results/Val Tartano_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
            GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vt_dir+"3.results/Val Tartano_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
            NEURAL_NETWORK_MODEL_LABEL: {"path": vt_dir+"3.results/NNLogistic/Val Tartano_Neural Network.pkl", "color": 'tab:pink'},
        },
        "ensemble": {
            ENSEMBLE_STACK_MODEL_LABEL: {"path": vt_dir+"3.results/ensemble/Val Tartano_Ensemble Stacking.pkl", "color": 'tab:orange', "linestyle": 'dashed'},
            ENSEMBLE_BLEND_MODEL_LABEL: {"path": vt_dir+"3.results/ensemble/Val Tartano_Ensemble Blending.pkl", "color": 'tab:green', "linestyle": 'dashed'},
            ENSEMBLE_SOFT_VOTING_MODEL_LABEL: {"path": vt_dir+"3.results/ensemble/Val Tartano_Ensemble Soft Voting.pkl", "color": 'tab:brown', "linestyle": 'dashed'},
        },
    }

uv_dir = base_dir + r"Upper Valtellina/"
uv_clfs = {
        "basic": {
            BAGGING_MODEL_LABLE: {"path": uv_dir+"3.results/Upper Valtellina_Bagging.pkl", "color": 'tab:orange'},
            RANDOMFOREST_MODEL_LABLE: {"path": uv_dir+"3.results/Upper Valtellina_Fortests of randomized trees.pkl", "color": 'tab:green'},
            ADABOOST_MODEL_LABLE: {"path": uv_dir+"3.results/Upper Valtellina_AdaBoost.pkl", "color": 'tab:red'},
            CALIBRATED_ADABOOST_MODEL_LABLE: {"path": uv_dir+"3.results.tune/AdaboostCalibrated/Upper Valtellina_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
            GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": uv_dir+"3.results/Upper Valtellina_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
            NEURAL_NETWORK_MODEL_LABEL: {"path": uv_dir+"3.results.tune/NNLogistic/Upper Valtellina_Neural Network.pkl", "color": 'tab:pink'},
        },
        "ensemble": {
            ENSEMBLE_STACK_MODEL_LABEL: {"path": uv_dir+"3.results/ensemble/Upper Valtellina_Ensemble Stacking.pkl", "color": 'tab:orange', "linestyle": 'dashed'},
            ENSEMBLE_BLEND_MODEL_LABEL: {"path": uv_dir+"3.results/ensemble/Upper Valtellina_Ensemble Blending.pkl", "color": 'tab:green', "linestyle": 'dashed'},
            ENSEMBLE_SOFT_VOTING_MODEL_LABEL: {"path": uv_dir+"3.results/ensemble/Upper Valtellina_Ensemble Soft Voting.pkl", "color": 'tab:brown', "linestyle": 'dashed'},
        }
    }

vc_dir = base_dir + r"ValChiavenna/"
vc_clfs = {
        "1_without": {
            "testset_path": vc_dir+"/2.samples/1st_without/Valchiavenna_LSM_testing_points.csv",
            "result_path": {"basic": vc_dir+"3.results/1st_without/", "ensemble": vc_dir+"3.results/1st_without/ensemble"},
            "clfs": {
                "basic": {
                    BAGGING_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_Bagging.pkl", "color": 'tab:orange'},
                    RANDOMFOREST_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_Forests of randomized trees.pkl", "color": 'tab:green'},
                    ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_AdaBoost.pkl", "color": 'tab:red'},
                    CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/1st_without/Valchiavenna_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
                    GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vc_dir+"3.results/1st_without/Valchiavenna_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
                    NEURAL_NETWORK_MODEL_LABEL: {"path": vc_dir+"3.results/1st_without/NNLogistic/Valchiavenna_Neural Network.pkl", "color": 'tab:pink'},
                },
                "ensemble": {
                    ENSEMBLE_STACK_MODEL_LABEL: {"path": vc_dir+"3.results/1st_without/ensemble/Valchiavenna_Ensemble Stacking.pkl", "color": 'tab:orange', "linestyle": 'dashed'},
                    ENSEMBLE_BLEND_MODEL_LABEL: {"path": vc_dir+"3.results/1st_without/ensemble/Valchiavenna_Ensemble Blending.pkl", "color": 'tab:green', "linestyle": 'dashed'},
                    ENSEMBLE_SOFT_VOTING_MODEL_LABEL: {"path": vc_dir+"3.results/1st_without/ensemble/Valchiavenna_Ensemble Soft Voting.pkl", "color": 'tab:brown', "linestyle": 'dashed'},
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
                    ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/2nd_with/Valchiavenna_AdaBoost.pkl", "color": 'tab:red'},
                    CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/2nd_with/Valchiavenna_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
                    GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vc_dir+"3.results/2nd_with/Valchiavenna_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
                    NEURAL_NETWORK_MODEL_LABEL: {"path": vc_dir+"3.results/2nd_with/NNLogistic/Valchiavenna_Neural Network.pkl", "color": 'tab:pink'},
                },
                "ensemble": {
                    ENSEMBLE_STACK_MODEL_LABEL: {"path": vc_dir+"3.results/2nd_with/ensemble/Valchiavenna_Ensemble Stacking.pkl", "color": 'tab:orange', "linestyle": 'dashed'},
                    ENSEMBLE_BLEND_MODEL_LABEL: {"path": vc_dir+"3.results/2nd_with/ensemble/Valchiavenna_Ensemble Blending.pkl", "color": 'tab:green', "linestyle": 'dashed'},
                    ENSEMBLE_SOFT_VOTING_MODEL_LABEL: {"path": vc_dir+"3.results/2nd_with/ensemble/Valchiavenna_Ensemble Soft Voting.pkl", "color": 'tab:brown', "linestyle": 'dashed'},
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
                    ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_AdaBoost.pkl", "color": 'tab:red'},
                    CALIBRATED_ADABOOST_MODEL_LABLE: {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_AdaBoost Calibrated.pkl", "color": 'tab:purple'},
                    GRADIENT_TREE_BOOSTING_MODEL_LABLE:  {"path": vc_dir+"3.results/3rd_onlyVC/Valchiavenna_Gradient Tree Boosting.pkl", "color": 'tab:brown'},
                    NEURAL_NETWORK_MODEL_LABEL: {"path": vc_dir+"3.results/3rd_onlyVC/NNLogistic/Valchiavenna_Neural Network.pkl", "color": 'tab:pink'},
                },
                "ensemble": {
                    ENSEMBLE_STACK_MODEL_LABEL: {"path": vc_dir+"3.results/3rd_onlyVC/ensemble/Valchiavenna_Ensemble Stacking.pkl", "color": 'tab:orange', "linestyle": 'dashed'},
                    ENSEMBLE_BLEND_MODEL_LABEL: {"path": vc_dir+"3.results/3rd_onlyVC/ensemble/Valchiavenna_Ensemble Blending.pkl", "color": 'tab:green', "linestyle": 'dashed'},
                    ENSEMBLE_SOFT_VOTING_MODEL_LABEL: {"path": vc_dir+"3.results/3rd_onlyVC/ensemble/Valchiavenna_Ensemble Soft Voting.pkl", "color": 'tab:brown', "linestyle": 'dashed'},
                },
            }
        },
    }

ld_dir = base_dir + r"Lombardy/"