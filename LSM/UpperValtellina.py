from processing import *
from config import base_dir

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def UpperValtellina():
    uv_dir = base_dir + r"Upper Valtellina/"
    factor_dir = uv_dir+"1.factors"
    trainset_path = uv_dir+"/2.samples/UpperValtellina_LSM_training_points.csv"
    testset_path = uv_dir+"/2.samples/UpperValtellina_LSM_testing_points.csv"
    result_path = uv_dir+"3.results/"
    preprocess = None

    def RF_wrapper(X, Y, Xtest, Ytest, save_to=None):
        model_paras = DEFAULT_RANDOMFOREST_MODEL_PARAS
        model_paras['n_estimators'] = 500
        return ensemble_randomforest(X, Y, Xtest, Ytest, model_paras=model_paras, save_to=save_to)
    
    def Adaboost_wrapper(X, Y, Xtest, Ytest, save_to=None):
        model_paras = DEFAULT_ADABOOST_MODEL_PARAS
        model_paras['estimator'] = SVC(probability=True , kernel='linear')#LogisticRegression()
        return ensemble_adaboost(X, Y, Xtest, Ytest, model_paras=model_paras, save_to=save_to)

    algorithms = {
        CALIBRATED_ADABOOST_MODEL_LABLE: ensemble_calibrated_adaboost,
        #NEURAL_NETWORK_MODEL_LABEL: NN_wrapper,
        #ENSEMBLE_STACK_MODEL_LABEL: ensemble_stack,
        #ENSEMBLE_BLEND_MODEL_LABEL: ensemble_blend,
        #ENSEMBLE_SOFT_VOTING_MODEL_LABEL: ensemble_soft_voting,
    }

    #LSM('Upper Valtellina', factor_dir, trainset_path, testset_path, result_path, preprocess)
    LSM('Upper Valtellina', factor_dir, trainset_path, testset_path, result_path, preprocess, algorithms=algorithms)

def UpperValtellina_PredictMap_WithChunk():
    uv_dir = base_dir + r"Upper Valtellina/"
    factor_dir = uv_dir+"1.factors"
    result_path = uv_dir+"3.results.v2/"

    clfs_dir = uv_dir + r"3.results.v1//"
    clfs = {
        BAGGING_MODEL_LABLE: {"path": clfs_dir+"Upper Valtellina_Bagging.pkl"},
        RANDOMFOREST_MODEL_LABLE: {"path": clfs_dir+"Upper Valtellina_Fortests of randomized trees.pkl",},
        ADABOOST_MODEL_LABLE: {"path": clfs_dir+"Upper Valtellina_AdaBoost.pkl"},
        GRADIENT_TREE_BOOSTING_MODEL_LABLE: {"path": clfs_dir+"Upper Valtellina_Gradient Tree Boosting.pkl"},
    }

    LSM_PredictMap_WithChunk(clfs, factor_dir, result_path, need_chunk=True, column_types=None, chunk_size=19970900//10, pred_batch_size=10**4)

def UpperValtellina_SVM_NN():
    uv_dir = base_dir + r"Upper Valtellina/"
    factor_dir = uv_dir+"1.factors"
    trainset_path = uv_dir+"/2.samples/UpperValtellina_LSM_training_points.csv"
    testset_path = uv_dir+"/2.samples/UpperValtellina_LSM_testing_points.csv"
    result_path = uv_dir+"3.results.svm.nn"
    preprocess = None
    
    algorithms = {
        #SVM_MODEL_LABLE:svm_svc, 
        #NEURAL_NETWORK_MODEL_LABEL: neural_network,
        GAUSSIAN_PROCESS_MODEL_LABEL: gaussian_process,
    }

    LSM('Upper Valtellina', factor_dir, trainset_path, testset_path, result_path, preprocess, algorithms=algorithms)

def check_factors():
    uv_dir = base_dir + r"Upper Valtellina/"
    factor_dir = uv_dir+"1.factors"
    #load_rasters(factor_dir)
    get_factors_meta(factor_dir)

def evaluation(testset_path, model_path, model_label):
    print(f"# Evaluation of {model_label} after modify the calculation of F1score for Optimal PRC case")
    clf = load_model(model_path)
    ## testing samples
    _, testingPoints = get_train_test(
        None,
        testset_path
    )
    test_xs, test_y = get_X_Y(testingPoints)
    test_xs = test_xs[clf.feature_names_in_]
    # test
    Ytest_pred = clf.predict_proba(test_xs)
    evaluation_report(test_y, Ytest_pred, model_label, save_to=base_dir + r"Upper Valtellina/3.results")

def plot_evaluation():
    testset_path = uv_dir+"2.samples/UpperValtellina_LSM_testing_points.csv"

    save_info = [('basic', uv_dir+"3.results/"), ("ensemble", uv_dir+"3.results/ensemble/")]
    plot_evaluation_with_testset(testset_path, uv_clfs, save_info)

if __name__ == '__main__':
    UpperValtellina()
    #UpperValtellina_SVM_NN()
    #UpperValtellina_PredictMap_WithChunk()
    #check_factors()

    #uv_dir = base_dir + r"Upper Valtellina/"
    #testset_path = uv_dir+"/2.samples/UpperValtellina_LSM_testing_points.csv"
    #evaluation(testset_path,uv_dir + "/3.results/Upper Valtellina_Gradient Tree Boosting.pkl","Model Ensemble / Gradient Tree Boosting")

    #plot_evaluation()
    
