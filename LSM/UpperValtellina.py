from processing import *
from config import base_dir

def UpperValtellina():
    uv_dir = base_dir + r"Upper Valtellina/"
    factor_dir = uv_dir+"1.factors"
    trainset_path = uv_dir+"/2.samples/UpperValtellina_LSM_training_points.csv"
    testset_path = uv_dir+"/2.samples/UpperValtellina_LSM_testing_points.csv"
    result_path = uv_dir+"3.results"
    preprocess = None

    LSM('Upper Valtellina', factor_dir, trainset_path, testset_path, result_path, preprocess)

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


def check_factors():
    uv_dir = base_dir + r"Upper Valtellina/"
    factor_dir = uv_dir+"1.factors"
    #load_rasters(factor_dir)
    get_factors_meta(factor_dir)


if __name__ == '__main__':
    #UpperValtellina()
    UpperValtellina_PredictMap_WithChunk()
    #check_factors()