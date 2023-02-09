from processing import LSM, load_rasters, get_factors_meta
from config import base_dir

def UpperValtellina():
    uv_dir = base_dir + r"Upper Valtellina/"
    factor_dir = uv_dir+"1.factors"
    trainset_path = uv_dir+"/2.samples/UpperValtellina_LSM_training_points.csv"
    testset_path = uv_dir+"/2.samples/UpperValtellina_LSM_testing_points.csv"
    result_path = uv_dir+"3.results"
    preprocess = None

    LSM('Upper Valtellina', factor_dir, trainset_path, testset_path, result_path, preprocess)

def check_factors():
    uv_dir = base_dir + r"Upper Valtellina/"
    factor_dir = uv_dir+"1.factors"
    #load_rasters(factor_dir)
    get_factors_meta(factor_dir)


if __name__ == '__main__':
    #UpperValtellina()
    check_factors()