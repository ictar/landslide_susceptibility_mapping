from processing import LSM, load_rasters
from config import base_dir

def ValTartano():
    vt_dir = base_dir + r"Val Tartano/"
    factor_dir = vt_dir+"1.factors"
    trainset_path = vt_dir+"/2.samples/ValTartano_Training_Points.csv"
    testset_path = vt_dir+"/2.samples/ValTartano_Testing_Points.csv"
    result_path = vt_dir+"3.results"

    def preprocess(train_xs, train_y, test_xs, test_y, target_xs):
        # 暂时处理：去掉dusaf_13
        target_xs = target_xs.drop(labels="dusaf_13", axis=1)
        return train_xs, train_y, test_xs, test_y, target_xs
        
    LSM('Val Tartano', factor_dir, trainset_path, testset_path, result_path, preprocess)

def check_factors():
    vt_dir = base_dir + r"Val Tartano/"
    factor_dir = vt_dir+"1.factors"
    load_rasters(factor_dir)

if __name__ == '__main__':
    #ValTartano()
    check_factors()