from processing import LSM
from config import base_dir

def Valchiavenna_v1_without():
    vc_dir = base_dir + r"ValChiavenna/"
    factor_dir = vc_dir+"1.factors"
    trainset_path = vc_dir+"/2.samples/1st_without/UpperValtellina_ValTartano_training_points.csv"
    testset_path = vc_dir+"/2.samples/1st_without/Valchiavenna_LSM_testing_points.csv"
    result_path = vc_dir+"3.results/1st_without/"
        
    LSM('Valchiavenna', factor_dir, trainset_path, testset_path, result_path, preprocess=None)

def Valchiavenna_v2_with():
    vc_dir = base_dir + r"ValChiavenna/"
    factor_dir = vc_dir+"1.factors"
    trainset_path = vc_dir+"/2.samples/2nd_with/UpperValtellina_ValTartano_Valchiavenna_LSM_training_points.csv"
    testset_path = vc_dir+"/2.samples/2nd_with/Valchiavenna_LSM_testing_points.csv"
    result_path = vc_dir+"3.results/2nd_with/"
        
    clfs = LSM('Valchiavenna', factor_dir, trainset_path, testset_path, result_path, preprocess=None)


if __name__ == '__main__':
    Valchiavenna_v1_without()
    #Valchiavenna_v2_with()