from time import time
from config import *
from processing import *

import os
import shutil
def preparation(factor_path, M=6, N=5):
    # list factors
    fps = [os.path.join(factor_path,ele) for ele in rFactors if os.path.isdir(os.path.join(factor_path,ele))]
    print(fps)
    # for each {factor}, move {factor}/{factor}_i_j to dir factor_i_j
    factor_dirs = []
    for i in range(M):
        for j in range(N):
            cur_dir = os.path.join(factor_path, f"factor_{i}_{j}")
            os.makedirs(cur_dir)
            factor_dirs.append(cur_dir)

            for fp in fps:
                tfn = fp.split("/")[-1]
                to_move = os.path.join(fp, f"{tfn}_{i}_{j}.tif")
                shutil.move(to_move, cur_dir)
                # rename
                if os.path.exists(os.path.join(cur_dir, f"{tfn}.tif")):
                    os.remove(os.path.join(cur_dir, f"{tfn}.tif"))
                os.rename(os.path.join(cur_dir, f"{tfn}_{i}_{j}.tif"), os.path.join(cur_dir, f"{tfn}.tif"))
            print(f"Processing {cur_dir} DONE")

    return factor_dirs

# TOFIX: 2023.01.30 Warp(reproject) - align ndvi extent to dtm, and change ndvi pixel size to 5px
# 2023.01.31
## TOFIX: Manage big rasters (https://www.giscourse.com/easy-way-to-manage-big-raster-layers-in-qgis-raster-divider-and-easy-raster-splitter/)
## TOFIX: 使用 'saga:addrastervaluestopoints' 对 category factor 进行采样的时候出现浮点值 --> 使用 gdal.translate 将这些 factor 转换成 uint16。特别的，对于 dusaf，设置 0 为 NODATA --> 不行，会出现值偏差。重新rasterize，这次直接选择output data type。但是在采样过程中，tif转grid的resampling方式是“B-Spline Interpolation”，这样会不会导致采样的值发生改变呢？不清楚，试试直接调用 saga_cmd （e.g. saga_cmd io_gdal 0 -TRANSFORM 1 -RESAMPLING 0 -GRIDS "./dusaf.sgrd" -FILES "./dusaf.tif"），分别选择“Resampling: Nearest Neighbour” 和 “Resampling: B-Spline Interpolation” 对比下结果 --> 确实不行，修改为先“Resampling: Nearest Neighbour”，再采样
def Lombardy(clfs, ld_dir, factor_dirs, testset_path):
    result_path = ld_dir+"3.results/testingpoints_without_3regions"
    #result_path = ld_dir+"3.results/testingpoints_in_whole_region"

    print(f"""# Lombardy
    [DIR]
    Factor dirs: {factor_dirs}
    Train set path: None
    Test set path: {testset_path}
    Result path: {result_path}
    """)

    if testset_path:
        start = time()  
        # testing data
        _, ld_testingPoints = get_train_test(
                None,
                testset_path
            )
        ld_test_xs, ld_test_y = get_X_Y(ld_testingPoints)
        end = time()
        print(f"""
        [INPUT]
        Testing points shape:
            X = {ld_test_xs.shape},
            Y = {ld_test_y.shape},
            Columns = {ld_test_xs.columns}
            Column Types = {ld_test_xs.dtypes}

        Time Cost: {end-start} seconds
        """)
    
    print("## Processing and Evaluation")
    for model_label, model_path in clfs.items():
        start = time()
        clf = load_model(model_path)
        print(f"""### {model_label}
        Feature Names (Names of features seen during fit):
        {clf.feature_names_in_}
        """)
        # test
        if testset_path:
            ## TOFIX: The feature names should match those that were passed during fit. Feature names must be in the same order as they were in fit.
            test_xs = ld_test_xs[clf.feature_names_in_]
            Ytest_pred = clf.predict_proba(test_xs)
            # test result evaluation
            evaluation_report(ld_test_y, Ytest_pred, model_label, result_path)

        end = time()
        print("\n\nTime Cost:  %.3f seconds" % (end-start))

        clfs[model_label] = (model_path, clf)

    # mapping
    print("## Mapping")
    for factor_dir in factor_dirs:
        ## target
        start = time()
        ld_target_xs, ld_raster_info, ld_mask = get_targets(factor_dir)
        print(f"""### {factor_dir}
        Target:
        X = {ld_target_xs.shape},
        Columns = {ld_target_xs.columns}
        Column Types = {ld_target_xs.dtypes}

        Get Target Time cost: {time()-start}
        """)
        tmp_result_path = os.path.join(result_path, factor_dir.split('/')[-1]+'/')
        if not os.path.exists(tmp_result_path):
            os.makedirs(tmp_result_path)
        for model_label, model_info in clfs.items():
            start = time()
            print(f"#### {model_label}")
            clf = model_info[1]
            target_xs = ld_target_xs[clf.feature_names_in_]
            mapping(target_xs, clf, model_label, ld_raster_info, ld_mask, tmp_result_path)
            end = time()
            print("Mapping Time cost: %.3f seconds" % (end-start))


def Lombardy_WithChunk():
    ld_dir = base_dir + r"Lombardy/"
    factor_dir = ld_dir+"1.factors/"
    result_path = ld_dir+"3.results/"

    clfs_dir = base_dir + r"ValChiavenna/3.results/2nd_with/"
    clfs = {
        BAGGING_MODEL_LABLE: {"path": clfs_dir+"Valchiavenna_Bagging.pkl", "skip": True},
        RANDOMFOREST_MODEL_LABLE: {"path": clfs_dir+"Valchiavenna_Forests of randomized trees.pkl", "skip": True, "skip_predict": True},
        GRADIENT_TREE_BOOSTING_MODEL_LABLE: {"path": clfs_dir+"Valchiavenna_Gradient Tree Boosting.pkl", "skip": True},
        ADABOOST_MODEL_LABLE: {"path": clfs_dir+"Valchiavenna_AdaBoost.pkl", "skip_predict": True},
    }

    column_types={'dtm': 'float32', 'east': 'float32', 'ndvi': 'float32', 'north': 'float32', 'faults': 'float32', 'rivers': 'float32', 'roads': 'float32', 'dusaf': 'float32', 'plan': 'float32', 'profile': 'float32', 'twi': 'float32'}

    LSM_PredictMap_WithChunk(clfs, factor_dir, result_path, need_chunk=False, column_types=column_types, chunk_size=PROCESS_BATCH_SIZE, pred_batch_size=PREDICT_BATCH_SIZE)



def check_factors():
    ld_dir = base_dir + r"Lombardy/"
    factor_dir = ld_dir+"1.factors"
    #load_rasters(factor_dir)
    get_factors_meta(factor_dir)

if __name__ == '__main__':

    ld_dir = base_dir + r"Lombardy/"
    #factor_dirs = ld_dir+"1.factors"
    #factor_dirs = preparation(ld_dir+"1.factors")
    factor_dirs =  []
    
    clfs_dir = base_dir + r"ValChiavenna/3.results/2nd_with/"
    clfs = {
        #BAGGING_MODEL_LABLE: clfs_dir+"Valchiavenna_Bagging.pkl",
        #RANDOMFOREST_MODEL_LABLE: clfs_dir+"Valchiavenna_Forests of randomized trees.pkl",
        #GRADIENT_TREE_BOOSTING_MODEL_LABLE: clfs_dir+"Valchiavenna_Gradient Tree Boosting.pkl",
        #ADABOOST_MODEL_LABLE: clfs_dir+"Valchiavenna_AdaBoost.pkl",
        CALIBRATED_ADABOOST_MODEL_LABLE: clfs_dir+"Valchiavenna_AdaBoost Calibrated.pkl",
        NEURAL_NETWORK_MODEL_LABEL: clfs_dir+"Valchiavenna_Neural Network.pkl",

    }
    #testset_path = ld_dir+"/2.samples/Lombardy_LSM_testing_points.csv"
    testset_path = ld_dir+"/2.samples/Lombardy_LSM_testing_points_without_3regions.csv"
    #preparation(ld_dir+"1.factors", M=1, N=5)
    Lombardy(clfs, ld_dir, factor_dirs, testset_path)
    #Lombardy_WithChunk()
    #check_factors()