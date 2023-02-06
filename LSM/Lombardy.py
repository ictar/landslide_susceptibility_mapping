import joblib
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

def load_model(model_path):
    return joblib.load(model_path)

# TOFIX: 2023.01.30 Warp(reproject) - align ndvi extent to dtm, and change ndvi pixel size to 5px
# 2023.01.31
## TOFIX: Manage big rasters (https://www.giscourse.com/easy-way-to-manage-big-raster-layers-in-qgis-raster-divider-and-easy-raster-splitter/)
## TOFIX: 使用 'saga:addrastervaluestopoints' 对 category factor 进行采样的时候出现浮点值 --> 使用 gdal.translate 将这些 factor 转换成 uint16。特别的，对于 dusaf，设置 0 为 NODATA --> 不行，会出现值偏差。重新rasterize，这次直接选择output data type。但是在采样过程中，tif转grid的resampling方式是“B-Spline Interpolation”，这样会不会导致采样的值发生改变呢？不清楚，试试直接调用 saga_cmd （e.g. saga_cmd io_gdal 0 -TRANSFORM 1 -RESAMPLING 0 -GRIDS "./dusaf.sgrd" -FILES "./dusaf.tif"），分别选择“Resampling: Nearest Neighbour” 和 “Resampling: B-Spline Interpolation” 对比下结果 --> 确实不行，修改为先“Resampling: Nearest Neighbour”，再采样
def Lombardy(clfs, ld_dir, factor_dirs, testset_path):
    result_path = ld_dir+"3.results/"

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

# This is used for big data mapping
def get_targets_raw(layer_dir):
    # load rasters
    factors, meta, mask = load_rasters(layer_dir)
    mask = mask.astype(np.float16)

    for layer in continuous_factors:
        raster_info = {
            "transform": meta['transform'],
            "shape": factors[layer].shape,
            "crs": meta['crs'],
        }
        factors[layer] = factors[layer].flatten()

    for cat in categorical_factors:    
        factors[cat] = factors[cat].flatten()

    return factors, raster_info, mask

# SIZE = raster_info["shape"][0] * raster_info["shape"][1]
def get_target_batch(factors, SIZE, batch_size, begin=0):   
    for ifrom in range(begin, SIZE, batch_size):
        ito = min(ifrom+batch_size, SIZE)
        yield (pd.DataFrame({layer: factors[layer][ifrom:ito] for layer in factors}), ifrom, ito)

import json
# target_y_pred： np.array
def _save_result(target_y_pred, envs, error_x, save_to):
    if target_y_pred:
        pred_path = os.path.join(save_to, "target_y_pred.tmp")
        #target_y_pred.to_csv(pred_path, index=False)
        np.savetxt(pred_path, target_y_pred)

    with open(os.path.join(save_to, "envs.tmp"), 'w') as f:
        f.write(json.dumps(envs))

    if error_x is not None:
        error_x.to_csv(os.path.join(save_to, "error_x.tmp"), index=False)

def _load_result(rpath):
    envs = None
    env_path = os.path.join(rpath, "envs.tmp")
    if os.path.exists(env_path):
        with open(env_path) as f:
            envs = json.loads(f.read())

    target_y_pred = []
    target_path = os.path.join(rpath, "target_y_pred.tmp")
    if os.path.exists(target_path):
        #target_y_pred.append(pd.read_csv(target_path))
        target_y_pred.append(np.loadtxt(target_path))

    return target_y_pred, envs


def bigdata_mapping(clfs, factor_dir, save_to, batch_size=10**7):
    print(f"""# Mapping
        [DIR]
        Factor dir: {factor_dir}
        Result path: {save_to}
    """)

    factors, raster_info, mask = get_targets_raw(factor_dir)

    # for each classifier, batch handling (TC=Time Consuming)
    for model_label, model_path in clfs.items():
        startM = time()
        print(f"## {model_label}")
        clf = load_model(model_path)

        target_y_pred, envs = _load_result(save_to)
        if envs is None or envs['model_label'] != model_label:
            target_y_pred, envs = [], {}
        print(f"""target_y_pred before: {len(target_y_pred)}
        now begin from: {envs.get('ito', 0)}""")

        # 1. for a block of factor
        print("### Input Data\n")
        for target_xy, ifrom, ito in get_target_batch(factors, raster_info["shape"][0] * raster_info["shape"][1], batch_size, begin=envs.get('ito', 0)):
            print(f"""Handling data from index {ifrom} to index {ito-1}...
            
            Target:
                XY = {target_xy.shape},
                Columns = {target_xy.columns}
            """)
            # 1.1 [TC] process categorical factors
            start = time()
            target_xs, _ = get_X_Y(target_xy)
            # TOAVOID: cannot allocate memory
            #target_xs = target_xs.astype(DATA_COLUMN_TYPES)
            print(f"""
            [Feature Info]
            Target:
                XY = {target_xs.shape},
                Columns = {target_xs.columns}

            Get XY Time Cost: {time()-start}""")
            # 1.2 make sure the feature names match the classifier feature name
            for feature in clf.feature_names_in_:
                if feature not in target_xs.columns:
                    print(f"Feature {feature} is missing in target, set default to 0.")
                    target_xs[feature] = 0
            target_xs = target_xs[clf.feature_names_in_]
            # 1.3 [TC] predict
            start = time()
            try:
                block_target_y_pred = clf.predict_proba(target_xs)
            except Exception as e:
                # check
                print(f"""[ERROR]
            Target:
                X = {target_xs.shape},
                Columns = {target_xs.dtypes}
                Max = {target_xs.max(axis=0)}
                Min = {target_xs.min(axis=0)}
                Non Check: {np.count_nonzero(np.isnan(target_xs))}
                Infinity Check: { np.count_nonzero(np.isinf(target_xs))}""")
                # save target_y_pred
                envs = {
                    'ito': ifrom,
                    'model': model_label,
                }
                target_y_pred = np.concatenate(target_y_pred) if len(target_y_pred) > 0 else []
                _save_result(target_y_pred, envs, target_xs, save_to)
                raise e

            print(f"Predict Time Cost: {time()-start}")
            # 1.4 append the predict result
            target_y_pred.append(block_target_y_pred)
        
        # 2. generate and save the LSM
        start = time()
        target_y_pred = np.concatenate(target_y_pred)
        try:
            _save_result(target_y_pred, envs={'model': model_label,}, error_x=None, save_to=save_to)
        except Exception as e:
            print(e)
        outmap = plot_LSM_prediction(target_y_pred, raster_info, mask, model_label, save_to)
        print(f"Plot Time Cost: {time()-start}")
        # save
        with rasterio.open(
            os.path.join(save_to, f'LSM_{model_label}.tif'),
            'w',
            driver='GTiff',
            height=outmap.shape[0],
            width=outmap.shape[1],
            count=1,
            dtype=outmap.dtype,
            crs=raster_info['crs'],
            transform=raster_info['transform'],
        ) as dst:
            dst.write(outmap, 1)

        print(f"\n\nTime Cost: {time()-startM}")

def Lombardy_Mapping(clfs):
    ld_dir = base_dir + r"Lombardy/"
    factor_dir = ld_dir+"1.factors/"
    result_path = ld_dir+"3.results/"
    bigdata_mapping(clfs, factor_dir, result_path, batch_size=PROCESS_BATCH_SIZE)


def check_factors():
    ld_dir = base_dir + r"Lombardy/"
    factor_dir = ld_dir+"1.factors"
    load_rasters(factor_dir)

if __name__ == '__main__':
    clfs_dir = base_dir + r"ValChiavenna/3.results/2nd_with/"
    clfs = {
        BAGGING_MODEL_LABLE: clfs_dir+"Valchiavenna_Bagging.pkl",
        RANDOMFOREST_MODEL_LABLE: clfs_dir+"Valchiavenna_Fortests of randomized trees.pkl",
        ADABOOST_MODEL_LABLE: clfs_dir+"Valchiavenna_AdaBoost.pkl",
        GRADIENT_TREE_BOOSTING_MODEL_LABLE: clfs_dir+"Valchiavenna_Gradient Tree Boosting.pkl",
    }

    ld_dir = base_dir + r"Lombardy/"
    #factor_dirs = ld_dir+"1.factors"
    #factor_dirs = preparation(ld_dir+"1.factors")
    factor_dirs =  [f'/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Lombardy/1.factors/factor_{i}_{j}' for i in range(6) for j in range(5)]
    
    testset_path = ld_dir+"/2.samples/Lombardy_LSM_testing_points.csv"
    #preparation(ld_dir+"1.factors", M=1, N=5)
    #Lombardy(clfs, ld_dir, factor_dirs, testset_path)
    #Lombardy(clfs, ld_dir, factor_dirs, "")
    Lombardy_Mapping(clfs)