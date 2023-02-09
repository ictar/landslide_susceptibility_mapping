import joblib
from time import time
from config import *
from processing import *

import os
import shutil
import gc
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
    raster_info = None

    for layer in continuous_factors:
        if raster_info is None:
            raster_info = {
                "transform": meta['transform'],
                "shape": factors[layer].shape, # size=2
                "crs": meta['crs'],
            }
        factors[layer] = factors[layer].flatten()

    for cat in categorical_factors:    
        factors[cat] = factors[cat].flatten()

    return factors, raster_info, mask

def get_raster_meta(layer_path):
    column_types, raster_info, mask = None, None, None
    with rasterio.open(layer_path) as ds:
        layer = ds.read(1)
        layer = np.where(layer==ds.nodatavals,np.nan,layer).astype(np.float32)
        meta = ds.meta
        raster_info = {
            "transform": meta['transform'],
            "shape": layer.shape, # size=2
            "crs": meta['crs'],
        }
        mask = np.where(np.isnan(layer),np.nan,1).astype(np.float16)

    return raster_info, mask

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

def bigdata_chunk(factor_dir, save_to, chunk_size=10**7):
    print(f"""# Chunk
    [CONFIG]
    Factor dir: {factor_dir}
    Result path: {save_to}
    Chunk size: {chunk_size}
    """)
    start = time()
    factors, raster_info, mask = get_targets_raw(factor_dir)

    print(f"Load Factors Time Cost: {time()-start} seconds\nRaster info: {raster_info}")
    row_size = raster_info["shape"][0] * raster_info["shape"][1]
    print(f"Begin to chunkchunk {row_size} rows")
    start = time()
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    column_types, chunk_idxs = None, []

    for target_xy, ifrom, ito in get_target_batch(factors, row_size, chunk_size, begin=0):
        print(f"""Chunk from idx {ifrom} to {ito-1}
        XY = {target_xy.shape},
        Columns = {target_xy.dtypes}
        Non Check = {np.count_nonzero(np.isnan(target_xy))}
        Infinity Check = { np.count_nonzero(np.isinf(target_xy))}
        """)
        # target_xy is pd.DataFrame
        fn = os.path.join(save_to, f"target_{int(ifrom)}_{int(ito-1)}.csv")
        target_xy.to_csv(fn, index=False)

        if column_types is None:
            column_types = target_xy.dtypes.apply(lambda x: x.name).to_dict()
        chunk_idxs.append((int(ifrom), int(ito-1)))

    print(f"Time cost: {time()-start} seconds")

    return column_types, raster_info, mask, chunk_idxs
    

# ref: [Reading and Writing Pandas DataFrames in Chunks](https://janakiev.com/blog/python-pandas-chunks/)
def bigdata_predict(clf, model_label, target_path, dtype, chunk_size, clf_pred_dir, batch_size=10**6):
    # load target
    target_xs_iter = pd.read_csv(target_path, dtype=dtype, chunksize=batch_size)

    start = time()
    row_cnt = 0
    target_y_pred = []
    for i, target_xs in enumerate(target_xs_iter):
        if i == 0:
            print(f"""[{model_label}] Handling target chunck {target_path}...

            Target:
                Columns = {target_xs.dtypes}
            """)

        row_cnt += target_xs.shape[0]

        # 1 [TC] process categorical factors
        target_xs, _ = get_X_Y(target_xs)
        if i == 0:
            print(f"""
            [Feature Info]
            Target:
                Columns = {target_xs.dtypes}

            Get XY Time Cost: {time()-start}""")

        # 2.1 make sure the feature names match the classifier feature name
        for feature in clf.feature_names_in_:
            if feature not in target_xs.columns:
                #print(f"Feature {feature} is missing in target, set default to 0.")
                target_xs[feature] = 0
        # 2.2 [TC] predict
        target_xs = target_xs[clf.feature_names_in_]
        target_y_pred_chunk = clf.predict_proba(target_xs) # ndarray of shape (n_samples, n_classes)
        target_y_pred.append(target_y_pred_chunk)

    print(f"Predict Time Cost: {time()-start}, Row cnt: {row_cnt}")

    # 2.3 Save
    pred_chunk_fn = os.path.join(clf_pred_dir, target_path.split("/")[-1].replace("target", "pred"))
    target_y_pred = np.concatenate(target_y_pred, axis=0)
    np.savetxt(pred_chunk_fn, target_y_pred)
    print(f"[DONE] save chunk predict result (shape: {target_y_pred.shape}) to {pred_chunk_fn}")

def bigdata_mapping(model_label, clf_pred_dir, save_to, raster_info, mask, chunk_idxs):
    print(f"""[DIR]
        Classifier Predict Result dir: {clf_pred_dir}
        Result path: {save_to}
        Chunk_idx: {chunk_idxs}
    """)

    # batch handling (TC=Time Consuming)
    startM = time()

    # read result and concat
    target_y_pred = []
    for ifrom, ito in chunk_idxs:
        print(f"Loading chunk {ifrom} ~ {ito}")
        pred_chunk_fn = os.path.join(clf_pred_dir, f"pred_{ifrom}_{ito}.csv")
        pred_chunk = np.loadtxt(pred_chunk_fn)
        # note that only pred_chunk[:,1] is used
        target_y_pred.append(pred_chunk)
    
    # 2. generate and save the LSM
    start = time()
    target_y_pred = np.concatenate(target_y_pred, axis=0)
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

def Lombardy_Mapping(clfs, need_chunk=True, need_predict=True, column_types=None):
    ld_dir = base_dir + r"Lombardy/"
    factor_dir = ld_dir+"1.factors/"
    result_path = ld_dir+"3.results/"

    chunk_save_to = os.path.join(factor_dir, "chunk")
    raster_info, mask, chunk_idxs = None, None, []
    if need_chunk:
        column_types, raster_info, mask, chunk_idxs = bigdata_chunk(factor_dir, chunk_save_to, chunk_size=PROCESS_BATCH_SIZE)
    else:
        print("Skip Chunk!!!")
        # get raster_info, mask
        raster_info, mask = get_raster_meta(os.path.join(factor_dir, 'dtm.tif'))
        row_size = raster_info["shape"][0] * raster_info["shape"][1]
        for ifrom in range(0, row_size, PROCESS_BATCH_SIZE):
            ito = min(ifrom+PROCESS_BATCH_SIZE, row_size)
            chunk_idxs.append((int(ifrom), int(ito-1)))

        print(f"""[CONFIG]
        Factor dir: {factor_dir}
        Result path: {result_path}
        
        Chunk size: {PROCESS_BATCH_SIZE}
        Raster info: {raster_info}
        Row Size: {row_size}
        """)
    
    # load clfs
    print(f"""# Predict and Mapping
    [INPUT DATA]
    Column info: {column_types}
    """)

    
    clf_pred_dir = {}
    load_clfs = {}
    for model_label, model_path in clfs.items():
        start = time()
        print(f"## {model_label}")

        clf_pred_dir = os.path.join(result_path, model_label)
        if not os.path.exists(clf_pred_dir):
            os.makedirs(clf_pred_dir)

        if need_predict and model_label not in (BAGGING_MODEL_LABLE, RANDOMFOREST_MODEL_LABLE):
            clf = load_model(model_path)
            print(f"[DONE] Load model {model_label}. Features are {clf.feature_names_in_}")
            
            for ifrom, ito in chunk_idxs:
                print(f"\nFrom idx {ifrom} to idx {ito}")
                chunk_target_path = os.path.join(chunk_save_to, f"target_{ifrom}_{ito}.csv")

                bigdata_predict(clf, model_label, chunk_target_path, column_types, chunk_size=PROCESS_BATCH_SIZE, clf_pred_dir=clf_pred_dir)
        else:
            print("Skip Predict!!")
        print(f"Predict Time cost: {time()-start}")

        gc.collect()
        
        print("### Mapping")
        bigdata_mapping(model_label, clf_pred_dir, clf_pred_dir, raster_info, mask, chunk_idxs)

        gc.collect()

    


def check_factors():
    ld_dir = base_dir + r"Lombardy/"
    factor_dir = ld_dir+"1.factors"
    #load_rasters(factor_dir)
    get_factors_meta(factor_dir)

if __name__ == '__main__':
    clfs_dir = base_dir + r"ValChiavenna/3.results/2nd_with/"
    clfs = {
        #BAGGING_MODEL_LABLE: clfs_dir+"Valchiavenna_Bagging.pkl",
        #RANDOMFOREST_MODEL_LABLE: clfs_dir+"Valchiavenna_Forests of randomized trees.pkl",
        ADABOOST_MODEL_LABLE: clfs_dir+"Valchiavenna_AdaBoost.pkl",
        GRADIENT_TREE_BOOSTING_MODEL_LABLE: clfs_dir+"Valchiavenna_Gradient Tree Boosting.pkl",
    }

    ld_dir = base_dir + r"Lombardy/"
    #factor_dirs = ld_dir+"1.factors"
    #factor_dirs = preparation(ld_dir+"1.factors")
    factor_dirs =  []
    
    testset_path = ld_dir+"/2.samples/Lombardy_LSM_testing_points.csv"
    #preparation(ld_dir+"1.factors", M=1, N=5)
    #Lombardy(clfs, ld_dir, factor_dirs, testset_path)
    #Lombardy(clfs, ld_dir, factor_dirs, "")
    Lombardy_Mapping(clfs,
        need_chunk=False,
        column_types={'dtm': 'float32', 'east': 'float32', 'ndvi': 'float32', 'north': 'float32', 'faults': 'float32', 'rivers': 'float32', 'roads': 'float32', 'dusaf': 'float32', 'plan': 'float32', 'profile': 'float32', 'twi': 'float32'}
    )
    #check_factors()