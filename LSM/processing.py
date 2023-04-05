import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
np.set_printoptions(precision=14)
from matplotlib import pyplot as plt

from sklearn import metrics
from time import time
from datetime import datetime
import os
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from config import *
from preprocessing import *
from models import *
from evaluation import *
from utils import _modelname2filename, load_model

# TOAVOID: https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model
os.environ['JOBLIB_TEMP_FOLDER'] = JOBLIB_TEMP_FOLDER
    
############################ MAPPING ############################ 
def plot_prediction(target_pred, raster_info, mask, title, save_to=None):
    responsesG = target_pred.reshape(raster_info['shape'])
    outmapG = np.where(mask==1,responsesG, np.nan)

    plt.figure(figsize=(10,10))
    plt.imshow(outmapG,cmap='RdYlGn_r',vmin=0,vmax= 1)
    plt.title(title)
    plt.colorbar()
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()
    
    return outmapG

def plot_LSM_prediction(target_pred, raster_info, mask, model_label, save_to=None, LSM_idx=1):
    outmap = {}
    # LSM
    title = f'Probability of Landslide - {model_label}'
    save_lsm_to = os.path.join(save_to, f"LSM_{_modelname2filename(model_label)}")
    outmap['LSM'] = plot_prediction(target_pred[:,LSM_idx], raster_info, mask, title, save_lsm_to)

    # NLZ
    if target_pred.shape[1] > 1:
        title = f'Probability of No Landslide - {model_label}'
        save_nlz_to = os.path.join(save_to, f"NLZ_{_modelname2filename(model_label)}")
        outmap['NLZ'] = plot_prediction(target_pred[:,0], raster_info, mask, title, save_nlz_to)

        # TOCHECK
        print(f"LSM+NLZ == {np.unique(outmap['LSM']+outmap['NLZ'])}")

    return outmap
    
def mapping(target_xs, clf, model_label,raster_info, mask, save_to):
    #print(f"[Mapping] Model: {model_label}")
    target_y_pred = clf.predict_proba(target_xs)
    outmap = plot_LSM_prediction(target_y_pred, raster_info, mask, model_label, save_to)
    # save
    try:
        with rasterio.open(
            os.path.join(save_to, f'LSM_{model_label}.tif'),
            'w',
            driver='GTiff',
            height=outmap['LSM'].shape[0],
            width=outmap['LSM'].shape[1],
            count=2,
            dtype=outmap['LSM'].dtype,
            crs=raster_info['crs'],
            transform=raster_info['transform'],
        ) as dst:
            dst.write(outmap['LSM'], 1)
            dst.write(outmap['NLZ'], 2)
    except Exception as e:
        print(e, f'\nResave to current')
        with rasterio.open(f'LSM_{model_label}.tif',
            'w',
            driver='GTiff',
            height=outmap['LSM'].shape[0],
            width=outmap['LSM'].shape[1],
            count=2,
            dtype=outmap['LSM'].dtype,
            crs=raster_info['crs'],
            transform=raster_info['transform'],
        ) as dst:
            dst.write(outmap['LSM'], 1)
            dst.write(outmap['NLZ'], 2)

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
        target_y_pred.append(pred_chunk[:,1].reshape((-1,1)))
    
    # 2. generate and save the LSM
    start = time()
    target_y_pred = np.concatenate(target_y_pred, axis=0)
    print(f"the shape of target_y_pred is {target_y_pred.shape}")
    outmap = plot_LSM_prediction(target_y_pred, raster_info, mask, model_label, save_to, LSM_idx=0)
    print(f"[{model_label}] Plot Time Cost: {time()-start}")
    # save
    try:
        with rasterio.open(
            os.path.join(save_to, f'LSM_{model_label}.tif'),
            'w',
            driver='GTiff',
            height=outmap['LSM'].shape[0],
            width=outmap['LSM'].shape[1],
            count=len(outmap),
            dtype=outmap['LSM'].dtype,
            crs=raster_info['crs'],
            transform=raster_info['transform'],
        ) as dst:
            dst.write(outmap['LSM'], 1)
            if 'NLZ' in outmap:
                dst.write(outmap['NLZ'], 2)
    except Exception as e:
        tmp_path = r"/tmp/LSM"
        print(e, f'\nResave to {tmp_path}')
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        with rasterio.open(
            os.path.join(tmp_path, f'LSM_{model_label}.tif'),
            'w',
            driver='GTiff',
            height=outmap['LSM'].shape[0],
            width=outmap['LSM'].shape[1],
            count=len(outmap),
            dtype=outmap['LSM'].dtype,
            crs=raster_info['crs'],
            transform=raster_info['transform'],
        ) as dst:
            dst.write(outmap['LSM'], 1)
            if 'NLZ' in outmap:
                dst.write(outmap['NLZ'], 2)


    print(f"\n\n[{model_label}] Time Cost: {time()-startM}")

# ref: [Reading and Writing Pandas DataFrames in Chunks](https://janakiev.com/blog/python-pandas-chunks/)
def bigdata_predict(clf, model_label, target_path, dtype, chunk_size, clf_pred_dir, batch_size=PREDICT_BATCH_SIZE):
    # load target
    target_xs_iter = pd.read_csv(target_path, dtype=dtype, chunksize=batch_size)

    start = time()
    row_cnt = 0
    target_y_pred = []
    for i, target_xs in enumerate(target_xs_iter):
        if i == 0:
            print(f"""[{model_label}] {datetime.now()} Handling target chunck {target_path}...

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
        print(f"...{100*(i+1)*batch_size/chunk_size:.4f}%")

    print(f"[{model_label}] Predict Time Cost: {time()-start}, Row cnt: {row_cnt}")

    # 2.3 Save
    pred_chunk_fn = os.path.join(clf_pred_dir, target_path.split("/")[-1].replace("target", "pred"))
    target_y_pred = np.concatenate(target_y_pred, axis=0)
    np.savetxt(pred_chunk_fn, target_y_pred)
    print(f"[DONE] save chunk predict result (shape: {target_y_pred.shape}) to {pred_chunk_fn}")




############################ PUT THEM TOGETHER ############################ 

import joblib

def LSM(ROI_label, factor_dir, trainset_path, testset_path, result_path, preprocess=None, algorithms=ALGORITHMS):

    print(f"""# {ROI_label} ({datetime.now()})
    [DIR]
    Factor dir: {factor_dir}
    Train set path: {trainset_path}
    Test set path: {testset_path}
    Result path: {result_path}
    """)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # 1. input data
    print("## Input Data")
    ## targets
    start = time()
    target_xs, raster_info, mask = get_targets(factor_dir)
    ## training and testing samples
    trainingPoints, testingPoints = get_train_test(
        trainset_path,
        testset_path
    )
    end = time()
    print(f"""
    [INPUT]
    Training points:
        Shape = {trainingPoints.shape},
        Columns = {trainingPoints.columns}
        Column Types = {trainingPoints.dtypes}
    
    Testing points:
        Shape = {testingPoints.shape},
        Columns = {testingPoints.columns}
        Column Types = {testingPoints.dtypes}

    Target:
        X = {target_xs.shape},
        Columns = {target_xs.columns}
        Column Types = {target_xs.dtypes}
    """)
    print("\n\nTime Cost:  %.3f seconds" % (end-start))
    # 2. preprocessing
    print("## Preprocessing")
    start = time()
    ## get features and label
    train_xs, train_y = get_X_Y(trainingPoints)
    test_xs, test_y = get_X_Y(testingPoints)
    
    ### TOAVOID: The feature names should match those that were passed during fit.
    test_xs = test_xs[train_xs.columns]
    target_xs = target_xs[train_xs.columns]
    
    #if preprocess:
        #train_xs, train_y, test_xs, test_y, target_xs = preprocess(train_xs, train_y, test_xs, test_y, target_xs)
    end = time()
    print(f"""
    [AFTER PREPROCESSING]
    Training points shape:
        X = {train_xs.shape},
        Y = {train_y.shape},
        Columns = {train_xs.columns}
        Column Types = {train_xs.dtypes}
        Non Check = {np.count_nonzero(np.isnan(train_xs))}
        Infinity Check = { np.count_nonzero(np.isinf(train_xs))}
    
    Testing points shape:
        X = {test_xs.shape},
        Y = {test_y.shape},
        Columns = {test_xs.columns}
        Column Types = {test_xs.dtypes}
        Non Check = {np.count_nonzero(np.isnan(test_xs))}
        Infinity Check = { np.count_nonzero(np.isinf(test_xs))}

    Target:
        X = {target_xs.shape},
        Columns = {target_xs.columns}
        Column Types = {target_xs.dtypes}
        Non Check = {np.count_nonzero(np.isnan(target_xs))}
        Infinity Check = { np.count_nonzero(np.isinf(target_xs))}
    """)
    print("\n\nPreprocessing Time Cost:  %.3f seconds" % (end-start))

    # 3. Processing
    print("## Processing and Evaluation")
    clfs = {}
    for alg_label, alg in algorithms.items():
        start = time()
        ## train
        clf = alg(train_xs, train_y, test_xs, test_y, save_to=result_path)
        clfs[alg_label] = clf
        print("Training and Evaluation Time cost: %.3f seconds" % (time()-start))
        ## save model
        joblib.dump(clf, os.path.join(result_path, f'{ROI_label}_{alg_label}.pkl'))
        ## mapping
        startM = time()
        mapping(target_xs, clf, alg_label, raster_info, mask, result_path)
        endM = time()
        print("Mapping Time cost: %.3f seconds" % (endM-startM))
        end = time()
        print("\n\n Total Time Cost:  %.3f seconds\n\n" % (end-start))
        
        
    return clfs


def LSM_PredictMap_WithChunk(clfs, factor_dir, result_path, need_chunk=True, column_types=None, chunk_size=PROCESS_BATCH_SIZE, pred_batch_size=PREDICT_BATCH_SIZE):

    chunk_save_to = os.path.join(factor_dir, "chunk")
    raster_info, mask, chunk_idxs = None, None, []
    if need_chunk:
        column_types, raster_info, mask, chunk_idxs = bigdata_chunk(factor_dir, chunk_save_to, chunk_size=chunk_size)
    else:
        print("Skip Chunk!!!")
        # get raster_info, mask
        raster_info, mask = get_raster_meta(os.path.join(factor_dir, 'dtm.tif'))
        row_size = raster_info["shape"][0] * raster_info["shape"][1]
        for ifrom in range(0, row_size, chunk_size):
            ito = min(ifrom+chunk_size, row_size)
            chunk_idxs.append((int(ifrom), int(ito-1)))

        print(f"""[CONFIG]
        Factor dir: {factor_dir}
        Result path: {result_path}
        
        Chunk size: {chunk_size}
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
    for model_label, model_info in clfs.items():
        skip = model_info.get("skip", False)
        if skip:
            print(f"Skip model {model_label}")
            continue

        model_path = model_info['path']
        start = time()
        print(f"## {model_label}")

        clf_pred_dir = os.path.join(result_path, model_label)
        if not os.path.exists(clf_pred_dir):
            os.makedirs(clf_pred_dir)

        skip_predict = model_info.get("skip_predict", False)
        predict_from = model_info.get("predict_from", 0)

        if not skip_predict:
            clf = load_model(model_path)
            print(f"[DONE] Load model {model_label}. Features are {clf.feature_names_in_}")
            
            chunk_target_paths = []
            for ifrom, ito in chunk_idxs:
                if ifrom < predict_from:
                    print(f"Skip {ifrom} ~ {ito}, because predict from {predict_from}")
                    continue

                print(f"\nFrom idx {ifrom} to idx {ito}")
                #chunk_target_path = os.path.join(chunk_save_to, f"target_{ifrom}_{ito}.csv")
                chunk_target_paths.append(os.path.join(chunk_save_to, f"target_{ifrom}_{ito}.csv"))
                if len(chunk_target_paths) == 3:
                    from joblib import Parallel, delayed
                    Parallel(n_jobs=len(chunk_target_paths))(delayed(bigdata_predict)(clf, model_label, chunk_target_path, column_types, chunk_size=chunk_size, clf_pred_dir=clf_pred_dir,batch_size=pred_batch_size) for chunk_target_path in chunk_target_paths)
                    #bigdata_predict(clf, model_label, chunk_target_path, column_types, chunk_size=chunk_size, clf_pred_dir=clf_pred_dir,batch_size=pred_batch_size)
                    chunk_target_paths = []
            if chunk_target_paths:
                from joblib import Parallel, delayed
                Parallel(n_jobs=len(chunk_target_paths))(delayed(bigdata_predict)(clf, model_label, chunk_target_path, column_types, chunk_size=chunk_size, clf_pred_dir=clf_pred_dir,batch_size=pred_batch_size) for chunk_target_path in chunk_target_paths)

        else:
            print("Skip Predict!!")
        print(f"[{model_label}] Predict Time cost: {time()-start}")

        gc.collect()
        
        print("### Mapping")
        if not model_info.get("skip_mapping", False):
            bigdata_mapping(model_label, clf_pred_dir, clf_pred_dir, raster_info, mask, chunk_idxs)

        gc.collect()

