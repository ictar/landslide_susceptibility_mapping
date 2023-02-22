import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from matplotlib import pyplot as plt
import joblib

from time import time
import os
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from config import *

# TOAVOID: https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model
os.environ['JOBLIB_TEMP_FOLDER'] = JOBLIB_TEMP_FOLDER

def _modelname2filename(mn):
    mn = mn.replace(":", " ").replace("/", " ")
    return "_".join([s for s in mn.split() if s])
    
# load 
def load_rasters(layer_dir):
    factors, meta = {}, None
    for rLayer in rFactors:
        with rasterio.open(f'{layer_dir}/{rLayer}.tif') as ds:
            factors[rLayer] = ds.read(1)
            # after the operation below, the dtype will become 'float64', because np.nan is introduced
            # ref: https://appdividend.com/2022/01/28/np-nan/
            factors[rLayer] = np.where(factors[rLayer]==ds.nodatavals,np.nan,factors[rLayer])
            
            # TOFIX: no space. handling some special type to reduce space
            if factors[rLayer].dtype.name in DTYPE_MAPPING:
               factors[rLayer] = factors[rLayer].astype(DTYPE_MAPPING[factors[rLayer].dtype.name])

            print(f"""[INFO of {rLayer}]
            {list(zip(ds.indexes, ds.dtypes, ds.nodatavals))}
            Factor shape: {factors[rLayer].shape}
            Factor type: {factors[rLayer].dtype}
            Non Check: {np.count_nonzero(np.isnan(factors[rLayer]))}
            Infinity Check: { np.count_nonzero(np.isinf(factors[rLayer]))}
            Min: {np.min(factors[rLayer])}
            Max: {np.max(factors[rLayer])}
            """)

            if meta is None: meta = ds.meta 

    # TOFIX: no space.    
    mask = np.where(np.isnan(factors['dtm']),np.nan,1).astype(np.float16)
    
    return factors, meta, mask

def get_train_test(train_csv, test_csv):
    trainingPoints, testingPoints = None, None
    if train_csv: trainingPoints = pd.read_csv(train_csv)
    if test_csv: testingPoints = pd.read_csv(test_csv)
    return trainingPoints, testingPoints

# 处理 roads/rivers/faults/dusaf 信息
import timeit
def categorical_factors_preprocessing(df):
    transt = {1: 50, 2: 100, 3: 250, 4: 500, 5: 9999}
    #print(f"[BEFORE] river: {df.rivers.unique()}, roads: {df.roads.unique()}, faults: {df.faults.unique()}, dusaf: {df.dusaf.unique()}")
    #start_time = timeit.default_timer()
    #df['rivers'] = df.apply(lambda x: transt.get(x['rivers'], x['rivers']), axis=1)#.astype('Int32')
    #df['roads'] = df.apply(lambda x: transt.get(x['roads'], x['roads']), axis=1)#.astype('Int32')
    #df['faults'] = df.apply(lambda x: transt.get(x['faults'], x['faults']), axis=1)#.astype('Int32')
    #df['dusaf'] = df['dusaf'].astype('Int32')
    for k, v in transt.items():
        for column in ['rivers', 'roads', 'faults']:
            df.loc[df[column]==k, column] = v
    #print(f"[AFTER ({timeit.default_timer()-start_time})] river: {df.rivers.unique()}, roads: {df.roads.unique()}, faults: {df.faults.unique()}, dusaf: {df.dusaf.unique()}")
    
    return df

# 增加定性数据
# ref: https://www.datalearner.com/blog/1051637141445141
def add_categorical(df):
    for cat in categorical_factors:
        df[cat] = df[cat].astype('object')
        df = df.join(pd.get_dummies(df[cat], prefix=cat))
        # rename column names
        # testing_data_df.rename(columns={"Hazard": "True",}, inplace=True)
        prefix_ = cat+"_"
        #print("Column Mapping: ", {cn: f"{prefix_}{int(float(cn[len(prefix_):]))}" for cn in list(df.columns) if cn.startswith(prefix_)})
        df.rename(
            columns={cn: f"{prefix_}{int(float(cn[len(prefix_):]))}"
                     for cn in list(df.columns) if cn.startswith(prefix_)},
            inplace=True
        )
             
    return df.drop(columns=categorical_factors)

# 获取 features 和 label
def get_X_Y(df):
    # transfer data
    df = categorical_factors_preprocessing(df)
    # drop fields which are not needed for the classification
    dropped_columns = list(set(df.columns)-set(rFactors))
    # TOFIX: dusaf = -99999
    # 当前处理的数据，dusaf 取值范围为 [11,51]，直接删掉训练集中对应项
    #df = df[df.dusaf != -99999]
    X, Y = df.drop(columns=dropped_columns), None
    if "hazard" in df.columns: Y = df.hazard  
    # handle categorical field
    X = add_categorical(X)
    # TOFIX: input X cannot contain NaN
    X=X.fillna(NaN)
    # TOFIX: save space
    X = X.astype(MODEL_DATA_COLUMN_TYPES)   
    return X, Y

# return pd.dataframes
def get_targets(layer_dir):
    factors, meta, mask = load_rasters(layer_dir)
    target, raster_info ={}, None

    for layer in continuous_factors:
        target[layer] = factors[layer].flatten()
        if raster_info is None:
            raster_info = {
                "transform": meta['transform'],
                "shape": factors[layer].shape,
                "crs": meta['crs'],
            }

    for cat in categorical_factors:    
        target[cat] = factors[cat].flatten()

    target_xs, _ = get_X_Y(pd.DataFrame(target))
    
    return target_xs, raster_info, mask

# SIZE = raster_info["shape"][0] * raster_info["shape"][1]
def get_target_batch(factors, SIZE, batch_size, begin=0):   
    for ifrom in range(begin, SIZE, batch_size):
        ito = min(ifrom+batch_size, SIZE)
        yield (pd.DataFrame({layer: factors[layer][ifrom:ito] for layer in factors}), ifrom, ito)

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

############################ EXPLORATION ############################ 

def plotit(x, title, vmin, cmap, save_to=None):
    ax = plt.subplot()
    im = ax.imshow(x,cmap=cmap,vmin=vmin)
    plt.title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()

def get_factors_meta(layer_dir):
    for rLayer in rFactors:
        with rasterio.open(f'{layer_dir}/{rLayer}.tif') as ds:
            print(f"""[{rLayer}]
            dtype: {ds.meta['dtype']}
            nodata: {ds.meta['nodata']}
            width: {ds.meta['width']}
            height: {ds.meta['height']}
            crs: {ds.meta['crs']}
            transform: {ds.meta['transform']}
            """)
            # save image
            img_name = os.path.join(layer_dir, f"{rLayer}.png")

            outmap = ds.read(1)
            if rLayer in categorical_factors:
                print(f"unique value: {np.unique(outmap)}")
            outmap = np.where(outmap==ds.nodatavals,np.nan,outmap)
            min_val = np.nanmin(outmap)
            max_val = np.nanmax(outmap)
            print(f"""
            min value: {min_val}
            max value: {max_val}
            """)
            continue

            plt.figure(figsize=(10,10))
            plt.imshow(outmap,cmap='RdYlGn_r',vmin=min_val,vmax=max_val)
            plt.title(f'Factor - {rLayer}')
            plt.colorbar()
            plt.savefig(img_name)

            del outmap
        
        del ds
        gc.collect()



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

############################ MODELLING (with validation) ############################ 

from sklearn.ensemble import BaggingClassifier
BAGGING_MODEL_LABLE = "Bagging"
'''
estimator: The base estimator to fit on random subsets of the dataset.
                If None, then the base estimator is a DecisionTreeClassifier.
n_estimators: The number of base estimators in the ensemble.
n_jobs: The number of jobs to run in parallel for both fit and predict.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors. 
'''
DEFAULT_RANDOMFOREST_MODEL_PARAS = {"n_estimators": 100, "n_jobs": -1}
def ensemble_bagging(X, Y, Xtest, Ytest, model_paras=DEFAULT_RANDOMFOREST_MODEL_PARAS, save_to=None):
    # train
    clf = BaggingClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model: Ensemble / Bagging", save_to)
    
    return clf

from sklearn.ensemble import RandomForestClassifier
RANDOMFOREST_MODEL_LABLE = "Forests of randomized trees"
'''
n_estimators: The number of trees in the forest. (指定森林中树的颗数，越多越好，只是不要超过内存)
                default: 100
max_depth: The maximum depth of the tree. 
            If None, then nodes are expanded until all leaves are pure or 
            until all leaves contain less than min_samples_split samples.
n_jobs: The number of jobs to run in parallel. (指定并行使用的进程数)
        -1 means using all processors.
'''
DEFAULT_RANDOMFOREST_MODEL_PARAS = {'n_estimators': 100, 'n_jobs': -1,}
def ensemble_randomforest(X, Y, Xtest, Ytest, model_paras=DEFAULT_RANDOMFOREST_MODEL_PARAS, save_to=None):
    # train
    print(f"[BEGIN] Training using parameters {model_paras}")
    clf = RandomForestClassifier(**model_paras).fit(X, Y)
    print(f"[END] Training using parameters {model_paras}")
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Random Forest", save_to)
    
    return clf

from sklearn.ensemble import AdaBoostClassifier
ADABOOST_MODEL_LABLE = "AdaBoost"
'''
estimator: object. The base estimator from which the boosted ensemble is built.
                    Support for sample weighting is required, as well as proper classes_ and n_classes_ attributes.
                    If None, then the base estimator is DecisionTreeClassifier initialized with max_depth=1.

                    New in version 1.2: base_estimator was renamed to estimator.
n_estimators: int. The maximum number of estimators at which boosting is terminated.
                    In case of perfect fit, the learning procedure is stopped early. Values must be in the range [1, inf).

learning_rate: float, default=1.0
                Weight applied to each classifier at each boosting iteration.
                A higher learning rate increases the contribution of each classifier.
                There is a trade-off between the learning_rate and n_estimators parameters.
                Values must be in the range (0.0, inf).
'''
DEFAULT_ADABOOST_MODEL_PARAS = {'n_estimators': 300, 'learning_rate': 0.8}
def ensemble_adaboost(X, Y, Xtest, Ytest,
                      model_paras=DEFAULT_ADABOOST_MODEL_PARAS, save_to=None):
    # train
    clf = AdaBoostClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Adaboost", save_to)
    
    return clf


from sklearn.ensemble import GradientBoostingClassifier
GRADIENT_TREE_BOOSTING_MODEL_LABLE = "Gradient Tree Boosting"
DEFAULT_GRADIENTBOOSTING_MODEL_PARAS = {'n_estimators': 300, 'learning_rate': 0.8}
def ensemble_gradienttreeboosting(X, Y, Xtest, Ytest,
                                  model_paras=DEFAULT_GRADIENTBOOSTING_MODEL_PARAS, save_to=None):
    # train
    clf = GradientBoostingClassifier(**model_paras).fit(X, Y)
    # test
    Ytest_pred = clf.predict_proba(Xtest)
    ## test result evaluation
    evaluation_report(Ytest, Ytest_pred, "Model Ensemble / Gradient Tree Boosting", save_to)
    return clf

def load_model(model_path):
    return joblib.load(model_path)

############################ MAPPING ############################ 
def plot_LSM_prediction(target_pred, raster_info, mask, model_label, save_to=None):
    responsesG = target_pred[:,1].reshape(raster_info['shape'])
    outmapG = np.where(mask==1,responsesG, np.nan)

    plt.figure(figsize=(10,10))
    plt.imshow(outmapG,cmap='RdYlGn_r',vmin=0,vmax= 1)
    plt.title(f'Probability of Landslide class - {model_label}')
    plt.colorbar()
    if save_to:
        plt.savefig(os.path.join(save_to, f"LSM_{_modelname2filename(model_label)}"))
    else:
        plt.show()
    
    return outmapG

def mapping(target_xs, clf, model_label,raster_info, mask, save_to):
    #print(f"[Mapping] Model: {model_label}")
    target_y_pred = clf.predict_proba(target_xs)
    outmap = plot_LSM_prediction(target_y_pred, raster_info, mask, model_label, save_to)
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
    print(f"[{model_label}] Plot Time Cost: {time()-start}")
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

    print(f"[{model_label}] Predict Time Cost: {time()-start}, Row cnt: {row_cnt}")

    # 2.3 Save
    pred_chunk_fn = os.path.join(clf_pred_dir, target_path.split("/")[-1].replace("target", "pred"))
    target_y_pred = np.concatenate(target_y_pred, axis=0)
    np.savetxt(pred_chunk_fn, target_y_pred)
    print(f"[DONE] save chunk predict result (shape: {target_y_pred.shape}) to {pred_chunk_fn}")


############################ EVALUATION ############################ 
from sklearn import metrics
def optimalROCthreshold(y_true, y_pred, model_label, save_to=None):
    '''
    Function to automatically determine the optimal threshold based on computer
    probability and ROC curve.

    y_true: the y array from the testing dataset
    y_pred: the predicted probability of test_X
    model_label: (str) the name of the model needed for the plot

    '''
    print('\n ###  CLASSIFICATION BASED ON OPTIMAL THRESHOLD FROM ROC ### \n')

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred[:,1])

    # get the best threshold using the so called the Youden’s J statistic.
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]

    print(f"""True Positive Rate = {tpr[ix]}
False Positive Rate = {fpr[ix]}""")

    # plot the roc curve for the model with the best threshold
    plt.figure()
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, color='black',label=model_label)
    plt.scatter(fpr[ix], tpr[ix], color='red', label='Best Threshold')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True,'major',linewidth=0.3)
    plt.legend()
    # show the plot
    if save_to:
        tmp_save_path = os.path.join(save_to, f"optimalROCthreshold_{_modelname2filename(model_label)}")
        plt.savefig(tmp_save_path)
        print(f"![optimalROCthreshold]({tmp_save_path}.png)")
    else:
        plt.show()

    # get the new Y predicted using the best threshold
    y_predNewThreshold=[]
    for num,n in enumerate(y_pred[:,1]):
        if y_pred[:,1][num]<best_thresh:
            n=0 
        else:
            n=1
        y_predNewThreshold.append(n)

    y_predRnewT = y_predNewThreshold

    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(y_true,  y_predRnewT)
    FPR = 1- metrics.recall_score(y_true,  y_predRnewT, pos_label = 0)

    print(f"""
Classification report
{metrics.classification_report(y_true, y_predRnewT)}

Confussion matrix
{metrics.confusion_matrix(y_true,  y_predRnewT)}

Testing Accuracy 
Accuracy Score: {metrics.accuracy_score(y_true, y_predRnewT)}

Precision: {precision}
Recall: {recall}
F1 score: {fscore}
False Positive Rate: {FPR}

Best Threshold = {best_thresh}

AUCROC
{metrics.roc_auc_score(y_true, y_predRnewT)}
""")

def optimalPRCthreshold(y_true, y_pred, model_label, save_to=None):
    '''
    Function to automatically determine the optimal threshold based on computer
    probability and PRC curve.

    y_true: the y array from the testing dataset
    y_pred: the predicted probability of test_X
    model_label: (str) the name of the model needed for the plot

    '''
    print('\n ###  CLASSIFICATION BASED ON OPTIMAL THRESHOLD FROM PRC ### \n')

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred[:,1])
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    best_thresh = thresholds[ix]
    print(f'Best Threshold={best_thresh}, F-Score={fscore[ix]}' )

    # plot the roc curve for the model
    no_skill = len(y_true[y_true==1]) / len(y_true)
    plt.figure()
    plt.plot([0,1], [1,0], linestyle='--', label='No Skill')
    plt.plot(recall, precision, color='black', label=model_label)
    plt.scatter(recall[ix], precision[ix],  color='red', label='Best Threshold')
    plt.grid(True,'major',linewidth=0.3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # show the plot
    if save_to:
        tmp_save_path = os.path.join(save_to, f"optimalPRCthreshold{_modelname2filename(model_label)}")
        plt.savefig(tmp_save_path)
        print(f"![optimalPRCthreshold]({tmp_save_path}.png)")
    else:
        plt.show()

    # get the new Y predicted using the best threshold
    y_predNewThreshold=[]
    for num,n in enumerate(y_pred[:,1]):
        if y_pred[:,1][num]<best_thresh:
            n=0 
        else:
            n=1
        y_predNewThreshold.append(n)

    y_predRnewT = y_predNewThreshold

    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(y_true,  y_predRnewT)
    FPR = 1- metrics.recall_score(y_true,  y_predRnewT, pos_label = 0)

    print(f"""
Classification report: {metrics.classification_report(y_true, y_predRnewT)}

Confussion matrix:
{metrics.confusion_matrix(y_true,  y_predRnewT)}

Testing Accuracy 
Accuracy Score: {metrics.accuracy_score(y_true, y_predRnewT)}

Precision: {precision}
Recall: {recall}
F1 score: {fscore}
False Positive Rate: {FPR}

Best Threshold={best_thresh}

AUCROC: {metrics.roc_auc_score(y_true, y_predRnewT)}""")

def evaluation_report(y_true, y_pred, model_label, save_to):
    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(
        y_true, [round(num) for num in y_pred[:,1]])
    FPR = 1- metrics.recall_score(y_true,  [round(num) for num in y_pred[:,1]], pos_label = 0)
    
    print(f"""
## Evaluation for model {model_label}
    
    [Classification report]
    {metrics.classification_report(y_true,  [round(num) for num in y_pred[:,1]])}
    
    [Testing Accuracy]
    
    Confusion matrix:
    {metrics.confusion_matrix(y_true, [round(num) for num in y_pred[:,1]])}
    
    Accuracy Score: {metrics.accuracy_score(y_true,  [round(num) for num in y_pred[:,1]])}
    
    Precision: {precision}
    Recall: {recall}
    F1 score: {fscore}
    False Positive Rate: {FPR}
    
    [AUCROC]
    {metrics.roc_auc_score(y_true,  [round(num) for num in y_pred[:,1]])}
    """)
    # plot ROC and PRC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred[:,1])

    # plot the roc curve for the model with the best threshold
    plt.figure()
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, color='black',label=model_label)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True,'major',linewidth=0.3)
    plt.legend()
    tmp_save_path = os.path.join(save_to, f"ROC_{_modelname2filename(model_label)}")
    plt.savefig(tmp_save_path)
    print(f"![ROC]({tmp_save_path}.png)")

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred[:,1])
    # plot the roc curve for the model
    plt.figure()
    plt.plot([0,1], [1,0], linestyle='--', label='No Skill')
    plt.plot(recall, precision, color='black', label=model_label)
    plt.grid(True,'major',linewidth=0.3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    tmp_save_path = os.path.join(save_to, f"PRC{_modelname2filename(model_label)}")
    plt.savefig(tmp_save_path)
    print(f"![PRC]({tmp_save_path}.png)")

    print("\t[Optimal ROC Threshold]")
    optimalROCthreshold(y_true, y_pred, model_label, save_to)
    
    print("\t[Optimal AUC Threshold]")
    optimalPRCthreshold(y_true, y_pred, model_label, save_to)
    
def result_evaluation(testing_data_path, save_to, model_label):
    testing_data = np.genfromtxt(testing_data_path, delimiter=",", skip_header=1, filling_values=NaN)
    print(testing_data[0:10, :])

    # ADD: remove NODATA
    print(f"Before remove NODATA, testing data shape is {testing_data.shape}, NODATA count is {np.count_nonzero(testing_data[:,2]==NaN)}")
    testing_data = testing_data[testing_data[:,2]!=NaN]
    print(f"After remove NODATA, testing data shape is {testing_data.shape}")
    # ADD END

    y_true, y_pred = testing_data[:,1], testing_data[:,2]
    print(y_true[:10], y_pred[:10])
    y_pred = y_pred.reshape((y_pred.shape[0],1))
    y_pred = np.concatenate([y_pred, y_pred], axis=1)
    evaluation_report(y_true, y_pred, model_label, save_to)


############################ PUT THEM TOGETHER ############################ 

import joblib

ALGORITHMS = {
    BAGGING_MODEL_LABLE: ensemble_bagging,
    RANDOMFOREST_MODEL_LABLE: ensemble_randomforest,
    ADABOOST_MODEL_LABLE: ensemble_adaboost,
    GRADIENT_TREE_BOOSTING_MODEL_LABLE: ensemble_gradienttreeboosting,
}

def LSM(ROI_label, factor_dir, trainset_path, testset_path, result_path, preprocess=None, algorithms=ALGORITHMS):

    print(f"""# {ROI_label}
    [DIR]
    Factor dir: {factor_dir}
    Train set path: {trainset_path}
    Test set path: {testset_path}
    Result path: {result_path}
    """)

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
    print("\n\nTime Cost:  %.3f seconds" % (end-start))

    # 3. Processing
    print("## Processing and Evaluation")
    clfs = {}
    for alg_label, alg in algorithms.items():
        start = time()
        ## train
        clf = alg(train_xs, train_y, test_xs, test_y, save_to=result_path)
        clfs[alg_label] = clf
        ## save model
        joblib.dump(clf, os.path.join(result_path, f'{ROI_label}_{alg_label}.pkl'))
        ## mapping
        startM = time()
        mapping(target_xs, clf, alg_label, raster_info, mask, result_path)
        endM = time()
        print("Mapping Time cost: %.3f seconds" % (endM-startM))
        end = time()
        print("\n\nTime Cost:  %.3f seconds" % (end-start))
        
        
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
            
            for ifrom, ito in chunk_idxs:
                if ifrom < predict_from:
                    print(f"Skip {ifrom} ~ {ito}, because predict from {predict_from}")
                    continue

                print(f"\nFrom idx {ifrom} to idx {ito}")
                chunk_target_path = os.path.join(chunk_save_to, f"target_{ifrom}_{ito}.csv")

                bigdata_predict(clf, model_label, chunk_target_path, column_types, chunk_size=chunk_size, clf_pred_dir=clf_pred_dir,
                batch_size=pred_batch_size)
        else:
            print("Skip Predict!!")
        print(f"[{model_label}] Predict Time cost: {time()-start}")

        gc.collect()
        
        print("### Mapping")
        bigdata_mapping(model_label, clf_pred_dir, clf_pred_dir, raster_info, mask, chunk_idxs)

        gc.collect()

