import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from matplotlib import pyplot as plt

from time import time
import os
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
    factors = {}
    for rLayer in rFactors:
        with rasterio.open(f'{layer_dir}/{rLayer}.tif') as ds:
            factors[rLayer] = ds.read(1)
            # after the operation below, the dtype will become 'float64'
            # ref: https://appdividend.com/2022/01/28/np-nan/
            factors[rLayer] = np.where(factors[rLayer]==ds.nodatavals,np.nan,factors[rLayer])
            
            # TOFIX: no space. handling some special type to reduce space
            if rLayer == 'plan': # To avoid overflow
                factors[rLayer] = factors[rLayer].astype(np.float32)
            elif ds.dtypes[0] in DTYPE_MAPPING:
                factors[rLayer] = factors[rLayer].astype(DTYPE_MAPPING[ds.dtypes[0]])

            print(f"""[INFO of {rLayer}]
            {list(zip(ds.indexes, ds.dtypes, ds.nodatavals))}
            Factor shape: {factors[rLayer].shape}
            Factor type: {factors[rLayer].dtype}
            Non Check: {np.count_nonzero(np.isnan(factors[rLayer]))}
            Infinity Check: { np.count_nonzero(np.isinf(factors[rLayer]))}
            Min: {np.min(factors[rLayer])}
            Max: {np.max(factors[rLayer])}
            """)

            meta = ds.meta 
            
    mask = np.where(np.isnan(factors['dtm']),np.nan,1)
    
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
    df = df[df.dusaf != -99999]
    X, Y = df.drop(columns=dropped_columns), None
    if "hazard" in df.columns: Y = df.hazard  
    # handle categorical field
    X = add_categorical(X)
    # TOFIX: input X cannot contain NaN
    X=X.fillna(NaN)
    return X, Y

# return pd.dataframes
def get_targets(layer_dir):
    factors, meta, mask = load_rasters(layer_dir)
    target={}

    for layer in continuous_factors:
        target[layer] = factors[layer].flatten()
        raster_info = {
            "transform": meta['transform'],
            "shape": factors[layer].shape,
            "crs": meta['crs'],
            }

    for cat in categorical_factors:    
        target[cat] = factors[cat].flatten()

    target_xs, _ = get_X_Y(pd.DataFrame(target))
    #target_xs=target_xs.fillna(NaN)
    
    #print(f"[Target] X = {target_xs.shape}, Columns = {target_xs.columns}")
    
    return target_xs, raster_info, mask

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
RANDOMFOREST_MODEL_LABLE = "Fortests of randomized trees"
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
False Positive Rate = {fpr[ix]}
Best Threshold = {best_thresh}""")

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
        plt.savefig(os.path.join(save_to, f"optimalROCthreshold_{_modelname2filename(model_label)}"))
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
 Testing Accuracy 
Accuracy Score: {metrics.accuracy_score(y_true, y_predRnewT)}

Classification report
{metrics.classification_report(y_true, y_predRnewT)}

Confussion matrix
{metrics.confusion_matrix(y_true,  y_predRnewT)}

AUCROC
{metrics.roc_auc_score(y_true, y_predRnewT)}

Precision: {precision}
Recall: {recall}
False Positive Rate: {FPR}
F1 score: {fscore}
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

    # y_predR = rclf.predict_proba(test_xs)
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
        plt.savefig(os.path.join(save_to, f"optimalPRCthreshold{_modelname2filename(model_label)}"))
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

    print(f""" Testing Accuracy 
Accuracy Score: {metrics.accuracy_score(y_true, y_predRnewT)}

Classification report: {metrics.classification_report(y_true, y_predRnewT)}

Confussion matrix: {metrics.confusion_matrix(y_true,  y_predRnewT)}

AUCROC: {metrics.roc_auc_score(y_true, y_predRnewT)}

Precision: {precision}
Recall: {recall}
False Positive Rate: {FPR}
F1 score: {fscore}""")

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
    print("\t[Optimal ROC Threshold]")
    optimalROCthreshold(y_true, y_pred, model_label, save_to)
    
    print("\t[Optimal AUC Threshold]")
    optimalPRCthreshold(y_true, y_pred, model_label, save_to)
    

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
    
    Testing points shape:
        X = {test_xs.shape},
        Y = {test_y.shape},
        Columns = {test_xs.columns}
        Column Types = {test_xs.dtypes}

    Target:
        X = {target_xs.shape},
        Columns = {target_xs.columns}
        Column Types = {target_xs.dtypes}
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