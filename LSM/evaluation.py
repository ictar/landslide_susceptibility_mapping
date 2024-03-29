############################ EVALUATION ############################ 
import os
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn import metrics

from utils import _modelname2filename, load_model
from preprocessing import get_X_Y, get_train_test
from config import *

def optimalROCthreshold(y_true, y_pred, model_label, save_to=None):
    '''
    Function to automatically determine the optimal threshold based on computer
    probability and ROC curve.

    y_true: the y array from the testing dataset
    y_pred: the predicted probability of test_X
    model_label: (str) the name of the model needed for the plot

    '''
    print('\n###  CLASSIFICATION BASED ON OPTIMAL THRESHOLD FROM ROC\n')

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
    report = {
        "report":metrics.classification_report(y_true, y_predRnewT),
        "confusion_matrix": metrics.confusion_matrix(y_true,  y_predRnewT).tolist(),
        "accuracy": 1.*metrics.accuracy_score(y_true, y_predRnewT),
        "precision": precision.tolist(), "recall": recall.tolist(),
        "fscore": fscore.tolist(), "FPR": 1.*FPR,
        "aucroc": 1.*metrics.roc_auc_score(y_true, y_predRnewT),
        "best_threshold": 1.*best_thresh,
        "best_threshold_fpr": fpr[ix],
        "best_threshold_tpr": tpr[ix],
    }
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
    return report

def optimalPRCthreshold(y_true, y_pred, model_label, save_to=None):
    '''
    Function to automatically determine the optimal threshold based on computer
    probability and PRC curve.

    y_true: the y array from the testing dataset
    y_pred: the predicted probability of test_X
    model_label: (str) the name of the model needed for the plot

    '''
    print('\n###  CLASSIFICATION BASED ON OPTIMAL THRESHOLD FROM PRC\n')

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred[:,1])
    print(f"""
    Precision: {precision[:10]}
    Recall: {recall[:10]}
    Thresholds: {thresholds[:10]}
    """)
    # convert to f score
    ## TOFIX: when precision=0 and recall=0
    tmp_prec_rec = precision+recall
    tmp_precision = precision[tmp_prec_rec!=0]
    tmp_recall = recall[tmp_prec_rec!=0]
    fscore = (2 * tmp_precision * tmp_recall) / (tmp_precision + tmp_recall)
    #fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    best_thresh, best_threshold_recall, best_threshold_precision = thresholds[ix], recall[ix], precision[ix]
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

    report = {
        "report":metrics.classification_report(y_true, y_predRnewT),
        "confusion_matrix": metrics.confusion_matrix(y_true,  y_predRnewT).tolist(),
        "accuracy": 1.*metrics.accuracy_score(y_true, y_predRnewT),
        "precision": precision.tolist(), "recall": recall.tolist(),
        "fscore": fscore.tolist(), "FPR": 1.*FPR,
        "aucroc": 1.*metrics.roc_auc_score(y_true, y_predRnewT),
        "best_threshold": 1.*best_thresh,
        "best_threshold_recall": best_threshold_recall, 
        "best_threshold_precision": best_threshold_precision,
    }
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
    return report

def evaluation_report(y_true, y_pred, model_label, save_to):
    if save_to and not os.path.exists(save_to):
        os.makedirs(save_to)

    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(
        y_true, [round(num) for num in y_pred[:,1]])
    # FPR = 1 - TNR and TNR = specificity
    # to calculate TNR we need to set the positive label to the other class, so pos_label=0
    FPR = 1- metrics.recall_score(y_true,  [round(num) for num in y_pred[:,1]], pos_label = 0)
    
    report = {"basic": {
        "report": metrics.classification_report(y_true,  [round(num) for num in y_pred[:,1]]),
        "confusion_matrix": metrics.confusion_matrix(y_true, [round(num) for num in y_pred[:,1]]).tolist(),
        "accuracy": 1.*metrics.accuracy_score(y_true,  [round(num) for num in y_pred[:,1]]),
        "precision": precision.tolist(), "recall": recall.tolist(),
        "fscore": fscore.tolist(), "FPR": 1.*FPR,
        "aucroc": 1.*metrics.roc_auc_score(y_true,  [round(num) for num in y_pred[:,1]]),
    }}
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
    if save_to:
        tmp_save_path = os.path.join(save_to, f"ROC_{_modelname2filename(model_label)}")
        plt.savefig(tmp_save_path)
        print(f"![ROC]({tmp_save_path}.png)")
    else:
        plt.show()

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred[:,1])
    print(f"""
    Precision: {precision[-10:]}
    Recall: {recall[-10:]}
    Thresholds: {thresholds[-10:]}
    Count of probability = 1: {np.count_nonzero(y_pred == 1)}
    Count of probability = 0: {np.count_nonzero(y_pred == 0)}
    """)
    # plot the roc curve for the model
    plt.figure()
    plt.plot([0,1], [1,0], linestyle='--', label='No Skill')
    plt.plot(recall, precision, color='black', label=model_label)
    plt.grid(True,'major',linewidth=0.3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    if save_to:
        tmp_save_path = os.path.join(save_to, f"PRC{_modelname2filename(model_label)}")
        plt.savefig(tmp_save_path)
        print(f"![PRC]({tmp_save_path}.png)")
    else: plt.show()

    report["optimalROC"] = optimalROCthreshold(y_true, y_pred, model_label, save_to)

    report["optimalPRC"] = optimalPRCthreshold(y_true, y_pred, model_label, save_to)

    return report
    
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
    return evaluation_report(y_true, y_pred, model_label, save_to)

def evaluation_with_testset(testset_path, model_path, model_label, save_to):
    print(f"""# Evaluation of {model_label} ({datetime.now()})
    
    [DIR]
    testset path: {testset_path}
    model path: {model_path}
    result path: {save_to}""")

    clf = load_model(model_path)
    print(f'classifer feature: {clf.feature_names_in_}')
    ## testing samples
    _, testingPoints = get_train_test(
        None,
        testset_path
    )
    test_xs, test_y = get_X_Y(testingPoints)
    test_xs = test_xs[clf.feature_names_in_]
    # test
    Ytest_pred = clf.predict_proba(test_xs)
    return evaluation_report(test_y, Ytest_pred, model_label, save_to=save_to)

import os
import rasterio
def plot_dataset_histogram(infos, save_to, labels=rFactors):
    N = len(infos)
    for name, info in infos.items():
        info['df'] = pd.read_csv(info['path'])

    colors = plt.cm.get_cmap("Dark2")
    gs = GridSpec(2, N//2) if N % 2 == 0 else GridSpec(2, N//2+1)
    grid_positions = [(i//2, i%2) for i in range(N)]
    for label in labels:
        fig = plt.figure(figsize=(10, 10))
        i = 0
        for title, info in infos.items():
            row, col = grid_positions[i]
            ax = fig.add_subplot(gs[row, col])
            _, _, patches = ax.hist(
                    info['df'][label],
                    bins=10,
                    label=label,
                    rwidth=0.8
                )
            # change to different colors
            for j in range(len(patches)):
                patches[j].set_facecolor(colors(j))
            ax.set(title=title, xlabel=label, ylabel="Frequency")
            i += 1

        plt.tight_layout()
        if save_to:
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            plt.savefig(os.path.join(save_to, label))
        else: plt.show()

def plot_factors_histogram(title, fpath, save_to, labels=rFactors):
    N = len(labels)

    colors = plt.cm.get_cmap("Dark2")

    for label in labels:
        with rasterio.open(f"{fpath}/{label}.tif") as ds:
            dataset = ds.read(1).reshape((-1,1))
            plt.figure()
            _, _, patches = plt.hist(
                    dataset,
                    bins=10,
                    label=label,
                    rwidth=0.8
                )
            # change to different colors
            for j in range(len(patches)):
                patches[j].set_facecolor(colors(j))
            plt.title(title)
            plt.xlabel(label)
            plt.ylabel("Frequency")

            plt.tight_layout()
            if save_to:
                if not os.path.exists(save_to):
                    os.makedirs(save_to)
                plt.savefig(os.path.join(save_to, label))
            else: plt.show()

from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay
def plot_LSM_evaluation(testset_path, clfs, save_to=None, skip_models=[ADABOOST_MODEL_LABLE,]):
    ## testing samples
    _, testingPoints = get_train_test(
        None,
        testset_path
    )
    # plot
    plot_dataset_histogram({"Number of Points - Testing":{"path":testset_path}}, save_to=os.path.join(save_to, "hist"))
    test_xs, y_true = get_X_Y(testingPoints)

    for model_label, info in clfs.items():
        clf = load_model(info["path"])
        test_xs = test_xs[clf.feature_names_in_]
        info["y_pred"] = clf.predict_proba(test_xs)
        info['clf'] = clf
        
    
    # plot ROC
    plt.figure()
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    for model_label, info in clfs.items():
        if model_label in skip_models: continue
        y_pred = info["y_pred"]
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred[:,1])

        # plot the roc curve for the model
        auc = metrics.roc_auc_score(y_true,  [round(num) for num in y_pred[:,1]])
        if 'linestyle' in info:
             plt.plot(fpr, tpr, color=info["color"],linestyle=info['linestyle'], label=f"{model_label} (AUC = {auc:.2f})")
        else: plt.plot(fpr, tpr, color=info["color"],label=f"{model_label} (AUC = {auc:.2f})")

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve")
    plt.grid(True,'major',linewidth=0.3)
    plt.legend()
    if save_to:
        plt.savefig(os.path.join(save_to, "ROC_models"))
    else: plt.show()

    # plot the PRC
    plt.figure()
    plt.plot([0,1], [1,0], linestyle='--', label='No Skill')
    for model_label, info in clfs.items():
        if model_label in skip_models: continue
        y_pred = info["y_pred"]
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred[:,1])
        auc = metrics.auc(recall, precision)
        if 'linestyle' in info:
            plt.plot(recall, precision, color=info["color"],linestyle=info['linestyle'], label=f"{model_label} (AUC = {auc:.2f})")
        else: plt.plot(recall, precision, color=info["color"], label=f"{model_label} (AUC = {auc:.2f})")
    #plt.grid(True,'major',linewidth=0.3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall curve")
    plt.legend()
    if save_to:
        plt.savefig(os.path.join(save_to, "PRC_models"))
    else: plt.show()

    # plot calibration curves
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(5, 3)

    ax_calibration_curve = fig.add_subplot(gs[:2, :3])
    calibration_displays = {}
    for model_label, info in clfs.items():
        calibration_displays[model_label] = CalibrationDisplay.from_predictions(
            y_true,
            info['y_pred'][:,-1],
            n_bins=10,
            name=model_label,
            ax=ax_calibration_curve,
            color=info["color"],
        )

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]
    i = 0
    for model_label, info in clfs.items():
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[model_label].y_prob,
            range=(0, 1),
            bins=10,
            label=model_label,
            color=info['color'],
        )
        ax.set(title=model_label, xlabel="Mean predicted probability", ylabel="Count")
        i += 1

    plt.tight_layout()
    if save_to:
        plt.savefig(os.path.join(save_to, "Calibration_plots"))
    else: plt.show()

def plot_single_model_calibration(infos, save_to=None, model_label='Neural Network'):
    # plot calibration curves
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(7, 3)
    colors = plt.cm.get_cmap("tab20")

    ax_calibration_curve = fig.add_subplot(gs[:3, :3])
    calibration_displays = {}
    for i, (case_label, info) in enumerate(infos.items()):
        clf = load_model(info["clf"]['path'])
        ## testing samples
        print(case_label, info['testset_path'])
        _, testingPoints = get_train_test(
            None,
            info['testset_path']
        )
        test_xs, y_true = get_X_Y(testingPoints)

        test_xs = test_xs[clf.feature_names_in_]
        calibration_displays[case_label] = CalibrationDisplay.from_estimator(clf,
            test_xs,
            y_true,
            n_bins=10,
            name=case_label,
            ax=ax_calibration_curve,
            color=colors(i)
        )

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title(f"Calibration plots using {model_label}")

    # Add histogram
    grid_positions = [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2), (6, 0), (6, 1), (6, 2), ]
    for i, (case_label, info) in enumerate(infos.items()):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[case_label].y_prob,
            range=(0, 1),
            bins=10,
            label=case_label,
            color=colors(i),
        )
        ax.set(title=case_label, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    if save_to:
        plt.savefig(os.path.join(save_to, "Calibration_plots"))
    else: plt.show()

def plot_evaluation_with_testset(testset_path, clfs_info, save_info):
    for clf_type, result_path in save_info:
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        clfs = clfs_info[clf_type]
        plot_LSM_evaluation(testset_path, clfs, result_path)
        reports = {}
        for model_lable in clfs:
            reports[model_lable] = evaluation_with_testset(testset_path, clfs[model_lable]["path"], model_lable, save_to=result_path)
        #print(reports)
        import json
        with open(os.path.join(result_path, "report.json"), "w") as f:
            json.dump(reports, f)

def evaluation_main():
    # Val Tartano
    testset_path = vt_dir+"/2.samples/ValTartano_Testing_Points.csv"
    print("Handle ", testset_path)
    save_info = [('basic', vt_dir+"3.results/"), ("ensemble", vt_dir+"3.results/ensemble/")]
    #plot_evaluation_with_testset(testset_path, vt_clfs, save_info)
    # Upper Valtellina
    testset_path = uv_dir+"2.samples/UpperValtellina_LSM_testing_points.csv"
    print("Handle ", testset_path)
    save_info = [('basic', uv_dir+"3.results/"), ("ensemble", uv_dir+"3.results/ensemble/")]
    #plot_evaluation_with_testset(testset_path, uv_clfs, save_info)
    # Valchiavenna
    for key, info in vc_clfs.items():
        print(f"Handling {key}")
        testset_path = info["testset_path"]
        save_info = [(label, info['result_path'][label]) for label in info['result_path']]
        #plot_evaluation_with_testset(testset_path, info['clfs'], save_info)
    # Lombardy
    test_info = [
        {
            "testset_path": ld_dir+"/2.samples/Lombardy_LSM_testing_points_northern.csv",
            "result_path": ld_dir + "/3.results/testingpoints_northern/",
        },
        {
            "testset_path": ld_dir+"/2.samples/Lombardy_LSM_testing_points_without_3regions.csv",
            "result_path": ld_dir + "/3.results/testingpoints_without_3regions/",
        },
    ]
    all_clfs = {"Valchiavenna_"+key:info["clfs"] for key, info in vc_clfs.items()}
    all_clfs["Val Tartano"] = vt_clfs
    all_clfs["UpperValtellina"] = uv_clfs

    for info in test_info:
        print(info)
        testset_path = info["testset_path"]
        for region, clfs in all_clfs.items():
            save_info = [
                ("basic", os.path.join(info["result_path"], f"{region}/")),
                ("ensemble", os.path.join(info["result_path"], f"{region}/ensemble")),
            ]

            plot_evaluation_with_testset(testset_path, clfs, save_info)

def evaluation_allmodels_main():
    # Val Tartano
    testset_path = vt_dir+"/2.samples/ValTartano_Testing_Points.csv"
    print("Handle ", testset_path)
    save_info = vt_dir+"3.results/allmodels/"
    clfs = vt_clfs['basic']
    clfs.update(vt_clfs['ensemble'])
    plot_LSM_evaluation(testset_path, clfs, save_info)
    # Upper Valtellina
    testset_path = uv_dir+"2.samples/UpperValtellina_LSM_testing_points.csv"
    print("Handle ", testset_path)
    save_info = uv_dir+"3.results/allmodels/"
    clfs = uv_clfs['basic']
    clfs.update(uv_clfs['ensemble'])
    plot_LSM_evaluation(testset_path, clfs, save_info)
    # Valchiavenna
    for key, info in vc_clfs.items():
        print(f"Handling {key}")
        testset_path = info["testset_path"]
        save_info = info['result_path']['basic']+'/allmodels/'
        clfs = info['clfs']['basic']
        clfs.update(info['clfs']['ensemble'])
        plot_LSM_evaluation(testset_path, clfs, save_info)
    # Lombardy
    test_info = [
        {
            "testset_path": ld_dir+"/2.samples/Lombardy_LSM_testing_points_northern.csv",
            "result_path": ld_dir + "/3.results/testingpoints_northern/",
        },
        {
            "testset_path": ld_dir+"/2.samples/Lombardy_LSM_testing_points_without_3regions.csv",
            "result_path": ld_dir + "/3.results/testingpoints_without_3regions/",
        },
    ]
    all_clfs = {"Valchiavenna_"+key:info["clfs"] for key, info in vc_clfs.items()}
    all_clfs["Val Tartano"] = vt_clfs
    all_clfs["UpperValtellina"] = uv_clfs

    for info in test_info:
        print(info)
        testset_path = info["testset_path"]
        for region, clfs in all_clfs.items():
            save_info = os.path.join(info["result_path"], f"{region}/allmodels/")

            tmp_clfs = clfs['basic']
            tmp_clfs.update(clfs['ensemble'])
            plot_LSM_evaluation(testset_path, tmp_clfs, save_info)


def evaluation_on_trainingset():
    '''
    Evaluation the model on training dataset
    '''
    # Val Tartano
    pointset_path = vt_dir+"/2.samples/ValTartano_Training_Points.csv"
    print("Handle ", pointset_path)
    save_info = [('basic', vt_dir+"3.results/training/"), ("ensemble", vt_dir+"3.results/training/ensemble/")]
    plot_evaluation_with_testset(pointset_path, vt_clfs, save_info)
    # Upper Valtellina
    pointset_path = uv_dir+"2.samples/UpperValtellina_LSM_training_points.csv"
    print("Handle ", pointset_path)
    save_info = [('basic', uv_dir+"3.results/training/"), ("ensemble", uv_dir+"3.results/training/ensemble/")]
    plot_evaluation_with_testset(pointset_path, uv_clfs, save_info)
    # Valchiavenna
    pointset_path = vc_dir+"/2.samples/3rd_onlyVC/ValChiavenna_LSM_training_points.csv"
    print("Handle ", pointset_path)
    save_info = [('basic', vc_dir+"3.results/training/"), ("ensemble", vc_dir+"3.results/training/ensemble/")]
    plot_evaluation_with_testset(pointset_path, vc_clfs['3_onlywith']['clfs'], save_info)

if __name__ == '__main__':
    #plot_factors_histogram("Number of Points - Raster (ValChiavenna)", vc_dir+"1.factors", save_to=os.path.join(vc_dir+"1.factors", "hist"))
    #plot_factors_histogram("Number of Points - Raster (Lombardy)", ld_dir+"1.factors", save_to=os.path.join(ld_dir+"1.factors", "hist"))
    #evaluation_allmodels_main()
    evaluation_on_trainingset()
    '''
    plot_single_model_calibration(infos={
        'LC1+VT': {'clf':vt_clfs['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_north},
        'LC1+UV': {'clf':uv_clfs['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_north},
        'LC1+VT+UV': {'clf':vc_clfs['1_without']['clfs']['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_north},
        'LC1+VCC2': {'clf':vc_clfs['2_with']['clfs']['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_north},
        'LC1+VCC3': {'clf':vc_clfs['3_onlywith']['clfs']['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_north},
        'LC2+VT': {'clf': vt_clfs['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_whole},
        'LC2+UV': {'clf':uv_clfs['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_whole},
        'LC2+VT+UV': {'clf':vc_clfs['1_without']['clfs']['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_whole},
        'LC2+VCC2': {'clf':vc_clfs['2_with']['clfs']['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_whole},
        'LC2+VCC3': {'clf':vc_clfs['3_onlywith']['clfs']['basic'][NEURAL_NETWORK_MODEL_LABEL], 'testset_path': ld_testset_path_whole},
    }, save_to=ld_dir+'/3.results/')
    '''