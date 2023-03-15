############################ EVALUATION ############################ 
import os
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn import metrics

from utils import _modelname2filename, load_model
from preprocessing import get_X_Y, get_train_test
from config import NaN

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

    report = {
        "report":metrics.classification_report(y_true, y_predRnewT),
        "confusion_matrix": metrics.confusion_matrix(y_true,  y_predRnewT).tolist(),
        "accuracy": 1.*metrics.accuracy_score(y_true, y_predRnewT),
        "precision": precision.tolist(), "recall": recall.tolist(),
        "fscore": fscore.tolist(), "FPR": 1.*FPR,
        "aucroc": 1.*metrics.roc_auc_score(y_true, y_predRnewT),
        "best_threshold": 1.*best_thresh,
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
