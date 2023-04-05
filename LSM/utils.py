import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import pandas as pd
pd.set_option('display.max_columns', None)
from matplotlib import pyplot as plt
from config import base_dir

from collections import defaultdict
import joblib
def load_model(model_path):
    return joblib.load(model_path)

# print in table for evaluation result
def print_evaluation_result_base(eval_js_path):
    print(eval_js_path)
    results = {}
    with open(eval_js_path) as f:
        results = json.load(f)
    print("Methods\tOA\tPrecision\tRecall\tF-measure")
    for model_label in results:
        info = results[model_label]["basic"]
        print(f'{model_label}\t{info["accuracy"]*100:.2f}%\t{info["precision"][-1]*100:.2f}%\t{info["recall"][-1]*100:.2f}%\t{info["fscore"][-1]*100:.2f}%')

def print_evaluation_result_base_inregions(eval_js_paths, groups):
    reports = {}
    model_labels = None
    for key, p in eval_js_paths.items():
        with open(p) as f:
            reports[key] = json.load(f)
            if model_labels is None:
                model_labels = list(reports[key].keys())

    tmp = ('\t'*4).join(model_labels)
    print(f"Case\t{tmp}")
    print("\tOA\tPrecision\tRecall\tF-measure"*len(model_labels))

    cases = reports.keys()
    OAs, Precisions, Recalls, Fmeasures = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for key in cases:
        OAs[key], Precisions[key], Recalls[key], Fmeasures[key] = {}, {}, {}, {}
        tlst = []
        for model_label in model_labels:
            info = reports[key][model_label]['basic']
            tlst.append(f'{info["accuracy"]*100:.2f}%\t{info["precision"][-1]*100:.2f}%\t{info["recall"][-1]*100:.2f}%\t{info["fscore"][-1]*100:.2f}%')
            OAs[key][model_label] = reports[key][model_label]['basic']["accuracy"]
            Precisions[key][model_label] = reports[key][model_label]['basic']["precision"][-1]
            Recalls[key][model_label] = reports[key][model_label]['basic']["recall"][-1]
            Fmeasures[key][model_label] = reports[key][model_label]['basic']["fscore"][-1]
        tmp = "\t".join(tlst)
        print(f"""{key}\t{tmp}""")
    # construct dfs
    OAs = pd.DataFrame(OAs).transpose()
    Precisions = pd.DataFrame(Precisions).transpose()
    Recalls = pd.DataFrame(Recalls).transpose()
    Fmeasures = pd.DataFrame(Fmeasures).transpose()

    print("Overall Accuracy")
    print(OAs.to_csv(sep='\t'))
    print("Precision")
    print(Precisions.to_csv(sep='\t'))
    print("Recall")
    print(Recalls.to_csv(sep='\t'))
    print("Fmeasure")
    print(Fmeasures.to_csv(sep='\t'))
    # plot
    for group in groups:
        OA = OAs.filter(items=group, axis=0)
        Precision = Precisions.filter(items=group, axis=0)
        Recall = Recalls.filter(items=group, axis=0)
        Fmeasure = Fmeasures.filter(items=group, axis=0)
        fig, axes = plt.subplots(nrows=4, ncols=1)
        OA.plot(ax=axes[0], style='-o', linewidth=1, ms=2, grid=True, ylabel="OA", xlabel="", legend=False, fontsize=9)
        axes[0].set_xticks(range(len(OA)))
        axes[0].set_xticklabels(OA.index.tolist())
        Precision.plot(ax=axes[1], style='-o', linewidth=1, ms=2, grid=True, ylabel="Precision", xlabel="", legend=False, fontsize=9)
        axes[1].set_xticks(range(len(Precision)))
        axes[1].set_xticklabels(Precision.index.tolist())
        Recall.plot(ax=axes[2], style='-o', linewidth=1, ms=2, grid=True, ylabel="Recall", xlabel="", legend=False, fontsize=9)
        axes[2].set_xticks(range(len(Recall)))
        axes[2].set_xticklabels(Recall.index.tolist())
        Fmeasure.plot(ax=axes[3], style='-o', linewidth=1, ms=2, grid=True, ylabel="Fmeasure", xlabel="Case", legend=False, fontsize=9)
        axes[3].set_xticks(range(len(Fmeasure)))
        axes[3].set_xticklabels(Fmeasure.index.tolist())
        #axes[3].legend(bbox_to_anchor=(1.0, 1.0))
        handels, labels = axes[3].get_legend_handles_labels()
        fig.legend(handels, labels, loc="upper center", ncol=4, fancybox=True)
        #fig.tight_layout()
        plt.show()

    return OAs, Precisions, Recalls, Fmeasures

def print_evaluation_result_OA_Threshold(eval_js_path):
    print(eval_js_path)
    results = {}
    with open(eval_js_path) as f:
        results = json.load(f)
    print("Methods\tOA(threshold=0.5)\tThreshold (optimal ROC)\tOA (optimal ROC)\tThreshold (optimal PRC)\tOA (optimal PRC)")
    for model_label in results:
        basic_info = results[model_label]["basic"]
        optroc_info = results[model_label]["optimalROC"]
        optprc_info = results[model_label]["optimalPRC"]
        
        print(f'{model_label}\t{basic_info["accuracy"]*100:.2f}%\t{optroc_info["best_threshold"]:.2f}\t{optroc_info["accuracy"]*100:.2f}%\t{optprc_info["best_threshold"]:.2f}\t{optprc_info["accuracy"]*100:.2f}%')

def print_OA_inregions(eval_js_paths):
    reports = {}
    model_labels = None
    for key, p in eval_js_paths.items():
        with open(p) as f:
            reports[key] = json.load(f)
            if model_labels is None:
                model_labels = list(reports[key].keys())

    tmp = ('\t'*5).join(model_labels)
    print(f"Cases\t{tmp}")
    print("\tBase OA\tOptimal ROC Threshold\tOptimal ROC OA\tOptimal PRC Threshold\tOptimal PRC OA"*len(model_labels))

    cases = reports.keys()
    OAs, optROCThrs, optROCOAs, optPRCThrs, optPRCOAs = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for key in cases:
        OAs[key], optROCThrs[key], optROCOAs[key], optPRCThrs[key], optPRCOAs[key] = {}, {}, {}, {}, {}
        tlist = []
        for model_label in model_labels:
            info = reports[key][model_label]
            tlist.append(f'{info["basic"]["accuracy"]*100:.2f}\t{info["optimalROC"]["best_threshold"]:.2f}\t{info["optimalROC"]["accuracy"]*100:.2f}\t{info["optimalPRC"]["best_threshold"]:.2f}\t{info["optimalPRC"]["accuracy"]*100:.2f}')
            
            OAs[key][model_label] = info["basic"]["accuracy"]
            optROCThrs[key][model_label] = info["optimalROC"]["best_threshold"]
            optROCOAs[key][model_label] = info["optimalROC"]["accuracy"]
            optPRCThrs[key][model_label] = info["optimalPRC"]["best_threshold"]
            optPRCOAs[key][model_label] = info["optimalPRC"]["accuracy"]

        tmp = '\t'.join(tlist)
        print(f"{key}\t{tmp}")

    # another print schema
    tmp = ('\t'*5).join(model_labels)
    print(f"\nCases\t{tmp}")
    print("\tBase OA\tOptimal ROC OA \t(Threshold)\tOptimal PRC OA \t(Threshold)"*len(model_labels))
    for key in cases:
        tlist = []
        for model_label in model_labels:
            info = reports[key][model_label]
            tlist.append(f'{info["basic"]["accuracy"]*100:.2f}%\t{info["optimalROC"]["accuracy"]*100:.2f}%\t{info["optimalROC"]["best_threshold"]:.4f}\t{info["optimalPRC"]["accuracy"]*100:.2f}%\t{info["optimalPRC"]["best_threshold"]:.4f}')

        tmp = '\t'.join(tlist)
        print(f"{key}\t{tmp}")

    # construct dfs
    OAs = pd.DataFrame(OAs).transpose()
    optROCThrs = pd.DataFrame(optROCThrs).transpose()
    optROCOAs = pd.DataFrame(optROCOAs).transpose()
    optPRCThrs = pd.DataFrame(optPRCThrs).transpose()
    optPRCOAs = pd.DataFrame(optPRCOAs).transpose()

    print("Basic Accuracy")
    print(OAs.to_csv(sep='\t'))
    print("Optimal ROC Threshold")
    print(optROCThrs.to_csv(sep='\t'))
    print("Optimal ROC Accuracy")
    print(optROCOAs.to_csv(sep='\t'))
    print("Optimal PRC Threshold")
    print(optPRCThrs.to_csv(sep='\t'))
    print("Optimal PRC Accuracy")
    print(optPRCOAs.to_csv(sep='\t'))
    # plot

    # save

    return OAs, optROCThrs, optROCOAs, optPRCThrs, optPRCOAs


def _modelname2filename(mn):
    mn = mn.replace(":", " ").replace("/", " ")
    return "_".join([s for s in mn.split() if s])

def main():
    eval_js_paths = {
        "basic": {
            "VT": base_dir + r"/Val Tartano/3.results/report.json",
            "UV": base_dir + r"/Upper Valtellina/3.results/report.json",
            # ValChiavenna
            "VCC1": base_dir + r"/ValChiavenna/3.results/1st_without/report.json",
            "VCC2": base_dir + r"/ValChiavenna/3.results/2nd_with/report.json",
            "VCC3": base_dir + r"/ValChiavenna/3.results/3rd_onlyVC/report.json",
            # Lombardy
            ## testingpoints_northern
            "LC1+VT": base_dir + r"/Lombardy/3.results/testingpoints_northern/Val Tartano/report.json",
            "LC1+UV": base_dir + r"/Lombardy/3.results/testingpoints_northern/UpperValtellina/report.json",
            "LC1+VT+UV": base_dir + r"/Lombardy/3.results/testingpoints_northern/Valchiavenna_1_without/report.json",
            "LC1+VCC2": base_dir + r"/Lombardy/3.results/testingpoints_northern/Valchiavenna_2_with/report.json",
            "LC1+VCC3": base_dir + r"/Lombardy/3.results/testingpoints_northern/Valchiavenna_3_onlywith/report.json",
            ## testingpoints_without_3regions
            "LC2+VT": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/Val Tartano/report.json",
            "LC2+UV": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/UpperValtellina/report.json",
            "LC2+VT+UV": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/Valchiavenna_1_without/report.json",
            "LC2+VCC2": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/Valchiavenna_2_with/report.json",
            "LC2+VCC3": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/Valchiavenna_3_onlywith/report.json",
        },
        "ensemble": {
            "VT": base_dir + r"/Val Tartano/3.results/ensemble/report.json",
            "UV": base_dir + r"/Upper Valtellina/3.results/ensemble/report.json",
            # ValChiavenna
            "VCC1": base_dir + r"/ValChiavenna/3.results/1st_without/ensemble/report.json",
            "VCC2": base_dir + r"/ValChiavenna/3.results/2nd_with/ensemble/report.json",
            "VCC3": base_dir + r"/ValChiavenna/3.results/3rd_onlyVC/ensemble/report.json",
            # Lombardy
            ## testingpoints_northern
            "LC1+VT": base_dir + r"/Lombardy/3.results/testingpoints_northern/Val Tartano/ensemble/report.json",
            "LC1+UV": base_dir + r"/Lombardy/3.results/testingpoints_northern/UpperValtellina/ensemble/report.json",
            "LC1+VT+UV": base_dir + r"/Lombardy/3.results/testingpoints_northern/Valchiavenna_1_without/ensemble/report.json",
            "LC1+VCC2": base_dir + r"/Lombardy/3.results/testingpoints_northern/Valchiavenna_2_with/ensemble/report.json",
            "LC1+VCC3": base_dir + r"/Lombardy/3.results/testingpoints_northern/Valchiavenna_3_onlywith/ensemble/report.json",
            ## testingpoints_without_3regions
            "LC2+VT": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/Val Tartano/ensemble/report.json",
            "LC2+UV": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/UpperValtellina/ensemble/report.json",
            "LC2+VT+UV": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/Valchiavenna_1_without/ensemble/report.json",
            "LC2+VCC2": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/Valchiavenna_2_with/ensemble/report.json",
            "LC2+VCC3": base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/Valchiavenna_3_onlywith/ensemble/report.json",
        },
    }
    groups = [
        #["VT", "UV", "VCC2", "VCC3"],
        ["VT", "UV", "VCC2", "VCC3", "VCC1", "LC1+VT", "LC1+UV", "LC1+VT+UV", "LC1+VCC2", "LC1+VCC3", "LC2+VT", "LC2+UV", "LC2+VT+UV", "LC2+VCC2", "LC2+VCC3"],
    ]
    infos = {}
    for key in eval_js_paths:
        print("\n\n", key)
        infos[key] = {}
        OAs, optROCThrs, optROCOAs, optPRCThrs, optPRCOAs = print_OA_inregions(eval_js_paths[key])
        infos[key]["OA"] = {'OAs': OAs, 'optROCThrs': optROCThrs, 'optROCOAs': optROCOAs, 'optPRCThrs': optPRCThrs, 'optPRCOAs': optPRCOAs}
        OAs, Precisions, Recalls, Fmeasures = print_evaluation_result_base_inregions(eval_js_paths[key], groups)
        infos[key]['base'] = {'OAs': OAs, 'Precisions': Precisions, 'Recalls': Recalls, 'Fmeasures': Fmeasures}
        """for p in eval_js_paths[key]:
            print('*'*20)
            #print_evaluation_result(p)
            print_evaluation_result_OA_Threshold(p)
            print('*'*20)"""
    return infos
        
if __name__ == '__main__':
    main()