import json
from config import base_dir

import joblib
def load_model(model_path):
    return joblib.load(model_path)

# print in table for evaluation result
def print_evaluation_result(eval_js_path):
    print(eval_js_path)
    results = {}
    with open(eval_js_path) as f:
        results = json.load(f)
    print("Methods\tOA\tPrecision\tRecall\tF-measure")
    for model_label in results:
        info = results[model_label]["basic"]
        print(f'{model_label}\t{info["accuracy"]*100:.4f}%\t{info["precision"][-1]*100:.4f}%\t{info["recall"][-1]*100:.4f}%\t{info["fscore"][-1]*100:.4f}%')

def print_ensemble_evaluation_result(eval_js_path):
    print(eval_js_path)
    results = {}
    with open(eval_js_path) as f:
        results = json.load(f)
    print("Methods\tOA(threshold=0.5)\tThreshold (optimal ROC)\tOA (optimal ROC)\tThreshold (optimal PRC)\tOA (optimal PRC)")
    for model_label in results:
        basic_info = results[model_label]["basic"]
        optroc_info = results[model_label]["optimalROC"]
        optprc_info = results[model_label]["optimalPRC"]
        
        print(f'{model_label}\t{basic_info["accuracy"]*100:.4f}%\t{optroc_info["best_threshold"]:.4f}\t{optroc_info["accuracy"]*100:.4f}%\t{optprc_info["best_threshold"]:.4f}\t{optprc_info["accuracy"]*100:.4f}%')

def _modelname2filename(mn):
    mn = mn.replace(":", " ").replace("/", " ")
    return "_".join([s for s in mn.split() if s])

if __name__ == '__main__':
    eval_js_paths = {
        "basic": [
            base_dir + r"/Val Tartano/3.results/report.json",
            base_dir + r"/Upper Valtellina/3.results/report.json",
            base_dir + r"/ValChiavenna/3.results/1st_without/report.json",
            base_dir + r"/ValChiavenna/3.results/2nd_with/report.json",
            base_dir + r"/ValChiavenna/3.results/3rd_onlyVC/report.json",
            base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/2nd_with/report.json",
            base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/3rd_onlyVC/report.json",
            base_dir + r"/Lombardy/3.results/testingpoints_northern/2nd_with/report.json",
        ],
        "ensemble": [
            base_dir + r"/Val Tartano/3.results/ensemble/report.json",
            base_dir + r"/Upper Valtellina/3.results/ensemble/report.json",
            base_dir + r"/ValChiavenna/3.results/1st_without/ensemble/report.json",
            base_dir + r"/ValChiavenna/3.results/2nd_with/ensemble/report.json",
            base_dir + r"/ValChiavenna/3.results/3rd_onlyVC/ensemble/report.json",
            base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/2nd_with/ensemble/report.json",
            base_dir + r"/Lombardy/3.results/testingpoints_without_3regions/3rd_onlyVC/ensemble/report.json",
            base_dir + r"/Lombardy/3.results/testingpoints_northern/2nd_with/ensemble/report.json",
        ],
    }
    for key in eval_js_paths:
        print("\n\n", key)
        for p in eval_js_paths[key]:
            print('*'*20)
            #print_evaluation_result(p)
            print_ensemble_evaluation_result(p)
            print('*'*20)