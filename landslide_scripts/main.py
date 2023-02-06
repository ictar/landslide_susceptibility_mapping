config_path = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/Materials/landslide_scripts"
import subprocess, sys
if config_path not in sys.path: sys.path.append(config_path)
from config import *
from utils import *
from data_preparation import prepare_data
from factor_sampling import factor_sampling
from preprocessing_result_check import check_preprocess

def preprocess():
    prepare_data()
    factor_sampling()
    check_preprocess()

def process(): pass

def evaluation(): pass

def main():
    preprocess()

main()