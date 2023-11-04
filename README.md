# Landslide susceptibility mapping

This repository contains all the Python scripts used in the paper "Landslide susceptibility mapping using ensemble Machine learning methods: a case study in Lombardy, Northern Italy"

## Directory structure
```sh
.
├── LSM # This folder contains all the scripts used to generate the models and LSM
│   ├── Lombardy.py # for case Lombardy
│   ├── UpperValtellina.py # for case Upper Valtellina
│   ├── ValTartano.py # for case Val Tartano
│   ├── Valchiavenna.py # for case Val Chiavenna
│   ├── config.py # global configuration, including the base dir, factor names, etc
│   ├── evaluation.py # methods for model evaluation
│   ├── models.py # model construction methods
│   ├── preprocessing.py # method for preprocessing, mainly for loading sample and target data.
│   ├── processing.py # contains methods for complete workflow
│   └── utils.py # utilities to generate different evaluation reports
└── landslide_scripts # This folder contains all the scripts used to preprocess the raw data source, mainly running in QGIS
    ├── config.py # global configuration, including the base dir, factor path, etc
    ├── data_preparation.py # generate raster factor layers based on the original raster/vector data.
    ├── factor_sampling.py # contains methods to do the factor sampling
    ├── preprocessing_result_check.py
    ├── print_layout.py # methods to generate layout images
    └── utils.py # some utilities
```

## Usage
### Data preparation: From original data to factors and samples
Steps:
1. Modify the configuration file `landslide_scripts/config.py` according to the actual situation
2. To convert the original data to the factors used, run `landslide_scripts/data_preparation.py` directly in QGIS
3. To do the factor sampling:
- call method `factor_sampling` in `landslide_scripts
/factor_sampling.py` in QGIS to sample training and testing data
- call method `sampling_test` in `landslide_scripts
/factor_sampling.py` in QGIS to sample only testing data
- call method `sample_with_avg_precipitation` in `landslide_scripts
/factor_sampling.py` in QGIS to sample only average precipitation data
- call method `sample_with_90th_precipitation` in `landslide_scripts
/factor_sampling.py` in QGIS to sample only 90th-percentile precipitation data
- call method `sample_with_precipitations` in `landslide_scripts
/factor_sampling.py` in QGIS to sample both average and 90th-percentile precipitation data

### LSM
Steps:
1. Modify the configuration file `LSM/config.py` according to the actual situation
2. To do the LSM, call method `LSM` in `LSM/processing.py`

Example:
- `LSM/Lombardy.py`
- `LSM/UpperValtellina.py`
- `LSM/ValTartano.py`
