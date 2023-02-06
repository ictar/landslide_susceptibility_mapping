import numpy as np

#base_dir = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/"
base_dir = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/"

rFactors = ['dtm', 'east', 'ndvi', 'north', 'faults',
        'rivers','roads','dusaf',
        'plan','profile','twi']

categorical_factors = ['dusaf','faults','rivers','roads']
continuous_factors = [x for x in rFactors if x not in categorical_factors]
#print(f"{len(rFactors)} = {len(categorical_factors)} + {len(continuous_factors)}")

NaN = -9999

#JOBLIB_TEMP_FOLDER = r'/Users/elexu/tmp'
JOBLIB_TEMP_FOLDER = r'/Volumes/Another/tmp'
# TOAVOID: save memory
DATA_COLUMN_TYPES = {}
for f in continuous_factors:
       DATA_COLUMN_TYPES[f] =  np.float16

# when deal with float->int, 'RuntimeWarning: invalid value encountered in cast'
'''
>>> arr = np.array([np.nan, 1, 2, ])
>>> arr.dtype
dtype('float64')
>>> arr1 = arr.astype('uint8')
<stdin>:1: RuntimeWarning: invalid value encountered in cast
>>> arr1
array([0, 1, 2], dtype=uint8)
>>> arr2 = arr.astype('float16')
>>> arr2
array([nan,  1.,  2.], dtype=float16)
'''
DTYPE_MAPPING = {'float32': np.float16, 'float64': np.float16, 'uint16': np.uint16}


##### for processing
PROCESS_BATCH_SIZE = 5*(10**6)