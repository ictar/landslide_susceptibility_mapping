import os
import gc
import rasterio
import pandas as pd
from matplotlib import pyplot as plt
from time import time
from config import *
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

    # TOCHECK: change profile and plan to categorical factor to see what happen => No performance increase, abandom!
    #df.loc[df["plan"]<0, 'plan'] = -1
    #df.loc[df["plan"]>0, 'plan'] = 1
    #df.loc[df["profile"]<0, 'profile'] = -1
    #df.loc[df["profile"]>0, 'profile'] = 1
    #print(f"After change profile and plan, now the unique value of plan is {np.unique(df['plan'])}, the unique value of profile is {np.unique(df['profile'])}")
    
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
                count = outmap.shape[0]*outmap.shape[1]
                uniq_val = np.unique(outmap, return_counts=True)
                print(f"unique value: {uniq_val}\n")
                for i in range(len(uniq_val[0])):
                    print(f"\tFor value {uniq_val[0][i]}: {uniq_val[1][i]} / {count} = {uniq_val[1][i]*100/count :.4f} %")


            outmap = np.where(outmap==ds.nodatavals,np.nan,outmap)
            min_val = np.nanmin(outmap)
            max_val = np.nanmax(outmap)
            print(f"""
            min value: {min_val}
            max value: {max_val}
            """)

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
