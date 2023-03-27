from warnings import catch_warnings
import processing
from qgis.core import QgsRasterLayer, QgsVectorLayer, NULL, edit
from osgeo import osr
import os
import traceback
import numpy as np
from osgeo import gdal

config_path = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/Materials/master_thesis/landslide_scripts"
import sys
if config_path not in sys.path: sys.path.append(config_path)
from config import *
from utils import *

''' Verify that all the data is correctly preprocessed.
For that, check all the environmental rasters to consistent and to have the same
- CRS
- Extent
- Resolution
'''

def _get_crs(rast):
    sr = osr.SpatialReference(wkt=rast.crs().toWkt())
    logging.info(sr)
    return sr.GetAttrValue('AUTHORITY',0) + ":"+sr.GetAttrValue('AUTHORITY',1)
def _get_extent(rast):
    ext = rast.extent()
    return "%d,%d,%d,%d" % (ext.xMinimum(), ext.xMaximum(), ext.yMinimum(), ext.yMaximum())
def _get_resolution(rast):
    return "%f,%f,%f,%f" % (rast.width(), rast.height(), rast.rasterUnitsPerPixelX(), rast.rasterUnitsPerPixelY())

def check_env_raster_properties():
    # use the elevation as reference
    ref_rast = dtm_layer
    ref_ext_ = _get_extent(ref_rast)
    ref_resolution = _get_resolution(ref_rast)

    info = f"""Reference (use DTM) information:
    CRS: {ref_crs}
    Extent: {ref_ext_}
    Resolution (Width, Height, PixelSizeX, PixelSizeY): {ref_resolution}
    """
    logging.info(info)

    for k, v in env_raster.items():
        logging.info(f"Checking {v}")
        try:
            # CRS
            rast = QgsRasterLayer(v, 'raster')
            crs = _get_crs(rast)
            if crs != ref_crs:
                raise Exception(f"{k}'s CRS ({crs}) is not consistente with reference CRS ({ref_crs})")
            # extent
            ext = _get_extent(rast)
            if ext != ref_ext_:
                raise Exception(f"{k}'s Extent ({ext}) is not consistente with reference Extent ({ref_ext_})")
            # resolution
            resolution = _get_resolution(rast)
            if resolution != ref_resolution:
                raise Exception(f"{k}'s Resolution ({resolution}) is not consistente with reference Resolution ({ref_resolution})")
            logging.info(f"""{k} PASS
            CRS: {crs}
            Extent: {ext}
            Resolution: {resolution}
            """)

        except Exception as e:
            logging.error(f"Exception raise when processing {k}: {e}")
            logging.error(traceback.format_exc())

def check_train_test_data():
    # training/testingPointsSampled attribute: not to be any null cells
    # otherwise, fill the cells with {NODATA}
    for layer_name in [train_points_sampled, test_points_sampled]:
        layer = QgsVectorLayer(layer_name)
        with edit(layer):
            for feature in layer.getFeatures():
                attrs = feature.attributeMap()
                for field, attr in attrs.items():
                    if attr == NULL: feature.setAttribute(field, NODATA)

# the pixels of A overlapping with B
def _calculate_raster_unique_value(layer_name, bandnum=1):
    ds = gdal.Open(layer_name)
    band = np.array(ds.GetRasterBand(bandnum).ReadAsArray())
    # get unique values and counts of each value
    unique, counts = np.unique(band, return_counts=True)
    res = {}
    for i in range(len(unique)):
        res[unique[i]] = counts[i]
    return band, res

def overlay_analysis(A, B, Afs, Bfs):
    # convert both into raster
    a_tmp_r = data_dir + "tmp_a.tif"
    vec2raster(A, a_tmp_r, Afs[0])
    b_tmp_r = data_dir + "tmp_b.tif"
    vec2raster(B, b_tmp_r, Bfs[0])

    # load
    a, a_info = _calculate_raster_unique_value(a_tmp_r)
    b, b_info = _calculate_raster_unique_value(b_tmp_r)
    ## diff
    diff = 1 * np.logical_and(a==1, b==1) + 0 * np.logical_not(np.logical_and(a==1, b==1))
    unique, counts = np.unique(diff, return_counts=True)
    diff_info = {unique[i]: counts[i] for i in range(len(unique))}

    report = f"""Layer info:
    a.shape = {a.shape}, b.shape = {b.shape}, diff.shape = {diff.shape}

    -----------------------------------------------------
    Value info:
    A: {a_info}

    B: {b_info}

    diff: {diff_info}
    -----------------------------------------------------
    The overlay estimate:
    #A(value=1) = {a_info.get(1, 0)}
    #B(value=1) = {b_info.get(1, 0)}
    #diff(value=1) = {diff_info.get(1, 0)}

    diff/A (value=1) = {diff_info.get(1, 0)/a_info.get(1, 0)}
    diff/B (value=1) = {diff_info.get(1, 0)/b_info.get(1, 0)}
    """
    print(report)
    logging.info(report)
    # remove
    #for ele in [a_tmp_r, b_tmp_r]: os.remove(ele)


def check_preprocess():
    check_env_raster_properties()
    #check_train_test_data()

base_data_dir = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/ValChiavenna/data/"
#overlay_analysis(base_data_dir+"NLS_fixed.gpkg", base_data_dir+"LS_union.gpkg", [isnlz_field, ], [hazard_field,])
#overlay_analysis(data_dir+"NLS_fixed.gpkg", ls, [isnlz_field, ], [hazard_field,])

#check_env_raster_properties()
