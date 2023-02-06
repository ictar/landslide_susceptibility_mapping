config_path = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/Materials/landslide_scripts"
import sys, os
if config_path not in sys.path: sys.path.append(config_path)
from config import *

import processing

## ref: https://docs.qgis.org/3.22/en/docs/user_manual/processing_algs/gdal/vectorconversion.html
def vec2raster(input, output, field, dtype=5, nodata=0):
    processing.runAndLoadResults("gdal:rasterize", {
                    'INPUT': input,
                    'FIELD': field,
                    'UNITS': 0, # pixels
                    'WIDTH': ref_width,
                    'HEIGHT': ref_height,
                    'EXTENT': ref_extent,
                    'NODATA': nodata,
                    'DATA_TYPE': dtype,
                    'OUTPUT': output,
                }, feedback=MyFeedback())

## sdat to tif
def sdat2tif(sdats, removed=False):
    for ele in sdats:
        processing.runAndLoadResults("gdal:translate", {
            'INPUT': ele[0],
            'TARGET_CRS':None,
            'NODATA':None,
            'COPY_SUBDATASETS':False,
            'OPTIONS':'',
            'EXTRA':'',
            'DATA_TYPE':0,
            'OUTPUT':ele[1]
            }, feedback=MyFeedback())
        if removed:
            remove_sdat(ele[0][:-5])


def remove_sdat(sdat_name):
    posts = [".prj", ".mgrd", ".sdat", ".sgrd", ".sdat.aux.xml"]
    for ele in map(lambda p: sdat_name+p, posts):
        try:
            os.remove(ele)
        except Exception as e:
            print(e)