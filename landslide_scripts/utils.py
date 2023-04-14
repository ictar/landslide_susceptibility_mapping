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

LSM_4CLASSES_RANGE = '(0 <"{layer}" and "{layer}" <=0.25)*1 + (0.25 <"{layer}" and "{layer}" <=0.5)*2 + (0.5 <"{layer}" and "{layer}" <=0.75)*3 + (0.75 <"{layer}" and "{layer}" <=1)*4'
LSM_5CLASSES_RANGE = '(0 <"{layer}" and "{layer}" <=0.2)*1 + (0.2 <"{layer}" and "{layer}" <=0.4)*2 + (0.4 <"{layer}" and "{layer}" <=0.6)*3 + (0.6 <"{layer}" and "{layer}" <=0.9)*4 + (0.9 <"{layer}" and "{layer}" <=1)*5'

def cal_classes_area(layer, classlayer, classarealayer, classsqkmarealayer, exp):
    # 1. transfer layer to classes: Raster calculator
    processing.runAndLoadResults('qgis:rastercalculator', {
                        'LAYERS': [layer],
                        'EXPRESSION':'("slope@1"<=0)*1+("slope@1">0)*"slope@1"',
                        'OUTPUT': classlayer,
                    }, feedback=MyFeedback())
    # 2. calculate class areas:  Raster Layer Unique Values Report
    processing.runAndLoadResults("native:rasterlayeruniquevaluesreport", {
        'INPUT':classlayer,
        'BAND':1,
        'OUTPUT_HTML_FILE':'TEMPORARY_OUTPUT',
        'OUTPUT_TABLE':classarealayer,
    }, feedback=MyFeedback())
    # 3. convert the area to s.q. kms: Vector table ‣ Field Calculator
    processing.runAndLoadResults("native:fieldcalculator", {
        'INPUT':classarealayer,
        'FIELD_NAME':'area_sqkm',
        'FIELD_TYPE':0,
        'FIELD_LENGTH':0,
        'FIELD_PRECISION':0,
        'FORMULA':'round("m2" / 1e6, 2)',
        'OUTPUT':classsqkmarealayer
    }, feedback=MyFeedback())
