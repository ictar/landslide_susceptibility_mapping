# INPUT
#base_dir = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/ValChiavenna/"
#base_dir = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/Lombardy/"
base_dir = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Lombardy/"
#data_dir = base_dir + r"data/"
data_dir = base_dir + r"1.factors/"
processing_dir = base_dir + r"processing/"
dtm = data_dir + "dtm.tif"

road_buffer = data_dir + 'road_buffer.gpkg'
river_buffer = data_dir + 'river_buffer.gpkg'
faults_buffer = data_dir + 'faults_buffer.gpkg'
landuse = data_dir + 'dusaf_lombardia.gpkg'
lithology = data_dir + 'litologia.gpkg'
ndvi = data_dir + 'NDVI.tif'

ls = data_dir + "LS_inventory.gpkg"

# SAVE
slope = data_dir + "slope.tif"
aspect = data_dir + "aspect.tif"
plan = data_dir + "plan.tif"
profile = data_dir + "profile.tif"
curv = data_dir + 'TEMPORARY_OUTPUT'
tang_curv = 'TEMPORARY_OUTPUT' #data_dir + "tang_curv.tif"
long_curv = 'TEMPORARY_OUTPUT' #data_dir + "long_curv.tif"
cros_curv = 'TEMPORARY_OUTPUT' #data_dir + "cros_curv.tif"
mini_curv = 'TEMPORARY_OUTPUT' #data_dir + "mini_curv.tif"
maxi_curv = 'TEMPORARY_OUTPUT' #data_dir + "maxi_curv.tif"
tota_curv = 'TEMPORARY_OUTPUT' #data_dir + "tota_curv.tif"
roto_curv = 'TEMPORARY_OUTPUT' #data_dir + "roto_curv.tif"

eastness = data_dir + "east.tif"
northness = data_dir + "north.tif"

twi = data_dir + "twi.tif"


road_dist = data_dir + 'roads.tif'
river_dist = data_dir + 'rivers.tif'
faults_dist = data_dir + 'faults.tif'

landuse_r = data_dir + 'dusaf.tif'

lithology_r = data_dir + 'lithologia.tif'

env_raster = {
    'elevation': dtm,
    'slope': slope,
    'aspect': aspect,
    'eastness': eastness,
    'northness': northness,
    'road_dist': road_dist,
    'river_dist': river_dist,
    'faults_dist': faults_dist,
    'twi': twi,
    'ndvi': ndvi,
    'landuse': landuse_r,
    'lithology': lithology_r,
    'plan_curv': plan,
    'profile_curv': profile,
    #'precipitation': data_dir + 'precipitation.tif',
}

nls = data_dir + "NLS_final.gpkg"
ls_nls = data_dir + 'LS_NLS.gpkg'

# LAYER PROPERTIES
from qgis.core import QgsRasterLayer
from osgeo import osr

dtm_layer = None
ref_pixel_size, ref_width, ref_height = None, None, None
ref_ext, ref_sr, ref_crs, ref_extent = None, None, None, None

def set_reference_layer_properties(layer_name):
    global dtm_layer, ref_pixel_size, ref_width, ref_height, ref_ext, ref_sr, ref_crs, ref_extent

    dtm_layer = QgsRasterLayer(layer_name, 'raster')

    ref_pixel_size = max(dtm_layer.rasterUnitsPerPixelX(), dtm_layer.rasterUnitsPerPixelY())
    ref_width, ref_height = dtm_layer.width(), dtm_layer.height()
    ref_ext = dtm_layer.extent()
    ref_sr = osr.SpatialReference(wkt=dtm_layer.crs().toWkt())
    print(ref_sr.GetAttrValue('AUTHORITY',0), ref_sr.GetAttrValue('AUTHORITY',1))
    ref_crs = ref_sr.GetAttrValue('AUTHORITY',0) + ":" + ref_sr.GetAttrValue('AUTHORITY',1)
    ref_extent = "%f,%f,%f,%f [%s]" % (ref_ext.xMinimum(), ref_ext.xMaximum(), ref_ext.yMinimum(), ref_ext.yMaximum(), ref_crs)

import traceback
try:
    set_reference_layer_properties(dtm)
except Exception as e:
    logging.error(e)
    logging.error(traceback.format_exc())
    
# FIELD and ATTRIBUTE
isnlz_field = 'IS_NLZ'
hazard_field = 'Hazard'
train_test_field = 'Train_Test'
train_attr, test_attr = 'Training', 'Testing'

# CONSTANT
sample_point_cnt = 80*1000 + 25000
NODATA = -9999
TRAIN_PER, TEST_PER = 0.8, 0.2
assert TRAIN_PER + TEST_PER == 1.0

# OTHERS
train_points_sampled = data_dir+'trainingPointsSampled.gpkg'
test_points_sampled = data_dir+'testingPointsSampled.gpkg'
test_points = data_dir + 'testingPoints.gpkg'
train_points = data_dir + 'trainingPoints.gpkg'

# Log
import logging
import logging.handlers
import os
from qgis.core import QgsProcessingFeedback

log_path = os.path.join(data_dir,"run.log")
handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", log_path))
formatter = logging.Formatter("[%(asctime)s] " + logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "DEBUG"))
root.addHandler(handler)
class MyFeedback(QgsProcessingFeedback):
    def setProgressText(self, text):
        logging.info(text)

    def pushInfo(self, info):
        logging.info(info)

    def pushCommandInfo(self, info):
        logging.info(info)

    def pushDebugInfo(self, info):
        logging.debug(info)

    def pushConsoleInfo(self, info):
        logging.info(info)

    def reportError(self, error, fatalError=False):
        logging.error(error)

glog = MyFeedback()
# external command information
import os
SAGA_MLB=r"/Applications/QGIS-LTR.app/Contents/MacOS/bin/../lib/saga"
PATH=r"/Applications/QGIS-LTR.app/Contents/MacOS/bin:{}"
try:
    PATH = PATH.format(os.environ['PATH'])
    os.environ['SAGA_MLB'] = SAGA_MLB
    os.environ['PATH'] = PATH
except Exception as e:
    print(e)