import processing
from qgis.core import (
    QgsRasterLayer, QgsVectorLayer, edit, QgsField, 
    QgsProcessingFeatureSourceDefinition, QgsProject,
    QgsTask, QgsFeatureRequest
    )
from qgis.PyQt.QtCore import QVariant

config_path = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/Materials/landslide_scripts"
import sys
if config_path not in sys.path: sys.path.append(config_path)
from config import *
from utils import *

# define No Landslide Zones: (slope<5) or [(5<slope<20 or slope > 70) and (IUCS>100)]
def NLZ():
    # extract "litologia_mod" with condition "Min_Strength" > 100 and convert it into raster
    lith_mod = data_dir + 'litologia_mod.gpkg|layername=litologia'
    IUCS = data_dir + 'IUCS_min.tif'
    vec2raster(lith_mod, IUCS, "Min_Strength")

    cond = 'logical_or(A<5, logical_and(B>=100, logical_or(A>70, logical_and(A>5,A<20))))'
    NLS = data_dir + 'NLS_zone.tif'
    NLS_paras = {
        'INPUT_A': slope,
        'BAND_A': 1,
        'INPUT_B': IUCS,
        'BAND_B': 1,
        'FORMULA': f'1*{cond} + 0*logical_not({cond})',
        'NO_DATA': NODATA,
        'OUTPUT': NLS,
    }
    processing.runAndLoadResults('gdal:rastercalculator', NLS_paras, feedback=MyFeedback())
    os.remove(IUCS)
                    
    # (Optional) remove small NLS patches of pixels (e.g. 10, 20, 50, etc.)
    ## sieve
    NLS_sieve = data_dir + 'NLS_zone_sieve_{}.tif'
    for threshold in [50,]:#[10, 20, 50]:
        sieve_fname =  NLS_sieve.format(threshold)
        processing.runAndLoadResults('gdal:sieve', {
                            'INPUT': NLS,
                            'THRESHOLD': threshold,
                            'OUTPUT': sieve_fname,
                        }, feedback=MyFeedback())

        # vectorize the resulted NLS raster to obtain the polygons of NLZ
        NLS_v = data_dir + 'NLS_zone_sieve_{}.gpkg'.format(threshold)
        processing.run('grass7:r.to.vect', {
                            'input': sieve_fname,
                            'type': 2, # feature type, 2: area
                            'column': isnlz_field,
                            '-v': True, #Use raster values as categories instead of unique sequence
                            'output': NLS_v,
                        }, feedback=MyFeedback())
        os.remove(sieve_fname)

        # delete the data with value "{NODATA}"
        vt = QgsVectorLayer(NLS_v)
        to_delete = [f.id() for f in vt.getFeatures() if f[isnlz_field] == NODATA]
        if to_delete:
            with edit(vt):
                vt.deleteFeatures(to_delete)
        # create an index against {isnlz_field} to speed up queries
        processing.run('native:createattributeindex', {
            'INPUT':vt,
            'FIELD': isnlz_field,
        }, feedback=MyFeedback())

    os.remove(NLS)
    return NLS_v

from preprocessing_result_check import overlay_analysis
def remove_nls_overlay(LS, NLS):
    # 1. remove the NLS zones that are overlapping with the Landslide Inventory polygons
    # ref: https://gis.stackexchange.com/questions/401235/qgis-difference-tool-feature-has-invalid-geometry-returns-empty-layer
    ## need to fix geometries beforehand
    nls_fixed = data_dir + "NLS_fixed.gpkg"
    glog.pushInfo("[BEGIN] remove_nls_overlay / fix geometries")
    processing.runAndLoadResults("native:fixgeometries", {
        'INPUT': NLS,
        'OUTPUT': nls_fixed,
    }, feedback=glog)
    glog.pushInfo("[END] remove_nls_overlay / fix geometries")
    ## in QGIS: Processing > Vector Overlay > Difference
    nls_rm_overlay = data_dir + 'NLS_rm_overlay.gpkg'
    glog.pushInfo("[BEGIN] remove_nls_overlay / difference")
    processing.runAndLoadResults('native:difference', {
        'INPUT': nls_fixed,
        'OVERLAY': LS,
        'OUTPUT': nls_rm_overlay,
    }, feedback=glog)
    glog.pushInfo("[END] remove_nls_overlay / difference")

    # calculate the overlay ratio
    overlay_analysis(nls_fixed, LS, [isnlz_field, ], [hazard_field,])
    return nls_rm_overlay

# Create new field '{hazard_field}' in the attribute tables of Landslide Inventory or NLS
def _creat_hazard_field(layer_name, val):
    glog.pushInfo(f"[BEGIN] create {hazard_field} for layer {layer_name} ")
    layer = QgsVectorLayer(layer_name)
    # check if there exists the hazard field
    if hazard_field in layer.fields().names():
        glog.pushInfo(f"[SKIP] layer {layer_name} alreay has the field {hazard_field}")
        return
        
    with edit(layer):
        layer.dataProvider().addAttributes([QgsField(hazard_field, QVariant.Int),])
    with edit(layer):
        fidx = layer.fields().indexOf(hazard_field)
        for feat in layer.getFeatures():
            layer.changeAttributeValue(feat.id(), fidx, val)
    glog.pushInfo(f"[END] create {hazard_field} for layer {layer_name} ")

# NOTE: please make sure if this one is needed before doing the sampling!!!!
def preprocess_ls_nls(LS, NLS):
    glog.pushInfo("[BEGIN] preprocess_ls_nls")
    NLS = remove_nls_overlay(LS, NLS)
    ## ADDITION: extract only the NLS zone
    glog.pushInfo("[BEGIN] preprocess_ls_nls / extract the NLS zone")
    processing.run('native:extractbyattribute', {
        'INPUT': NLS,
        'FIELD': isnlz_field,
        'OPERATOR': 0, # 0 — =
        'VALUE': '1',
        'OUTPUT': nls,
    }, feedback=glog)
    glog.pushInfo("[END] preprocess_ls_nls / extract the NLS zone")
    NLS = nls

    glog.pushInfo("[BEGIN] preprocess_ls_nls / create hazard field for LS and NLS")
    _creat_hazard_field(LS, 1)
    _creat_hazard_field(NLS, 0)
    glog.pushInfo("[END] preprocess_ls_nls / create hazard field for LS and NLS")

    # Perform a Union Operation on the LS polygons
    ls_union = data_dir + 'LS_union.gpkg'
    glog.pushInfo("[BEGIN] preprocess_ls_nls / union")
    processing.runAndLoadResults('native:union', {
        'INPUT': LS,
        'OUTPUT': ls_union,
    }, feedback=glog)
    glog.pushInfo("[END] preprocess_ls_nls / union")
    glog.pushInfo("[END] preprocess_ls_nls")
    
    return ls_union, NLS

def _set_train_test_polygon(layer):
    processing.run('qgis:randomselection', {
        'INPUT': layer,
        'METHOD': 1, # 1 — Percentage of selected features
        'NUMBER': TRAIN_PER*100,
    }, feedback=MyFeedback())
    with edit(layer):
        fidx = layer.fields().indexOf(train_test_field)
        for featid in layer.selectedFeatureIds(): # set training
            layer.changeAttributeValue(featid, fidx, train_attr)
        for feat in layer.getFeatures(): # set test
            if feat[train_test_field] != train_attr:
                layer.changeAttributeValue(feat.id(), fidx, test_attr)

def LS_NLS_polygon_sampling(LS, NLS, skip_train_test_create=False):
    # 1. select the percentage of polygons for training/testing accordingly for both LS and NLS
    ## Create new text attribute 'Train_Test' and assign the value 'Training' or 'Testing' accordingly
    ls_layer = QgsVectorLayer(LS)
    nls_layer = QgsVectorLayer(NLS)
    if not skip_train_test_create:
        with edit(ls_layer):
            ls_layer.dataProvider().addAttributes([QgsField(train_test_field, QVariant.String),])
    
        with edit(nls_layer):
            nls_layer.dataProvider().addAttributes([QgsField(train_test_field, QVariant.String),])

    ## In QGIS: Processing > Vector Selection > Random Selection. Invert the selection for moving between training and testing polygons
    ## ---> LS
    _set_train_test_polygon(ls_layer)
    ## ---> NLS
    _set_train_test_polygon(nls_layer)

    # 2. Merge the processed LS layer with the one of NLS
    processing.runAndLoadResults("native:mergevectorlayers", {
        'LAYERS': [ls_layer, nls_layer],
        #'CRS': '',
        'OUTPUT': ls_nls,
    }, feedback=MyFeedback())
    ## # create an index against {hazard_field} and {train_test_field} to speed up queries
    vt = QgsVectorLayer(ls_nls)
    for fld in [hazard_field, train_test_field]:
        processing.run('native:createattributeindex', {
            'INPUT':vt,
            'FIELD': fld,
        }, feedback=MyFeedback())

def _point_sampling(layer, exp, point_num, min_distance, output):
    if point_num == 0: return
    # 1. "Select Features by Value" according to "{hazard_field}" and '{train_test_field}' field
    processing.run('qgis:selectbyexpression', {
        'INPUT': layer,
        'EXPRESSION': exp,
        'METHOD': 0,# 0 — creating new selection
    }, feedback=MyFeedback())
    # 2. selection: Processing > Vector Creation > Random Points in layer bounds
    ## ref: https://gis.stackexchange.com/questions/311336/running-pyqgis-algorithm-on-selected-features-in-layer
    '''processing.runAndLoadResults('qgis:randompointsinlayerbounds', {
        'INPUT': QgsProcessingFeatureSourceDefinition(layer.id(), True), # selected feature only
        'POINTS_NUMBER': point_num,
        'MIN_DISTANCE': min_distance, # it should be the DTM layer resolution in meters
        'OUTPUT': output,
    }, feedback=MyFeedback())'''
    processing.runAndLoadResults("grass7:v.random", {
        'npoints':point_num,'restrict':QgsProcessingFeatureSourceDefinition(layer.id(), selectedFeaturesOnly=True, featureLimit=-1, geometryCheck=QgsFeatureRequest.GeometryAbortOnInvalid),
        'where':'','zmin':0,'zmax':0,'seed':None,'column':'z','column_type':0,'-z':False,'-a':False,
        'output': output,
        'GRASS_REGION_PARAMETER':None,'GRASS_SNAP_TOLERANCE_PARAMETER':-1,'GRASS_MIN_AREA_PARAMETER': min_distance, #Minimum size of area to be imported (square meters). Smaller areas and islands are ignored
        'GRASS_OUTPUT_TYPE_PARAMETER':1, # 1: point
        'GRASS_VECTOR_DSCO':'','GRASS_VECTOR_LCO':'','GRASS_VECTOR_EXPORT_NOCAT':False
    }, feedback=MyFeedback())
    

# Create random points inside the polygons.
def LS_NLS_point_sampling(LS_NLS):
    train_cnt = int(TRAIN_PER * sample_point_cnt)
    test_cnt = sample_point_cnt - train_cnt
    train_hazard_cnt, train_no_hazard_cnt = int(train_cnt/2), train_cnt-int(train_cnt/2)
    test_hazard_cnt, test_no_hazard_cnt = int(test_cnt/2), test_cnt-int(test_cnt/2)
    log = MyFeedback()
    log.pushInfo(f"train_cnt: {train_cnt} = {TRAIN_PER} * {sample_point_cnt}\ntrain_hazard_cnt={train_hazard_cnt}, train_no_hazard_cnt={train_no_hazard_cnt}, test_hazard_cnt={test_hazard_cnt}, test_no_hazard_cnt={test_no_hazard_cnt}")

    # sampling
    layer = QgsVectorLayer(LS_NLS)
    QgsProject.instance().addMapLayer(layer, True)
    
    train_hazard = data_dir + 'train_hazard.gpkg'
    _point_sampling(layer, f""""{hazard_field}"=1 and "{train_test_field}"='{train_attr}'""", train_hazard_cnt, ref_pixel_size, output=train_hazard)
    train_no_hazard = data_dir + 'train_no_hazard.gpkg'
    _point_sampling(layer, f""""{hazard_field}"=0 and "{train_test_field}"='{train_attr}'""", train_no_hazard_cnt, ref_pixel_size, output=train_no_hazard)
    test_hazard = data_dir + 'test_hazard.gpkg'
    _point_sampling(layer, f""""{hazard_field}"=1 and "{train_test_field}"='{test_attr}'""", test_hazard_cnt, ref_pixel_size, output=test_hazard)
    test_no_hazard = data_dir + 'test_no_hazard.gpkg'
    _point_sampling(layer, f""""{hazard_field}"=0 and "{train_test_field}"='{test_attr}'""", test_no_hazard_cnt, ref_pixel_size, output=test_no_hazard)
    
    # merge separately the training layers and testing layers into two point layers
    processing.runAndLoadResults("native:mergevectorlayers", {
        'LAYERS': [train_hazard, train_no_hazard],
        #'CRS': '',
        'OUTPUT': train_points,
    }, feedback=MyFeedback())

    processing.runAndLoadResults("native:mergevectorlayers", {
        'LAYERS': [test_hazard, test_no_hazard],
        #'CRS': '',
        'OUTPUT': test_points,
    }, feedback=MyFeedback())

    _add_hazard_to_point_sampling()

# add {hazard_field}
def _add_hazard_to_point_sampling():
    # train
    layer = QgsVectorLayer(train_points)
    with edit(layer):
        layer.dataProvider().addAttributes([QgsField(hazard_field, QVariant.Int),])
    with edit(layer):
        fidx = layer.fields().indexOf(hazard_field)
        for feat in layer.getFeatures():
            if feat['layer'] == 'train_hazard':
                layer.changeAttributeValue(feat.id(), fidx, 1) 
            if feat['layer'] == 'train_no_hazard':
                layer.changeAttributeValue(feat.id(), fidx, 0)

    # test
    layer = QgsVectorLayer(test_points)
    with edit(layer):
        layer.dataProvider().addAttributes([QgsField(hazard_field, QVariant.Int),])
    with edit(layer):
        fidx = layer.fields().indexOf(hazard_field)
        for feat in layer.getFeatures():
            if feat['layer'] == 'test_hazard':
                layer.changeAttributeValue(feat.id(), fidx, 1) 
            if feat['layer'] == 'test_no_hazard':
                layer.changeAttributeValue(feat.id(), fidx, 0) 

# sampling the environment factors with the training and testeing point layers
# ref: https://gis.stackexchange.com/questions/3538/extracting-raster-values-at-points-using-open-source-gis
def env_point_sampling(layer_name, output):
    processing.runAndLoadResults('saga:addrastervaluestopoints', {
        'SHAPES': layer_name,
        'GRIDS': list(env_raster.values()),
        'RESULT': output,
        'RESAMPLING': 0, # 0: [0] Nearest Neighbor
    }, feedback=MyFeedback())

# sampale points  used for training and testing
def factor_sampling():
    nls_tmp = NLZ()
    new_ls, new_nls = preprocess_ls_nls(ls, nls)
    LS_NLS_polygon_sampling(new_ls, new_nls, skip_train_test_create=False)
    LS_NLS_point_sampling(ls_nls)
    env_point_sampling(train_points, train_points_sampled)
    env_point_sampling(test_points, test_points_sampled)

# sample for test
def sampling_test(total):
    glog.pushInfo(f"[BEGIN] sampling_test: total={total}")
    hazard_cnt = int(total/2)
    no_hazard_cnt = total - hazard_cnt
    # sample
    layer = QgsVectorLayer(ls_nls)
    QgsProject.instance().addMapLayer(layer, True)

    glog.pushInfo(f"[BEGIN] sampling_test / hazard = {hazard_cnt}, no_hazard = {no_hazard_cnt}")
    hazard = data_dir + 'hazard.gpkg'
    #_point_sampling(layer, f'"{hazard_field}"=1', hazard_cnt, ref_pixel_size, output=hazard)
    _point_sampling(layer, f""""{hazard_field}"=1 and "{train_test_field}"='{test_attr}'""", hazard_cnt, ref_pixel_size, output=hazard)
    no_hazard = data_dir + 'no_hazard.gpkg'
    #_point_sampling(layer, f'"{hazard_field}"=0', no_hazard_cnt, ref_pixel_size, output=no_hazard)
    _point_sampling(layer, f""""{hazard_field}"=0 and "{train_test_field}"='{test_attr}'""", no_hazard_cnt, ref_pixel_size, output=no_hazard)
    glog.pushInfo(f"[END] sampling_test / hazard = {hazard_cnt}, no_hazard = {no_hazard_cnt}")

    # merge separately the hazard layers and no_hazard layers into one point layer
    glog.pushInfo(f"[BEGIN] sampling_test / merge hazard layers and no_hazard layers ")
    processing.runAndLoadResults("native:mergevectorlayers", {
        'LAYERS': [hazard, no_hazard],
        'OUTPUT': test_points,
    }, feedback=MyFeedback())

    # add {hazard_field}
    layer = QgsVectorLayer(test_points)
    with edit(layer):
        layer.dataProvider().addAttributes([QgsField(hazard_field, QVariant.Int),])
    with edit(layer):
        fidx = layer.fields().indexOf(hazard_field)
        for feat in layer.getFeatures():
            if feat['layer'] == 'hazard':
                layer.changeAttributeValue(feat.id(), fidx, 1) 
            if feat['layer'] == 'no_hazard':
                layer.changeAttributeValue(feat.id(), fidx, 0) 
    glog.pushInfo(f"[END] sampling_test / merge hazard layers and no_hazard layers ")
    
    glog.pushInfo(f"[END] sampling_test")

factor_sampling()
#sampling_test(2*200*1000)
