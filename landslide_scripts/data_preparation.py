import processing
from qgis.core import QgsRasterLayer

config_path = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/Materials/master_thesis/landslide_scripts"
import subprocess, sys, os
if config_path not in sys.path: sys.path.append(config_path)
from config import *
from utils import *

# From DTM to Slope/Aspect/Eastness/Northness/ProfileCurvature/PlanCurvature/TWI

def dtm_process():
    # interpolate voids in the DTM: GDAL|Fill nodata
    dtm_raw = data_dir + 'dtm_raw.tif'
    processing.runAndLoadResults("gdal:fillnodata", {
                        'INPUT': dtm_raw,
                        'BAND': 1,
                        'DISTANCE': 10,
                        'OUTPUT': dtm,
                    }, feedback=MyFeedback())
    # set reference layer properties
    set_reference_layer_properties(dtm)
    

def slope_aspect_curvature():
    mylog = MyFeedback()
    ## ref: https://docs.qgis.org/3.22/en/docs/user_manual/processing/console.html
    method = 6
    #slope_sdat = data_dir + "slope.sdat"
    #aspect_sdat = data_dir + "aspect.sdat"
    plan_sdat = data_dir + "plan.sdat"
    profile_sdat = data_dir + "profile.sdat"
    """PROBLEM WHEN HANDLING LARGE DTM
    processing.run('saga:slopeaspectcurvature', {
                    'ELEVATION': dtm,
                    'UNIT_SLOPE': 1, # 1-degree
                    'UNIT_ASPECT': 1, # 1-degree
                    'SLOPE': slope_sdat,
                    'ASPECT': aspect_sdat,
                    'C_PLAN': plan_sdat,
                    'C_PROF': profile_sdat,
                    'C_GENE': " ", #curv,
                    'C_TANG': " ", #tang_curv,
                    'C_LONG': " ", #long_curv,
                    'C_CROS': " ", #cros_curv,
                    'C_MINI': " ", #mini_curv,
                    'C_MAXI': " ", #maxi_curv,
                    'C_TOTA': " ", #tota_curv,
                    'C_ROTO': " ", #roto_curv,
                }, feedback=MyFeedback())"""
    ### run using the command directly
    dtm_grid = data_dir + "dtm.sgrd"
    # f"""saga_cmd io_gdal 0 -TRANSFORM 1 -RESAMPLING 3 -GRIDS "{dtm_grid}" -FILES "{dtm}" &>> {log_path} """
    cmds = ["saga_cmd", "io_gdal", "0", "-TRANSFORM", "1", "-RESAMPLING", "3", "-GRIDS", f'"{dtm_grid}"', "-FILES", f'"{dtm}"']
    mylog.pushInfo(f"SAGA CMD: {' '.join(cmds)}")
    called = subprocess.run(cmds,  capture_output=True, text=True)
    mylog.pushInfo(called.stdout)
    if called.stderr: mylog.reportError(called.stderr)
    #f"""saga_cmd ta_morphometry "Slope, Aspect, Curvature"  -ELEVATION "{dtm_grid}" -UNIT_SLOPE 1 -UNIT_ASPECT 1 -SLOPE "{slope_sdat}" -ASPECT "{aspect_sdat}" -C_PLAN "{plan_sdat}" -C_PROF "{profile_sdat}" &>> {log_path}"""
    #cmds = ["saga_cmd", "ta_morphometry", "0",  "-ELEVATION", f'"{dtm_grid}"', "-UNIT_SLOPE", "1", "-UNIT_ASPECT", "1", "-SLOPE", f'"{slope_sdat}"', "-ASPECT", f'"{aspect_sdat}"', "-C_PLAN", f'"{plan_sdat}"', "-C_PROF", f'"{profile_sdat}"']
    cmds = ["saga_cmd", "ta_morphometry", "0",  "-ELEVATION", f'"{dtm_grid}"', "-UNIT_SLOPE", "1", "-UNIT_ASPECT", "1", "-C_PLAN", f'"{plan_sdat}"', "-C_PROF", f'"{profile_sdat}"']
    mylog.pushInfo(f"SAGA CMD: {' '.join(cmds)}")
    called = subprocess.run(cmds, capture_output=True, text=True)
    mylog.pushInfo(called.stdout)
    if called.stderr: mylog.reportError(called.stderr)
    ## delete dtm.sgrd
    remove_sdat(dtm_grid[:-5])

    ### note that SAGA tools in processing always output grids as SDAT format or a temporary file.
    ## sdat to tif
    sdat2tif([
        #(slope_sdat, slope),
        #(aspect_sdat, aspect),
        (plan_sdat, plan),
        (profile_sdat, profile),   
    ], removed=True)

    # slope
    processing.run("native:slope", {
        'INPUT': dtm,
        'OUTPUT': slope,
    }, feedback=MyFeedback())

    # aspect
    processing.run("native:aspect", {
        'INPUT': dtm,
        'OUTPUT': aspect,
    }, feedback=MyFeedback())

# TOFIX: generate by aspect (in radian, not degree), not dtm. Notice that by now, the aspect generated is in radian
def east_north():      
    ## eastness and northness
    east_paras = {
        'INPUT_A': aspect,
        'BAND_A': 1,
        'FORMULA': 'sin(A*0.01745)',
        'OUTPUT': eastness,
    }
    processing.runAndLoadResults('gdal:rastercalculator', east_paras, feedback=MyFeedback())

    north_paras = {
        'INPUT_A': aspect,
        'BAND_A': 1,
        'FORMULA': 'cos(A*0.01745)',
        'OUTPUT': northness,
    }
    processing.runAndLoadResults('gdal:rastercalculator', north_paras, feedback=MyFeedback())

## TWI: ln(a/tan(b)), where a=uplope contributing area, b=slope in radians
### calcuate TWI using SAGA
## ref: https://courses.gisopencourseware.org/mod/book/view.php?id=41
def TWI():
    # convert slope in degree to radians
    ## in order to calculate "ln", we need to make sure there is no pixels with a sope of 0 degrees
    slope_n0 = data_dir + "slope_n0.tif"
    processing.runAndLoadResults('qgis:rastercalculator', {
                        'LAYERS': [slope],
                        'EXPRESSION':'("slope@1"<=0)*1+("slope@1">0)*"slope@1"',
                        'OUTPUT': slope_n0,
                    }, feedback=MyFeedback())
    ## degree to radian
    slope_rad = data_dir + "slope_radians.tif"
    processing.runAndLoadResults('gdal:rastercalculator', {
                        'INPUT_A': slope_n0,
                        'BAND_A': 1,
                        'FORMULA': "A*0.01745", # radians(A)
                        'OUTPUT': slope_rad,
                    }, feedback=MyFeedback())
    os.remove(slope_n0)
    # calculate the contributing upslope area for each pixel of the DTM
    area = data_dir + "contributing_upslope_area.tif"
    """#THIS PART IS VERY SLOW
    processing.runAndLoadResults('saga:flowaccumulationqmofesp', {
                        'DEM': dtm_voidfilled,
                        'PREPROC': 1,
                        'DZFILL': 0.01,
                        'FLOW': area,
                    }, feedback=MyFeedback())"""
    processing.runAndLoadResults("wbt:FD8FlowAccumulation", {
        'dem':dtm,
        'out_type':1, # 1: specific contributing area
        'exponent':1.1,
        'threshold':None, # Convergence Threshold (grid cells; blank for none)
        'log':False,
        'clip':False,
        'output':area}, feedback=MyFeedback())

    # calculate TWI
    """processing.runAndLoadResults('gdal:rastercalculator', {
                    'INPUT_A': area,
                    'BAND_A': 1,
                    'INPUT_B': slope_rad,
                    'INPUT_B': 1,
                    'FORMULA': "log((A*5*5)/tan(B))", # ln((A*5*5)/tan(B))
                    'OUTPUT': twi,
                }, feedback=MyFeedback())"""
    processing.runAndLoadResults("wbt:WetnessIndex", {
        'sca':area,
        'slope': slope_rad,
        'output':twi}, feedback=MyFeedback())

    # remove for save space
    for ele in [area, slope_rad]:
        os.remove(ele)
    

def TWI_grass():
    processing.runAndLoadResults('grass7:r.topidx', {
        "input": dtm,
        "output": twi,
    }, feedback=MyFeedback())

def vector_data_handling():
    ## road network
    road_tmp = data_dir + "roads_tmp.tif"
    vec2raster(road_buffer, road_dist, 'distance', dtype=2)

    ## river network
    river_tmp = data_dir + "rivers_tmp.tif"
    vec2raster(river_buffer, river_dist, 'distance', dtype=2)
                    
    ## fault lines
    faults_tmp = data_dir+"faults_tmp.tif"
    vec2raster(faults_buffer, faults_dist, 'distance', dtype=2)
              
    ## land use: categorize the features and then convert to raster
    vec2raster(landuse, landuse_r, '2-CODICE', dtype=2)

    ## lithology: categorize the features and then convert to raster
    vec2raster(lithology, lithology_r, 'TIPO_EL', dtype=2)


def prepare_data():
    dtm_process()
    slope_aspect_curvature()
    east_north()
    TWI() #TWI_grass()
    # Convert vector data to raster with same pixel size
    vector_data_handling()


#prepare_data()

east_north()