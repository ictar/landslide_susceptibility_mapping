import processing
import numpy as np

config_path = r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/Materials/master_thesis/landslide_scripts"
import sys
if config_path not in sys.path: sys.path.append(config_path)
from config import *


def sample_with_LSM(points, lsm, output):
    # convert image to tif
    # sampling
    output = processing_dir + "testingPointsSampled_with_LSM.gpkg"
    processing.run("sagang:addrastervaluestopoints", {
            'SHAPES':points,
            'GRIDS':[lsm,],
            'RESULT': output,
            'RESAMPLING':0
        }, feedback=MyFeedback())
    
    # MANUAL: export vector to csv

from qgis.core import QgsCoordinateReferenceSystem, QgsVectorFileWriter, QgsVectorLayer
def add_landslide_records_to_LSM():
    for tmp in ['Valchiavenna_2_with', 'Valchiavenna_4th_avgprecip', 'Valchiavenna_5th_90thprecipp', 'Valchiavenna_6_precips']:
        output = f'/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Lombardy/3.results/{tmp}/Neural Network/piff_LSM.gpkg'
        processing.run("sagang:addrastervaluestopoints", {
            'SHAPES':'/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/Lombardy/data/frane_piff_lombardia_opendata/frane_piff_opendataPoint.shp',
            'GRIDS':[f'/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Lombardy/3.results/{tmp}/Neural Network/LSM_Neural Network.tif'],
            'RESULT':output,
            'RESAMPLING':0
        }, feedback=MyFeedback())
        layer = QgsVectorLayer(output)
        output = output.replace('.gpkg', '.csv')
        QgsVectorFileWriter.writeAsVectorFormat(layer, output,"utf-8",driverName = "CSV" , layerOptions = ['GEOMETRY=AS_XY'])

def main():
    sample_with_LSM(
        r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/ValChiavenna/data/v1_WithoutTrainingPoints/testingPoints.gpkg",
        r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/ValChiavenna/processing/v2_WithTrainingPoints/result/ValChiavenna_map.img",
        r"/Users/elexu/Education/Politecnico(GIS-CS)/Thesis/practice/ValChiavenna/processing/v2_WithTrainingPoints/testingPointsSampled_with_LSM_v1_points.gpkg")

add_landslide_records_to_LSM()

if __name__ == '__main__':
    # Supply path to qgis install location
    # use QgsApplication.prefixPath() to see what the first parameter should be
    QgsApplication.setPrefixPath('/Applications/QGIS-LTR.app/Contents/MacOS', True)

    # Create a reference to the QgsApplication.  Setting the
    # second argument to False disables the GUI.
    qgs = QgsApplication([], False)

    # Load providers
    qgs.initQgis()

    # Add the path to Processing framework
    sys.path.append(r"/Applications/QGIS-LTR.app/Contents/PlugIns") # QGIS ver. 3.22
    # Import and initialize Processing framework
    from qgis.analysis import QgsNativeAlgorithms
    import processing
    from processing.core.Processing import Processing
    Processing.initialize() # needed to be able to use the functions afterwards
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

    print("[BEGIN] LET'S PLAY!!!!")
    main()
    
    # Finally, exitQgis() is called to remove the
    # provider and layer registries from memory
    qgs.exitQgis()
    print("[END]CONGRATUATION!!!")