import sys
from qgis.core import QgsApplication

def main():
    #sampling_test(2*200*1000)
    #overlay_analysis(data_dir+"NLS_fixed.gpkg", ls, [isnlz_field, ], [hazard_field,])
    check_env_raster_properties()


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
    """for alg in QgsApplication.processingRegistry().algorithms():
        print(f"{alg.provider().name()}:{alg.name()} --> {alg.displayName()}")"""


    print("[BEGIN] LET'S PLAY!!!!")

    from preprocessing_result_check import *
    from factor_sampling import *
    main()

    # Finally, exitQgis() is called to remove the
    # provider and layer registries from memory
    qgs.exitQgis()
    print("[END]CONGRATUATION!!!")