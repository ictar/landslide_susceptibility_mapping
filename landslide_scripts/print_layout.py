'''
This script aims to generate the layout for different map, such as different factors of different regions.

Please use it in QGIS.

Life is short, don't waste time on duplication!

ref: https://data.library.virginia.edu/how-to-create-and-export-print-layouts-in-python-for-qgis-3/
ref: https://www.geodose.com/2022/02/pyqgis-tutorial-automating-map-layout.html
'''
import os
from qgis.core import (
    QgsProject, QgsUnitTypes, QgsLayerTree, QgsMapLayerLegendUtils,
    QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLegend,
    QgsLayoutItemScaleBar, QgsLayoutItemPicture, QgsLayoutExporter,
    QgsLayoutPoint, QgsLayoutSize, QgsLayoutItemMapGrid,
    QgsUnitTypes, QgsLayoutItemMapGrid,
    QgsCoordinateReferenceSystem,
)
from qgis.utils import iface
from PyQt5.QtGui import QFont, QColor, QPainter
from PyQt5.QtCore import QRectF

"""
Layout contains
- map
- legend
- scale bar
- north arrow
"""
def print_layout(layoutName, boundaryLegendName, legendName, config, save_to):
    """This creates a new print layout"""
    project = QgsProject.instance()             #gets a reference to the project instance
    manager = project.layoutManager()           #gets a reference to the layout manager

    layouts_list = manager.printLayouts()
    for layout in layouts_list:
        if layout.name() == layoutName:
            manager.removeLayout(layout)
            
    layout = QgsPrintLayout(project) #makes a new print layout object, takes a QgsProject as argument
    layout.initializeDefaults()                 #create default map canvas
    layout.setName(layoutName)
    manager.addLayout(layout)

    """This adds a map item to the Print Layout"""
    map = QgsLayoutItemMap(layout)
    map.setRect(QRectF(config["map"]["position"][0], config["map"]["position"][1], config["map"]["size"][0], config["map"]["size"][1]))
    map.zoomToExtent(iface.mapCanvas().extent())   #sets map extent to current map canvas
    #map.setFrameEnabled(True)
    layout.addLayoutItem(map)
    #Move & Resize
    map.attemptMove(QgsLayoutPoint(config["map"]["position"][0], config["map"]["position"][1], QgsUnitTypes.LayoutMillimeters))
    map.attemptResize(QgsLayoutSize(config["map"]["size"][0], config["map"]["size"][1], QgsUnitTypes.LayoutMillimeters))
    # add grid
    map.grid().setEnabled(True)
    ## Appearance
    map.grid().setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
    map.grid().setUnits(QgsLayoutItemMapGrid.MapUnit)
    map.grid().setIntervalX(config["map"]["grid_interval"][0]) # Sets the interval between grid lines in the x-direction.
    map.grid().setIntervalY(config["map"]["grid_interval"][1]) # Sets the interval between grid lines in the y-direction.
    map.grid().setGridLineColor(QColor(197, 197, 197)) # Sets the color of grid lines.
    map.grid().setGridLineWidth(0.3)
    map.grid().setBlendMode(QPainter.CompositionMode_SourceOver)
    ## Frame
    map.grid().setFrameStyle(QgsLayoutItemMapGrid.Zebra)
    #map.grid().setFrameWidth(1.5)
    map.grid().setFramePenSize(0.5)
    map.grid().setFramePenColor(QColor(0, 0, 0))
    map.grid().setFrameFillColor1(QColor(0, 0, 0))
    map.grid().setFrameFillColor2(QColor(255, 255, 255))
    map.grid().setFrameSideFlag(QgsLayoutItemMapGrid.FrameLeft, True)
    map.grid().setFrameSideFlag(QgsLayoutItemMapGrid.FrameRight, True)
    map.grid().setFrameSideFlag(QgsLayoutItemMapGrid.FrameTop, True)
    map.grid().setFrameSideFlag(QgsLayoutItemMapGrid.FrameBottom, True)
    ## Draw Coordinates
    ### ref: https://gis.stackexchange.com/questions/389070/qgis-3-16-print-layout-grid-coordinates-in-dms
    map.grid().setAnnotationEnabled(True)
    map.grid().setAnnotationFormat(QgsLayoutItemMapGrid.DegreeMinuteSecond)
    map.grid().setAnnotationPrecision(0)
    #map.grid().setAnnotationFrameDistance(0.1)
    map.grid().setAnnotationPosition(QgsLayoutItemMapGrid.OutsideMapFrame, QgsLayoutItemMapGrid.Top)
    #map.grid().setAnnotationDirection(QgsLayoutItemMapGrid.Horizontal, QgsLayoutItemMapGrid.Top)
    #map.grid().setAnnotationDirection(QgsLayoutItemMapGrid.Horizontal, QgsLayoutItemMapGrid.Bottom)
    map.grid().setAnnotationPosition(QgsLayoutItemMapGrid.OutsideMapFrame, QgsLayoutItemMapGrid.Left)
    map.grid().setAnnotationDirection(QgsLayoutItemMapGrid.Vertical, QgsLayoutItemMapGrid.Left)
    #map.grid().setAnnotationPosition(QgsLayoutItemMapGrid.InsideMapFrame, QgsLayoutItemMapGrid.Right)
    map.grid().setAnnotationDirection(QgsLayoutItemMapGrid.Vertical, QgsLayoutItemMapGrid.Right)
    map.grid().setAnnotationFont(QFont("Arial",8))
    map.grid().setAnnotationFontColor(QColor(0, 0, 0))
    
    map.updateBoundingRect()
    
    """Gathers active layers to add to legend"""
    #Checks layer tree objects and stores them in a list. This includes csv tables
    checked_layers = [layer.name() for layer in QgsProject().instance().layerTreeRoot().children() if layer.isVisible()]
    print(f"Adding {checked_layers} to legend." )
    #get map layer objects of checked layers by matching their names and store those in a list
    layersToAdd = [layer for layer in QgsProject().instance().mapLayers().values() if layer.name() in checked_layers]
    root = QgsLayerTree()
    for layer in layersToAdd:
        #add layer objects to the layer tree
        treeLayer = root.addLayer(layer)
        treeLayer.setUseLayerName(False)
        if layer.name() == layoutName:
            treeLayer.setName(legendName)
        else:
            if boundaryLegendName:
                treeLayer.setName(boundaryLegendName)
            

    """This adds a legend item to the Print Layout"""
    legend = QgsLayoutItemLegend(layout)
    legend.setTitle("Legend")
    legend.setFrameEnabled(True)
    legend.model().setRootGroup(root)
    layout.addLayoutItem(legend)
    legend.attemptMove(QgsLayoutPoint(config["legend"]["position"][0], config["legend"]["position"][1], QgsUnitTypes.LayoutMillimeters))
    
    model = legend.model()
    tmp_root = model.rootGroup().findLayer(project.mapLayersByName(layoutName)[0])
    nodes = model.layerLegendNodes(tmp_root)
    if nodes[0].data(0) in ('Band 1 (Gray)', 'Band 1 (Palette)'):
        indexes = list(range(1, len(nodes)))
        QgsMapLayerLegendUtils.setLegendNodeOrder(tmp_root, indexes)
        model.refreshLayerLegend(tmp_root)


    """This adds a scale bar to the Print Layout"""
    scale = QgsLayoutItemScaleBar(layout)
    # Main properties
    scale.setLinkedMap(map)
    scale.setStyle('Single Box')
    scale.setUnits(QgsUnitTypes.DistanceKilometers) # Sets the distance units used by the scalebar.
    scale.setUnitsPerSegment(2.5) # Sets the number of scalebar units per segment.
    scale.setNumberOfSegments(2)
    scale.setFont(QFont("Arial",15))
    scale.setFontColor(QColor("Black"))
    scale.setFillColor(QColor("Black"))
    scale.applyDefaultSize(QgsUnitTypes.DistanceMeters)
    scale.setMapUnitsPerScaleBarUnit(1000.0)
    scale.setUnitLabel("km")
    scale.setFrameEnabled(True)
    scale.setBackgroundEnabled(True)
    scale.update()
    layout.addLayoutItem(scale)
    scale.attemptMove(QgsLayoutPoint(config["scale"]["position"][0], config["scale"]["position"][1],QgsUnitTypes.LayoutMillimeters))


    """This adds a north arrow to the Print Layout"""
    north=QgsLayoutItemPicture(layout)
    north.setMode(QgsLayoutItemPicture.FormatSVG)
    north.setPicturePath(":/images/north_arrows/layout_default_north_arrow.svg")
    north.attemptMove(QgsLayoutPoint(config["arrow"]["position"][0], config["arrow"]["position"][1], QgsUnitTypes.LayoutMillimeters))
    north.attemptResize(QgsLayoutSize(*config["arrow"]["size"], QgsUnitTypes.LayoutMillimeters))
    layout.addLayoutItem(north)

    """This exports a Print Layout as an image"""
    exporter = QgsLayoutExporter(layout)                #this creates a QgsLayoutExporter object
    exporter.exportToImage(save_to, QgsLayoutExporter.ImageExportSettings())  #this exports an image of the layout object


def print_factor_layout(config):
    prj = QgsProject.instance()
    # set all layer unvisible
    for layer in prj.layerTreeRoot().children(): # layer type: QgsLayerTreeLayer
        layer.setItemVisibilityCheckedParentRecursive(False)
        # make boundary layer visible
        # ref: https://stackoverflow.com/questions/59720855/pyqgis-make-layers-visible-and-invisible
        if layer.name() == config['boundary']:
            layer.setItemVisibilityCheckedParentRecursive(True)

    factors = config['factors']
    for factor_name in factors:
        # find layer
        layer = prj.mapLayersByName(factor_name)[0]
        # set visibility
        prj.layerTreeRoot().findLayer(layer.id()).setItemVisibilityCheckedParentRecursive(True)
        # load style
        # ref: https://gis.stackexchange.com/questions/386890/symbolizing-raster-layer-with-python-qgis
        layer.loadNamedStyle(factors[factor_name]["style"])
        iface.layerTreeView().refreshLayerSymbology(layer.id())
        layer.triggerRepaint()

        # print
        print_layout(factor_name, config["boundaryLegendName"], factors[factor_name]["legendname"], config["layout"], os.path.join(config['save_to'], f"{factor_name}.png"))

        # set invisibility
        prj.layerTreeRoot().findLayer(layer.id()).setItemVisibilityCheckedParentRecursive(False)

def print_LSM_layout(config):
    prj = QgsProject.instance()
    # set all layer unvisible
    for layer in prj.layerTreeRoot().children(): # layer type: QgsLayerTreeLayer
        layer.setItemVisibilityCheckedParentRecursive(False)

    for lsm_name, info in config['lsms'].items():
        # find layer
        layer = prj.mapLayersByName(lsm_name)[0]
        # set visibility
        prj.layerTreeRoot().findLayer(layer.id()).setItemVisibilityCheckedParentRecursive(True)
        # load style
        # ref: https://gis.stackexchange.com/questions/386890/symbolizing-raster-layer-with-python-qgis
        layer.loadNamedStyle(config["style"])
        iface.layerTreeView().refreshLayerSymbology(layer.id())
        layer.triggerRepaint()

        # print
        print_layout(lsm_name, "", info["legendname"], config["layout"], os.path.join(config['save_to'], f"{lsm_name}.png"))

        # set invisibility
        prj.layerTreeRoot().findLayer(layer.id()).setItemVisibilityCheckedParentRecursive(False)


def Upper_Valtellina():
    style_path = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Upper Valtellina/symbology/"
    aoi_name = "Upper Valtellina"
    config = {
        "save_to": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Upper Valtellina/1.factors/",
        "boundary": "UpperValtellina",
        "boundaryLegendName": f"{aoi_name} AOI",
        "factors": {
            'dtm': {"style": os.path.join(style_path, 'dtm.qml'), "legendname": f"{aoi_name} DTM [m]"},
            'east': {"style": os.path.join(style_path, 'east.qml'), "legendname": f"{aoi_name} eastness"},
            'faults': {"style": os.path.join(style_path, 'faults.qml'), "legendname": f"{aoi_name}\n distance from faults [m]"},
            'dusaf': {"style": os.path.join(style_path, 'landuse.qml'), "legendname": f"{aoi_name} land use"},
            'ndvi': {"style": os.path.join(style_path, 'ndvi.qml'), "legendname": f"{aoi_name} NDVI"},
            'north': {"style": os.path.join(style_path, 'north.qml'), "legendname": f"{aoi_name} northness"},
            'plan': {"style": os.path.join(style_path, 'plan.qml'), "legendname": f"{aoi_name} plan curvature"},
            'profile': {"style": os.path.join(style_path, 'profile.qml'), "legendname": f"{aoi_name} profile curvature"},
            'rivers': {"style": os.path.join(style_path, 'rivers.qml'), "legendname": f"{aoi_name}\n distance from rivers [m]"},
            'roads': {"style": os.path.join(style_path, 'roads.qml'), "legendname": f"{aoi_name}\n distance from roads [m]"},
            'twi': {"style": os.path.join(style_path, 'twi.qml'), "legendname": f"{aoi_name} TWI"},
        },
        "layout": {
           "map": {"position": (10, 10), "size": (200, 170), "grid_interval": (0.2, 0.1)},
            "arrow": {"position": (20, 15), "size": [22,27]},
            "legend": {"position": (230, 10), "size": ()},
            "scale": {"position": (230,164), "size": ()},
        },
    }

    print_factor_layout(config)
   
#Upper_Valtellina()

def Upper_Valtellina_LSM(style_paths):
    aoi_name = 'Upper Valtellina'
    save_to = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Upper Valtellina/3.results/"
    config = {
        "save_to": save_to,
        "lsms": {
            'LSM_Bagging': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_AdaBoost': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_AdaBoost Calibrated': {"legendname": f"{aoi_name}\nSusceptibility Map"}, 
            'LSM_Fortests of randomized trees': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Gradient Tree Boosting': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Neural Network': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Ensemble Soft Voting': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Ensemble Blending': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Ensemble Stacking': {"legendname": f"{aoi_name}Susceptibility Map"},       
        },
        "layout": {
           "map": {"position": (10, 10), "size": (200, 170), "grid_interval": (0.1, 0.05)}, # VT
            "arrow": {"position": (20, 25), "size": [22,27]},
            "legend": {"position": (230, 10), "size": ()},
            "scale": {"position": (230,164), "size": ()},
        },
    }

    for style_name, style_path in style_paths.items():
        config['save_to'] = os.path.join(save_to, style_name)
        config['style'] = style_path
        print_LSM_layout(config)

#Upper_Valtellina_LSM({"LSM_4class": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/symbology/LSM_4classes.qml", "LSM_5class": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/symbology/LSM_5classes.qml",})
#Upper_Valtellina_LSM({"LSM_5class": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/symbology/LSM_5classes.qml",})

def Val_Tartano():
    style_path = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Val Tartano/symbology/"
    aoi_name = "Val Tartano"
    config = {
        "save_to": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Val Tartano/1.factors/",
        "boundary": "ValTartano_Boundary",
        "boundaryLegendName": f"{aoi_name} AOI",
        "factors": {
            'dtm': {"style": os.path.join(style_path, 'dtm.qml'), "legendname": f"{aoi_name} DTM [m]"},
            'east': {"style": os.path.join(style_path, 'east.qml'), "legendname": f"{aoi_name} eastness"},
            'faults': {"style": os.path.join(style_path, 'faults.qml'), "legendname": f"{aoi_name}\n distance from faults [m]"},
            'dusaf': {"style": os.path.join(style_path, 'landuse.qml'), "legendname": f"{aoi_name} land use"},
            'ndvi': {"style": os.path.join(style_path, 'ndvi.qml'), "legendname": f"{aoi_name} NDVI"},
            'north': {"style": os.path.join(style_path, 'north.qml'), "legendname": f"{aoi_name} northness"},
            'plan': {"style": os.path.join(style_path, 'plan.qml'), "legendname": f"{aoi_name} plan curvature"},
            'profile': {"style": os.path.join(style_path, 'profile.qml'), "legendname": f"{aoi_name} profile curvature"},
            'rivers': {"style": os.path.join(style_path, 'rivers.qml'), "legendname": f"{aoi_name}\n distance from rivers [m]"},
            'roads': {"style": os.path.join(style_path, 'roads.qml'), "legendname": f"{aoi_name}\n distance from roads [m]"},
            'twi': {"style": os.path.join(style_path, 'twi.qml'), "legendname": f"{aoi_name} TWI"},
        },
        "layout": {
           "map": {"position": (10, 10), "size": (200, 170), "grid_interval": (0.1, 0.05)}, # VT
            "arrow": {"position": (20, 25), "size": [22,27]},
            "legend": {"position": (230, 10), "size": ()},
            "scale": {"position": (230,164), "size": ()},
        },
    }

    print_factor_layout(config)
    
#Val_Tartano()

def Val_Tartano_LSM(style_paths):
    aoi_name = 'Val Tartano'
    save_to = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Val Tartano/3.results/"
    config = {
        "save_to": save_to,
        "lsms": {
            'LSM_Bagging': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_AdaBoost': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_AdaBoost Calibrated': {"legendname": f"{aoi_name}\nSusceptibility Map"}, 
            'LSM_Fortests of randomized trees': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Gradient Tree Boosting': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Neural Network': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Ensemble Soft Voting': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Ensemble Blending': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Ensemble Stacking': {"legendname": f"{aoi_name}Susceptibility Map"},       
        },
        "layout": {
           "map": {"position": (10, 10), "size": (200, 170), "grid_interval": (0.1, 0.05)}, # VT
            "arrow": {"position": (20, 25), "size": [22,27]},
            "legend": {"position": (230, 10), "size": ()},
            "scale": {"position": (230,164), "size": ()},
        },
    }

    for style_name, style_path in style_paths.items():
        config['save_to'] = os.path.join(save_to, style_name)
        config['style'] = style_path
        print_LSM_layout(config)

#Val_Tartano_LSM({"LSM_5class": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/symbology/LSM_5classes.qml",})

def Val_Chiavenna():
    style_path = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/ValChiavenna/symbology/"
    aoi_name = "Val Chiavenna"
    config = {
        "save_to": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/ValChiavenna/1.factors/",
        "boundary": "valChiavenna — valChiavennaROI",
        "boundaryLegendName": f"{aoi_name} AOI",
        "factors": {
            'dtm': {"style": os.path.join(style_path, 'dtm.qml'), "legendname": f"{aoi_name} DTM [m]"},
            'east': {"style": os.path.join(style_path, 'east.qml'), "legendname": f"{aoi_name} eastness"},
            'faults': {"style": os.path.join(style_path, 'faults.qml'), "legendname": f"{aoi_name}\n distance from faults [m]"},
            'dusaf': {"style": os.path.join(style_path, 'landuse.qml'), "legendname": f"{aoi_name} land use"},
            'ndvi': {"style": os.path.join(style_path, 'ndvi.qml'), "legendname": f"{aoi_name} NDVI"},
            'north': {"style": os.path.join(style_path, 'north.qml'), "legendname": f"{aoi_name} northness"},
            'plan': {"style": os.path.join(style_path, 'plan.qml'), "legendname": f"{aoi_name} plan curvature"},
            'profile': {"style": os.path.join(style_path, 'profile.qml'), "legendname": f"{aoi_name} profile curvature"},
            'rivers': {"style": os.path.join(style_path, 'rivers.qml'), "legendname": f"{aoi_name}\n distance from rivers [m]"},
            'roads': {"style": os.path.join(style_path, 'roads.qml'), "legendname": f"{aoi_name}\n distance from roads [m]"},
            'twi': {"style": os.path.join(style_path, 'twi.qml'), "legendname": f"{aoi_name} TWI"},
        },
        "layout": {
           "map": {"position": (10, 10), "size": (200, 170), "grid_interval": (0.1, 0.05)},
            "arrow": {"position": (20, 25), "size": [22,27]},
            "legend": {"position": (230, 10), "size": ()},
            "scale": {"position": (230,164), "size": ()},
        },
    }

    print_factor_layout(config)
    
#Val_Chiavenna()

def Val_Chiavenna_LSM(style_paths):
    aoi_name = 'Val Chiavenna'
    #save_to = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/ValChiavenna/3.results/1st_without/"
    #save_to = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/ValChiavenna/3.results/2nd_with/"
    save_to = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/ValChiavenna/3.results/3rd_onlyVC/"
    config = {
        "save_to": save_to,
        "lsms": {
            'LSM_Bagging': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_AdaBoost': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_AdaBoost Calibrated': {"legendname": f"{aoi_name}\nSusceptibility Map"}, 
            'LSM_Forests of randomized trees': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Gradient Tree Boosting': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Neural Network': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Ensemble Soft Voting': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Ensemble Blending': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Ensemble Stacking': {"legendname": f"{aoi_name}Susceptibility Map"},       
        },
        "layout": {
           "map": {"position": (10, 10), "size": (200, 170), "grid_interval": (0.1, 0.05)},
            "arrow": {"position": (20, 25), "size": [22,27]},
            "legend": {"position": (230, 10), "size": ()},
            "scale": {"position": (230,164), "size": ()},
        },
    }

    for style_name, style_path in style_paths.items():
        config['save_to'] = os.path.join(save_to, style_name)
        config['style'] = style_path
        print_LSM_layout(config)

#Val_Chiavenna_LSM({"LSM_5class": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/symbology/LSM_5classes.qml",})


def Lombardy():
    style_path = r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Lombardy/symbology/"
    aoi_name = "Lombardy"
    config = {
        "save_to": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Lombardy/1.factors/",
        "boundary": "Lombardi_boundaries — Lombardy_Reg_boundaries",
        "boundaryLegendName": f"{aoi_name} AOI",
        "factors": {
            'dtm': {"style": os.path.join(style_path, 'dtm.qml'), "legendname": f"{aoi_name} DTM [m]"},
            'east': {"style": os.path.join(style_path, 'east.qml'), "legendname": f"{aoi_name} eastness"},
            'faults': {"style": os.path.join(style_path, 'faults.qml'), "legendname": f"{aoi_name}\n distance from faults [m]"},
            'dusaf': {"style": os.path.join(style_path, 'landuse.qml'), "legendname": f"{aoi_name} land use"},
            'ndvi': {"style": os.path.join(style_path, 'ndvi.qml'), "legendname": f"{aoi_name} NDVI"},
            'north': {"style": os.path.join(style_path, 'north.qml'), "legendname": f"{aoi_name} northness"},
            'plan': {"style": os.path.join(style_path, 'plan.qml'), "legendname": f"{aoi_name} plan curvature"},
            'profile': {"style": os.path.join(style_path, 'profile.qml'), "legendname": f"{aoi_name} profile curvature"},
            'rivers': {"style": os.path.join(style_path, 'rivers.qml'), "legendname": f"{aoi_name}\n distance from rivers [m]"},
            'roads': {"style": os.path.join(style_path, 'roads.qml'), "legendname": f"{aoi_name}\n distance from roads [m]"},
            'twi': {"style": os.path.join(style_path, 'twi.qml'), "legendname": f"{aoi_name} TWI"},
        },
        "layout": {
           "map": {"position": (10, 10), "size": (200, 170), "grid_interval": (1, 0.5)},
            "arrow": {"position": (20, 25), "size": [22,27]},
            "legend": {"position": (230, 10), "size": ()},
            "scale": {"position": (230,164), "size": ()},
        },
    }

    print_factor_layout(config)
    
#Lombardy()

def Lombardy_LSM(style_paths, save_to):
    aoi_name = 'Lombardy'
    config = {
        "save_to": save_to,
        "lsms": {
            #'LSM_Bagging': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            #'LSM_AdaBoost': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            #'LSM_AdaBoost Calibrated': {"legendname": f"{aoi_name}\nSusceptibility Map"}, 
            #'LSM_Forests of randomized trees': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            #'LSM_Gradient Tree Boosting': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            'LSM_Neural Network': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            #'LSM_Ensemble Soft Voting': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            #'LSM_Ensemble Blending': {"legendname": f"{aoi_name}\nSusceptibility Map"},
            #'LSM_Ensemble Stacking': {"legendname": f"{aoi_name}Susceptibility Map"},       
        },
        "layout": {
           "map": {"position": (10, 10), "size": (200, 170), "grid_interval": (1, 0.5)},
            "arrow": {"position": (20, 25), "size": [22,27]},
            "legend": {"position": (230, 10), "size": ()},
            "scale": {"position": (230,164), "size": ()},
        },
    }

    for style_name, style_path in style_paths.items():
        config['save_to'] = os.path.join(save_to, style_name)
        config['style'] = style_path
        print_LSM_layout(config)

Lombardy_LSM({"LSM_5class": r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/symbology/LSM_5classes.qml"}, r"/Volumes/Another/3. Education/Politecnico(GIS-CS)/3 Thesis/practice/Lombardy/3.results/Valchiavenna_6_precips/")