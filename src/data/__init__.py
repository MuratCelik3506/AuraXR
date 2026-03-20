# Data Loaders Package for Intent-Aware XR Framework
# Includes H2O and HOT3D dataset loaders and fusion utilities.

from src.data.h2o_dataset     import H2ODataset, get_dataloaders, NUM_CLASSES as H2O_NUM_CLASSES
from src.data.hot3d_dataset   import HOT3DDataset, get_hot3d_dataloaders, NUM_CLASSES_HOT3D
from src.data.combined_dataset import get_combined_dataloaders, num_classes_for_fusion
