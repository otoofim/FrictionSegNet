"""
Generic dataloader utilities and class definitions for FrictionSegNet.
"""

# Cityscapes class IDs mapping
classIds = {
    0: "road",
    1: "sidewalk", 
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic_light",
    7: "traffic_sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle"
}

def get_class_names():
    """Returns list of class names in order."""
    return list(classIds.values())

def get_num_classes():
    """Returns number of classes."""
    return len(classIds)