import os
from os.path import isdir, join
import shutil

BOX = ["Phone", "Remote", "StaplerFlat", "USBFlat"]
TALL = ["Bottle"]
NON_CONVEX = ["Camera", "FoldingKnife", "Pliers", "Scissors", "USB"]
SIDE_GRASP = ["Eyeglasses", "Stapler", "OpenedBottle", "Dispenser"]
FOLDED_CONTAINER = ["Suitcase", "Box"]
OPENED_SPACE = ["Safe"]

models = {

    ## --------------- BOX --------------- ##
    "Phone": {
        'models': ['103251', '103813', '103828', '103892', '103916',
                   '103927'],
    },
    "Remote": {
        'models': ['100269', '100270', '100392', '100394', '100405',
                   '100809', '100819', '100997', '101034'],
    },
    "StaplerFlat": {
        'models': ['103095', '103104', '103271', '103273', '103280',
                   '103292', '103297', '103792'],
    },
    "USBFlat": {
        'models': ['100085', '100073', '100095', '100071', '101950',
                   '102063'],

    },
    "USB": {
        'models': ['100086', '100109', '100082', '100078', '100065',
                   '101999', '102008'],

    },

    ## --------------- TALL --------------- ##
    "Bottle": {
        'models': ['3520', '3596', '3625', '4216', '4403', '4514',
                   '6771'],
    },

    ## --------------- NON_CONVEX --------------- ##
    "Camera": {
        'models': ['101352', '102417', '102434', '102536', '102852',
                   '102873', '102890'],
        'max-height': 0.5,
    },
    "FoldingKnife": {
        'models': ['101068', '101079', '101107', '101108', '101245',
                   '103740'],
    },
    "Pliers": {
        'models': ['100144', '100146', '100179', '100182', '102243',
                   '102288'],
    },
    "Scissors": {
        'models': ['10495', '10502', '10537', '10546', '10567',
                   '11021', '11029'],
    },

    ## --------------- SIDE_GRASP --------------- ##
    "Eyeglasses": {
        'models': ['101284', '101287', '101293', '101291', '101303',
                   '101326', '101328', '101838'],
    },
    "Stapler": {
        'models': ['102990', '103099', '103113', '103283', '103299',
                   '103307'],
    },
    "OpenedBottle": {
        'models': ['3574', '3571', '3593', '3763', '3517', '3868',
                   '3830', '3990', '4043'],

    },
    "Dispenser": {
        'models': ['101458', '101517', '101533', '101563', '103397',
                   '103416'],
    },

    ## --------------- FOLDED_CONTAINER --------------- ##
    "Suitcase": {
        'models': ['100550', '100767', '101668', '103755', '103761']
    },
    "Box": {
        'models': ['100426', '100767', '100154', '100247', '100243']
    },

    ## --------------- FOLDED_CONTAINER --------------- ##
    "Safe": {
        'models': ['101363', '102373', '101584', '101591', '101611',
                   '102316']
    }
}


def download_category(indices, category_dir):
    """ models are initially inside a dataset folder without class hierarchy """
    partnet_dataset_path = '../dataset'
    for i in indices:
        from_dir = join(partnet_dataset_path, str(i))
        to_dir = join(category_dir, str(i))
        if isdir(to_dir):
            continue
        if isdir(from_dir):
            shutil.copytree(from_dir, to_dir)
        else:
            print(f"Warning: {from_dir} does not exist")


def download_models():
    for name, data in models.items():
        category_dir = os.path.join('models', name)
        if not isdir(category_dir):
            os.makedirs(category_dir)
        download_category(data['models'], category_dir)


if __name__ == '__main__':
    download_models()