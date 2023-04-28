import os
from os.path import isdir, join, abspath, isfile, dirname
import shutil
import numpy as np
from functools import lru_cache
import json
import pybullet_planning as pp
import pybullet as p


from .pybullet_utils import add_text, draw_fitted_box, get_aabb, draw_points
from hacl.engine.bullet.world import JointState


CATEGORIES_BOX = ["Phone", "Remote", "StaplerFlat", "USBFlat", "Bowl", "Cup", "Mug"]
CATEGORIES_TALL = ["Bottle"]
CATEGORIES_NON_CONVEX = ["Camera", "FoldingKnife", "Pliers", "Scissors", "USB"]
CATEGORIES_SIDE_GRASP = ["Eyeglasses", "Stapler", "OpenedBottle", "Dispenser"]
CATEGORIES_FOLDED_CONTAINER = ["Suitcase", "Box"]
CATEGORIES_OPENED_SPACE = ["Safe"]

models = {

    ## --------------- BOX --------------- ##
    "Phone": {
        'models': ['103251', '103813', '103828', '103916', '103927'],  ## '103892',
        'length-range': [0.18, 0.2],
    },
    "Remote": {
        'models': ['100269', '100270', '100392', '100394',
                   '100809', '100819', '100997', '101034'],
        'length-range': [0.22, 0.24],
        'width-range': [0, 0.06],
    },
    "StaplerFlat": {
        'models': ['103095', '103104', '103271', '103273', '103280',
                   '103297', '103792'],
        'length-range': [0.2, 0.22],
    },
    "USBFlat": {
        'models': ['100085', '100073', '100095', '100071', '101950',
                   '102063'],
        'length-range': [0.05, 0.07],
    },
    "Bowl": {
        'models': ['7000', '7001', '7002', '7003', '7004'],
        'length-range': [0.07, 0.17],
    },
    "Cup": {
        'models': ['7004', '7005', '7006', '7007'],
        'length-range': [0.07, 0.09],
    },
    "Mug": {
        'models': ['7008', '7009', '7010', '7011'],
        'length-range': [0.05, 0.07],
    },

    ## --------------- TALL --------------- ##
    "Bottle": {
        'models': ['3520', '3596', '3625', '4216', '4403', '4514',
                   '6771'],
        'length-range': [0.04, 0.06],
    },

    ## --------------- NON_CONVEX --------------- ##
    "Camera": {
        'models': ['101352', '102417', '102434', '102536', '102852',
                   '102873', '102890'],
        'height-range': [0.1, 0.12],
    },
    "FoldingKnife": {
        'models': ['101068', '101079', '101107', '101245', '103740'],
        'length-range': [0.06, 0.12],
        'width-range': [0.06, 0.12],
    },
    "Pliers": {
        'models': ['100144', '100146', '100179', '100182', '102243',
                   '102288'],
        'length-range': [0.15, 0.17],
    },
    "Scissors": {
        'models': ['10495', '10502', '10537', '10567', '11021', '11029'],
        'length-range': [0.13, 0.15],
    },
    "USB": {
        'models': ['100086', '100109', '100082', '100078', '100065',
                   '101999', '102008'],
        'length-range': [0.05, 0.07],
    },

    ## --------------- SIDE_GRASP --------------- ##
    "Eyeglasses": {
        'models': ['101284', '101287', '101293', '101291', '101303',
                   '101326', '101328', '101838'],
        'length-range': [0.1, 0.12],
    },
    "Stapler": {
        'models': ['102990', '103099', '103113', '103283', '103299', '103307'],
        'length-range': [0.18, 0.2],
    },
    "OpenedBottle": {
        'models': ['3574', '3571', '3763', '3517', '3868',
                   '3830', '3990', '4043'],
        'length-range': [0.04, 0.06],
    },
    "Dispenser": {
        'models': ['101458', '101517', '101533', '101563', '103397',
                   '103416'],
        'length-range': [0.04, 0.06],
    },

    ## --------------- FOLDED_CONTAINER --------------- ##
    "Suitcase": {
        'models': ['100550', '100767', '101668', '103755', '103761'],
        'y-range': [0.3, 0.4],
    },
    "Box": {
        'models': ['100426', '100154', '100243'],  ## '100247',
        'y-range': [0.3, 0.4],
    },

    ## --------------- FOLDED_CONTAINER --------------- ##
    "Safe": {
        'models': ['101363', '102373', '101584', '101591', '101611',
                   '102316']
    }
}


@lru_cache(maxsize=None)
def get_model_path(category, model_id):
    return join(abspath(dirname(__file__)), 'models', category, str(model_id), 'mobility.urdf')


@lru_cache(maxsize=None)
def get_model_ids(category):
    return models[category]['models']


@lru_cache(maxsize=None)
def get_model_natural_extent(c, model_path):
    """" store and load the aabb when scale = 1, so it's easier to scale according to given range """
    data_file = join(dirname(__file__), 'aabb_extents.json')
    if not isfile(data_file):
        with open(data_file, 'w') as f:
            json.dump({}, f)
    data = json.load(open(data_file, 'r'))
    model_name = model_path.replace(join(abspath(dirname(__file__)), 'models/'), '')
    if model_name not in data:
        body = c.load_urdf(model_path, (0, 0, 0), body_name='tmp')
        extent = pp.get_aabb_extent(get_aabb(c.client_id, body))
        data[model_name] = tuple(extent)
        c.remove_body(body)
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=4)
    return data[model_name]


def get_model_scale_from_constraint(c, category, model_id):
    """ get the scale according to height_range, length_range (longer side), and width_range (shorter side) """
    model_path = get_model_path(category, model_id)
    extent = get_model_natural_extent(c, model_path)
    keys = {'length-range': 0, 'width-range': 1, 'height-range': 2, 'x-range': 0, 'y-range': 1}
    if extent[0] < extent[1]:
        keys.update({
            'length-range': 1, 'width-range': 0
        })
    criteria = [k for k in models[category] if k in keys]
    if len(criteria) == 0:
        return 1

    scale_range = [-np.inf, np.inf]
    for k in criteria:
        r = [models[category][k][i] / extent[keys[k]] for i in range(2)]
        scale_range[0] = max(scale_range[0], r[0])
        scale_range[1] = min(scale_range[1], r[1])
    scale = np.random.uniform(*scale_range)
    return scale


def load_asset_to_pdsketch(c, category, model_id, floor=None, **kwargs):
    """ load a model from the dataset into the bullet environment though PDSketch API """
    model_path = get_model_path(category, model_id)

    name = f'{category}_{model_id}'
    print('load_asset_to_pdsketch.loading', name)

    with c.disable_rendering():
        scale = get_model_scale_from_constraint(c, category, model_id)

        pos = kwargs.pop('pos', (0, 0, 0))
        if floor is not None:
            extent = get_model_natural_extent(c, model_path)
            pos = list(pos[:2]) + [get_aabb(c.client_id, floor).upper[2] + extent[2] * scale / 2]

        body = c.load_urdf(model_path, pos=pos, body_name=name, scale=scale, **kwargs)

    ## open suitcases
    if category in ['Suitcase', 'Box']:
        for ji in c.w.get_joint_info_by_body(body):
            j = ji.joint_index
            if ji.joint_type == p.JOINT_REVOLUTE:
                pstn = 1.57
                if category == 'Suitcase':
                    pstn = ji.joint_lower_limit+1.57
                elif category == 'Box':
                    if model_id == '100426':
                        pstn = 1.46
                    if model_id == '100154':
                        pstn = 0.8
                c.w.set_joint_state_by_id(body, j, JointState(pstn, 0))

    draw_fitted_box(c, body, draw_box=True, draw_centroid=False, draw_points=False)
    add_text(c.client_id, name, body)
    return body


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