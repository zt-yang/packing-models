import os
import random
from os import listdir
from os.path import isdir, join, abspath, isfile, dirname
import shutil
import numpy as np
from functools import lru_cache
import json
import pybullet_planning as pp
import pybullet as p
import sys
import functools
err = functools.partial(print, flush=True, file=sys.stderr)

from bullet_utils import add_text, draw_fitted_box, get_aabb, draw_points, get_pose, \
    set_pose, nice, get_grasp_db_file
from hacl.engine.bullet.world import JointState

MODEL_PATH = abspath(join(dirname(abspath(__file__)), 'models'))

CATEGORIES_BOX = ["Phone", "Remote", "StaplerFlat", "USBFlat", "Bowl", "Cup", "Mug", "Bottle"]
CATEGORIES_TALL = ["BottleOpened"]
CATEGORIES_NON_CONVEX = ["Eyeglasses", "Camera", "FoldingKnife", "Pliers", "Scissors", "USB"]
CATEGORIES_SIDE_GRASP = ["Stapler", "BottleOpened", "Dispenser", "Bowl"]
CATEGORIES_FOLDED_CONTAINER = ["Suitcase", "Box"]
CATEGORIES_OPENED_SPACE = ["Safe"]
CATEGORIES_BANDU = ["Bandu", "engmikedset"]  ##
CATEGORIES_DIFFUSION_CSP = ['Dispenser', 'Bowl', 'StaplerFlat', 'Eyeglasses',
                            'Pliers', 'Scissors', 'Camera', 'Bottle', 'BottleOpened', 'Mug']

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
        'models': ['7001', '7002'],
        'length-range': [0.13, 0.18],
    },
    "Cup": {
        'models': ['7004', '7005', '7006', '7007'],
        'length-range': [0.07, 0.09],
    },
    "Mug": {
        'models': ['7009'],  ## '7008', '7011', '7010',
        'length-range': [0.08, 0.09],
    },

    ## --------------- TALL --------------- ##
    "Bottle": {
        'models': ['3520', '3596', '4216', '4403', '4514', '6771'],  ## '3625',
        'length-range': [0.03, 0.045],
    },

    ## --------------- NON_CONVEX --------------- ##
    "Camera": {
        'models': ['101352', '102417', '102536', '102852', '102873'],  ## '102434', '102890',
        'height-range': [0.08, 0.09],
    },
    "FoldingKnife": {
        'models': ['101068', '101079', '101107', '101245', '103740'],
        'length-range': [0.06, 0.12],
        'width-range': [0.06, 0.12],
    },
    "Pliers": {
        'models': ['100144', '100146', '100179', '100182', '102243',
                   '102288'],
        'length-range': [0.18, 0.2],
    },
    "Scissors": {
        'models': ['10495', '10502', '10537', '10567', '11021', '11029'],
        'length-range': [0.18, 0.2],
    },
    "USB": {
        'models': ['100086', '100109', '100082', '100078', '100065',
                   '101999', '102008'],
        'length-range': [0.05, 0.07],
    },

    ## --------------- SIDE_GRASP --------------- ##
    "Eyeglasses": {
        'models': ['101284', '101287', '101293', '101291', '101303', '101326', '101838'],  ## '101328',
        'length-range': [0.13, 0.15],
    },
    "Stapler": {
        'models': ['102990', '103099', '103113', '103283', '103299', '103307'],
        'length-range': [0.18, 0.2],
    },
    "BottleOpened": {
        'models': ['3571', '3574', '3763', '3517', '3830', '3990', '4043'],  ## '3593',
        'length-range': [0.04, 0.06],
    },
    "Dispenser": {
        'models': ['101458', '101517', '101533', '101563', '103397', '103416'],
        'length-range': [0.06, 0.07],
    },

    ## --------------- FOLDED_CONTAINER --------------- ##
    "Suitcase": {
        'models': ['100550', '101668', '103755', '103761'],  ## , '100767'
        'y-range': [0.6, 0.65],
    },
    "Box": {
        'models': ['100426', '100243'],  ## bad '100247', ## too tall '100154'
        'y-range': [0.4, 0.5],
    },

    ## --------------- FOLDED_CONTAINER --------------- ##
    "Safe": {
        'models': ['101363', '102373', '101584', '101591', '101611',
                   '102316'],
        'height-range': [0.8, 0.9]
    }
}


@lru_cache()
def get_packing_assets():
    assets = {}
    for cat, data in models.items():
        if cat not in CATEGORIES_DIFFUSION_CSP:
            continue
        for model_id in data['models']:
            extent = np.array(get_model_natural_extent(get_model_path(cat, model_id)))
            scale_range = np.array(get_model_scale_from_constraint(cat, model_id))
            extent_range = np.outer(extent, scale_range)
            area = np.prod(extent_range[:, 0])
            assets[(cat, model_id)] = (extent_range, scale_range, extent, area)
    ## sort assets by decreasing area
    assets = {k: v[:3] for k, v in sorted(assets.items(), key=lambda x: x[1][-1], reverse=True)}
    return assets


def fit_object_assets(region, assets, w, l, h, padding=0.1):
    """ return the fitted asset, sampled scale, and rotation """
    b = 1 - padding
    found = []
    for identifier, (extent_range, scale_range, extent) in assets.items():
        x_range, y_range = extent_range[:2]
        ratio = x_range[0] / y_range[0]
        if x_range[0] < region[2] and y_range[0] < region[3]:
            scale_range[1] = min(scale_range[1], region[2] * b / ratio)
            theta = random.choice([0, np.pi])
        elif x_range[0] < region[3] and y_range[0] < region[2]:
            scale_range[1] = min(scale_range[1], region[3] * b * ratio)
            theta = random.choice([np.pi/2, -np.pi/2])
        else:
            continue
        scale = np.random.uniform(scale_range[0], scale_range[1])
        extent *= scale
        x = region[0] + region[2] / 2 - w / 2
        y = region[1] + region[3] / 2 - l / 2
        z = extent[2] / 2 + h * 3 / 2 + 0.01
        pose = ((x, y, z), pp.quat_from_euler((theta, 0, 0)))
        found.append([identifier, scale, extent, pose, theta])

    n = len(found)
    first = min([n, 5])
    second = n - first
    weights = np.log(np.arange(first).astype(float) + 2)
    weights = 1 / weights
    weights = np.concatenate([weights, np.ones(second) * weights[-1]])
    weights /= weights.sum().astype(float)
    # print(len(weights), [round(w, 3) for w in weights])
    return random.choices(found, weights=weights.tolist(), k=1)[0]


@lru_cache()
def get_model_path(category, model_id):
    model_dir = join(MODEL_PATH, category, str(model_id))
    return [join(model_dir, f) for f in listdir(model_dir) if f.endswith('.urdf')][0]


@lru_cache()
def get_pointcloud_path(category, model_id):
    model_path = get_model_path(category, model_id)
    return join(dirname(model_path), 'pointcloud.ply')


@lru_cache()
def get_model_ids(category):
    if category in models:
        return models[category]['models']
    return [f for f in listdir(join(MODEL_PATH, category)) if isdir(join(MODEL_PATH, category, f))]


def get_instance_name(path):
    if not isfile(path): return None
    rows = open(path, 'r').readlines()
    if len(rows) > 50: rows = rows[:50]

    def from_line(r):
        r = r.replace('\n', '')[13:]
        return r[:r.index('"')]

    name = [from_line(r) for r in rows if '<robot name="' in r]
    if len(name) == 1:
        return name[0]
    return None


@lru_cache()
def get_model_natural_extent(model_path, c=None):
    """" store and load the aabb when scale = 1, so it's easier to scale according to given range """
    data_file = join(dirname(__file__), 'aabb_extents.json')
    if not isfile(data_file):
        with open(data_file, 'w') as f:
            json.dump({}, f)
    data = json.load(open(data_file, 'r'))
    model_name = model_path.replace(MODEL_PATH+'/', '')
    if model_name not in data and c is not None:
        body = c.load_urdf(model_path, (0, 0, 0), body_name='tmp')
        extent = pp.get_aabb_extent(get_aabb(c.client_id, body))
        data[model_name] = tuple(extent)
        c.remove_body(body)
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=4)
    return data[model_name]


def get_model_scale_from_constraint(category, model_id, c=None):
    """ get the scale according to height_range, length_range (longer side), and width_range (shorter side) """
    if category not in models:
        return 1, 1
    model_path = get_model_path(category, model_id)
    extent = get_model_natural_extent(model_path, c=c)
    keys = {'length-range': 0, 'width-range': 1, 'height-range': 2, 'x-range': 0, 'y-range': 1}
    if extent[0] < extent[1]:
        keys.update({
            'length-range': 1, 'width-range': 0
        })
    criteria = [k for k in models[category] if k in keys]
    if len(criteria) == 0:
        return 1, 1

    scale_range = [-np.inf, np.inf]
    for k in criteria:
        r = [models[category][k][i] / extent[keys[k]] for i in range(2)]
        scale_range[0] = max(scale_range[0], r[0])
        scale_range[1] = min(scale_range[1], r[1])
    return scale_range


def sample_model_scale_from_constraint(category, model_id, c=None, scale=None):
    scale_range = get_model_scale_from_constraint(category, model_id, c)
    if scale == 'max':
        return scale_range[1]
    elif scale == 'min':
        return scale_range[0]
    elif scale is not None:
        amount = float(scale)
        return scale_range[0] + (scale_range[1] - scale_range[0]) * amount
    return np.random.uniform(*scale_range)


def bottom_to_center(cid, body):
    return get_pose(cid, body)[0][2] - get_aabb(cid, body).lower[2]


def is_array(x, length=None):
    result = isinstance(x, np.ndarray) or isinstance(x, list) or isinstance(x, tuple)
    if length is not None:
        result = result and len(x) == length
    return result


def load_asset_to_pdsketch(c, category, model_id, scale=None, name=None, floor=None,
                           pos=None, draw_bb=False, **kwargs):
    """ load a model from the dataset into the bullet environment though PDSketch API """
    model_path = get_model_path(category, model_id)

    if name is None:
        name = f'{category}_{model_id}'
    print('load_asset_to_pdsketch.loading', name)

    adjust = False
    with c.disable_rendering():
        gap = 0.01

        if scale is None or isinstance(scale, str):
            scale = sample_model_scale_from_constraint(category, model_id, c, scale)

        if len(pos) == 2 and is_array(pos[0], length=3) and is_array(pos[1], length=4):
            pos, quat = pos
        else:
            quat = (0, 0, 0, 1)
            if len(pos) == 2 and isinstance(pos[0], tuple):
                pos, quat = pos
            if floor is not None:
                extent = get_model_natural_extent(model_path, c)
                pos = tuple(list(pos[:2]) + [get_aabb(c.client_id, floor).upper[2] + extent[2] * scale / 2 + gap])
            adjust = True

        body = c.load_urdf(model_path, pos=pos, quat=quat, body_name=name, scale=scale, **kwargs)

        ## adjust because sometimes the model is not centered on z axis
        if floor is not None and adjust:
            bottom_to_ceter = bottom_to_center(c.client_id, body) + gap
            pose = get_pose(c.client_id, body)
            pose = (list(pose[0][:2]) + [get_aabb(c.client_id, floor).upper[2] + bottom_to_ceter], pose[1])
            set_pose(c.client_id, body, pose)

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

    ## drawing bounding boxes
    if draw_bb and category not in CATEGORIES_BANDU:
        draw_fitted_box(c.client_id, body, draw_box=True, draw_centroid=False, draw_points=False)
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


def check_model_simulatable(cat, model_id):
    asset_urdf = get_model_path(cat, model_id)
    lines = ''.join(open(asset_urdf).readlines())
    return 'mass' in lines and 'friction' in lines


@lru_cache()
def get_grasp_data():
    """ cat_model: grasps """
    grasp_file = join(dirname(__file__), 'grasps', f'hand_grasps_PandaRobot.json')
    data = json.load(open(grasp_file)).values()
    return {d['name']: d for d in data}


def get_cat_models(cats=CATEGORIES_DIFFUSION_CSP):
    for cat in cats:
        model_ids = get_model_ids(cat)
        for model_id in model_ids:
            yield cat, model_id


def check_grasps_exist(cat, model_id, names=None):
    if names is None:
        names = list(get_grasp_data().keys())
    return names, f'{cat}_{model_id}' in names


def check_simulatable():
    names = None
    for cat, model_id in get_cat_models():
        if not check_model_simulatable(cat, model_id):
            print(f'{cat} {model_id} is not simulatable')
        names, exist = check_grasps_exist(cat, model_id, names)
        if not exist:
            print(f'{cat} {model_id} does not have grasps stored')


def get_grasps(name):
    return get_grasp_data()[name]['grasps']


def get_grasps_info(name):
    return get_grasp_data()[name]


def get_grasp_id(name, nice_grasp):
    grasps = [nice(x, 2)[-3:] for x in get_grasps(name)]
    if nice_grasp not in grasps:
        return 'x'
    return grasps.index(nice_grasp)


def get_grasp_side_by_id(name, grasp_id):
    return get_grasps_info(name)['grasp_sides'][grasp_id]


if __name__ == '__main__':
    # download_models()
    check_simulatable()
