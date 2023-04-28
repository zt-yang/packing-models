import os
from os.path import isdir, join, abspath, isfile, dirname
import shutil
import sys
import numpy as np
import math
import pybullet as p
import pybullet_planning as pp
from collections import namedtuple
from itertools import product, combinations


RED = (1, 0, 0, 1)
YELLOW = (1, 1, 0, 1)
GREEN = (0, 1, 0, 1)
BLUE = (0, 0, 1, 1)
PURPLE = (1, 0, 1, 1)
BLACK = (0, 0, 0, 1)
BASE_LINK = -1


CollisionShapeData = namedtuple('CollisionShapeData', ['object_unique_id', 'linkIndex',
                                                       'geometry_type', 'dimensions', 'filename',
                                                       'local_frame_pos', 'local_frame_orn'])

VisualShapeData = namedtuple('VisualShapeData', ['objectUniqueId', 'linkIndex',
                                                 'visualGeometryType', 'dimensions', 'meshAssetFileName',
                                                 'localVisualFrame_position', 'localVisualFrame_orientation',
                                                 'rgbaColor'])

LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])

AABB = namedtuple('AABB', ['lower', 'upper'])


def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])


def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])


def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return point, pp.quat_from_euler(euler)


class HideOutput(object):
    """ A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    """

    DEFAULT_ENABLE = True

    def __init__(self, enable=None):
        if enable is None:
            enable = self.DEFAULT_ENABLE
        self.enable = enable
        if not self.enable:
            return
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        if not self.enable:
            return
        self.fd = 1
        self._newstdout = os.dup(self.fd)
        os.dup2(self._devnull, self.fd)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        sys.stdout.close()
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, self.fd)
        os.close(self._oldstdout_fno)


def add_text(c, text, body_id, link_id=BASE_LINK, position=(0., 0., 0.), color='r'):
    if isinstance(color, str):
        color = {'b': (0, 0, 0), 'r': (1, 0, 0)}[color]
    with HideOutput():
        return p.addUserDebugText(str(text), textPosition=position, textColorRGB=color[:3],  # textSize=1,
                                  lifeTime=0, parentObjectUniqueId=body_id, parentLinkIndex=link_id,
                                  physicsClientId=c)


def add_line(c, start, end, color=BLACK, width=1, parent=-1, parent_link=BASE_LINK):
    assert (len(start) == 3) and (len(end) == 3)
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              lifeTime=0, parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=c)


def draw_point(c, point, size=0.01, **kwargs):
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size/2 * axis
        p2 = np.array(point) + size/2 * axis
        lines.append(add_line(c, p1, p2, **kwargs))
    return lines


def get_pose(c, body):
    return p.getBasePositionAndOrientation(body, physicsClientId=c)


def get_collision_data(c, body, link=BASE_LINK):
    return [CollisionShapeData(*tup) for tup in p.getCollisionShapeData(body, link, physicsClientId=c)]


def get_visual_data(c, body, link=BASE_LINK):
    visual_data = [VisualShapeData(*tup) for tup in p.getVisualShapeData(body, physicsClientId=c)]
    list(filter(lambda d: d.linkIndex == link, visual_data))


def equal_float(a, b, epsilon=0.0):
    return abs(a - b) <= epsilon


def equal(tup_a, tup_b, epsilon=0.001):
    if isinstance(tup_a, float) or isinstance(tup_a, int):
        return equal_float(tup_a, tup_b, epsilon)

    elif isinstance(tup_a, tuple):
        a = list(tup_a)
        b = list(tup_b)
        return all([equal(a[i], b[i], epsilon) for i in range(len(a))])

    return None


ROTATIONAL_MATRICES = {}


def get_rotation_matrix(c, body, verbose=False):
    import untangle
    r = pp.unit_pose()
    collision_data = get_collision_data(c, body, link=0)
    if len(collision_data) > 0:
        urdf_file = dirname(collision_data[0].filename.decode())
        count = 0
        while len(urdf_file.strip()) == 0:
            count += 1
            urdf_file = dirname(collision_data[count].filename.decode())
        urdf_file = urdf_file.replace('/textured_objs', '').replace('/base_objs', '').replace('/vhacd', '')
        if urdf_file not in ROTATIONAL_MATRICES:
            if verbose:
                print('get_rotation_matrix | urdf_file = ', abspath(urdf_file))
            joints = untangle.parse(join(urdf_file, 'mobility.urdf')).robot.joint
            if isinstance(joints, list):
                for j in joints:
                    if j.parent['link'] == 'base':
                        joint = j
                        break
            else:
                joint = joints
            rpy = joint.origin['rpy'].split(' ')
            rpy = tuple([eval(e) for e in rpy])
            if equal(rpy, (1.57, 1.57, -1.57), epsilon=0.1):
                r = Pose(euler=Euler(math.pi / 2, 0, -math.pi / 2))
            elif equal(rpy, (3.14, 3.14, -1.57), epsilon=0.1):
                r = Pose(euler=Euler(0, 0, math.pi / 2))
            elif equal(rpy, (1.57, 0, -1.57), epsilon=0.1):
                r = Pose(euler=Euler(math.pi/2, 0, -math.pi / 2))
            ROTATIONAL_MATRICES[urdf_file] = r
        r = ROTATIONAL_MATRICES[urdf_file]
    return r


def get_link_pose(c, body, link):
    if link == BASE_LINK:
        return get_pose(body)
    link_state = get_link_state(c, body, link)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation


def get_link_state(c, body, link):
    return LinkState(*p.getLinkState(body, link, physicsClientId=c))


def get_num_joints(c, body):
    return p.getNumJoints(body, physicsClientId=c)


def get_joints(c, body):
    return list(range(get_num_joints(c, body)))


get_links = get_joints
get_num_links = get_num_joints


def get_model_pose(c, body, link=None, **kwargs):
    if link is None:
        body_pose = pp.multiply(get_pose(c, body), get_rotation_matrix(c, body, **kwargs))
    else:
        body_pose = get_link_pose(c, body, link)
    return body_pose


def implies(p1, p2):
    return not p1 or p2


def vertices_from_link(c, body, link=BASE_LINK, collision=True):
    # TODO: get_mesh_data(body, link=link)
    # In local frame
    vertices = []
    # PyBullet creates multiple collision elements (with unknown_file) when nonconvex
    get_data = get_collision_data if collision else get_visual_data
    for data in get_data(body, link):
        vertices.extend(apply_affine(c, get_data_pose(data), pp.vertices_from_data(data)))
    return vertices


def get_data_pose(data):
    if isinstance(data, CollisionShapeData):
        return (data.local_frame_pos, data.local_frame_orn)
    return (data.localVisualFrame_position, data.localVisualFrame_orientation)


def get_model_points(c, body, link=None, draw_all_points=False, body_pose=None):
    if link is None:
        links = get_links(c, body)
    else:
        links = [link]

    vertices = []
    colors = [RED, YELLOW, GREEN, BLUE, PURPLE, BLACK]
    for i, link in enumerate(links):
        vv = pp.vertices_from_rigid(body, link)
        if len(vv) > 0:
            cdata = get_collision_data(c, body, link=link)
            if len(cdata) > 0:
                cdata = cdata[0]
            pose = (cdata.local_frame_pos, cdata.local_frame_orn)
            new_vertices = apply_affine(c, pp.invert(pose), vv)
            vertices.extend(new_vertices)
            if draw_all_points and body_pose is not None:
                draw_points(c, new_vertices, body_pose, color=colors[i])

                link_aabb = aabb_from_points(new_vertices)
                draw_bounding_box(c, link_aabb, body_pose, color=colors[i])
    return vertices


def aabb_from_points(points):
    return AABB(np.min(points, axis=0), np.max(points, axis=0))


def get_all_links(c, body):
    return [BASE_LINK] + list(get_links(c, body))


def can_collide(c, body, link=BASE_LINK):
    return len(get_collision_data(c, body, link=link)) != 0


def get_aabbs(c, body, links=None, only_collision=True):
    if links is None:
        links = get_all_links(c, body)
    if only_collision:
        # TODO: return the null bounding box
        links = [link for link in links if can_collide(c, body, link)]
    return [get_aabb(c, body, link=link) for link in links]


def get_aabb(c, body: int, link: int = None):
    if link is None:
        return pp.aabb_union(get_aabbs(c, body=body))
    return AABB(*p.getAABB(body, linkIndex=link, physicsClientId=c))


def draw_fitted_box(c, body, link=None, draw_box=False, draw_centroid=False,
                    draw_points=False, verbose=False, **kwargs):
    body_pose = get_model_pose(c.client_id, body, link=link, verbose=verbose)
    vertices = get_model_points(c, body, link=link, draw_all_points=draw_points, body_pose=body_pose)
    c = c.client_id

    ## form the aabb
    if link is None:
        link = -1
    data = get_collision_data(c, body, link)
    if len(data) == 0 or data[0].geometry_type == p.GEOM_MESH:
        aabb = aabb_from_points(vertices)
    else:
        aabb = get_aabb(c, body)

    ## other visualization options
    handles = []
    if draw_box:
        handles += draw_bounding_box(c, aabb, body_pose, **kwargs)
    if draw_centroid:
        handles += draw_face_points(c, aabb, body_pose, dist=0.04)
    return aabb, handles


def draw_face_points(c, aabb, body_pose, dist=0.08):
    center = pp.get_aabb_center(aabb)
    w, l, h = pp.get_aabb_extent(aabb)
    faces = [(w/2+dist, 0, 0), (0, l/2+dist, 0), (0, 0, h/2+dist)]
    faces += [minus(0, f) for f in faces]
    faces = [add(f, center) for f in faces]
    faces = apply_affine(c, body_pose, faces)
    handles = []
    for f in faces:
        handles.extend(draw_point(c, f, size=0.02, color=RED))
    return handles


def draw_points(c, vertices, body_pose, size=0.01, **kwargs):
    vertices = apply_affine(c, body_pose, vertices)
    handles = []
    num_vertices = 20
    if len(vertices) > num_vertices:
        gap = int(len(vertices)/num_vertices)
        vertices = vertices[::gap]
    for v in vertices:
        handles.append(draw_point(c, v, size=size, **kwargs))
    return handles


def add(elem1, elem2):
    return tuple(np.asarray(elem1)+np.asarray(elem2))


def minus(elem1, elem2):
    return tuple(np.asarray(elem1)-np.asarray(elem2))


def dist(elem1, elem2):
    return np.linalg.norm(np.asarray(elem1)-np.asarray(elem2))


def tform_points(c, affine, points):
    tform = tform_from_pose(c, affine)
    points_homogenous = np.vstack([np.vstack(points).T, np.ones(len(points))])
    return tform.dot(points_homogenous)[:3, :].T


def tform_from_pose(c, pose):
    (point, quat) = pose
    tform = np.eye(4)
    tform[:3, 3] = point
    tform[:3, :3] = matrix_from_quat(c, quat)
    return tform


def matrix_from_quat(c, quat):
    return np.array(p.getMatrixFromQuaternion(quat, physicsClientId=c)).reshape(3, 3)


apply_affine = tform_points


def draw_bounding_box(c, aabb, body_pose, **kwargs):
    handles = []
    for a, b in get_aabb_edges(aabb):
        p1, p2 = apply_affine(c, body_pose, [a, b])
        handles.append(add_line(c, p1, p2, **kwargs))
    return handles


def get_aabb_edges(aabb):
    d = len(aabb[0])
    vertices = list(product(range(len(aabb)), repeat=d))
    lines = []
    for i1, i2 in combinations(vertices, 2):
        if sum(i1[k] != i2[k] for k in range(d)) == 1:
            p1 = [aabb[i1[k]][k] for k in range(d)]
            p2 = [aabb[i2[k]][k] for k in range(d)]
            lines.append((p1, p2))
    return lines