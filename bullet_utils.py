import os
from os.path import isdir, join, abspath, isfile, dirname
import shutil
import sys
import numpy as np
import time
import json
import math
from datetime import datetime
import pybullet as p
import pybullet_planning as pp
from collections import namedtuple
from itertools import product, combinations
import functools
err = functools.partial(print, flush=True, file=sys.stderr)


GRASPS_DIR = abspath(join(dirname(__file__), 'grasps'))
COLLISION_FILE = join(GRASPS_DIR, 'collisions.json')
PP_RAINBOW_COLORS = [pp.RED, pp.BROWN, pp.YELLOW, pp.GREEN, pp.BLUE, pp.TAN, pp.GREY, pp.WHITE, pp.BLACK]


def add_text(cid, text, body_id, link_id=pp.BASE_LINK, position=(0., 0., 0.), color='r'):
    if isinstance(color, str):
        color = {'b': (0, 0, 0), 'r': (1, 0, 0)}[color]
    with pp.HideOutput():
        return p.addUserDebugText(str(text), textPosition=position, textColorRGB=color[:3],  # textSize=1,
                                  lifeTime=0, parentObjectUniqueId=body_id, parentLinkIndex=link_id,
                                  physicsClientId=cid)


def add_line(cid, start, end, color=pp.BLACK, width=1, parent=-1, parent_link=pp.BASE_LINK):
    assert (len(start) == 3) and (len(end) == 3)
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              lifeTime=0, parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=cid)


def draw_point(cid, point, size=0.01, **kwargs):
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size/2 * axis
        p2 = np.array(point) + size/2 * axis
        lines.append(add_line(cid, p1, p2, **kwargs))
    return lines


def get_pose(cid, body):
    return p.getBasePositionAndOrientation(body, physicsClientId=cid)


def set_pose(cid, body, pose):
    p.resetBasePositionAndOrientation(body, pose[0], pose[1], physicsClientId=cid)


def set_joint_position(cid, body, joint, value):
    p.resetJointState(body, joint, targetValue=value, targetVelocity=0, physicsClientId=cid)


def set_joint_positions(cid, body, joints, values):
    for joint, value in zip(joints, values):
        set_joint_position(cid, body, joint, value)


def get_collision_data(cid, body, link=pp.BASE_LINK):
    return [pp.CollisionShapeData(*tup) for tup in p.getCollisionShapeData(body, link, physicsClientId=cid)]


def get_visual_data(cid, body, link=pp.BASE_LINK):
    visual_data = [pp.VisualShapeData(*tup) for tup in p.getVisualShapeData(body, physicsClientId=cid)]
    list(filter(lambda d: d.linkIndex == link, visual_data))


def get_bodies(cid):
    return [p.getBodyUniqueId(i, physicsClientId=cid)
            for i in range(p.getNumBodies(physicsClientId=cid))]


#################################


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


def get_rotation_matrix(cid, body, verbose=False):
    import untangle
    r = pp.unit_pose()
    collision_data = get_collision_data(cid, body, link=0)
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
                r = pp.Pose(euler=pp.Euler(math.pi / 2, 0, -math.pi / 2))
            elif equal(rpy, (3.14, 3.14, -1.57), epsilon=0.1):
                r = pp.Pose(euler=pp.Euler(0, 0, math.pi / 2))
            elif equal(rpy, (1.57, 0, -1.57), epsilon=0.1):
                r = pp.Pose(euler=pp.Euler(math.pi/2, 0, -math.pi / 2))
            ROTATIONAL_MATRICES[urdf_file] = r
        r = ROTATIONAL_MATRICES[urdf_file]
    return r


def get_link_pose(cid, body, link):
    if link == pp.BASE_LINK:
        return get_pose(cid, body)
    link_state = get_link_state(cid, body, link)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation


def get_link_state(cid, body, link):
    return pp.LinkState(*p.getLinkState(body, link, physicsClientId=cid))


def get_num_joints(cid, body):
    return p.getNumJoints(body, physicsClientId=cid)


def get_joints(cid, body):
    return list(range(get_num_joints(cid, body)))


get_links = get_joints
get_num_links = get_num_joints


def get_model_pose(cid, body, link=None, **kwargs):
    if link is None:
        body_pose = pp.multiply(get_pose(cid, body), get_rotation_matrix(cid, body, **kwargs))
    else:
        body_pose = get_link_pose(cid, body, link)
    return body_pose


def implies(p1, p2):
    return not p1 or p2


def vertices_from_link(cid, body, link=pp.BASE_LINK, collision=True):
    # TODO: get_mesh_data(body, link=link)
    # In local frame
    vertices = []
    # PyBullet creates multiple collision elements (with unknown_file) when nonconvex
    get_data = get_collision_data if collision else get_visual_data
    for data in get_data(body, link):
        vertices.extend(apply_affine(cid, get_data_pose(data), pp.vertices_from_data(data)))
    return vertices


def get_data_pose(data):
    if isinstance(data, pp.CollisionShapeData):
        return (data.local_frame_pos, data.local_frame_orn)
    return (data.localVisualFrame_position, data.localVisualFrame_orientation)


def get_model_points(cid, body, link=None, draw_all_points=False, body_pose=None):
    if link is None:
        links = get_links(cid, body)
    else:
        links = [link]

    vertices = []
    colors = [pp.RED, pp.YELLOW, pp.GREEN, pp.BLUE, pp.TAN, pp.BLACK]
    for i, link in enumerate(links):
        vv = pp.vertices_from_rigid(body, link)
        if len(vv) > 0:
            cdata = get_collision_data(cid, body, link=link)
            if len(cdata) > 0:
                cdata = cdata[0]
            pose = (cdata.local_frame_pos, cdata.local_frame_orn)
            new_vertices = apply_affine(cid, pp.invert(pose), vv)
            vertices.extend(new_vertices)
            if draw_all_points and body_pose is not None:
                draw_points(cid, new_vertices, body_pose, color=colors[i])

                link_aabb = aabb_from_points(new_vertices)
                draw_bounding_box(cid, link_aabb, body_pose, color=colors[i])
    return vertices


def aabb_from_points(points):
    return pp.AABB(np.min(points, axis=0), np.max(points, axis=0))


def aabb_from_extent_center(extent, center):
    if center is None:
        center = np.zeros(len(extent))
    else:
        center = np.array(center)
    half_extent = np.array(extent) / 2.
    lower = center - half_extent
    upper = center + half_extent
    return pp.AABB(lower, upper)


def get_all_links(cid, body):
    return [pp.BASE_LINK] + list(get_links(cid, body))


def can_collide(cid, body, link=pp.BASE_LINK):
    return len(get_collision_data(cid, body, link=link)) != 0


def get_aabbs(cid, body, links=None, only_collision=True):
    if links is None:
        links = get_all_links(cid, body)
    if only_collision:
        # TODO: return the null bounding box
        links = [link for link in links if can_collide(cid, body, link)]
    return [get_aabb(cid, body, link=link) for link in links]


def get_aabb(cid, body: int, link: int = None):
    if link is None:
        return pp.aabb_union(get_aabbs(cid, body=body))
    return pp.AABB(*p.getAABB(body, linkIndex=link, physicsClientId=cid))


def remove_debug(cid, debug):
    p.removeUserDebugItem(debug, physicsClientId=cid)


remove_handle = remove_debug


def remove_handles(cid, handles):
    for handle in handles:
        remove_debug(cid, handle)


def draw_fitted_box(cid, body, link=None, draw_box=False, draw_centroid=False,
                    draw_points=False, verbose=False, **kwargs):
    body_pose = get_model_pose(cid, body, link=link, verbose=verbose)
    vertices = get_model_points(cid, body, link=link, draw_all_points=draw_points, body_pose=body_pose)
    # c = c.client_id

    ## form the aabb
    if link is None:
        link = -1
    data = get_collision_data(cid, body, link)
    if len(data) == 0 or data[0].geometry_type == p.GEOM_MESH:
        aabb = aabb_from_points(vertices)
    else:
        aabb = get_aabb(cid, body)

    ## other visualization options
    handles = []
    if draw_box:
        handles += draw_bounding_box(cid, aabb, body_pose, **kwargs)
    if draw_centroid:
        handles += draw_face_points(cid, aabb, body_pose, dist=0.04)
    return aabb, handles


def draw_face_points(cid, aabb, body_pose, dist=0.08):
    center = pp.get_aabb_center(aabb)
    w, l, h = pp.get_aabb_extent(aabb)
    faces = [(w/2+dist, 0, 0), (0, l/2+dist, 0), (0, 0, h/2+dist)]
    faces += [minus(0, f) for f in faces]
    faces = [add(f, center) for f in faces]
    faces = apply_affine(cid, body_pose, faces)
    handles = []
    for f in faces:
        handles.extend(draw_point(cid, f, size=0.02, color=pp.RED))
    return handles


def draw_points(cid, vertices, body_pose, size=0.01, **kwargs):
    vertices = apply_affine(cid, body_pose, vertices)
    handles = []
    num_vertices = 20
    if len(vertices) > num_vertices:
        gap = int(len(vertices)/num_vertices)
        vertices = vertices[::gap]
    for v in vertices:
        handles.append(draw_point(cid, v, size=size, **kwargs))
    return handles


def add(elem1, elem2):
    return tuple(np.asarray(elem1)+np.asarray(elem2))


def minus(elem1, elem2):
    return tuple(np.asarray(elem1)-np.asarray(elem2))


def dist(elem1, elem2):
    return np.linalg.norm(np.asarray(elem1)-np.asarray(elem2))


def tform_points(cid, affine, points):
    tform = tform_from_pose(cid, affine)
    points_homogenous = np.vstack([np.vstack(points).T, np.ones(len(points))])
    return tform.dot(points_homogenous)[:3, :].T


def tform_from_pose(cid, pose):
    (point, quat) = pose
    tform = np.eye(4)
    tform[:3, 3] = point
    tform[:3, :3] = matrix_from_quat(cid, quat)
    return tform


def matrix_from_quat(cid, quat):
    return np.array(p.getMatrixFromQuaternion(quat, physicsClientId=cid)).reshape(3, 3)


apply_affine = tform_points


def draw_bounding_box(cid, aabb, body_pose, **kwargs):
    handles = []
    for a, b in get_aabb_edges(aabb):
        p1, p2 = apply_affine(cid, body_pose, [a, b])
        handles.append(add_line(cid, p1, p2, **kwargs))
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


def has_gui(c):
    return pp.get_connection(c) == p.GUI


def set_renderer(cid, enable):
    if not has_gui(cid):
        return
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(enable), physicsClientId=cid)


def set_camera_target_body(body, link=None, dx=None, dy=None, dz=None, distance=1):
    # if isinstance(body, tuple):
    #     link = BODY_TO_OBJECT[body].handle_link
    #     body = body[0]
    aabb = get_aabb(body, link)
    x = (aabb.upper[0] + aabb.lower[0]) / 2
    y = (aabb.upper[1] + aabb.lower[1]) / 2
    z = (aabb.upper[2] + aabb.lower[2]) / 2
    if dx is None and dy is None and dz is None:
        dx = pp.get_aabb_extent(aabb)[0] * 2 * distance
        dy = pp.get_aabb_extent(aabb)[1] * 2 * distance
        dz = pp.get_aabb_extent(aabb)[2] * 2 * distance
    camera_point = [x + dx, y + dy, z + dz]
    target_point = [x, y, z]
    pp.set_camera_pose(camera_point=camera_point, target_point=target_point)
    return camera_point, target_point


####################################################################################

def nice_float(ele, round_to=3):
    if isinstance(ele, int) and '.' not in str(ele):
        return int(ele)
    else:
        return round(ele, round_to)


def nice_tuple(tup, round_to=3):
    new_tup = []
    for ele in tup:
        new_tup.append(nice_float(ele, round_to))
    return tuple(new_tup)


def nice(tuple_of_tuples, round_to=3, one_tuple=True):
    ## float, int
    if isinstance(tuple_of_tuples, float) or isinstance(tuple_of_tuples, int):
        return nice_float(tuple_of_tuples, round_to)

    elif len(tuple_of_tuples) == 0:
        return []

    ## position, pose
    elif isinstance(tuple_of_tuples[0], tuple) or isinstance(tuple_of_tuples[0], list) \
            or isinstance(tuple_of_tuples[0], np.ndarray):

        ## pose = (point, quat) -> (point, euler)
        if len(tuple_of_tuples[0]) == 3 and len(tuple_of_tuples[1]) == 4:
            if one_tuple:
                one_list = list(tuple_of_tuples[0]) + list(pp.euler_from_quat(tuple_of_tuples[1]))
                return nice(tuple(one_list), round_to)
            return nice( (tuple_of_tuples[0], pp.euler_from_quat(tuple_of_tuples[1])) , round_to)
            ## pose = (point, quat) -> (x, y, z, yaw)
            # return pose_to_xyzyaw(tuple_of_tuples)

        new_tuple = []
        for tup in tuple_of_tuples:
            new_tuple.append(nice_tuple(tup, round_to))
        return tuple(new_tuple)

    ## AABB
    elif isinstance(tuple_of_tuples, pp.AABB):
        lower, upper = tuple_of_tuples
        return pp.AABB(nice_tuple(lower, round_to), nice_tuple(upper, round_to))

    ## point, euler, conf
    return nice_tuple(tuple_of_tuples, round_to)


def get_datetime_form(year=True):
    form = "%m%d_%H%M%S"
    if year:
        form = "%y" + form
    return form


def get_datetime(year=True):
    form = get_datetime_form(year)
    return datetime.now().strftime(form)


def parse_datetime(string, year=True):
    form = get_datetime_form(year)
    return datetime.strptime(string, form).timestamp()


def draw_pose(cid, pose, length=0.1, d=3, **kwargs):
    origin_world = pp.tform_point(pose, np.zeros(3))
    handles = []
    for k in range(d):
        axis = np.zeros(3)
        axis[k] = 1
        axis_world = pp.tform_point(pose, length*axis)
        handles.append(add_line(cid, origin_world, axis_world, color=axis, **kwargs))
    return handles


def aabb_from_extent_center(extent, center=None):
    if center is None:
        center = np.zeros(len(extent))
    else:
        center = np.array(center)
    half_extent = np.array(extent) / 2.
    lower = center - half_extent
    upper = center + half_extent
    return pp.AABB(lower, upper)


def visualize_point(point):
    z = 0
    if len(point) == 3:
        x, y, z = point
    else:
        x, y = point
    body = pp.create_box(.05, .05, .05, mass=1, color=(1, 0, 0, 1))
    set_pose(body, pp.Pose(point=pp.Point(x, y, z)))
    return body


def get_loaded_scale(cid, body):
    data = get_collision_data(cid, body, 0)
    scale = None
    if len(data) > 0:
        scale = data[0].dimensions[0]
    else:
        print('get_scale | no collision data for body', body)
        # wait_unlocked()
    return scale


def get_grasp_db_file(robot):
    robot_name = robot.__class__.__name__
    db_file_name = f'hand_grasps_{robot_name}.json'
    db_file = join(GRASPS_DIR, db_file_name)
    if not isfile(db_file):
        os.makedirs(dirname(db_file), exist_ok=True)
        with open(db_file, 'w') as f:
            json.dump({}, f)
    return db_file


def find_grasp_in_db(db_file, instance_name, scale=None, verbose=True):
    """ find saved json files, prioritize databases/ subdir """
    db = json.load(open(db_file, 'r')) if isfile(db_file) else {}

    def rewrite_grasps(data):
        ## the newest format has poses written as (x, y, z, roll, pitch, row)
        if len(data[0]) == 6:
            found = [(tuple(e[:3]), pp.quat_from_euler(e[3:])) for e in data]
        elif len(data[0][1]) == 3:
            found = [(tuple(e[0]), pp.quat_from_euler(e[1])) for e in data]
        elif len(data[0][1]) == 4:
            found = [(tuple(e[0]), tuple(e[1])) for e in data]
        if verbose:
            print(f'    bullet_utils.find_grasp_in_db returned {len(found)}'
                f' grasps for {instance_name} | scale = {scale}')
        return found

    found = None
    if instance_name in db:
        all_data = db[instance_name]
        ## the newest format has attr including 'name', 'grasps', 'grasps_length_variants'
        if '::' not in instance_name or ('scale' in all_data and scale == all_data['scale']):
            data = all_data['grasps']
            if len(data) > 0:
                found = rewrite_grasps(data)
                ## scale the grasps for object grasps but not handle grasps
                if scale is not None and 'scale' in all_data and scale != all_data['scale']:
                    found = [(tuple(scale/all_data['scale'] * np.array(p)), q) for p, q in found]
                    # new_found = []
                    # for p, q in found:
                    #     p = np.array(p)
                    #     p[:2] *= scale / all_data['scale']
                    #     p[2] *= scale * 1.4 / all_data['scale']
                    #     new_found.append((tuple(p), q))
                    # found = new_found
        elif 'other_scales' in all_data and str(scale) in all_data['other_scales']:
            data = all_data['other_scales'][str(scale)]
            found = rewrite_grasps(data)

    return found, db, db_file


def dump_json(db, db_file, indent=2, width=160, sort_dicts=True, **kwargs):
    """ don't break lines for list elements """
    import pprint
    with open(db_file, 'w') as f:
        # pprint(db, f, indent=2, width=120) ## single quote
        f.write(pprint.pformat(db, indent=indent, width=width, sort_dicts=sort_dicts,
                               **kwargs).replace("'", '"'))


def save_grasp_db(db, db_file):
    keys = {k: v['datetime'] for k, v in db.items()}
    keys = sorted(keys.items(), key=lambda x: x[1])
    db = {k: db[k] for k, v in keys}
    if isfile(db_file): os.remove(db_file)
    dump_json(db, db_file, sort_dicts=False)


def add_grasp_in_db(db, db_file, instance_name, grasps, name=None, scale=None, grasp_sides=None):
    if instance_name is None: return

    add_grasps = []
    for g in grasps:
        add_grasps.append(list(nice(g, 4)))
    if len(add_grasps) == 0:
        return

    ## -------- save to json
    if name is None:
        name = 'None'

    if instance_name in db:
        if 'other_scales' not in db[instance_name]:
            db[instance_name]['other_scales'] = {}
        db[instance_name]['other_scales'][str(scale)] = add_grasps
    else:
        db[instance_name] = {
            'name': name,
            'grasps' : add_grasps,
            'datetime': get_datetime(),
            'scale': scale,
        }
        if grasp_sides is not None:
            db[instance_name]['grasp_sides'] = grasp_sides
    save_grasp_db(db, db_file)
    print('\n    bullet_utils.add_grasp_in_db saved', instance_name, '\n')


def set_gripper_pose(c, body, robot, grasp_pose, try_length=False):
    pose = c.w.get_body_state_by_id(body)[:2]
    lengths = [0] if try_length else [0]
    for dz in lengths:
        result = True
        new_point = np.array(grasp_pose[0])
        which_dim = np.where(np.abs(new_point) != 0)
        new_point[which_dim] -= np.sign(new_point[which_dim]) * dz
        grasp = (new_point.tolist(), grasp_pose[1])
        pick_pose = pp.multiply(pose, grasp)

        robot.open_gripper_free()

        ## grasp ik should be solvable
        pick_q = robot.ikfast(pick_pose[0], pick_pose[1], error_on_fail=False)
        if pick_q is None:
            continue

        ## shouldn't collide with anything when opened gripper at grasp conf
        colliding = robot.is_colliding(pick_q)
        if colliding:
            continue
        """ 
        add the following to urdf:
        <contact>
          <lateral_friction value="1"/>
          <rolling_friction value="0.0001"/>
          <inertia_scaling value="3.0"/>
        </contact>
        <inertial>
          <origin rpy="0 0 0" xyz="0 0 0"/>
           <mass value="0.2"/>
           <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
        
        for finding strange grasp poses:
            g = pp.multiply(((0.04, 0, 0.036), (0.5, -0.5, -0.5, 0.5)), (pp.unit_point(), pp.quat_from_euler((np.pi/4, 0, 0))))
            pick_pose = pp.multiply(pose, g)
            colliding = robot.is_colliding(robot.ikfast(pick_pose[0], pick_pose[1], error_on_fail=False))
        
        finding other symmetrically grasp poses:
            gg = pp.multiply(g, (pp.unit_point(), pp.quat_from_euler((0, np.pi/4, 0))))
            gg = ((-0.04, 0.0, 0.036), pp.quat_from_euler((-2.356, 0.0, -1.571)))
            gg = ((0, -0.04, 0.036), pp.quat_from_euler((2.356, 0, 3.1415)))
            pick_pose = pp.multiply(pose, gg)
            colliding = robot.is_colliding(robot.ikfast(pick_pose[0], pick_pose[1], error_on_fail=False))
        
        for finding partnet name
            p.getBodyInfo(body, c.client_id)[1]
            get_loaded_scale(c.client_id, body)
            add_grasp_in_db(db, db_file, instance_name, [gg], name=c.w.get_body_name(body), scale=scale)
        """
        # colliding = c.w.get_contact(robot.panda)
        # colliding = [c for c in colliding if c.body_b != robot.panda]
        # if len(colliding) > 0:
        #     continue

        ## should collide with the object when closed gripper at grasp conf
        robot.close_gripper_free()
        colliding = [cl for cl in c.w.get_contact(robot.panda) if cl.body_b == body]
        if len(colliding) == 0:
            continue
        gripper_pstn = sum(c.w.get_batched_qpos_by_id(robot.panda, robot.gripper_joints))
        if np.sum(np.abs(gripper_pstn)) < 0.01:
            continue
        if result:
            return grasp
    return None


def create_attachment(c, body, container, body_pose=None):
    if body_pose is None:
        body_pose = get_pose(c.client_id, body)
    container_pose = get_pose(c.client_id, container)
    grasp_pose = pp.multiply(pp.invert(container_pose), body_pose, ((0, 0, 0), pp.quat_from_euler((np.pi/2, 0, np.pi/2))))
    c.p.createConstraint(
        parentBodyUniqueId=container,
        parentLinkIndex=-1,
        childBodyUniqueId=body,
        childLinkIndex=0,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=grasp_pose[0],
        parentFrameOrientation=grasp_pose[1],
        childFramePosition=(0, 0, 0),
        childFrameOrientation=(0, 0, 0)
    )


def get_grasp_poses(c, robot, body, instance_name='test', link=None, grasp_length=0.02,
                    HANDLE_FILTER=False, visualize=False, verbose=True, faces=None):
    cid = c.client_id
    body_name = (body, link) if link is not None else body
    title = f'bullet_utils.get_hand_grasps({body_name}) | '
    dist = grasp_length
    scale = get_loaded_scale(cid, body)
    print("get hand grasps scale", scale)

    body_pose = get_model_pose(cid, body, link=link, verbose=verbose)
    print("get hand grasps, body pose", body_pose)

    if link is None:  ## grasp the whole body
        r = pp.Pose(euler=pp.Euler(math.pi / 2, 0, -math.pi / 2))
        body_pose = pp.multiply(body_pose, pp.invert(r))
    else:  ## grasp handle links
        body_pose = pp.multiply(body_pose, pp.invert(robot.tool_from_hand))

    ## retrieve from database
    grasp_db_file = get_grasp_db_file(robot)
    found, db, db_file = find_grasp_in_db(grasp_db_file, instance_name, verbose=verbose, scale=scale)
    if found is not None:
        if visualize:
            bodies = []
            for g in found:
                bodies.append(robot.visualize_grasp(body_pose, g, verbose=verbose))
            # set_renderer(True)
            set_camera_target_body(body)
            # wait_unlocked()
            for b in bodies:
                pp.remove_body(b)
        # remove_handles(cid, handles)
        return found

    ## hack to get generate grasps with known variations
    add_grasp_in_db(db, db_file, instance_name,
                    [((0.04, 0, 0.036), (0.5, -0.5, -0.5, 0.5))],
                    name=c.w.get_body_name(body), scale=scale)
    return

    aabb, handles = draw_fitted_box(cid, body, link=link, verbose=verbose, draw_box=True, draw_centroid=True)

    ## get the points in hand frame to be transformed to the origin of object frame in different directions
    center = pp.get_aabb_center(aabb)
    w, l, h = dimensions = pp.get_aabb_extent(aabb)
    if faces is None:
        faces = [(w/2+dist, 0, 0), (0, l/2+dist, 0), (0, 0, h/2+dist)]
        faces += [minus(0, f) for f in faces]
    else:
        faces = [(np.array(f) * np.array((w/2+dist, l/2+dist, h/2+dist))).tolist() for f in faces]

    ## for finding the longest dimension
    max_value = max(dimensions)
    filter = [int(x != max_value) for x in dimensions]

    P = math.pi
    rots = {
        (1, 0, 0): [(P/2, 0, -P/2), (P/2, P, -P/2), (P/2, -P/2, -P/2), (P/2, P/2, -P/2)],
        (-1, 0, 0): [(P/2, 0, P/2), (P/2, P, P/2), (P/2, -P/2, P/2), (P/2, P/2, P/2), (-P, -P/2, 0), (-P, -P/2, -P)],
        (0, 1, 0): [(0, P/2, -P/2), (0, -P/2, P/2), (P/2, P, 0), (P/2, 0, 0)],
        (0, -1, 0): [(0, P/2, P/2), (0, -P/2, -P/2), (-P/2, P, 0), (-P/2, 0, 0)],
        (0, 0, 1): [(P, 0, P/2), (P, 0, -P/2), (P, 0, 0), (P, 0, P)],
        (0, 0, -1): [(0, 0, -P/2), (0, 0, P/2), (0, 0, 0), (0, 0, P)],
    }
    grasps = []
    for f in faces:
        p = np.array(f)
        p = p / np.linalg.norm(p)

        ## only attempt the bigger surfaces
        on_longest = sum([filter[i]*p[i] for i in range(3)]) != 0
        if HANDLE_FILTER and not on_longest:
            continue

        ang = tuple(p)
        # f = add(f, center)
        # r = rots[ang][0] ## random.choice(rots[tuple(p)]) ##

        for r in rots[ang]:
            grasp = pp.multiply(pp.Pose(point=f), pp.Pose(euler=r))
            grasp = set_gripper_pose(c, body, robot, grasp, try_length=True)
            if grasp is None:
                continue
            grasps.append(grasp)

        # ## just to look at the orientation
        # if debug_del:
        #     set_camera_target_body(body, dx=0.3, dy=0.3, dz=0.3)
        #     print(f'bullet_utils.get_hand_grasps | rots[{ang}]', len(rots[ang]), [nice(n) for n in rots[ang]])
        #     print(f'bullet_utils.get_hand_grasps -> ({len(these)})', [nice(n[1]) for n in these])
        #     print('bullet_utils.get_hand_grasps')

    # set_renderer(True)
    if verbose:
        print(f"{title} ({len(grasps)}) {[nice(g) for g in grasps]}")
        if len(grasps) == 0:
            print(title, 'no grasps found')

    ## lastly store the newly sampled grasps
    name = c.w.get_body_name(body)
    grasp_sides = get_grasp_sides(c, body, grasps)
    add_grasp_in_db(db, db_file, instance_name, grasps, name=name, scale=scale, grasp_sides=grasp_sides)
    remove_handles(cid, handles)
    # if len(grasps) > num_samples:
    #     random.shuffle(grasps)
    #     return grasps[:num_samples]
    return grasps  ##[:1]


def load_floating_gripper(c):
    FEG_FILE = 'assets://franka_description/robots/hand.urdf'
    with c.disable_rendering():
        feg = c.load_urdf(FEG_FILE, pos=pp.unit_point(), quat=pp.unit_quat(), static=True, body_name='feg', group=None)
        set_all_color(c.client_id, feg, color=pp.GREEN)
        return feg


CollisionInfo = namedtuple('CollisionInfo',
                           '''
                           contactFlag
                           bodyUniqueIdA
                           bodyUniqueIdB
                           linkIndexA
                           linkIndexB
                           positionOnA
                           positionOnB
                           contactNormalOnB
                           contactDistance
                           normalForce
                           lateralFriction1
                           lateralFrictionDir1
                           lateralFriction2
                           lateralFrictionDir2
                           '''.split())


CollisionPair = namedtuple('Collision', ['body', 'links'])


def get_closest_points(cid, body1, body2, link1=None, link2=None, max_distance=pp.MAX_DISTANCE, use_aabb=False):
    if use_aabb and not pp.aabb_overlap(get_buffered_aabb(body1, link1, max_distance=max_distance/2.),
                                        get_buffered_aabb(body2, link2, max_distance=max_distance/2.)):
        return []
    if (link1 is None) and (link2 is None):
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance, physicsClientId=cid)
    elif link2 is None:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=link1,
                                     distance=max_distance, physicsClientId=cid)
    elif link1 is None:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexB=link2,
                                     distance=max_distance, physicsClientId=cid)
    else:
        results = p.getClosestPoints(bodyA=body1, bodyB=body2, linkIndexA=link1, linkIndexB=link2,
                                     distance=max_distance, physicsClientId=cid)
    if results is None:  ## after reinstalling pybullet, problems occur
        return []
    return [CollisionInfo(*info) for info in results]


def parse_body(body, link=None):
    return body if isinstance(body, tuple) else CollisionPair(body, link)


def get_buffered_aabb(body, link=None, max_distance=pp.MAX_DISTANCE, **kwargs):
    if link is None:
        body, links = parse_body(body, link=link)
    else:
        links = [link]
    return buffer_aabb(pp.aabb_union(get_aabbs(body, links=links, **kwargs)), buffer=max_distance)


def buffer_aabb(aabb, buffer):
    if (aabb is None) or (buffer is None) or (np.isscalar(buffer) and (buffer == 0.)):
        return aabb
    extent = pp.get_aabb_extent(aabb)
    if np.isscalar(buffer):
        buffer = buffer * np.ones(len(extent))
    new_extent = np.add(2*buffer, extent)
    center = pp.get_aabb_center(aabb)
    return aabb_from_extent_center(new_extent, center)


def pairwise_link_collision(cid, body1, link1, body2, link2=pp.BASE_LINK, **kwargs):
    return len(get_closest_points(cid, body1, body2, link1=link1, link2=link2, **kwargs)) != 0


def body_collision(cid, body1, body2, **kwargs):
    results = get_closest_points(cid, body1, body2, **kwargs)
    return len(results) != 0


##########################################################################################


def get_reachability_dir():
    return abspath(join(dirname(__file__), 'reachability'))


def get_reachability_file(name, num_samples=50000, use_rrt=False):
    """ {name} = {cat}_{model_id} """
    output_dir = get_reachability_dir()
    name += f'_n={num_samples}'
    if use_rrt:
        name += '_rrt'
    return join(output_dir, f'{name}.csv')


def get_grasps_given_name_scale(name, scale):
    import pandas as pd
    df = pd.read_csv(get_reachability_file(name))
    df = df.loc[df['solved'] == True]
    scales = df.scale.unique()
    scale = min(scales, key=lambda x: abs(x - scale))
    df = df.loc[df['scale'] == scale]
    return df


def get_pose_from_row(row, z):
    return ((row['x'], row['y'], z), pp.quat_from_euler((0, 0, row['theta'])))


def sample_reachable_pose_nearest_neighbor(name, scale, pose_i, z, valid_grasps):
    df = get_grasps_given_name_scale(name, scale)
    df = df.loc[df['grasp_id'].isin(valid_grasps)]

    ## find the row in df that has the minimum distance to the given pose
    df['dist'] = df.apply(lambda row: np.linalg.norm(np.array([row['x'], row['y']]) - pose_i), axis=1)
    row = df.loc[df['dist'].idxmin()]
    pose = get_pose_from_row(row, z)
    grasp_id = row['grasp_id']
    return pose, grasp_id


def sample_reachable_pose_given_grasp(name, scale, grasp_id, z):
    df = get_grasps_given_name_scale(name, scale)
    df = df.loc[df['grasp_id'] == grasp_id]
    idx = np.random.randint(len(df))
    row = df.iloc[idx]
    return get_pose_from_row(row, z)


##########################################################################################


def load_body_five_sides(c, body):
    aabb = get_aabb(c.client_id, body)
    w, l, h = pp.get_aabb_extent(aabb)
    x, y, z = pp.get_aabb_center(aabb)
    t = 0.001
    boxes = [
        ('z+', (w, l, t), (x, y, z + h / 2 - t)),
        ('y+', (w, t, h), (x, y + l / 2 - t, z)),
        ('y-', (w, t, h), (x, y - l / 2 + t, z)),
        ('x+', (t, l, h), (x + w / 2 - t, y, z)),
        ('x-', (t, l, h), (x - w / 2 + t, y, z)),
    ]
    all_sides = {}
    for i, (label, dim, pose) in enumerate(boxes):
        side = c.load_urdf_template(
            'assets://box/box-template.urdf',
            {'DIM': dim, 'LATERAL_FRICTION': 1.0, 'MASS': 0.2,
             'COLOR': pp.apply_alpha(PP_RAINBOW_COLORS[i], 0.5)},
            pose, body_name=label, group='rigid', static=False
        )
        all_sides[side] = label
    return all_sides


def remove_body_five_sides(c, all_sides):
    for side in all_sides:
        c.remove_body(side)
    return False


def get_grasp_sides(c, name, robot, body, grasps):
    """ return a list of lists of sides of the object that the grasp is on """

    cid = c.client_id
    scale = get_loaded_scale(cid, body)
    c.w.set_batched_qpos_by_id(robot.panda, robot.gripper_joints, [0, 0])

    all_sides = None
    exist_sides = False
    pose = get_pose(cid, body)
    z = get_pose(cid, body)[0][2]
    all_grasp_sides = []
    for i, grasp in enumerate(grasps):
        grasp_sides = []

        ## re-place object if it could not be grasped
        pick_ee_pose = pp.multiply(pose, grasp)
        pick_q = robot.ikfast(pick_ee_pose[0], pick_ee_pose[1], error_on_fail=False)
        while pick_q is None:
            pose = sample_reachable_pose_given_grasp(name, scale, i, z)
            exist_sides = False
            pick_ee_pose = pp.multiply(pose, grasp)
            pick_q = robot.ikfast(pick_ee_pose[0], pick_ee_pose[1], error_on_fail=False)

        robot.set_qpos(pick_q)
        set_pose(cid, body, pose)

        ## put sides for collision checking
        if not exist_sides:
            if all_sides is not None:
                remove_body_five_sides(c, all_sides)
            all_sides = load_body_five_sides(c, body)

        for side, label in all_sides.items():
            d = {'x': 0, 'y': 1, 'z': 2}[label[0]]
            contacts = []
            for link_name in ['leftfinger', 'rightfinger', 'hand']:
                link = c.w.link_names[f'panda/panda_{link_name}'].link_id
                contacts += get_closest_points(cid, robot.panda, side, link)
            # contacts = [ct for ct in contacts if ct[3] < 0.01]
            if len(contacts) > 0:
                normals_on_b = [ct.contactNormalOnB[d] for ct in contacts]
                normal_on_b = sum(normals_on_b) / len(normals_on_b)
                grasp_sides.append([label, round(normal_on_b, 3)])
        all_grasp_sides.append(grasp_sides)
    err(all_grasp_sides)
    remove_body_five_sides(c, all_sides)
    return all_grasp_sides


def draw_aabb(cid, aabb, **kwargs):
    d = len(aabb[0])
    vertices = list(product(range(len(aabb)), repeat=d))
    lines = []
    for i1, i2 in combinations(vertices, 2):
        if sum(i1[k] != i2[k] for k in range(d)) == 1:
            p1 = [aabb[i1[k]][k] for k in range(d)]
            p2 = [aabb[i2[k]][k] for k in range(d)]
            lines.append(add_line(cid, p1, p2, **kwargs))
    return lines


def draw_goal_pose(cid, body, pose_g, **kwargs):
    aabb = aabb_from_extent_center(pp.get_aabb_extent(get_aabb(cid, body)), pose_g[0])
    # pose = get_pose(cid, body)
    # set_pose(cid, body, pose_g)
    # aabb = get_aabb(cid, body)
    draw_aabb(cid, aabb, **kwargs)
    # set_pose(cid, body, pose)


def save_pointcloud_to_ply(c, body, pcd_path, zero_center=True, points_per_geom=100):
    from pyntcloud import PyntCloud
    import pandas as pd

    start = time.time()
    points = c.w.get_pointcloud(body, zero_center=zero_center, points_per_geom=points_per_geom)
    colors = np.ones_like(points) * 0
    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=np.hstack((points, colors)), columns=["x", "y", "z", "red", "green", "blue"]))
    cloud.to_file(pcd_path)
    print(f'saved point cloud to {pcd_path} in {round(time.time() - start)} sec')


#################################################################


CameraImage = namedtuple('CameraImage', ['rgbPixels', 'depthPixels', 'segmentationMaskBuffer',
                                         'camera_pose', 'camera_matrix'])


def get_image_flags(segment=False, segment_links=False):
    if segment:
        if segment_links:
            return p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        return 0 # TODO: adjust output dimension when not segmenting links
    return p.ER_NO_SEGMENTATION_MASK


def compiled_with_numpy():
    return bool(p.isNumpyEnabled())


def get_focal_lengths(dims, fovs):
    return np.divide(dims, np.tan(fovs / 2)) / 2


def get_camera_matrix(width, height, fx, fy=None):
    if fy is None:
        fy = fx
    cx, cy = (width - 1) / 2., (height - 1) / 2.
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def get_image(cid, camera_pos, target_pos, width=640, height=480, vertical_fov=60.0, near=0.02, far=5.0,
              tiny=False, segment=False, **kwargs):
    # computeViewMatrixFromYawPitchRoll
    up_vector = [0, 0, 1] # up vector of the camera, in Cartesian world coordinates
    view_matrix = p.computeViewMatrix(cameraEyePosition=camera_pos, cameraTargetPosition=target_pos,
                                      cameraUpVector=up_vector, physicsClientId=cid)
    projection_matrix = pp.get_projection_matrix(width, height, vertical_fov, near, far)

    flags = get_image_flags(segment=segment, **kwargs)
    # DIRECT mode has no OpenGL, so it requires ER_TINY_RENDERER
    renderer = p.ER_TINY_RENDERER if tiny else p.ER_BULLET_HARDWARE_OPENGL
    width, height, rgb, d, seg = p.getCameraImage(width, height,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=projection_matrix,
                                                  shadow=False, # only applies to ER_TINY_RENDERER
                                                  flags=flags,
                                                  renderer=renderer,
                                                  physicsClientId=cid)
    if not compiled_with_numpy():
        rgb = np.reshape(rgb, [height, width, -1]) # 4
        d = np.reshape(d, [height, width])
        seg = np.reshape(seg, [height, width])

    depth = far * near / (far - (far - near) * d)
    segmented = seg

    camera_tform = np.reshape(view_matrix, [4, 4])
    camera_tform[:3, 3] = camera_pos
    view_pose = pp.multiply(pp.pose_from_tform(camera_tform), pp.Pose(euler=pp.Euler(roll=pp.PI)))

    focal_length = get_focal_lengths(height, vertical_fov) # TODO: horizontal_fov
    camera_matrix = get_camera_matrix(width, height, focal_length)

    return CameraImage(rgb, depth, segmented, view_pose, camera_matrix)


def dimensions_from_camera_matrix(camera_matrix):
    cx, cy = np.array(camera_matrix)[:2, 2]
    width, height = (2*cx + 1), (2*cy + 1)
    return width, height


def get_field_of_view(camera_matrix):
    dimensions = np.array(dimensions_from_camera_matrix(camera_matrix))
    focal_lengths = np.array([camera_matrix[i, i] for i in range(2)])
    return 2*np.arctan(np.divide(dimensions, 2*focal_lengths))


def get_camera_image_at_pose(camera_point, target_point, camera_matrix, far=5.0, **kwargs):
    # far is the maximum depth value
    width, height = map(int, dimensions_from_camera_matrix(camera_matrix))
    _, vertical_fov = get_field_of_view(camera_matrix)
    return pp.get_image(camera_point, target_point, width=width, height=height,
                        vertical_fov=vertical_fov, far=far, **kwargs)


TRANSPARENT = (0, 0, 0, 0)


def in_pycharm():
    import psutil
    return psutil.Process(os.getpid()).name() not in ['zsh', 'bash', 'python']  ## will ve 'java' if ran inside the IDE


def set_color(cid, body, color, link=pp.BASE_LINK, shape_index=pp.NULL_ID):
    if link is None:
        return set_all_color(cid, body, color)
    return p.changeVisualShape(body, link, shapeIndex=shape_index, rgbaColor=color,
                               physicsClientId=cid)


def set_all_color(cid, body, color):
    for link in get_all_links(cid, body):
        set_color(cid, body, color, link)


def get_bullet_image(cid, camera_pose, target_pose):
    rgb, depth, seg, pose, matrix = get_image(
        cid, camera_pose, target_pose, width=640, height=480, vertical_fov=30,
        near=0.02, far=5.0, tiny=False, segment=False)
    return rgb


def rotate_180(array):
    M, N = array.shape[:2]
    out = np.zeros_like(array)
    for i in range(M):
        for j in range(N):
            out[i, N-1-j] = array[M-1-i, j]


def take_bullet_image(cid, camera_pose, target_pose, png_name=None, tf='rot180'):
    rgb = get_bullet_image(cid, camera_pose, target_pose)
    if tf == 'rot180':
        rgb = np.rot90(rgb, 2)
    if png_name is not None:
        save_image(rgb, png_name)
    return rgb


def take_top_down_image(cid, robot, tray_point, png_name=None, goal_pose_only=False):
    if goal_pose_only:
        set_all_color(cid, robot.panda, TRANSPARENT)
        camera_pose = tuple((np.array(tray_point) + np.array([-0.01, 0, 0.5])).tolist())
        target_pose = tray_point
    else:
        q = [1.57, 1.8, -3.14, -1.57, -1.57, 1.57, 0]
        robot.set_qpos(q)
        camera_pose = [0, 0.01, 1.2]
        target_pose = (0, 0, 0)

    rgb = take_bullet_image(cid, camera_pose, target_pose, png_name)

    if goal_pose_only:
        set_all_color(cid, robot.panda, (1, 1, 1, 1))
    else:
        robot.set_qpos(robot.get_home_qpos())

    return rgb


def save_image(image, png_name):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(png_name, bbox_inches='tight', dpi=100)
    plt.close()
    # if in_pycharm():
    #     plt.show()
    # else:
    #     plt.close()


def images_to_gif(img_arrs, gif_file, repeat=1, crop=None):
    import imageio
    with imageio.get_writer(gif_file, mode='I') as writer:
        for img in img_arrs:
            if crop is not None:
                left, top, right, bottom = crop
                img = img[top:bottom, left:right]
            for i in range(repeat):
                writer.append_data(img)
    return gif_file


def merge_images(before, after, png_name):
    ## concatenate the left half of before array with the right half of after array
    before = before[:, :before.shape[1]//2, :]
    after = after[:, after.shape[1]//2:, :]
    image = np.concatenate((before, after), axis=1)
    save_image(image, png_name)


######################################################################################


def parallel_processing(process, inputs, parallel, debug=False):
    start_time = time.time()
    num_cases = len(inputs)

    outputs = []
    if parallel:
        import multiprocessing
        from multiprocessing import Pool

        max_cpus = 10
        num_cpus = min(multiprocessing.cpu_count(), max_cpus)
        print(f'\nusing {num_cpus} cpus\n')
        with Pool(processes=num_cpus) as pool:
            pool.map(process, inputs)
    else:

        for i in range(num_cases):
            outputs.append(process(inputs[i]))

    if debug:
        print(f'went through {num_cases} run_dirs (parallel={parallel}) in {round(time.time() - start_time, 3)} sec')
    return outputs
