import pybullet as p
import matplotlib.pyplot as pt
from ergo import PoppyErgoEnv
import numpy as np
import random
import os

in_campus = False
gs = 10
scale = 20
fall_down = True
c = 10
if in_campus:
    base_path = 'C:\\Users\\Borui He\\Desktop\\falling detection images'
else:
    base_path = 'C:\\Users\\hbrch\\Desktop\\falling detection images'

def random_destination(base_path, seed=None):
    # randomization on the destination
    des_x = random.uniform(env.boundary['x'][0], env.boundary['x'][1])
    des_y = random.uniform(env.boundary['y'][0], env.boundary['y'][1])
    des_z = random.uniform(env.boundary['z'][0], env.boundary['z'][1])
    path = base_path + '\\({0}, {1}, {2})'.format(des_x, des_y, des_z)
    return (des_x, des_y, des_z), path

# set the number of destinations (which is the variable c) you want by running this script once, default is 1
while c:
    env = PoppyErgoEnv(p.POSITION_CONTROL, show=False, use_fixed_base=False, global_scale=gs, gravity=True)
    angles = env.angle_dict(env.get_position())
    angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})
    env.set_position(env.angle_array(angles))

    if fall_down:
        path = base_path + '\\falling down'
    else:
        path = base_path + '\\vibration'
    destination, path = random_destination(path)

    # pos, ori, lin, ang, joint states at timestep 0
    pos, ori, lin, ang = env.get_base()
    ini_state = {'pos': pos, 'ori': ori, 'lin': lin, 'ang': ang}
    ini_joint_states = env.get_position()

    # the force on a random link
    # link_idx = random.randint(0, max(list(env.joint_index.values())))

    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
        f = open(path + '\\log.txt', 'w')
        c -= 1

    state_log = []
    
    # falling downs
    if fall_down:
        for t in range(406):
            current, _ = p.getBasePositionAndOrientation(env.robot_id)

            if t > 100:
                force = scale * (np.array(destination) - np.array(current))
                p.applyExternalForce(objectUniqueId=env.robot_id, linkIndex=-1, forceObj=force, posObj=current, flags=p.WORLD_FRAME)
            
            env.step(action= ini_joint_states)
            
            # this takes a picture of falling down
            rgba, view, proj = env.get_camera_image()
            pt.imsave(path + '\\{}.png'.format(t), rgba)
            state_log.append('(' + f'{t:03}' + ', 1)')

    # vibration
    else:
        for t in range(406):
            current, _ = p.getBasePositionAndOrientation(env.robot_id)

            if t > 100 and t % 2  == 0:
                force = scale * (np.array(destination) - np.array(current))
                p.applyExternalForce(objectUniqueId=env.robot_id, linkIndex=-1, forceObj=force, posObj=current, flags=p.WORLD_FRAME)
                
            elif t > 100 and t % 2 == 1:
                force = -scale * (np.array(destination) - np.array(current))
                p.applyExternalForce(objectUniqueId=env.robot_id, linkIndex=-1, forceObj=force, posObj=current, flags=p.WORLD_FRAME)

            env.step(action= ini_joint_states)

            # this takes a picture of standing
            rgba, view, proj = env.get_camera_image()
            pt.imsave(path + '\\{3}.png'.format(destination[0], destination[1], destination[2], t), rgba)
            state_log.append('(' + f'{t:03}' + ', 0)')

    env.close()
    f.writelines('\n'.join(state_log))
    f.close()