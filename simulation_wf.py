import pybullet as p
import matplotlib.pyplot as pt
from ergo import PoppyErgoEnv
import numpy as np
import random
import os


gs = 10
scale = 20
fall_down = True

def random_destination(seed=None):
    # randomization on the destination
    des_x = random.uniform(env.boundary['x'][0], env.boundary['x'][1])
    des_y = random.uniform(env.boundary['y'][0], env.boundary['y'][1])
    des_z = random.uniform(env.boundary['z'][0], env.boundary['z'][1])
    path = os.getcwd() + '\\camera_images\\({0}, {1}, {2})'.format(des_x, des_y, des_z)
    return (des_x, des_y, des_z), path

def will_fall(env, base_states=None):
    # pos, ori, _, _, _, _, lin, ang = p.getLinkState(env.robot_id, env.joint_index['r_hip_x'], computeLinkVelocity=1)
    pos, ori, lin, ang = env.get_base()
    # ori = p.getEulerFromQuaternion(ori)

    if base_states == None:
        return {'pos': pos, 'ori': ori, 'lin': lin, 'ang': ang}

    diff_pos = np.array(base_states['pos']) - np.array(pos)
    # diff_ori = np.array(base_states['ori']) - np.array(ori)
    diff_lin = np.array(base_states['lin']) - np.array(lin)
    diff_ang = np.array(base_states['ang']) - np.array(ang)
    # print(abs(diff_pos[2]))
    # print(abs(diff_lin[2]))
    # print(diff_ang)

    # TODO: set a threshod for each difference on Z axis?
    if abs(diff_pos[2]) >= 0 and abs(diff_lin[2]) >= 0 and abs(diff_ang[0] >= 0) and abs(diff_ang[1] >= 0):
        return True
    else:
        return False

# set the number of destinations you want at a time (running this code), default is 1
c = 1
while c:
    env = PoppyErgoEnv(p.POSITION_CONTROL, show=False, use_fixed_base=False, global_scale=gs, gravity=True)
    angles = env.angle_dict(env.get_position())
    angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})
    env.set_position(env.angle_array(angles))

    destination, path = random_destination()
    # destination = (1, 1, 0.4)
    # path = 'C:\\Users\\Borui He\\OneDrive - Syracuse University\\falling detection\\camera_images\\(1, 1, 0.4)'
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
        # os.mkdir(path + '\\standing')
        # os.mkdir(path + '\\falling down')
        f = open(path + '\\log.txt', 'w')
        c -= 1

    state_log = []
    
    # vibration
    if not fall_down:
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
            pt.imsave('camera_images\\({0}, {1}, {2})\\{3}.png'.format(destination[0], destination[1], destination[2], 2*t+1), rgba)
            state_log.append('(' + f'{t:03}' + ', 0)')


    # gradually fall down
    else:
        for t in range(406):
            current, _ = p.getBasePositionAndOrientation(env.robot_id)

            if t > 100:
                force = scale * (np.array(destination) - np.array(current))
                p.applyExternalForce(objectUniqueId=env.robot_id, linkIndex=-1, forceObj=force, posObj=current, flags=p.WORLD_FRAME)
            
            env.step(action= ini_joint_states)

            isfa = will_fall(env, ini_state)

            if isfa:
                # this takes a picture of falling down
                rgba, view, proj = env.get_camera_image()
                pt.imsave('camera_images\\({0}, {1}, {2})\\{3}.png'.format(destination[0], destination[1], destination[2], t), rgba)
                state_log.append('(' + f'{t:03}' + ', 1)')
            else:
                # this takes a picture of standing
                rgba, view, proj = env.get_camera_image()
                pt.imsave('camera_images\\({0}, {1}, {2})\\{3}.png'.format(destination[0], destination[1], destination[2], t), rgba) 
                state_log.append('(' + f'{t:03}' + ', 0)')
    env.close()

    f.writelines('\n'.join(state_log))
    f.close()