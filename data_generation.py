import pybullet as p
import matplotlib.pyplot as pt
from env.poppy_env import PoppyEnv
from datetime import datetime
import numpy as np
import shutil
import random
import os
import pickle
import argparse



def std_randomization(env, t=188):
    # # randomization on the destination
    # des_x = random.uniform(env.boundary['x'][0], env.boundary['x'][1])
    # des_y = random.uniform(env.boundary['y'][0], env.boundary['y'][1])
    # des_z = random.uniform(env.boundary['z'][0], env.boundary['z'][1])
    # standard deviation for nomarl distributions in vibration cases
    threshold = 188
    if t <= threshold:
        factor = 0.25
    else:
        factor = (t - threshold) / 3
    std = random.uniform(0.0015, 0.006 * factor) # 17: 186, 18: 194

    # path = os.path.join(root_path, '({0}, {1}, {2}), {3}'.format(des_x, des_y, des_z, std))

    return std

def start_generation(gs, c, tp):
    start = datetime.now()
    if tp == 'both':
        work_list = [False for _ in range(c)] + [True for _ in range(c)]
    elif tp == 'standing':
        work_list = [False for _ in range(c)]
    elif tp == 'fall':
        work_list = [True for _ in range(c)]
    root_path = os.path.join(os.getcwd(), 'virtual_poppy', 'fall_new')

    while c:
        env = PoppyEnv(p.POSITION_CONTROL, show=False, use_fixed_base=False, global_scale=gs, gravity=True)
        angles = env.angle_dict(env.get_position())
        angles.update({'l_elbow_y': -90, 'r_elbow_y': -90, 'head_y': 35})
        env.set_position(env.angle_array(angles))
        path = os.path.join(root_path, str(c))
        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path)

        # pos, ori, lin, ang, joint states at timestep 0
        ini_joint_states = env.get_position()
        
        # coordinate of head camera and base
        hc_z, base_z = [], []

        # In either fall or standing cases, first 50 pictures should not be taken into consideration
        # standing: False in the 'work' list
        if not work_list[len(work_list) - c]:
            print('---generating a video where Poppy will stand---{}/{}'.format(len(work_list)-c+1, len(work_list)))
            for t in range(300):
                if t > 50:
                    std = std_randomization(env)
                    target = ini_joint_states + np.random.normal(0, std, (36))
                    target[-2:] = 0
                    target[-11:-9] = 0
                    target[4], target[9] = 0, 0
                    env.step(target)
                else:
                    env.step(ini_joint_states)
                    
                rgba, _, _ = env.get_camera_image()
                pt.imsave(os.path.join(path, '{}.png'.format(t)), rgba)
                
                base_pos, _, _, _ = env.get_base()
                base_z.append(base_pos)
                hc_pos, _, _, _, _, _, _, _ = p.getLinkState(env.robot_id, env.joint_index['head_cam'], computeLinkVelocity=1)
                hc_z.append(hc_pos)

            with open(os.path.join(path, 'base_z.pkl'), 'wb') as pf:
                pickle.dump(base_z, pf)
            with open(os.path.join(path, 'hc_z.pkl'), 'wb') as pf:
                pickle.dump(hc_z, pf)

        # fall: True in the 'work' list
        else:
            print('---generating a video where Poppy will fall---{}/{}'.format(len(work_list)-c+1, len(work_list)))
            for t in range(300):
                if t > 50:
                    std = std_randomization(env, t)
                    if t == 51:
                        target = ini_joint_states + np.random.normal(0, std, (36))
                    else:
                        target = old_target + np.random.normal(0, std, (36))
                    target[-2:] = 0
                    target[-11:-9] = 0
                    target[4], target[9] = 0, 0
                    env.step(target)
                    old_target = target
                else:
                    env.step(ini_joint_states)

                rgba, _, _ = env.get_camera_image()
                pt.imsave(os.path.join(path, '{}.png'.format(t)), rgba)

                base_pos, _, _, _ = env.get_base()
                base_z.append(base_pos)
                hc_pos, _, _, _, _, _, _, _ = p.getLinkState(env.robot_id, env.joint_index['head_cam'], computeLinkVelocity=1)
                hc_z.append(hc_pos)

            with open(os.path.join(path, 'base_z.pkl'), 'wb') as pf:
                pickle.dump(base_z, pf)
            with open(os.path.join(path, 'hc_z.pkl'), 'wb') as pf:
                pickle.dump(hc_z, pf)

        env.close()
        c -= 1

    print('generating a new dataset takes {}'.format(datetime.now()-start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gs', default= 10, help= 'global scaling', type= int)
    parser.add_argument('n', default= 2, help= 'the number of videos of each case you want every time the script is executed', type= int)
    parser.add_argument('type', default= 2, help= '\'fall\' for falls videos, \'standing\' for standing videos, \'both\' for n fall videos and n standing videos', type= str)

    args = parser.parse_args()
    gs = args.gs
    n = args.n
    tp = args.type
    # gs = 10
    # n = 1
    # tp = 'fall'
    start_generation(gs, n, tp)