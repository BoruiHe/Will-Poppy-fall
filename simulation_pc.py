import pybullet as p
import matplotlib.pyplot as pt
from env.poppy_env import PoppyEnv
import numpy as np
import random
import os
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--mode', action= 'store_true', help= 'either falling down(True) or vibration(False)')
parser.add_argument('-t', '--dataset', action= 'store_true', help= 'generate training dataset(False) or testing dataset(True)')
parser.add_argument('gs', default= 10, help= 'global scaling', type= int)
# parser.add_argument('fs', default= 20, help= 'force scale', type= int)
parser.add_argument('nlv', default= 1, help= 'the number of long videos you want every time the script is executed', type= int)

args = parser.parse_args()
gs = args.gs
# scale = args.fs
train_test = args.dataset
fall_down = args.mode
c = args.nlv

# TODO: parser should include the random seed given by users, which would be necessory for the method called randomization

# gs = parser.parse_args(['-t', '-f', '10', '1']).gs
# # scale = parser.parse_args(['-f', '10', '1']).fs
# train_test = parser.parse_args(['-f', '10', '1']).dataset
# fall_down = parser.parse_args(['-t', '10', '1']).mode
# c = parser.parse_args(['-t', '-f', '10', '1']).nlv

if train_test:
    root_path = os.getcwd() + '\\data\\testing'
else:
    root_path = os.getcwd() + '\\data\\training'

def randomization(root_path, seed= None, fd= True):
    # randomization on the destination
    des_x = random.uniform(env.boundary['x'][0], env.boundary['x'][1])
    des_y = random.uniform(env.boundary['y'][0], env.boundary['y'][1])
    des_z = random.uniform(env.boundary['z'][0], env.boundary['z'][1])
    # standard deviation for nomarl distributions in vibration cases
    std = random.uniform(0.001, 0.01)
    if fd:
        path = root_path + '\\falling down\\({0}, {1}, {2}), {3}'.format(des_x, des_y, des_z, std)
    else:
        path = root_path + '\\vibration\\({0}, {1}, {2}), {3}'.format(des_x, des_y, des_z, std)
    return (des_x, des_y, des_z), path, std

# TODO: the thresold is set based on the statistics of multiple long vedioes.
# In real tasks, how could the simulation know the threshold when it generates the first long vedio? 
# Get multiple videoes, do analysis and finally label each frame? Not efficient.

def label(frame_index, target_pos, state_log):
    # labels are determined based on v_ave +/- 3*std

    # here is problem: if setting the threshold based on the statistics of vibration cases, the labels of frames of generated videos are probably unbalanced
    with open(os.getcwd() + '\\statistics\\v_ave.pkl', 'rb') as f:
        ave, std = pickle.load(f)

    if target_pos <= ave[frame_index] + 3*std[frame_index] and target_pos >= max(0, ave[frame_index] - 3*std[frame_index]):
        state_log.append('(' + f'{t:03}' + ', 0)') # label 0 -> vibration
    else:
        state_log.append('(' + f'{t:03}' + ', 1)') # label 1 -> falling down

while c:
    env = PoppyEnv(p.POSITION_CONTROL, show=False, use_fixed_base=False, global_scale=gs, gravity=True)
    angles = env.angle_dict(env.get_position())
    angles.update({"l_elbow_y": -90, "r_elbow_y": -90, "head_y": 35})
    env.set_position(env.angle_array(angles))
    destination, path, std = randomization(root_path, fd=fall_down)

    # pos, ori, lin, ang, joint states at timestep 0
    pos, ori, lin, ang = env.get_base()
    ini_state = {'pos': pos, 'ori': ori, 'lin': lin, 'ang': ang}
    ini_joint_states = env.get_position()

    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
        f = open(path + '\\log.txt', 'w')
        c -= 1

    # label list
    state_log = []

    # coordinate of head camera and base
    hc_z, base_z = [], []
    
    # In either fall_down or vibration cases, first 50 pictures should not be taken into consideration
    # vibration
    if not fall_down:
        for t in range(300):
            if t > 50:
                # TODO: this is one-step roatation, multi-step rotation should be taken into consideration
                target = ini_joint_states + np.random.normal(0, std, (36))
                target[-2:] = 0
                target[-11:-9] = 0
                target[4], target[9] = 0, 0
                env.step(target)
            else:
                env.step(ini_joint_states)
                
            rgba, view, proj = env.get_camera_image()
            pt.imsave(path + '\\{}.png'.format(t), rgba)
            
            base_pos, _, _, _ = env.get_base()
            base_z.append(base_pos)
            hc_pos, _, _, _, _, _, _, _ = p.getLinkState(env.robot_id, env.joint_index['head_cam'], computeLinkVelocity=1)
            hc_z.append(hc_pos)

            label(t, target_pos=hc_pos[2], state_log=state_log)
        
        with open(path + '\\base_z.pkl', 'wb') as pf:
            pickle.dump(base_z, pf)
        with open(path + '\\hc_z.pkl', 'wb') as pf:
            pickle.dump(hc_z, pf)

    # gradually fall down
    else:
        for t in range(300):

            if t > 50:
                if t == 51:
                    target = np.zeros(36) + np.random.normal(0, std, (36))
                else:
                    target = old_target + np.random.normal(0, std, (36))
                target[-2:] = 0
                target[-11:-9] = 0
                target[4], target[9] = 0, 0
                env.step(target)
                old_target = target
            else:
                env.step(ini_joint_states)

            rgba, view, proj = env.get_camera_image()
            pt.imsave(path + '\\{}.png'.format(t), rgba)

            base_pos, _, _, _ = env.get_base()
            base_z.append(base_pos)
            hc_pos, _, _, _, _, _, _, _ = p.getLinkState(env.robot_id, env.joint_index['head_cam'], computeLinkVelocity=1)
            hc_z.append(hc_pos)

            label(t, target_pos=hc_pos[2], state_log=state_log)
        
        with open(path + '\\base_z.pkl', 'wb') as pf:
            pickle.dump(base_z, pf)
        with open(path + '\\hc_z.pkl', 'wb') as pf:
            pickle.dump(hc_z, pf)

    env.close()

    f.writelines('\n'.join(state_log))
    f.close()