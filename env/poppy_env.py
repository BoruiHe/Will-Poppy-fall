import pybullet as pb
from pybullet_data import getDataPath
import time
import numpy as np
import os
import random

class PoppyEnv(object):

    def load_urdf(self, use_fixed_base=False):
        fpath = os.path.dirname(os.path.abspath(__file__))
        pb.setAdditionalSearchPath(fpath)
        robot_id = pb.loadURDF(
            'poppy_ergo.pybullet.urdf',
            basePosition = (0, 0, .43),
            baseOrientation = pb.getQuaternionFromEuler((0,0,0)),
            useFixedBase=use_fixed_base,)
        return robot_id 

    def __init__(self,
        control_mode=pb.POSITION_CONTROL,
        timestep=1/240,
        control_period=1,
        show=True,
        step_hook=None,
        use_fixed_base=False,
        global_scale=10,
        gravity = True
    ):

        # step_hook(env, action) is called in each env.step(action)
        if step_hook is None: step_hook = lambda env, action: None

        self.control_mode = control_mode
        self.timestep = timestep
        self.control_period = control_period
        self.show = show
        self.step_hook = step_hook

        self.client_id = pb.connect(pb.GUI if show else pb.DIRECT)
        if show: pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
        pb.setTimeStep(timestep)
        if gravity:
            pb.setGravity(0, 0, -9.8)
        else:
            pb.setGravity(0, 0, 0)
        pb.setAdditionalSearchPath(getDataPath())
        planeId = pb.loadURDF("plane.urdf")

        # load 5 cubes as background
        # Box size is [1,1,1]. z_pos(height) = globalScaling/2
        gs = global_scale
        # back, front, right, left, top(sky)
        pos_ls = [[0,gs,gs/2], [0,-gs,gs/2], [gs,0,gs/2], [-gs,0,gs/2], [0,0,3*gs/2]]
        # textures
        texture_path = os.getcwd() + '\\env\\textures'
        texture_files = os.listdir(texture_path)
        texture_files.remove('sky.png')
        texture_files = [f for f in texture_files if os.path.isfile(texture_path + '\\' + f)]
        texture_files = random.sample(texture_files, 4) + ['sky.png']
        for i in range(5):
            cubeid = pb.loadURDF('cube.urdf', useFixedBase=1, globalScaling= gs)
            pb.resetBasePositionAndOrientation(cubeid, pos_ls[i], pb.getQuaternionFromEuler([np.pi / 2, 0, 0]))
            pb.changeVisualShape(cubeid, -1, textureUniqueId= pb.loadTexture(texture_path + '\\' + texture_files[i]))
            # # rotate front cube for a reasonable environment
            # if i==1:
            #     quat = pb.getQuaternionFromEuler([np.pi / 2, 0, 0])
            #     pb.resetBasePositionAndOrientation(cubeid, pb.getBasePositionAndOrientation(cubeid)[0], quat)
            # # rotate right cube for a reasonable environment
            # elif i==2:
            #     quat = pb.getQuaternionFromEuler([np.pi / 2, 0, 0])
            #     pb.resetBasePositionAndOrientation(cubeid, pb.getBasePositionAndOrientation(cubeid)[0], quat)
            # # rotate left cube for a reasonable environment
            # elif i==3:
            #     quat = pb.getQuaternionFromEuler([np.pi / 2, 0, 0])
            #     pb.resetBasePositionAndOrientation(cubeid, pb.getBasePositionAndOrientation(cubeid)[0], quat)

        # use overridden loading logic
        self.robot_id = self.load_urdf(use_fixed_base)

        self.boundary = {'x': (-gs/2, gs/2), 'y':(-gs/2, gs/2), 'z':(0, pb.getBasePositionAndOrientation(self.robot_id)[0][2])}

        self.num_joints = pb.getNumJoints(self.robot_id)
        self.joint_name, self.joint_index, self.joint_fixed = {}, {}, {}
        for i in range(self.num_joints):
            info = pb.getJointInfo(self.robot_id, i)
            name = info[1].decode('UTF-8')
            self.joint_name[i] = name
            self.joint_index[name] = i
            self.joint_fixed[i] = (info[2] == pb.JOINT_FIXED)
        
        self.initial_state_id = pb.saveState(self.client_id)
    
    def reset(self):
        pb.restoreState(stateId = self.initial_state_id)
    
    def close(self):
        pb.disconnect()
        
    def step(self, action=None, sleep=None):
        
        self.step_hook(self, action)
    
        if action is not None:
            duration = self.control_period * self.timestep
            distance = np.fabs(action - self.get_position())
            pb.setJointMotorControlArray(
                self.robot_id,
                jointIndices = range(len(self.joint_index)),
                controlMode = self.control_mode,
                targetPositions = action,
                targetVelocities = [0]*len(action),
                positionGains = [.25]*len(action), # important for constant position accuracy
                # maxVelocities = distance / duration,
            )

        if sleep is None: sleep = self.show # True
        if sleep:
            for _ in range(self.control_period):
                start = time.perf_counter()
                pb.stepSimulation()
                duration = time.perf_counter() - start
                remainder = self.timestep - duration
                if remainder > 0: time.sleep(remainder)
        else:
            for _ in range(self.control_period):
                pb.stepSimulation()

    # base position/orientation and velocity/angular
    def get_base(self):
        pos, orn = pb.getBasePositionAndOrientation(self.robot_id)
        vel, ang = pb.getBaseVelocity(self.robot_id)
        return pos, orn, vel, ang

    def set_base(self, pos=None, orn=None, vel=None, ang=None):
        _pos, _orn, _vel, _ang = self.get_base()
        if pos == None: pos = _pos
        if orn == None: orn = _orn
        if vel == None: vel = _vel
        if ang == None: ang = _ang
        pb.resetBasePositionAndOrientation(self.robot_id, pos, orn)
        pb.resetBaseVelocity(self.robot_id, vel, ang)
    
    # get/set joint angles as np.array
    def get_position(self):
        states = pb.getJointStates(self.robot_id, range(len(self.joint_index)))
        # joint position, joint velocity, joint reaction forces(6), applied joint motor torque.
        # each state is a tuple of length 4
        return np.array([state[0] for state in states])

    def set_position(self, position):
        for p, angle in enumerate(position):
            pb.resetJointState(self.robot_id, p, angle)

    # convert a pypot style dictionary {... name:angle ...} to joint angle array
    # if convert == True, convert from degrees to radians
    def angle_array(self, angle_dict, convert=True):
        angle_array = np.zeros(self.num_joints)
        for name, angle in angle_dict.items():
            angle_array[self.joint_index[name]] = angle
        if convert: angle_array *= np.pi / 180
        return angle_array

    # convert back from dict to array
    def angle_dict(self, angle_array, convert=True):
        return {
            name: angle_array[j] * 180/np.pi
            for j, name in enumerate(self.joint_index)}

    # pypot-style command, goes to target joint position with given speed
    # target is a joint angle array
    # speed is desired joint speed
    # if hang==True, wait for user enter at each timestep of motion
    def goto_position(self, target, speed=1., hang=False):

        current = self.get_position()
        distance = np.sum((target - current)**2)**.5
        duration = distance / speed
        # TODO: ask the question: what's the definition of self.control_period
        num_steps = int(duration / (self.timestep * self.control_period) + 1)
        weights = np.linspace(0, 1, num_steps).reshape(-1,1)
        trajectory = weights * target + (1 - weights) * current

        positions = np.empty((num_steps, self.num_joints))
        for a, action in enumerate(trajectory):
            self.step(action)
            positions[a] = self.get_position()
            if hang: input('..')

        return positions

    # Get image from head camera
    def get_camera_image(self):

        # Get current pose of head camera
        # link index should be same as parent joint index?
        state = pb.getLinkState(self.robot_id, self.joint_index["head_cam"])
        pos, quat = state[:2]
        M = np.array(pb.getMatrixFromQuaternion(quat)).reshape((3,3)) # local z-axis is third column

        # Calculate camera target and up vector
        camera_position = tuple(p + d for (p,d) in zip(pos, .1*M[:,2]))
        target_position = tuple(p + d for (p,d) in zip(pos, .4*M[:,2]))
        up_vector = tuple(M[:,1])
        
        # Capture image
        width, height = 128, 128
        # width, height = 8, 8 # doesn't actually make much speed difference
        view = pb.computeViewMatrix(
            cameraEyePosition = camera_position,
            cameraTargetPosition = target_position, # focal point
            cameraUpVector = up_vector,
        )
        proj = pb.computeProjectionMatrixFOV(
            # fov = 135,
            fov = 90,
            aspect = height/width,
            nearVal = 0.01,
            # farVal should be large enough to eliminate the unexpected white area(because ur camera is not expected to see that far)
            farVal = 20.0,
        )
        # rgba shape is (height, width, 4)
        _, _, rgba, _, _ = pb.getCameraImage(
            width, height, view, proj,
            flags = pb.ER_NO_SEGMENTATION_MASK) # not much speed difference
        # rgba = np.empty((height, width, 4)) # much fafr than pb.getCameraImage
        return rgba, view, proj

if __name__ == '__main__':
    target = np.random.normal(0, 0.1, (36))
    target[-2:] = 0
    target[-11:-9] = 0
    print('target position:')
    print(target)

    env = PoppyEnv(pb.POSITION_CONTROL, show=False, use_fixed_base=False, global_scale=10, gravity=True)
    init_pos = env.get_position()

    print('real position:')
    print(env.goto_position(target, speed=2.)[-1])

    env.set_position(init_pos)

    print('real position:')
    print(env.goto_position(target, speed=2.)[-1])
    pass