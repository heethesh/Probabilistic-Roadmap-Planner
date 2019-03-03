# Import system libraries
import os
import sys
import time
from sets import Set

sys.path.append(os.getcwd())
try:
    from lib import vrep
except:
    print ('"vrep.py" could not be imported. Check for the library file')

# Import application libraries
import numpy as np

### Global variables ##########################################################

STREAMING_HANDLES_JOINT_POSITION = Set([])
STREAMING_HANDLES_JOINT_FORCE = Set([])

ARM_JOINT_NAMES = ['joint_1', # revolute / arm_base_link <- shoulder_link
                   'joint_2', # revolute / shoulder_link <- elbow_link
                   'joint_3', # revolute / elbow_link <- forearm_link
                   'joint_4', # revolute / forearm_link <- wrist_link
                   'joint_5', # revolute / wrist_link <- gripper_link
                   'joint_6', # prismatic / gripper_link <- finger_r
                   'joint_7'] # prismatic / gripper_link <- finger_l

N_ARM_JOINTS = len(ARM_JOINT_NAMES)
ARM_JOINT_HANDLES = None

### Utilities #################################################################

def connect_to_vrep():
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP
    if clientID == -1:
        print('Failed connecting to V-REP remote API server.')
        exit()
    else:
        vrep.simxSynchronous(clientID, True)
    return clientID

def step_sim(clientID, nrSteps=1):
    for _ in range(nrSteps):
        vrep.simxSynchronousTrigger(clientID)
        wait_for_sim(clientID)

def wait_for_sim(clientID):
    vrep.simxGetPingTime(clientID)

def stop_sim(clientID):
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
    # TBD - change to polling V-REP server instead of sleeping
    time.sleep(2)

def start_sim(clientID):
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

def reset_sim(clientID):
    stop_sim(clientID)
    start_sim(clientID)

def get_sim_status(clientID):
    response, info = vrep.simxGetInMessageInfo(clientID,
                                               vrep.simx_headeroffset_server_state)
    assert response != -1, 'Did not receive a valid response.'
    return info

def set_sim_dt(clientID, dt_seconds):
    # This only works if the custom time has been selected in the GUI.
    vrep.simxSetFloatingParameter(clientID, vrep.sim_floatparam_simulation_time_step,
                                  dt_seconds, vrep.simx_opmode_oneshot)

### Get and Set Methods #######################################################

def get_sim_time_seconds(clientID):
    return vrep.simxGetLastCmdTime(clientID)/1.e3

def get_handle_by_name(clientID, name):
    response, handle = vrep.simxGetObjectHandle(clientID, name,
                                                vrep.simx_opmode_blocking)
    assert response == 0, 'Expected return code of 0, but instead return code was {}'.format(response)
    return handle

def get_joint_position(clientID, handle):
    is_request_initial = handle not in STREAMING_HANDLES_JOINT_POSITION
    mode = vrep.simx_opmode_streaming if is_request_initial else vrep.simx_opmode_buffer
    valid_response = False
    while not valid_response:
        response, position = vrep.simxGetJointPosition(clientID, handle, mode)
        valid_response = response == 0
    if is_request_initial:
        STREAMING_HANDLES_JOINT_POSITION.add(handle)
    return position

def get_joint_force(clientID, handle):
    is_request_initial = handle not in STREAMING_HANDLES_JOINT_FORCE
    mode = vrep.simx_opmode_streaming if is_request_initial else vrep.simx_opmode_buffer
    valid_response = False
    while not valid_response:
        response, force = vrep.simxGetJointForce(clientID, handle, mode)
        valid_response = response == 0
    if is_request_initial:
        STREAMING_HANDLES_JOINT_FORCE.add(handle)
    return force

def set_joint_target_velocity(clientID, handle, target_velocity):
    response = vrep.simxSetJointTargetVelocity(clientID, handle, target_velocity,
                                               vrep.simx_opmode_oneshot)

def set_joint_force(clientID, handle, force):
    # This method does not always set the force in the way you expect.
    # It will depend on the control and dynamics mode of the joint.
    response = vrep.simxSetJointForce(clientID, handle, force,
                                      vrep.simx_opmode_oneshot)

def get_object_position(clientID, handle, relative_to_handle=-1):
    '''Return the object position in reference to the relative handle.'''
    response, position = vrep.simxGetObjectPosition(clientID, 
                                                    handle,
                                                    relative_to_handle)
    if response != 0:
        print("Error: Cannot query position for handle {} with reference to {}".
                format(handle, relative_to_handle))
    return position

def get_object_orientation(clientID, handle, reference_handle=-1):
    '''Return the object orientation in reference to the relative handle.'''
    response, orientation = vrep.simxGetObjectOrientation(clientID, 
                                                          handle,
                                                          relative_to_handle)
    if response != 0:
        print("Error: Cannot query position for handle {} with reference to {}".
                format(handle, reference_handle))
    return orientation

def get_object_bounding_box(clientID, handle):
    '''Return the bounding box for the given object handle.'''
    position_to_param_id = {
            'min_x': 15, 'min_y': 16, 'min_z': 17,
            'max_x': 18, 'max_y': 19, 'max_z': 20
    }
    position_to_value = {}
    for pos in position_to_param_id.keys():
        param_id = position_to_param_id[pos]
        response, value = vrep.simxGetObjectFloatParameter(
                clientID, handle, param_id)
        if response != 0:
            print("Error {}: Cannot get value for param {}".format(
                response, pos))
        position_to_value[pos] = value
    min_pos = (position_to_value['min_x'],
               position_to_value['min_y'],
               position_to_value['min_z'])
    max_pos = (position_to_value['max_x'],
               position_to_value['max_y'],
               position_to_value['max_z'])
    return min_pos, max_pos


### LocoBot Methods ###########################################################

def get_arm_joint_handles(clientID):
    global ARM_JOINT_HANDLES
    if not ARM_JOINT_HANDLES:
        # Cache arm joint handles to avoid repeated handle requests
        ARM_JOINT_HANDLES = [get_handle_by_name(clientID, j) for j in ARM_JOINT_NAMES]
    return ARM_JOINT_HANDLES

def get_arm_joint_positions(clientID):
    joint_handles = get_arm_joint_handles(clientID)
    joint_positions = [get_joint_position(clientID, j) for j in joint_handles]
    return joint_positions

def get_arm_joint_forces(clientID):
    joint_handles = get_arm_joint_handles(clientID)
    joint_forces = [get_joint_force(clientID, j) for j in joint_handles]
    return joint_forces

def set_arm_joint_target_velocities(clientID, target_velocities):
    joint_handles = get_arm_joint_handles(clientID)
    assert len(target_velocities) == len(joint_handles), \
        'Expected joint target velocities to be length {}, but it was length {} instead.'.format(len(target_velocities), len(joint_handles))
    for j, v in zip(joint_handles, target_velocities):
        set_joint_target_velocity(clientID, j, v)

def set_arm_joint_forces(clientID, forces):
    joint_handles = get_arm_joint_handles(clientID)
    assert len(forces) == len(joint_handles), \
        'Expected joint forces to be length {}, but it was length {} instead.'.format(len(forces), len(joint_handles))
    for j, f in zip(joint_handles, forces):
        set_joint_force(clientID, j, f)
