from typing import Callable, List, Type
import sys
sys.path.append('/')
# import gymnasium as gym
import numpy as np
# from mani_skill.envs.sapien_env import BaseEnv
# from mani_skill.utils import common, gym_utils
import argparse
import yaml
from configs.state_vec import STATE_VEC_IDX_MAPPING
# sys.path.append(str("/data/ylkong/code/RoboticsDiffusionTransformer-main/scripts"))
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
import torch
from collections import deque
from PIL import Image
import cv2
import imageio
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "--env-id", type=str, default="DrawerCleanUp", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=4, help="Number of trajectories to test.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to the pretrained model")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    return parser.parse_args()

import random
import os
import robosuite
from robosuite import load_composite_controller_config
import dexmimicgen
# set cuda 
args = parse_args()
# set random seeds
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

task2lang = {
    "TwoArmDrawerCleanup": "Place the cup inside the drawer.",
    "TwoArmBoxCleanup": "Place the lid on the box."
}

ACTION_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_dexhand_{i}"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"left_eef_angle_{i}"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"left_dexhand_{i}"] for i in range(6)
]

def fill_in_state(values):
    # Target indices corresponding to your state space
    # In this example: 6 joints + 1 gripper for each arm
    UNI_STATE_INDICES = [
        STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
    ] + [
        STATE_VEC_IDX_MAPPING[f"right_dexhand_{i}"] for i in range(12)
    ] + [
        STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(7)
    ] + [
        STATE_VEC_IDX_MAPPING[f"left_dexhand_{i}"] for i in range(12)
    ]
    uni_vec = np.zeros(values.shape[:-1] + (128,))
    uni_vec[..., UNI_STATE_INDICES] = values
    return uni_vec

ENV_ROBOTS = {
    "TwoArmThreading": ["Panda", "Panda"],
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmCanSortRandom": ["GR1ArmsOnly"],
}

env_kwargs = {
    "env_name": args.env,
    "robots": ENV_ROBOTS[args.env],
    "controller_configs": load_composite_controller_config(
        robot=ENV_ROBOTS[args.env][0]
    ),
    "has_renderer": False,
    "has_offscreen_renderer": True,
    "ignore_done": True,
    "use_camera_obs": True,
    "use_object_obs": True,
    "camera_names": ["robot0_eye_in_hand", "robot1_eye_in_hand","agentview"],
    "control_freq": 20,
}
env_id=args.env
# initialize the task
env = robosuite.make(
    **env_kwargs,
)

config_path = 'configs/base.yaml'
with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
pretrained_path = "checkpoints/rdt-finetune-1b-dexmimicgen-boxcleanup/checkpoint-35000/pytorch_model/mp_rank_00_model_states.pt"
policy = create_model(
    args=config, 
    dtype=torch.bfloat16,
    pretrained=pretrained_path,
    pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
    pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
)

if os.path.exists(f'text_embed_{env_id}.pt'):
    text_embed = torch.load(f'text_embed_{env_id}.pt')
    print('Already exist instruction embedding!')
else:
    text_embed = policy.encode_instruction(task2lang[env_id])
    torch.save(text_embed, f'text_embed_{env_id}.pt')

MAX_EPISODE_STEPS = 400 
total_episodes = args.num_traj  
success_count = 0  

def generate_proprio(obs):
    right_arm_state = obs['robot0_joint_pos']
    left_arm_state = obs['robot1_joint_pos']
    right_gripper_state = obs['robot0_gripper_qpos']
    left_gripper_state = obs['robot1_gripper_qpos']
    proprio = np.concatenate([right_arm_state, right_gripper_state, left_arm_state, left_gripper_state])
    proprio=proprio.reshape((1,*proprio.shape))
    proprio=fill_in_state(proprio)
    proprio=torch.tensor(proprio)
    return proprio

video_flag=False
video_writer = imageio.get_writer('rdt_dexmimicgen_demo.mp4', fps=20)
video_count=0
video_flag=True
base_seed = 20241201
import tqdm
for episode in tqdm.trange(total_episodes):
    # if episode==1:
        
    obs_window = deque(maxlen=2)
    obs = env.reset()
    policy.reset()
    # fetch images in sequence [front, right, left]
    img_agent=obs['agentview_image']
    img_right=obs['robot0_eye_in_hand_image']
    img_left=obs['robot1_eye_in_hand_image']
    obs_window.append([None,None,None])
    obs_window.append(np.array([img_agent,img_right,img_left]))
    proprio=generate_proprio(obs)

    global_steps = 0
    video_frames = []

    success_time = 0
    done = False

    while global_steps < MAX_EPISODE_STEPS and not done:
        image_arrs = []
        success=False
        for window_img in obs_window:
            image_arrs.append(window_img[0])
            image_arrs.append(window_img[1])
            image_arrs.append(window_img[2])
        images = [Image.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        actions = policy.step(proprio, images, text_embed).float().squeeze(0).cpu().numpy()
        # Take 8 steps since RDT is trained to predict interpolated 64 steps(actual 14 steps)
        actions = actions[::4, :]
        for idx in range(actions.shape[0]):
            action = actions[idx]
            # print(action.shape)
            # true_action=action[33:39]+action[103:109]+action[83:89]+action[113:119]
            # action_indices = MANISKILL_INDICES
            true_action = action[ACTION_INDICES]
            # print(true_action.shape)
            obs, reward, terminated, info = env.step(true_action)
            # img = env.render().squeeze(0).detach().cpu().numpy()
            img_agent=obs['agentview_image']
            img_right=obs['robot0_eye_in_hand_image']
            img_left=obs['robot1_eye_in_hand_image']
            obs_window.append(np.array([img_agent,img_right,img_left]))
            proprio = generate_proprio(obs)
            # video_frames.append(img)
            global_steps += 1
            if video_flag==True:
                if video_count % 5 == 0:
                    video_img = []
                    for cam_name in ['sideview','birdview']:
                        im = env.sim.render(height=512, width=512, camera_name=cam_name)[
                            ::-1
                        ]
                        video_img.append(im)
                    video_img = np.concatenate(
                        video_img, axis=1
                    )  # concatenate horizontally
                    video_writer.append_data(video_img)

                video_count += 1

            if env._check_success():
                success = True
                success_count += 1
                done = True
                break 
            if terminated:
                if env._check_success():
                    success = True
                    success_count += 1
                    done = True
                    break 
                # assert "success" in info, sorted(info.keys())
                # if info['success']:
                #     success_count += 1
                #     done = True
                #     break 
    print(f"Trial {episode+1} finished, success: {success}, steps: {global_steps}")
video_writer.close()
success_rate = success_count / total_episodes * 100
print(f"Success rate: {success_rate}%")
