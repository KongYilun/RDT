import os
import fnmatch
import json
import sys
import h5py
import yaml
import cv2
import numpy as np
from scipy.interpolate import interp1d
# sys.path.append(str("/data/ylkong/code/RoboticsDiffusionTransformer-main/configs"))
from configs.state_vec import STATE_VEC_IDX_MAPPING

def interpolate_action_sequence(action_sequence, target_size):
    """
    Extend the action sequece to `target_size` by linear interpolation.
    
    Args:
        action_sequence (np.ndarray): original action sequence, shape (N, D).
        target_size (int): target sequence length.
    
    Returns:
        extended_sequence (np.ndarray): extended action sequence, shape (target_size, D).
    """
    N, D = action_sequence.shape
    indices_old = np.arange(N)
    indices_new = np.linspace(0, N - 1, target_size)

    interp_func = interp1d(indices_old, action_sequence, 
                           kind='linear', axis=0, assume_sorted=True)
    action_sequence_new = interp_func(indices_new)

    return action_sequence_new

class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIR = "datasets/DexMG_BoxCleanup"
        self.DATASET_NAME = "DexMG_BoxCleanup"
        
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
    
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            right_arm_state = f['data']['obs']['robot0_joint_pos']
            left_arm_state = f['data']['obs']['robot1_joint_pos']
            right_gripper_state = f['data']['obs']['robot0_gripper_qpos']
            left_gripper_state = f['data']['obs']['robot1_gripper_qpos']
            qpos= np.concatenate([right_arm_state, right_gripper_state, left_arm_state, left_gripper_state], axis=1)
            num_steps = qpos.shape[0]
            # state_dim=qpos.shape[1]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # We randomly sample a timestep
            step_id = np.random.randint(first_idx-1, num_steps)
            
            # Load the instruction
            dir_path = os.path.dirname(file_path)
            with open(os.path.join(dir_path, 'instruction.json'), 'r') as f_instr:
                instruction_dict = json.load(f_instr)
            # We have 1/3 prob to use original instruction,
            # 1/3 to use simplified instruction,
            # and 1/3 to use expanded instruction.
            instruction_type = np.random.choice([
                'instruction'])#, 'simplified_instruction', 'expanded_instruction'
            instruction = instruction_dict[instruction_type]
            if isinstance(instruction, list):
                instruction = np.random.choice(instruction)
            # You can also use precomputed language embeddings (recommended)
            # instruction = "path/to/lang_embed.pt"
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            # Rescale gripper to [0, 1]
            # qpos = qpos / np.array(
            #    [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]] 
            # )
            runtime_chunksize = self.CHUNK_SIZE // 4
            target_qpos = f['data']['actions'][step_id:step_id+runtime_chunksize] 
            # / np.array(
            #    [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]] 
            # )
            
            # Parse the state and action
            state = qpos[step_id:step_id+1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < runtime_chunksize:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (runtime_chunksize-actions.shape[0], 1))
                ], axis=0)
            actions = interpolate_action_sequence(actions, self.CHUNK_SIZE)


            # Fill the state/action into the unified vector
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
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            def fill_in_action(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING[f"right_dexhand_{i}"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING[f"left_eef_angle_{i}"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING[f"left_dexhand_{i}"] for i in range(6)
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = fill_in_action(actions)
            
            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                    img = f['data']['obs'][key][i]
                    imgs.append(img)#cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            # `cam_high` is the external camera image
            cam_high = parse_img('agentview_image')
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = parse_img('robot1_eye_in_hand_image')
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img('robot0_eye_in_hand_image')
            cam_right_wrist_mask = cam_high_mask.copy()
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            right_arm_state = f['data']['obs']['robot0_joint_pos']
            left_arm_state = f['data']['obs']['robot1_joint_pos']
            right_gripper_state = f['data']['obs']['robot0_gripper_qpos']
            left_gripper_state = f['data']['obs']['robot1_gripper_qpos']
            qpos= np.concatenate([right_arm_state, right_gripper_state, left_arm_state, left_gripper_state], axis=1)
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # Rescale gripper to [0, 1]
            # qpos = qpos / np.array(
            #    [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]] 
            # )
            target_qpos = f['data']['actions'][:] 
            # / np.array(
            #    [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]] 
            # )
            
            # Parse the state and action
            state = qpos[first_idx-1:]
            action = target_qpos[first_idx-1:]
            
            # Fill the state/action into the unified vector
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
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            def fill_in_action(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING[f"right_dexhand_{i}"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING[f"left_eef_angle_{i}"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING[f"left_dexhand_{i}"] for i in range(6)
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            state = fill_in_state(state)
            action = fill_in_action(action)
            
            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        dic=ds.get_item(i)
        # print(dic)
        # exit()
