import os
from concurrent.futures import ThreadPoolExecutor

import cv2
from typing import Iterator, Tuple, Any
from scipy.spatial.transform import Rotation
import pickle

import PIL

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import re

from tqdm import tqdm


class Bridge(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'depth_0': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='image of depth camera or padding 1s, if has_depth_0 is false.',
                        ),
                        'image_0': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='image of main camera or padding 1s, if has_image_0 is false.',
                        ),
                        'image_1': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='image of second camera or padding 1s, if has_image_1 is false.',
                        ),
                        'image_2': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='image of third camera or padding 1s, if has_image_2 is false.',
                        ),
                        'image_3': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='image of forth camera or padding 1s, if has_image_3 is false.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Robot end-effector state. Consists of [3x pos, 3x orientation (euler: roll, pitch, yaw), 1x gripper width]',
                        ),
                        'full_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Robot end-effector state. Consists of [3x pos, 3x orientation (euler: roll, pitch, yaw), 1x gripper width]',
                        ),
                        'desired_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float64,
                            doc='Robot end-effector state. Consists of [3x pos, 3x orientation (euler: roll, pitch, yaw), 1x gripper width]',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Delta robot action, consists of [3x delta_end_effector_pos, '
                            '3x delta_end_effector_ori (euler: roll, pitch, yaw), 1x des_gripper_width].',
                    ),
                    'new_robot_transform': tfds.features.Tensor(
                        shape=(4, 4),
                        dtype=np.float64,
                        doc='Field new_robot_transform from bridge dataset, probably some form of quat (x,y,z,w) in second dim'
                            'no information was given, cant check further'
                    ),
                    'delta_robot_transform': tfds.features.Tensor(
                        shape=(4, 4),
                        dtype=np.float64,
                        doc='Field delta_robot_transform from bridge dataset, probably some form of quat (x,y,z,w) in second dim'
                            'no information was given, cant check further'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction_0': tfds.features.Text(
                        doc='Language Instruction. utf-8 encoded data from files'
                            'empty byte stream if has_language is false'
                    ),
                    'language_instruction_1': tfds.features.Text(
                        doc='Language Instruction. utf-8 encoded data from files'
                            'empty byte stream if has_language is false'
                    ),
                    'language_instruction_2': tfds.features.Text(
                        doc='Language Instruction. utf-8 encoded data from files'
                            'empty byte stream if has_language is false'
                    ),
                    'language_instruction_3': tfds.features.Text(
                        doc='Language Instruction. utf-8 encoded data from files'
                            'empty byte stream if has_language is false'
                    ),
                    'groundtruth_0': tfds.features.Text(
                        doc='Groundtruth Language Instruction. utf-8 encoded data from files'

                    ),
                    'groundtruth_1': tfds.features.Text(
                        doc='Groundtruth Language Instruction. utf-8 encoded data from files'

                    ),
                    'groundtruth_2': tfds.features.Text(
                        doc='Groundtruth Language Instruction. utf-8 encoded data from files'

                    ),

                    'language_embedding': tfds.features.Tensor(
                        shape=(1, 512),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    )
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.',
                    ),
                    'traj_length': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Number of samples in trajectorie'
                    ),
                    'has_depth_0': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had a depth img, false if none (padding 1s in depth_0)'
                    ),
                    'has_image_0': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had an img_0, false if none (padding 1s in image_0)'
                    ),
                    'has_image_1': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had an img_1, false if none (padding 1s in image_1)'
                    ),
                    'has_image_2': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had an img_2, false if none (padding 1s in image_2)'
                    ),
                    'has_image_3': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had an img_3, false if none (padding 1s in image_3)'
                    ),
                    'has_language': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='bool, true if dataset had language annotations, false if none (empty string in language_instruction as padding)'
                    )
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        data_path = "/home/marcelr/BridgeData/raw"
        return {
            'train': self._generate_examples(path=data_path),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # create list of all examples
        raw_dirs = []
        get_trajectorie_paths_recursive(path, raw_dirs)
        raw_dirs.reverse()

        # for smallish datasets, use single-thread parsing
        counter = 0
        end = len(raw_dirs)
        for raw_dir in raw_dirs:
            print(counter, " of ", end, " done")
            counter += 1
            for traj_group in os.listdir(raw_dir):
                traj_group_full_path = os.path.join(raw_dir, traj_group)
                if os.path.isdir(traj_group_full_path):
                    for traj_dir in os.listdir(traj_group_full_path):
                        traj_dir_full_path = os.path.join(traj_group_full_path, traj_dir)
                        if os.path.isdir(traj_dir_full_path):
                            example = _parse_example(traj_dir_full_path, self._embed)
                            if example is not None:
                                yield example
                        else:
                            print("non dir instead of traj found!")
                            yield traj_dir_full_path, {}
                else:
                    print("non dir instead of traj_group found!")
                    yield traj_group_full_path, {}

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

def _parse_example(episode_path, embed=None):
    data = {}

    lupus_path = os.path.join(episode_path, "annotations", "lang_lupus.txt")
    if not os.path.exists(lupus_path):
        return None

    # check if "lang_lupus" exists in traj
    lupus_path = os.path.join(episode_path, "annotations", "lang_lupus.txt")
    lang_txt_path = os.path.join(episode_path, "lang.txt")
    # if not os.path.isfile(lupus_path) and not os.path.isfile(lang_txt_path):
    #     return None

    for data_field in os.listdir(episode_path):
        if "agent_data" in data_field:
            continue
        data_field_full_path = os.path.join(episode_path, data_field)
        if os.path.isdir(data_field_full_path):
            if data_field == "annotations": # extract "lang_lupus"
                for lupus_annotation in os.listdir(data_field_full_path):
                    if lupus_annotation == "lang_lupus.txt":

                        with open(os.path.join(data_field_full_path, lupus_annotation), 'rb') as f:
                            lang = f.read()
                            lang = lang.decode("utf-8")
                            lang = lang.split("\n")
                            lang = [line.strip() for line in lang if "confidence" not in line]
                            lang_lupus = {"lang_lupus":lang}
                            data.update(lang_lupus)
            else:
                cam1_image_vector = create_img_vector(data_field_full_path)
                data.update({data_field: cam1_image_vector})
        elif data_field == "lang.txt":
            with open(data_field_full_path, 'rb') as f:

                lang = f.read()
                lang = lang.decode("utf-8")
                lang = lang.split("\n")
                lang = [line.strip() for line in lang if "confidence" not in line]
                lang_txt = {"lang":lang}
                data.update(lang_txt)
        else:

            data.update({data_field[:data_field.find(".")]: np.load(data_field_full_path, allow_pickle=True)})


    # agent_data : dict_keys(['traj_ok', 'camera_info', 'term_t', 'stats'])
    # policy_out : dict_keys(['actions', 'new_robot_transform', 'delta_robot_transform', 'policy_type'])
    # obs_dict   : dict_keys(['joint_effort', 'qpos', 'qvel', 'full_state', 'state', 'desired_state', 'time_stamp', 'eef_transform', 'high_bound', 'low_bound', 'env_done', 't_get_obs', 'task_stage'])
    # lang.txt   : b'take the silver pot and place it on the top left burner\nconfidence: 1\n'
    # for key, value in data.items():
    #     print(key)
    #     if isinstance(value, list):
    #         print(value[0].keys())
    #     elif isinstance(value, dict):
    #         print(value.keys())
    #     else:
    #         print(value)


    for i in range(len(data["policy_out"])-1):
        diff = data["obs_dict"]["state"][i+1] - data["obs_dict"]["state"][i]
        act = data["policy_out"][i]["actions"]

        print("diff: ", diff)
        print("act: ", act)


    trajectory_length = data["agent_data"]["term_t"] if "agent_data" in data else len(data["policy_out"])
    has_depth_0 = "depth_images0" in data
    has_image_0 = "images0" in data
    has_image_1 = "images1" in data
    has_image_2 = "images2" in data
    has_image_3 = "images3" in data
    has_language = "lang_lupus" in data
    has_groundtruth = "lang" in data

    if has_language:
        #lang_str = data["lang_lupus"].decode("utf-8")
        lupus_array = data["lang_lupus"]

        if len(lupus_array) < 3:
            to_fill = 3 - len(lupus_array)
            for i in range(to_fill):
                lupus_array.append(lupus_array[i])

        #sort array by length
        lupus_array = sorted(lupus_array, key=len)

    if has_groundtruth:
        #lang_str = data["lang"].decode("utf-8")
        lang_array = data["lang"]
        if len(lang_array) < 3:
            to_fill = 3 - len(lang_array)
            for i in range(to_fill):
                lang_array.append(lang_array[i])

    pad_img_tensor = tf.ones([480, 640, 3], dtype=np.uint8).numpy()
    # pad_depth_tensor = tf.ones([480, 640, 1], dtype=data["images0"][0].dtype).numpy()

    episode = []
    for i in range(trajectory_length):
        # compute Kona language embedding
        if embed is None:
            language_embedding = [np.zeros(512)]
        elif has_language:
            lang_str = lupus_array[0].decode("utf-8")
            language_embedding = embed([lang_str]).numpy()
        else:
            language_embedding = embed([""]).numpy()

        episode.append({
            'observation': {
                "depth_0": data['depth_images0'][i] if has_depth_0 else pad_img_tensor,
                "image_0": data['images0'][i] if has_image_0 else pad_img_tensor,
                "image_1": data['images1'][i] if has_image_1 else pad_img_tensor,
                "image_2": data['images2'][i] if has_image_2 else pad_img_tensor,
                "image_3": data['images3'][i] if has_image_3 else pad_img_tensor,
                "state": data["obs_dict"]["state"][i],
                "full_state": data["obs_dict"]["full_state"][i],
                "desired_state": data["obs_dict"]["desired_state"][i],
            },
            'action': data["policy_out"][i]["actions"],
            'new_robot_transform': data["policy_out"][i]["new_robot_transform"], # prbl quat, x,y,z,w
            'delta_robot_transform': data["policy_out"][i]["delta_robot_transform"], # prbl quat, x,y,z,w
            'discount': 1.0,
            'reward': float(i == (trajectory_length - 1)),
            'is_first': i == 0,
            'is_last': i == (trajectory_length - 1),
            'is_terminal': i == (trajectory_length - 1),
            'language_instruction_0': lupus_array[0] if has_language else b'',
            'language_instruction_1': lupus_array[1] if has_language else b'',
            'language_instruction_2': lupus_array[2] if has_language else b'',
            'groundtruth_0': lang_array[0] if has_groundtruth else b'',
            'groundtruth_1': lang_array[1] if has_groundtruth else b'',
            'groundtruth_2': lang_array[2] if has_groundtruth else b'',
            'language_embedding': language_embedding,
        })

    # create output data sample
    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': episode_path,
            'traj_length': trajectory_length,
            'has_depth_0': has_depth_0,
            'has_image_0': has_image_0,
            'has_image_1': has_image_1,
            'has_image_2': has_image_2,
            'has_image_3': has_image_3,
            'has_language': has_language,
        }
    }

    # if you want to skip an example for whatever reason, simply return None
    return episode_path, sample

def preprocess_string(unfiltered_str: str) -> list:
    lang_str = unfiltered_str[:unfiltered_str.find("\nconfidence:")]
    start = 0
    end = -1
    lang_array = []
    while True:
        end = lang_str[start:].find("\n")
        if end == -1:
            if len(lang_str[start:]) != 0:
                lang_array.append(lang_str[start:].encode("utf-8"))
            break
        lang_array.append(lang_str[start:start + end].encode("utf-8"))
        start += end + 1
      
    return lang_array





def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def load_image(image_path):
    image = PIL.Image.open(image_path)
    image = np.array(image)
    return image

def create_img_vector(img_folder_path):
    cam_list = []
    cam_path_list = []
    dir_list_sorted = sorted_alphanumeric(os.listdir(img_folder_path))

    #only keep files with img extension
    dir_list_sorted_images = [os.path.join(img_folder_path,path) for path in dir_list_sorted if path.endswith(('.png', '.jpg', '.jpeg'))]

    cam_list = [load_image(img_path) for img_path in dir_list_sorted_images]
    return cam_list

    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list((executor.map(load_image, dir_list_sorted_images)))

    cam_list = images
    return cam_list



    for img_name in dir_list_sorted:
        ext = img_name[img_name.find("."):]
        if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
            cam_path_list.append(img_name)
            img_path = os.path.join(img_folder_path, img_name)
            img_array = cv2.imread(img_path)
            cam_list.append(img_array)
    return cam_list

def get_trajectorie_paths_recursive(directory, sub_dir_list):
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            sub_dir_list.append(full_path) if entry == "raw" else get_trajectorie_paths_recursive(full_path, sub_dir_list)
    # return subdirectories

if __name__ == "__main__":
    data_path = "/home/DATA_SHARE/bridge_data"
    #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    embed = None
    raw_dirs = []
    counter = 0
    get_trajectorie_paths_recursive(data_path, raw_dirs)
    raw_dirs.reverse() # '/home/marcelr/BridgeData/raw/datacol1_toykitchen1/many_skills/09/2023-03-15_15-11-20/raw' '/home/marcelr/BridgeData/raw/datacol1_toykitchen1/many_skills/09/2023-03-15_15-11-20/raw'

    pb = tqdm(total=50000)

    for raw_dir in raw_dirs:
        for traj_group in os.listdir(raw_dir):
            traj_group_full_path = os.path.join(raw_dir, traj_group)
            if os.path.isdir(traj_group_full_path):
                for traj_dir in os.listdir(traj_group_full_path):
                    traj_dir_full_path = os.path.join(traj_group_full_path, traj_dir)
                    if os.path.isdir(traj_dir_full_path):
                        counter += 1
                        _parse_example(traj_dir_full_path, embed)
                        pb.update(1)
                    else:
                        print("non dir instead of traj found!")
            else:
                print("non dir instead of traj_group found!")
    # create list of all examples
    # episode_paths = glob.glob(data_path)
    # for episode in episode_paths:
    #     _, sample = _parse_example(episode, embed)