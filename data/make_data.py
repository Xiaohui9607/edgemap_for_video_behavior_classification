import os
import glob
import random
import numpy as np
import PIL.Image
import argparse
from PDBF import graypdbfs

# FIXED = True
# if FIXED:
#     train_test_split = '/home/golf/code/models/Experiement_object_based_all_behaviors/None/train_test_split'
#     train, test = open(train_test_split).readlines()[:2]
#     train = train[9:-3].split('\', \'')
#     test = test[8:-3].split('\', \'')

DATA_DIR = '../data/CY101'
OUT_DIR = '../data/CY101EDNPY'
# added for VIS splits
VIS_DIR = '../data/EDVIS/'


STRATEGY = 'object' # object | category | trial

CATEGORIES = ['basket', 'weight', 'smallstuffedanimal', 'bigstuffedanimal', 'metal', 'timber', 'pasta', 'tin', 'pvc',
              'cup', 'can', 'bottle', 'cannedfood', 'medicine', 'tupperware', 'cone', 'noodle', 'eggcoloringcup', 'egg',
              'ball']

OBJECTS = [
    'ball_base', 'can_coke', 'egg_rough_styrofoam', 'noodle_3', 'timber_square', 'ball_basket', 'can_red_bull_large',
    'egg_smooth_styrofoam', 'noodle_4', 'timber_squiggle', 'ball_blue', 'can_red_bull_small', 'egg_wood', 'noodle_5',
    'tin_pokemon',
    'ball_transparent', 'can_starbucks', 'eggcoloringcup_blue', 'pasta_cremette', 'tin_poker', 'ball_yellow_purple',
    'cannedfood_chili',
    'eggcoloringcup_green', 'pasta_macaroni', 'tin_snack_depot', 'basket_cylinder', 'cannedfood_cowboy_cookout',
    'eggcoloringcup_orange',
    'pasta_penne', 'tin_snowman', 'basket_funnel', 'cannedfood_soup', 'eggcoloringcup_pink', 'pasta_pipette', 'tin_tea',
    'basket_green',
    'cannedfood_tomato_paste', 'eggcoloringcup_yellow', 'pasta_rotini', 'tupperware_coffee_beans', 'basket_handle',
    'cannedfood_tomatoes',
    'medicine_ampicillin', 'pvc_1', 'tupperware_ground_coffee', 'basket_semicircle', 'cone_1', 'medicine_aspirin',
    'pvc_2', 'tupperware_marbles',
    'bigstuffedanimal_bear', 'cone_2', 'medicine_bilberry_extract', 'pvc_3', 'tupperware_pasta',
    'bigstuffedanimal_bunny', 'cone_3',
    'medicine_calcium', 'pvc_4', 'tupperware_rice', 'bigstuffedanimal_frog', 'cone_4', 'medicine_flaxseed_oil', 'pvc_5',
    'weight_1',
    'bigstuffedanimal_pink_dog', 'cone_5', 'metal_flower_cylinder', 'smallstuffedanimal_bunny', 'weight_2',
    'bigstuffedanimal_tan_dog',
    'cup_blue', 'metal_food_can', 'smallstuffedanimal_chick', 'weight_3', 'bottle_fuse', 'cup_isu',
    'metal_mix_covered_cup',
    'smallstuffedanimal_headband_bear', 'weight_4', 'bottle_google', 'cup_metal', 'metal_tea_jar',
    'smallstuffedanimal_moose',
    'weight_5', 'bottle_green', 'cup_paper_green', 'metal_thermos', 'smallstuffedanimal_otter', 'bottle_red',
    'cup_yellow', 'timber_pentagon', 'bottle_sobe', 'egg_cardboard', 'noodle_1', 'timber_rectangle', 'can_arizona',
    'egg_plastic_wrap', 'noodle_2', 'timber_semicircle', 'no_object'
]

# Objects
SORTED_OBJECTS = sorted(OBJECTS)

BEHAVIORS = ['crush', 'grasp', 'lift_slow', 'shake', 'poke', 'push', 'tap', 'low_drop', 'hold']

TRAILS = ['exec_1', 'exec_2', 'exec_3', 'exec_4', 'exec_5']

crop_stategy = {
    'crush': [16, -5],
    'grasp': [0, -10],
    'lift_slow': [0, -3],
    'shake': [0, -1],
    'poke': [2, -5],
    'push': [2, -5],
    'tap': [0, -5],
    'low_drop': [0, -1],
    'hold': [0, -1],
}


SEQUENCE_LENGTH = 10
STEP = 4
IMG_SIZE = (64, 64)


def read_dir():
    visions = glob.glob(os.path.join(DATA_DIR, 'vision*/*/*/*/*'))
    return visions


def generate_npy_vision(path, behavior, sequence_length):
    '''
    :param path: path to images folder,
    :return: numpy array with size [SUB_SAMPLE_SIZE, SEQ_LENGTH, ...]
    '''
    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    img_length = len(files)
    files = files[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    imglist = []
    for file in files:
        img = PIL.Image.open(file)
        img = img.resize(IMG_SIZE)
        img = graypdbfs(img, [3]).astype(np.bool)
        img = np.array(img).transpose([2, 0, 1])[np.newaxis, ...]
        imglist.append(img)
    ret = []
    for i in range(0, len(imglist) - sequence_length, STEP):
        ret.append(np.concatenate(imglist[i:i + sequence_length], axis=0))
    return ret, img_length


def split(strategy):
    '''
    :param strategy: object | category | trail
    :return: train -> list
             test -> list
    '''
    train_list = []
    test_list = []
    # if FIXED:
    #     #     return train, test
    if strategy == 'object':
        for i in range(len(SORTED_OBJECTS) // 5):
            random_number = np.random.randint(low=0, high=5)
            np.random.shuffle(SORTED_OBJECTS[5 * i:5 * i + 5])
            shuffled_category = SORTED_OBJECTS[5 * i:5 * i + 5]
            for item, object in enumerate(shuffled_category):
                if item == random_number:
                    test_list.append(object)
                else:
                    train_list.append(object)

    elif strategy == 'category':
        random.shuffle(CATEGORIES)
        train_list, test_list = CATEGORIES[:16], CATEGORIES[16:]

    elif strategy == 'trail':
        random.shuffle(TRAILS)
        train_list, test_list = TRAILS[:4], TRAILS[4:]
        train_list += ['exec_6', 'exec_7', 'exec_8', 'exec_9', 'exec_10']
    return train_list, test_list


def process(visions, chosen_behavior):
    CHOOSEN_BEHAVIORS = BEHAVIORS
    if chosen_behavior in CHOOSEN_BEHAVIORS:
        CHOOSEN_BEHAVIORS = [chosen_behavior]
    train_subdir = 'train'
    test_subdir = 'test'
    vis_subdir = 'vis'
    if not os.path.exists(os.path.join(OUT_DIR, train_subdir)):
        os.makedirs(os.path.join(OUT_DIR, train_subdir))

    if not os.path.exists(os.path.join(OUT_DIR, test_subdir)):
        os.makedirs(os.path.join(OUT_DIR, test_subdir))

    if not os.path.exists(os.path.join(OUT_DIR, vis_subdir)):
        os.makedirs(os.path.join(OUT_DIR, vis_subdir))

    # added for VIS splits
    if not os.path.exists(os.path.join(VIS_DIR, chosen_behavior)):
        os.makedirs(os.path.join(VIS_DIR, chosen_behavior))

    train_list, test_list = split(strategy=STRATEGY)

    train_test_split_dict = {
        'train': train_list,
        'test': test_list
    }
    with open(os.path.join(VIS_DIR, chosen_behavior, "train_test_split"), 'wt') as split_file:
        for k, v in train_test_split_dict.items():
            split_file.write('%s: %s\n' % (str(k), str(v)))

    split_base = train_list + test_list
    cutting = len(train_list)

    fail_count = 0
    for vision in visions:
        save = False
        behavior = ''
        for _bh in CHOOSEN_BEHAVIORS:
            if _bh in vision.split('/'):
                behavior = _bh
                save = True
                break
        if not save:
            continue
        subdir = ''
        for ct in split_base[:cutting]:
            if ct in vision:
                subdir = train_subdir
        for ct in split_base[cutting:]:
            if ct in vision:
                subdir = test_subdir
        if not subdir:
            continue
        out_sample_dir = os.path.join(OUT_DIR, subdir, '_'.join(vision.split('/')[-4:]))

        if subdir == test_subdir:
            vis_out_sample_dir = os.path.join(OUT_DIR, vis_subdir, '_'.join(vision.split('/')[-4:]))
            out_vision_npys, n_frames = generate_npy_vision(vision, behavior, SEQUENCE_LENGTH*2)

            out_behavior_npys = np.zeros(len(CHOOSEN_BEHAVIORS))
            out_behavior_npys[CHOOSEN_BEHAVIORS.index(behavior)] = 1

            for i, out_vision_npy in enumerate(
                    out_vision_npys):
                ret = {
                    'behavior': out_behavior_npys,
                    'vision': out_vision_npy,
                }
                np.save(vis_out_sample_dir + '_' + str(i), ret)

        out_vision_npys, n_frames = generate_npy_vision(vision, behavior, SEQUENCE_LENGTH)

        out_behavior_npys = np.zeros(len(CHOOSEN_BEHAVIORS))
        out_behavior_npys[CHOOSEN_BEHAVIORS.index(behavior)] = 1

        # make sure that all the lists are in the same length!
        for i, out_vision_npy in enumerate(
                out_vision_npys):
            ret = {
                'behavior': out_behavior_npys,
                'vision': out_vision_npy,
            }
            np.save(out_sample_dir + '_' + str(i), ret)
    print("fail: ", fail_count)


def run(chosen_behavior):
    print("start making data")
    visons = read_dir()
    process(visons, chosen_behavior)
    print("done!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--behavior', default='None', help='which behavior?')
    args = parser.parse_args()

    print("behavior: ", args.behavior)

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    run(chosen_behavior=args.behavior)