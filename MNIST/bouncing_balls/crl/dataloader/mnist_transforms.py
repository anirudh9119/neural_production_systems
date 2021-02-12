import torch
import copy
import itertools
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pprint
from torchvision import datasets, transforms
import torchsample
from tqdm import tqdm

from .base_dataloader import DataLoader
from .image_transforms import *
from .transformation_combination import TransformationCombiner
from .generative_recognition_mapping import GR_Map_full
from .image_transforms import *


torch.manual_seed(0)
np.random.seed(0)

def shrink_mnist_dataset(mnist_orig, bkgd_dim):
    mnist_shrunk = {}
    for k in mnist_orig.keys():
        v_data, v_labels = mnist_orig[k]
        mnist_shrunk[k] = (place_subimage_in_background(bkgd_dim)(v_data), 
            v_labels)
    return mnist_shrunk

def cuda_mnist_dataset(mnist_dataset):
    for key in mnist_dataset.keys():
        inputs = mnist_dataset[key][0]
        targets = mnist_dataset[key][1]
        mnist_dataset[key] = (inputs.cuda(), targets.cuda())
    return mnist_dataset

class BaseImageTransformDataLoader(DataLoader):
    def __init__(self, cuda):
        super(BaseImageTransformDataLoader, self).__init__()
        self.cuda = cuda

    def set_xform_combo_info(self, xform_combo_info):
        self.xform_combo_info = xform_combo_info

    def get_xform_combo_info(self):
        return copy.deepcopy(self.xform_combo_info)

    def initialize_data(self, splits):
        pass

    def reset(self, mode, bsize):
        return self.transformation_dataloader.reset(
            mode, bsize)

    def get_trace(self):
        return ''

    def change_mt(self):
        pass

class ImageTransformDataLoader_identity(BaseImageTransformDataLoader):
    def __init__(self, dataset, composition_depth, cuda=False):
        super(ImageTransformDataLoader_identity, self).__init__(cuda)
        self.transformations = [
            Identity(cuda=self.cuda)
        ]
        self.composition_depth = composition_depth
        self.transformation_dataloader = TransformationDataloader(
            dataset, self.transformations, self.composition_depth,
            self.set_xform_combo_info, splittype='all')
        self.num_train = self.transformation_dataloader.num_train
        self.num_test = self.transformation_dataloader.num_test
        self.xform_combo_info = None

    def get_composition_depth(self):
        return self.composition_depth

class TransformationCombinationDataLoader(BaseImageTransformDataLoader):
    def __init__(self, dataset, transformation_combinations, transform_config, num_transfer=1, cuda=False):
        super(TransformationCombinationDataLoader, self).__init__(cuda)
        self.dataset = dataset
        self.transformation_combinations = transformation_combinations
        self.transform_config = transform_config
        self.num_transfer = num_transfer
        self.dataloaders = {}
        for key in ['train', 'val', 'test']:
            len_dataset = len(dataset[key][0])
            num_transfer = self.num_transfer if key == 'train' else 1
            clipped_len_dataset = int(num_transfer*len_dataset)
            inputs = dataset[key][0][:clipped_len_dataset]
            targets = dataset[key][1][:clipped_len_dataset]
            print('mnist_transforms:line 91'+str(transformation_combinations))
            self.dataloaders[key] = BasicTransformDataLoader(
                inputs=inputs,
                targets=targets,
                transformation_combination=transformation_combinations[key],
                set_xform_combo_info_callback=self.set_xform_combo_info)
        self.num_train = len(self.dataloaders['train'])
        self.num_test = len(self.dataloaders['test'])
        self.verify_consistency()
        self._gr_map = GR_Map_full(self.transform_config).get_gr_map()

    def reset(self, mode, bsize):
        inputs, targets = self.dataloaders[mode].next(bsize)
        assert inputs.size(2) == inputs.size(3) == 64
        return inputs, targets

    def get_composition_depth(self):
        xform_combo_info = self.get_xform_combo_info()
        return xform_combo_info['depth']

    def get_trace(self):
        xform_combo_info = self.get_xform_combo_info()
        forward_parameters = xform_combo_info['forward_parameters']
        inverse_parameters = xform_combo_info['inverse_parameters']
        trace = 'forward parameters: {} inverse parameters: {}'.format(forward_parameters, inverse_parameters)
        return trace

    def update_curriculum(self):
        for key in ['train', 'val', 'test']:
            self.dataloaders[key].transformation_combination.update_curriculum()

    def verify_consistency(self):
        """
            supposed to verify that the transformation_combiner
            object for each mode has the same set of transformations
        """
        retrieve_tc = lambda x: self.dataloaders[x].transformation_combination.get_transform_config()
        train_tc_at = retrieve_tc('train').all_transformations
        val_tc_at = retrieve_tc('val').all_transformations
        test_tc_at = retrieve_tc('test').all_transformations
        assert train_tc_at == val_tc_at == test_tc_at == self.transform_config.all_transformations

    def get_gr_map(self):
        return copy.deepcopy(self._gr_map)

    def decode_tokens(self, x):
        return ''.join([str(y) for y in x])

class BasicDataLoader(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        super(BasicDataLoader, self).__init__(inputs, targets)
        self.inputs = inputs
        self.targets = targets
        self.counter = 0

    def size(self):
        return self.inputs.size()

    def permute(self):
        perm = torch.LongTensor(np.random.permutation(range(len(self.inputs))))
        if self.inputs.is_cuda:
            perm = perm.cuda()
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self, bsize):
        if self.counter >= len(self.inputs)-bsize:
            self.permute()
            self.counter = 0
        inputs = self.inputs[self.counter: self.counter+bsize]
        targets = self.targets[self.counter: self.counter+bsize]
        self.counter += bsize
        return inputs, targets

class BasicTransformDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, num_transforms = 8, transform_length = 3, gen = True, state = "Train"):
        """     
            inputs: (bsize, 1, H, W)
            targets: (bsize, 1)
            transformation_combination: TransformationCombiner object
        """
        #super(BasicTransformDataLoader, self).__init__(inputs, targets)
        self.inputs = inputs
        self.targets = targets
        self.state = state
        self.num_transforms = num_transforms
        self.transform_length = transform_length
        self.gen = gen

        self.transforms = {'rotate_right': Rotate(60)(),
                           'rotate_left': Rotate(-60)(),
                           'translate_up':Translate(0.2, 0)(),
                           'translate_down':Translate(-0.2, 0)(),
                           
                           #'scale_small':Scale(1.7)(),
                           #'scale_large':Scale(0.6)(),
                           
                           'translate_left':Translate(0, 0.2)(),
                           'translate_right':Translate(0, -0.2)(),
                           }

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, bsize):
        # get the raw data
        transforms = []
        images = []

        inputs, targets = self.inputs[bsize], self.targets[bsize]
        transforms = []
        images = [place_subimage_in_background((64, 64))(inputs)]
        transform_vectors = []


        for t in range(self.transform_length):
            #if not self.gen:
            transform_id = random.randint(0, self.num_transforms - 1)
            """
            else:
                #if self.state == "Train":

                    if targets % 2 == 0:
                        transform_id = random.randint(0, 1)
                    else:
                        transform_id = random.randint(2, 3)
                else:
                    if targets % 2 == 0:
                        transform_id = random.randint(2, 3)
                    else:
                        transform_id = random.randint(0, 1)
            """
            cur_vector = torch.zeros(self.num_transforms)
            cur_vector[transform_id] = 1.
            transform_vectors.append(cur_vector)
            transforms.append(list(self.transforms.keys())[transform_id])
            im_ = images[-1]
            im_ = self.transforms[transforms[-1]](im_)
            images.append(im_)
        
        transform_vectors = torch.stack(transform_vectors, dim = 0)
        images = torch.stack(images, dim = 0)
        return images, transforms, transform_vectors

    def update_curriculum(self):
        # print('BasicTransformDataLoader update_curriculum')
        self.transformation_combination.update_curriculum()



class ColorTransformDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, num_transforms = 8, transform_length = 3, state  = "Train"):
        """     
            inputs: (bsize, 1, H, W)
            targets: (bsize, 1)
            transformation_combination: TransformationCombiner object
        """
        #super(BasicTransformDataLoader, self).__init__(inputs, targets)
        self.inputs = inputs
        self.targets = targets
        self.num_transforms = num_transforms
        self.transform_length = transform_length
        self.state = state

        self.transforms = {'rotate_right': Rotate(60)(),
                           'translate_down':Translate(-0.2, 0)(),
                           'translate_up':Translate(0.2, 0)(),
                           'rotate_left': Rotate(-60)(),
                           
                           #'scale_small':Scale(1.7)(),
                           #'scale_large':Scale(0.6)(),
                           
                           'translate_left':Translate(0, 0.2)(),
                           'translate_right':Translate(0, -0.2)(),
                           }
        self.colors = [[0, 1,1], [0, 1, 0], [0, 0, 1], [1,1,0]]

    def color_image(self, im, target):
        if self.state == 'Train':
            if target % 2 == 0:
                color_index = 0#random.randint(0, 1)
            else:
                color_index = 0#random.randint(2, 3)
        else:
            if target % 2 == 0:
                color_index = 0#random.randint(0, 3)
            else:
                color_index = 0#random.randint(0, 3)
        im_1 = im.clone()
        im_1[im_1 > 0] = im_1[im_1 > 0] * self.colors[color_index][0]
        im_2 = im.clone() 
        im_2[im_2 > 0] = im_2[im_2 > 0] * self.colors[color_index][1]
        im_3 = im.clone()
        im_3[im_3 > 0] = im_3[im_3 > 0] * self.colors[color_index][2]

        im = torch.cat((im_1, im_2, im_3), dim = 0)
        return im


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, bsize):
        # get the raw data
        transforms = []
        images = []

        inputs, targets = self.inputs[bsize], self.targets[bsize]
        inputs = self.color_image(inputs, targets)
        
        transforms = []
        images = [place_subimage_in_background((64, 64))(inputs)]
        transform_vectors = []


        for t in range(self.transform_length):
            if self.state == 'Train':
                if targets % 2 == 0:
                    transform_id = random.randint(0, 1)
                else:
                    transform_id = random.randint(2, 3)
            else:
                if targets % 2 == 0:
                    transform_id = random.randint(2, 3)
                else:
                    transform_id = random.randint(0, 1)

            cur_vector = torch.zeros(self.num_transforms)
            cur_vector[transform_id] = 1.
            transform_vectors.append(cur_vector)
            transforms.append(list(self.transforms.keys())[transform_id])
            im_ = images[-1]
            im_ = self.transforms[transforms[-1]](im_)
            images.append(im_)
        
        transform_vectors = torch.stack(transform_vectors, dim = 0)
        images = torch.stack(images, dim = 0)
        return images, transforms, transform_vectors

    def update_curriculum(self):
        # print('BasicTransformDataLoader update_curriculum')
        self.transformation_combination.update_curriculum()


class TrainTransformDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, num_transforms = 8, transform_length = 3):
        """     
            inputs: (bsize, 1, H, W)
            targets: (bsize, 1)
            transformation_combination: TransformationCombiner object
        """
        #super(BasicTransformDataLoader, self).__init__(inputs, targets)
        self.inputs = inputs
        self.targets = targets
        self.num_transforms = num_transforms
        self.transform_length = transform_length

        self.transforms = {'rotate_right': Rotate(45)(),
                           'rotate_left': Rotate(-45)(),
                           'translate_up':Translate(0.2, 0)(),
                           'translate_down':Translate(-0.2, 0)(),
                           
                           #'scale_small':Scale(1.7)(),
                           #'scale_large':Scale(0.6)(),
                           
                           'translate_left':Translate(0, 0.2)(),
                           'translate_right':Translate(0, -0.2)(),
                           }

        self.input_images = []
        self.target_images = []
        self.transform_vectors = []
        self.transform_names = []

        for i in tqdm(range(self.inputs.size(0))):
            cur_transforms = []
            cur_images = [place_subimage_in_background((64, 64))(inputs[i])]
            cur_targets = [] 
            cur_transform_vectors = []
            for t in range(self.transform_length):
                transform_id = random.randint(0, self.num_transforms - 1)
                cur_vector = torch.zeros(self.num_transforms)
                cur_vector[transform_id] = 1.
                cur_transform_vectors.append(cur_vector)
                cur_transforms.append(list(self.transforms.keys())[transform_id])
                im_ = cur_images[-1]
                im_ = self.transforms[cur_transforms[-1]](im_)
                if t < self.transform_length - 1:
                    cur_images.append(im_)
                cur_targets.append(im_)

            assert len(cur_images) == len(cur_targets) == len(cur_transform_vectors) == len(cur_transforms)

            self.input_images.extend(cur_images)
            self.target_images.extend(cur_targets)
            self.transform_vectors.extend(cur_transform_vectors)
            self.transform_names.extend(cur_transforms)

        joint_list = list(zip(self.input_images, self.target_images, self.transform_vectors, self.transform_names))
        random.shuffle(joint_list)
        self.input_images, self.target_images, self.transform_vectors, self.transform_names = zip(*joint_list)
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, i):
        return self.input_images[i], self.target_images[i], self.transform_vectors[i], self.transform_names[i]








