from torch.utils.data import DataLoader, TensorDataset

from .dataloader.mnist_dataset import load_mnist_datasets
from .dataloader.mnist_transforms import shrink_mnist_dataset, cuda_mnist_dataset, \
    TransformationCombinationDataLoader, BasicTransformDataset, TrainTransformDataset, ColorTransformDataset
from .dataloader.transformation_combination import ConcatTransformationCombiner, TransformationCombiner, \
    SpatialImageTransformations
from .dataloader.image_transforms import *
import cv2


def show_trajectory(traj, transforms):
    images = []
    transforms =  transforms + ['nothing']
    traj = torch.split(traj, 1, dim = 0)
    for i, t in enumerate(traj):
        t_ = convert_image_np(t.squeeze(0))
        print(t_.shape)
        write = np.zeros((30, t_.shape[1],t_.shape[2]))
        
        write = cv2.putText(write, transforms[i], (10, 15), cv2.FONT_HERSHEY_SIMPLEX ,
                0.3, (255,255,255))
        t_ = np.concatenate((t_, write), axis = 0)
        zeros = np.zeros((t_.shape[0], 50, t_.shape[2]))
        images.append(t_)
        images.append(zeros)

    images = np.concatenate(images[:-1], axis = 1)
    cv2.imshow('img', images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def batchify(batch):
    images = []
    transforms = []
    transform_vector = []
    for b in batch:
        images.append(b[0])
        transforms.append(b[1])
        transform_vector.append(b[2])
    return torch.stack(images, dim = 0), transforms, torch.stack(transform_vector, dim = 0)

def get_dataloaders(num_transforms = 4, transform_length = 3, batch_size = 50, color = True, shuffle = True):
    mnist_orig = load_mnist_datasets('../data', normalize=False)
    train_data = mnist_orig['train']
    val_data = mnist_orig['val']
    test_data = mnist_orig['test']
    if not color:
        train_dataset = BasicTransformDataset(train_data[0], train_data[1], num_transforms = num_transforms, transform_length = transform_length, gen = False, state = "Train")

        val_dataset = BasicTransformDataset(val_data[0], val_data[1], num_transforms = num_transforms, transform_length = transform_length, gen = False, state = "Val")    

        test_dataset = BasicTransformDataset(test_data[0], test_data[1], num_transforms = num_transforms, transform_length = transform_length, gen = False, state = "Val")
    else:
        
        train_dataset = BasicTransformDataset(train_data[0], train_data[1], num_transforms = num_transforms, transform_length = transform_length, gen = True, state = "Train")

        val_dataset = BasicTransformDataset(val_data[0], val_data[1], num_transforms = num_transforms, transform_length = transform_length, gen = True, state = "Val")    

        test_dataset = BasicTransformDataset(test_data[0], test_data[1], num_transforms = num_transforms, transform_length = transform_length, gen = True, state = "Val")

    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 4, collate_fn = batchify)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 4, collate_fn = batchify)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 4, collate_fn = batchify)
    


    """mnist_shrunk = shrink_mnist_dataset(mnist_orig, (64, 64))
    if use_cuda: mnist_shrunk = cuda_mnist_dataset(mnist_shrunk)

    kwargs = {}

    transform_config = lambda: SpatialImageTransformations(cuda=use_cuda, **kwargs)

    if mix_in_normal:
        train_combiner = ConcatTransformationCombiner(transformation_combiners=[
            TransformationCombiner(transform_config(), name='3c2_RT', mode='train', cuda=use_cuda),
            TransformationCombiner(transform_config(), name='identity', mode='train', cuda=use_cuda)])
    else:
        train_combiner = TransformationCombiner(transform_config(), name='3c2_RT', mode='train', cuda=use_cuda)

    transformation_combinations = {
        'train': train_combiner,
        'val': TransformationCombiner(transform_config(), name='3c2_RT', mode='val', cuda=use_cuda),
        'test': TransformationCombiner(transform_config(), name='3c3_SRT', mode='test', cuda=use_cuda),
    }

    dataloader = TransformationCombinationDataLoader(
        dataset=mnist_shrunk,
        transformation_combinations=transformation_combinations,
        transform_config=transform_config(),
        cuda=use_cuda)  # although we can imagine not doing this""" 
    return train_dataloader, val_dataloader, test_dataloader




"""def get_dataloaders(args, use_cuda, should_shuffle=True, mix_in_normal=False):
    Method to return the dataloaders
    data = load_image_xforms_env(use_cuda=use_cuda,
                                 mix_in_normal=mix_in_normal)

    modes = ["train", "val", "test"]
    shuffle_list = [should_shuffle, False, False]

    def _collate_fn(batch):
        return batch

    def _get_dataloader(dataset, shuffle):
        print('dataset.py:line 51' + str(dataset[0].size()))
        return DataLoader(dataset=TensorDataset(dataset[0],
                                                dataset[1]),
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             num_workers=0,)
                          # collate_fn=_collate_fn)
    print('dataset.py:line 58:'+ str(data.dataloaders['test']))
    import torch
    return [torch.utils.data.DataLoader(data.dataloaders[mode], batch_size = 4) for
            mode, shuffle in zip(modes, shuffle_list)]"""

# get_dataloaders(use_cuda=False, mix_in_normal=False)

if __name__ == '__main__':
    get_dataloaders()