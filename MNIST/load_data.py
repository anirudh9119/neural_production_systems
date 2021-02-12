from __future__ import print_function
from abc import ABC, abstractmethod
import argparse
import torch
import torch.nn as nn
from typing import Generic, Iterable, Optional, Sequence, Tuple, Union, TypeVar
import torch.nn.functional as F
import torch.optim as optim
from prelude import Self
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.dataset import Dataset
import h5py
from torch import nn, Tensor
from .utils import Device
from .block_wrapper import BlockWrapper
from .init import lstm_bias, Initializer
import ipdb

class LoadDataset(Dataset):
	def __init__(self, mode, length=51, directory='/Volumes/Fred/Data'):
		'''
		Function to load in the data from the h5 file.
		mode is either 'test', 'train' or 'validation'
		'''
		self.length = length
		self.mode   = mode
		self.directory = directory
		datasets  = ['/atari.h5','/balls3curtain64.h5','/balls4mass64.h5','/balls678mass64.h5']
		hdf5_file = h5py.File(self.directory+'/balls4mass64.h5', 'r')
		self.input_data   = hdf5_file[self.mode]

	def __getitem__(self, index, out_list=('features', 'groups')):
		# ['collisions', 'events', 'features', 'groups', 'positions', 'velocities']
		print('Loading index: ',index)
		data_in_file = {
			data_name: self.input_data[data_name] for data_name in out_list
		}
		# Currently (51 ,64, 64, 1)
		features = 1.0*data_in_file['features'][:self.length,index,:,:,:] # True, False label, conert to int
		groups   = data_in_file['groups'][:self.length,index,:,:,:].astype(np.uint8) # int 0,1,2,3,4
		# Convert to tensors
		features = torch.tensor(features.reshape(51,1,64,64))
		groups   = torch.tensor(groups.reshape(51,1, 64,64))
		return (features.float(), groups.float())


	def __len__(self):
		return int(self.input_data['groups'].shape[1])


class Flatten(nn.Module):
        def forward(self, input):
                return input.view(input.size(0), -1)

class Encoder(nn.Module):
    """Encoder from https://openreview.net/pdf?id=ryH20GbRW
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.Network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=2),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ELU(),
            nn.BatchNorm2d(64),
            Flatten(),
            nn.Linear(4096, 512),
            nn.ELU(),
            nn.BatchNorm1d(512)
            )

    def forward(self, x):
        x = self.Network(x)
        return x

class RnnState(ABC):
     @abstractmethod
     def __getitem__(self, x: Union[Sequence[int], int]) -> Self:
         pass

     @abstractmethod
     def __setitem__(self, x: Union[Sequence[int], int], value: Self) -> None:
         pass

     @abstractmethod
     def fill_(self, f: float) -> None:
         pass

     @abstractmethod
     def mul_(self, x: Tensor) -> None:
         pass


RS = TypeVar('RS', bound=RnnState)


class RnnState(ABC):
    @abstractmethod
    def __getitem__(self, x: Union[Sequence[int], int]) -> Self:
        pass

    @abstractmethod
    def __setitem__(self, x: Union[Sequence[int], int], value: Self) -> None:
        pass

    @abstractmethod
    def fill_(self, f: float) -> None:
        pass

    @abstractmethod
    def mul_(self, x: Tensor) -> None:
        pass


RS = TypeVar('RS', bound=RnnState)


class RnnBlock(Generic[RS], nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor, hidden: RS, masks: Optional[Tensor]) -> Tuple[Tensor, RS]:
        pass

    @abstractmethod
    def initial_state(self, batch_size: int, device: Device) -> RS:
        pass


def _apply_mask(mask: Optional[Tensor], *args) -> Sequence[Tensor]:
    if mask is None:
        return tuple(map(lambda x: x.unsqueeze(0), args))
    else:
        m = mask.view(1, -1, 1)
        return tuple(map(lambda x: x * m, args))


def _reshape_batch(x: Tensor, mask: Optional[Tensor], nsteps: int) -> Tuple[Tensor, Tensor]:
    x = x.view(nsteps, -1, x.size(-1))
    if mask is None:
        return x, torch.ones_like(x[:, :, 0])
    else:
        return x, mask.view(nsteps, -1)


def _haszero_iter(mask: Tensor, nstep: int) -> Iterable[Tuple[int, int]]:
    has_zeros = (mask[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
    if has_zeros.dim() == 0:
        haszero = [has_zeros.item() + 1]
    else:
        haszero = (has_zeros + 1).tolist()
    return zip([0] + haszero, haszero + [nstep])


class LstmState(RnnState):
    def __init__(self, h: Tensor, c: Tensor, squeeze: bool = False) -> None:
        self.h = h
        self.c = c
        if squeeze:
            self.h.squeeze_(0)
            self.c.squeeze_(0)

    def __getitem__(self, x: Union[Sequence[int], int]) -> Self:
        return LstmState(self.h[x], self.c[x])

    def __setitem__(self, x: Union[Sequence[int], int], value: Self) -> None:
        self.h[x] = value.h[x]
        self.c[x] = value.c[x]

    def fill_(self, f: float) -> None:
        self.h.fill_(f)
        self.c.fill_(f)

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)
        self.c.mul_(x)


class LstmBlock(RnnBlock[LstmState]):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            initializer: Initializer = Initializer(),
            **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.lstm = BlockWrapper(input_dim, output_dim, output_dim, **kwargs)
        #initializer(self.lstm)
        #iself.lstm.myrnn.block_lstm)
        print('using lstm block!')

    def forward(
            self,
            x: Tensor,
            hidden: LstmState,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, LstmState]:
        in_shape = x.shape
        ipdb.set_trace()
        if in_shape == hidden.h.shape:
            out, (h, c) = self.lstm(x.unsqueeze(0), _apply_mask(mask, hidden.h, hidden.c))
            return out.squeeze(0), LstmState(h, c)
        # forward Nsteps altogether
        nsteps = in_shape[0] // hidden.h.size(0)
        x, mask = _reshape_batch(x, mask, nsteps)
        res, h, c = [], hidden.h, hidden.c
        for start, end in _haszero_iter(mask, nsteps):
            m = mask[start].view(1, -1, 1)
            processed, (h, c) = self.lstm(x[start:end], (h * m, c * m))
            print('h c min max', h.min(), h.max(), c.min(), c.max())
            res.append(processed)
        return torch.cat(res).view(in_shape), LstmState(h, c)

    def initial_state(self, batch_size: int, device: Device) -> LstmState:
        zeros = device.zeros((batch_size, self.input_dim))
        return LstmState(zeros, zeros, squeeze=False)

class GruState(RnnState):
    def __init__(self, h: Tensor) -> None:
        self.h = h

    def __getitem__(self, x: Union[Sequence[int], int]) -> Self:
        return GruState(self.h[x])

    def __setitem__(self, x: Union[Sequence[int], int], value: Self) -> None:
        self.h[x] = value.h[x]

    def fill_(self, f: float) -> None:
        self.h.fill_(f)

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)


class GruBlock(RnnBlock[GruState]):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            initializer: Initializer = Initializer(),
            **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        #self.gru = nn.GRU(input_dim, output_dim, **kwargs)
        #initializer(self.gru)

        self.gru = BlockWrapper(input_dim, output_dim, output_dim, **kwargs)
        initializer(self.gru.myrnn.block_lstm)

    def forward(
            self,
            x: Tensor,
            hidden: GruState,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, GruState]:
        in_shape = x.shape
        #print('compare in/h', in_shape, hidden.h.shape)
        if in_shape[0] == hidden.h.shape[0]:
            #print('inp shape', x.unsqueeze(0).shape, 'h shape', (_apply_mask(mask, hidden.h))[0].shape)
            out, h = self.gru(x.unsqueeze(0), *_apply_mask(mask, hidden.h))
            #print('out shape', out.shape, 'h shape', h.shape)
            return out.squeeze(0), GruState(h.squeeze_(0))
        # forward Nsteps altogether
        nsteps = in_shape[0] // hidden.h.size(0)
        x, mask = _reshape_batch(x, mask, nsteps)
        res, h = [], hidden.h
        for start, end in _haszero_iter(mask, nsteps):
            #print('inp shape', x[start:end].shape, 'h  loop shape', h.shape)
            processed, h = self.gru(x[start:end], h * mask[start].view(1, -1, 1))
            #print('h min max', h.min(), h.max())
            #print('process shape', processed.shape, 'h shape after', h.shape)
            res.append(processed)
        return torch.cat(res).view(in_shape), GruState(h.squeeze_(0))

    def initial_state(self, batch_size: int, device: Device) -> GruState:
        return GruState(device.zeros((batch_size, self.input_dim)))

if __name__=='__main__':
	encoder = Encoder()
	batch_size = 8
	sequence_length = 51
	directory = '.'
	training_data   = LoadDataset('training',sequence_length,directory)
	train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
	validation_data = LoadDataset('validation',sequence_length,directory)
	validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)
	test_data = LoadDataset('test',sequence_length,directory)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
	for data in train_loader:
		print('Shape is: ',data[0].shape)
		print('Shape is: ',data[1].shape)
		print('here1')
		encoded = encoder(data[0][:,0,:,:,:]).cuda() #.unsqueeze(0)
		print(encoded.shape)
		print(encoded)
		h0_blocks = torch.randn(8, 512).cuda()
		h0_blocks.h = torch.randn(8, 512).cuda()
		h0_blocks.c = torch.randn(8, 512).cuda()
		rnn_ = GruBlock(512, 512)
		#ipdb.set_trace()
		rnn_.forward(encoded, h0_blocks)
		exit()
