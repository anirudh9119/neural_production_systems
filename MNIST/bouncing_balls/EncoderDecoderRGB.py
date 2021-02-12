import itertools
from collections import namedtuple

import torch
from torch import nn
from operator import add

from model_components import GruBlock, LstmBlock, LayerNorm, UnFlatten, Flatten, Interpolate


class Model(nn.Module):
    """Encoder from https://openreview.net/pdf?id=ryH20GbRW
    Decoder is not copied
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        self.encoders = None
        self.single_encoder = None


        self.input_size = args.hidden_size // args.num_encoders
        input_size = self.input_size
        self.output_size = args.hidden_size #input_size * args.num_encoders
        self.init_encoders()

        h0_blocks = torch.randn(input_size, self.output_size).to(args.device)
        h0_blocks.h = torch.randn(input_size, self.output_size).to(args.device)
        h0_blocks.c = torch.randn(input_size, self.output_size).to(args.device)

        self.rnn_ = GruBlock(self.output_size, self.output_size, device=args.device,
                            num_blocks=args.num_blocks,
                            topk=args.topk,
                            memorytopk=args.memorytopk,
                            num_modules_read_input=args.num_modules_read_input,
                            inp_heads=args.inp_heads,
                            n_templates = args.n_templates,
                            do_rel = args.do_rel,
                            algo= args.algo,
                            dropout=args.dropout,
                            memory_slots= args.memory_slots,
                            num_memory_heads=args.num_memory_heads,
                            memory_head_size=args.memory_head_size,
                            share_inp = args.share_inp,
                            share_comm = args.share_comm,
                            memory_mlp=args.memory_mlp,
                            attention_out=args.attention_out,
                            version=args.version,
                            step_att=args.do_comm,
                            ).to(args.device)

        self.Decoder = None

        self.init_decoders()

    def forward(self, x, h_prev):

        if self.args.num_encoders == 1:
            encoded_input = self.single_encoder(x)#.repeat(1, self.args.num_blocks)

        else:
            encoded_input = [
                encoder(x) for encoder in self.encoders
            ]
            encoded_input = torch.cat(encoded_input, 1)
        something_, rnn_out_, extra_loss, block_mask, _ = self.rnn_.forward(encoded_input, h_prev)
        dec_out_ = self.Decoder(rnn_out_.h)
        return dec_out_, rnn_out_, extra_loss, block_mask


    def init_encoders(self):
        """Method to initialise the encoder"""

        self.encoders = nn.ModuleList(
            [self.make_encoder() for _ in range(self.args.num_encoders)]
        )
        if self.args.num_encoders == 1:
            self.single_encoder = self.make_encoder()

    def make_encoder(self):
        """Method to create an encoder"""
        print(self.input_size)
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            LayerNorm(),
            Flatten(),
            nn.Linear(2304, self.input_size),
            nn.ELU(),
            LayerNorm(),
        )

    def init_decoders(self):
        """Method to initialise the decoder"""
        self.Decoder = nn.Sequential(
            nn.Sigmoid(),
            LayerNorm(),
            nn.Linear(self.output_size, 4096),
            nn.ReLU(),
            LayerNorm(),
            UnFlatten(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(2),
            nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm(),
            Interpolate(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def load_state_dict(self, state_dict, shape_offset, strict=True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.
        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        _IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, shape_offset, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            shape_offset = custom_load_from_state_dict(
                module, state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs, shape_offset)
            for name, child in module._modules.items():
                if child is not None:
                    shape_offset = load(child, shape_offset=shape_offset, prefix=prefix + name + '.')
            return shape_offset

        shape_offset = load(self, shape_offset=shape_offset)

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            # if len(missing_keys) > 0:
            #     error_msgs.insert(
            #         0, 'Missing key(s) in state_dict: {}. '.format(
            #             ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys), shape_offset


def custom_load_from_state_dict(module, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs, shape_offset):
    r"""Copies parameters and buffers from :attr:`state_dict` into only
    this module, but not its descendants. This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.
    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.
    Arguments:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in module._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    local_name_params = itertools.chain(module._parameters.items(), module._buffers.items())
    local_state = {k: v.data for k, v in local_name_params if v is not None}

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]

            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]

            if input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                input_shape = input_param.shape
                if key in shape_offset:
                    offset = shape_offset[key]
                else:
                    offset = [0 for _ in input_shape]

                shape_offset[key] = []

                for current_offset, current_inp_dim, current_param_dim in \
                        zip(offset, input_shape, param.shape):
                    if current_inp_dim == current_param_dim:
                        shape_offset[key].append(0)
                    else:
                        shape_offset[key].append(current_offset + current_inp_dim)

                if len(input_shape) == 1:
                    param[offset[0]:input_shape[0] + offset[0]].copy_(input_param)
                elif len(input_shape) == 2:
                    # from ipdb import set_trace
                    # set_trace()
                    param[offset[0]:input_shape[0] + offset[0],
                    offset[1]:input_shape[1] + offset[1]].copy_(input_param)
                elif len(input_shape) == 3:

                    # special case
                    param[offset[0]:input_shape[0] + offset[0],
                    :input_shape[1], :input_shape[2]].copy_(input_param)
                else:
                    error_msgs.append('size mismatch for {}: copying a param '
                                      'with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                continue

            if isinstance(input_param, nn.Parameter):
                # backwards compatibility for serialized parameters
                input_param = input_param.data
            try:
                param.copy_(input_param)
            except Exception:
                error_msgs.append('While copying the parameter named "{}", '
                                  'whose dimensions in the model are {} and '
                                  'whose dimensions in the checkpoint are {}.'
                                  .format(key, param.size(), input_param.size()))
        elif strict:
            missing_keys.append(key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix):
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                if input_name not in module._modules and \
                        input_name not in local_state:
                    unexpected_keys.append(key)

    return shape_offset
