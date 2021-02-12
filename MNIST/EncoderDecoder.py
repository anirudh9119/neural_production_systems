import itertools
from collections import namedtuple

import torch
from torch import nn
from operator import add
from slot_attention import SlotAttention
from RuleNetwork import RuleNetwork

#from model_components import GruBlock, LstmBlock, LayerNorm, UnFlatten, Flatten, Interpolate

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    
    def forward(self, input):
        return input.view(input.size(0), 64, 8, 8)
class LayerNorm(nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
        # self.layernorm = nn.LayerNorm
        self.layernorm = nn.functional.layer_norm

    def forward(self, x):
        #print("SHAPE IS")
        #print(list(x.size()))
        x = self.layernorm(x, list(x.size()[1:]))
        return x

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class ConvEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        

        

        # slot attention encoder according to table 4 in tha paper
        self.f = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1),
            nn.ReLU(True),
            nn.Conv2d(64, args.slot_dim, 5, 1),
            nn.ReLU(True)
        )
        self.slot_dim = args.slot_dim
        # TODO: positional embeddings
        self.encode_positions = nn.Linear(4, args.slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(args.slot_dim, args.slot_dim),
            nn.ReLU(True),
            nn.Linear(args.slot_dim, args.slot_dim)
        )
        


    def forward(self, x):
        
        conv_out = self.f(x)
        
        d_1 = torch.linspace(0, 1, conv_out.size(-1)).to(conv_out.device).unsqueeze(0)
        d_2 = torch.linspace(1, 0, conv_out.size(-1)).to(conv_out.device).unsqueeze(0)

        x_1 = d_1.repeat(conv_out.size(-1), 1)
        x_2 = d_2.repeat(conv_out.size(-1), 1)
        y_1 = d_1.transpose(0, 1).repeat(1, conv_out.size(-1))
        y_2 = d_2.transpose(0, 1).repeat(1, conv_out.size(-1))
        positional = torch.stack((x_1, x_2, y_1, y_2), dim  = -1)
        positional = self.encode_positions(positional)
        positional = positional.unsqueeze(0)
        positional = positional.repeat(conv_out.size(0), 1,1,1)
        positional = positional.permute(0, 3, 1, 2)

        conv_out = conv_out + positional

        conv_out = conv_out.reshape(x.size(0), -1, self.slot_dim)
        out = self.mlp(conv_out)
        return out 
    
    
class BroadcastConvDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.im_size = 64 + 8
        self.init_grid()

        self.g = nn.Sequential(
                    nn.Conv2d(args.slot_dim+2, 64, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(64, 4, 1, 1, 0)
                    )

    def init_grid(self):
        x = torch.linspace(-1, 1, self.im_size)
        y = torch.linspace(-1, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)
        
        
    def broadcast(self, z):
        b = z.size(0)
        x_grid = self.x_grid.expand(b, 1, -1, -1).to(z.device)
        y_grid = self.y_grid.expand(b, 1, -1, -1).to(z.device)
        z = z.view((b, -1, 1, 1)).expand(-1, -1, self.im_size, self.im_size)
        z = torch.cat((z, x_grid, y_grid), dim=1)
        return z

    def forward(self, z):
        z = self.broadcast(z)
        x = self.g(z)
        x_k_mu = x[:, :3]
        m_k_logits = x[:, 3:]        
        return x_k_mu, m_k_logits



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
                            rule_dim = args.rule_dim,
                            num_rules = args.num_rules,
                            rule_time_steps = args.rule_time_steps,
                            rule_selection = args.rule_selection,
                            application_option = args.application_option,
                            ).to(args.device)
        self.slot_attention = SlotAttention(num_slots = 5,
                                            dim = 512,
                                            iters = 3)
        self.design_config = {'comm': True, 'grad': False,
                    'transformer': True, 'application_option': args.application_option, 'selection': 'gumble'}
        self.rule_network = RuleNetwork(args.slot_dim, 4, num_rules = args.num_rules, rule_dim = args.rule_dim, query_dim = 32, value_dim = 64, key_dim = 32, num_heads = 4, dropout = 0.1, design_config = self.design_config)
        self.mha = MultiHeadAttention(n_head=4, d_model_read=args.slot_dim, d_model_write=args.slot_dim, d_model_out=args.slot_dim, d_k=16, d_v=16, num_blocks_read=4, num_blocks_write=4, topk=4, grad_sparse=False)
          

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
        something_, rnn_out_, extra_loss, block_mask, template_attn, entropy = self.rnn_.forward(encoded_input, h_prev)
        dec_out_ = self.Decoder(rnn_out_.h)
        return dec_out_, rnn_out_, extra_loss, block_mask, template_attn, entropy


    def init_encoders(self):
        """Method to initialise the encoder"""
        self.encoders = nn.ModuleList(
            [self.make_encoder() for _ in range(self.args.num_encoders)]
        )
        if self.args.num_encoders == 1:
            self.single_encoder = self.make_encoder()

    def make_encoder(self):
        """Method to create an encoder"""
        if self.args.something == '4Balls':
            encoder = ConvEncoder(args)
            return encoder
        else:
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=4, stride=2),
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
        if self.args.something == '4Balls':
            self.Decoder = BroadcastConvDecoder(args)
        else:
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
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0),
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

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim, 128)
        self.mlp2 = nn.Linear(128, 128)
        self.mlp3 = nn.Linear(128, 128)
        self.mlp4 = nn.Linear(128, out_dim)
        #self.dropout = nn.Dropout(p = 0.5)

    def forward(self, x):
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))
        x = torch.relu(self.mlp3(x))
        x = self.mlp4(x)
        #x = torch.relu(self.mlp3(x))
        #x = self.mlp4(x)
        return x

class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input * 1.0
    def backward(ctx, grad_output):
        #print(torch.sqrt(torch.sum(torch.pow(grad_output,2))))
        print(grad_output)
        return grad_output * 1.0



class MNISTModel(nn.Module):
    """Encoder from https://openreview.net/pdf?id=ryH20GbRW
    Decoder is not copied
    """

    def __init__(self, args):
        super(MNISTModel, self).__init__()
        self.args = args

        self.num_channels = 1 if args.color else 1#self.args.num_channels
        print('NUM CHANNELS:', self.num_channels)

        self.encoders = None

        self.init_encoders()

        input_size = 100
        self.output_size = input_size * args.num_encoders

        h0_blocks = torch.randn(input_size, self.output_size).to(args.device)
        h0_blocks.h = torch.randn(input_size, self.output_size).to(args.device)
        h0_blocks.c = torch.randn(input_size, self.output_size).to(args.device)
        """self.rnn_ = GruBlock(self.output_size + args.num_transforms, self.output_size, device=args.device,
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
                            num_rules = args.num_rules,
                            rule_time_steps = args.rule_time_steps,
                            rule_selection = args.rule_selection,
                            application_option = args.application_option,
                            rule_dim = args.rule_dim
                            ).to(args.device)"""
        #self.rnn_ = GruBlock(self.output_size, self.output_size, device=args.device,
        #                     num_blocks=args.num_blocks,
        #                     topk=args.topk,
        #                     num_of_rules=args.num_of_rules,
        #                     num_of_step_unrolling=args.num_of_steps_unrolling
        #                     ).to(args.device)
        self.operation_rep = nn.Sequential(nn.Linear(4, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.output_size))
        self.num_rules = args.num_rules
        if self.num_rules > 0:
            self.design_config = {'comm': True, 'grad': False,
                        'transformer': True, 'application_option': '3.0.1.0', 'selection': 'gumble'}
            
            self.rule_network = RuleNetwork(self.output_size, 2, num_rules = args.num_rules, rule_dim = args.rule_dim, query_dim = 32, value_dim = 64, key_dim = 32, num_heads = 4, dropout = 0.1, design_config = self.design_config)

        self.mlp = MLP(2 * self.output_size, self.output_size)
        self.Decoder = None

        self.init_decoders()

    def forward(self, x, h_prev, inp_transforms, rule_mask = None):
        encoded_input = [
            encoder(x) for encoder in self.encoders
        ]
        operation = self.operation_rep(inp_transforms).unsqueeze(1)
        encoded_input = torch.cat(encoded_input, 1)
        encoded_input = torch.cat((encoded_input.unsqueeze(1), operation), dim = 1)


        #print(encoded_input.size())
        #print(inp_transforms.size())
        

        #something_, rnn_out_, extra_loss, block_mask, activation_frequency, \
        #block_rules_correlation_matrix = self.rnn_.forward(encoded_input, h_prev)
        #print(encoded_input.size())
        if self.num_rules > 0:
            dec_in, mask = self.rule_network(encoded_input)
            #dec_in = Identity().apply(dec_in)
            #encoded_input = encoded_input + rule_out.squeeze(1)
            #encoded_input = torch.cat((encoded_input, rule_out.squeeze(1)), dim = 1)
            #encoded_input = self.mlp(encoded_input)
        else:
            #print('here')
            encoded_input = torch.cat((encoded_input, inp_transforms), dim = 1)
        #print(dec_in.size())
        #dec_outs = []
        #for d in self.Decoder:
        #    dec_outs.append(d(dec_in))
        dec_outs = self.Decoder[0](dec_in)
        #dec_outs = torch.stack(dec_outs, dim = 1)
        #mask = mask.unsqueeze(2).unsqueeze(3)#.unsqueeze(4)
        #print(dec_outs.size())
        #print(mask.size())
        #dec_outs = dec_outs * mask
        #dec_outs = torch.sum(dec_outs, dim = 1)

        #dec_out_ = self.Decoder(encoded_input)
        
        return torch.sigmoid(dec_outs) #rnn_out_, extra_loss, block_mask, activation_frequency,#block_rules_correlation_matrix

    def init_encoders(self):
        """Method to initialise the encoder"""

        self.encoders = nn.ModuleList(
            [self.make_encoder() for _ in range(self.args.num_encoders)]
        )

    def make_encoder(self):
        """Method to create an encoder"""
        return nn.Sequential(
                    nn.Conv2d(self.num_channels, 16, kernel_size=4, stride=2),
                    nn.ELU(),
                    LayerNorm(),
                    nn.Conv2d(16, 32, kernel_size=4, stride=2),
                    nn.ELU(),
                    LayerNorm(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ELU(),
                    LayerNorm(),
                    Flatten(),
                    nn.Linear(2304, 100),
                    nn.ELU(),
                    LayerNorm(),
                )

    def init_decoders(self):
        """Method to initialise the decoder"""
        if self.num_rules > 0:
            in_channels = self.output_size
        else:
            in_channels = self.output_size + self.args.num_transforms
        
        self.Decoder = nn.ModuleList([nn.Sequential(
            nn.Sigmoid(),
            LayerNorm(),
            nn.Linear(in_channels, 4096),
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
            nn.Conv2d(16, self.num_channels, kernel_size=3, stride=1, padding=0),
            #nn.Sigmoid()
        )for _ in range(4)])


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
