#from rnn_models_bb import  RNNModel
#import baseline_lstm_model
import torch
import torch.nn as nn
from modularity import RIMv2, SCOFFv2
#from rnn_models_wiki import RNNModel as RNNModelRules

class Identity(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    return input * 1.0
  def backward(ctx, grad_output):
    print(grad_output)
    return grad_output * 1.0

class BlockWrapper(nn.Module):

    def __init__(self, ntokens, nhid, n_out, device = None, **kwargs):
        super(BlockWrapper, self).__init__()
        num_blocks = kwargs['num_blocks']
        update_topk = kwargs['topk']
        memory_topk = kwargs['memorytopk']
        num_modules_read_input = kwargs['num_modules_read_input']
        inp_heads=kwargs['inp_heads']
        n_templates = kwargs['n_templates']
        algo = kwargs['algo']
        dropout = kwargs['dropout']
        do_rel= kwargs['do_rel']
        memory_slots= kwargs['memory_slots']
        num_memory_heads= kwargs['num_memory_heads']
        memory_head_size= kwargs['memory_head_size']
        share_inp=   kwargs["share_inp"]
        share_comm=  kwargs["share_comm"]
        memory_mlp=  kwargs["memory_mlp"]
        attention_out=kwargs["attention_out"]
        version=kwargs["version"]
        step_att=kwargs["step_att"]
        num_rules = kwargs['num_rules']
        rule_time_steps = kwargs['rule_time_steps']
        rule_selection = kwargs['rule_selection']
        
        application_option = kwargs['application_option']
        rule_dim = kwargs['rule_dim']
        print("Number of blocks %s, Updating top K %s, number of input modules %s, number of input heads",  num_blocks, update_topk, num_modules_read_input, inp_heads)
        rule_config = {'rule_time_steps': rule_time_steps, 'num_rules': num_rules,
            'rule_emb_dim': 64, 'rule_query_dim':32, 'rule_value_dim':64, 'rule_key_dim': 32,
            'rule_heads':4,'rule_dropout': 0.5}
        design_config = {'comm': True , 'grad': False, 'transformer': True, 'application_option': 3}
        if algo == 'SCOFF':
            print('ntoken:' + str(ntokens))
            print('n_hid:' + str(nhid))
            print('n_templates:' + str(n_templates))
            print('dropout:' + str(dropout))
            print('num_blocks:' + str(num_blocks))
            print('update_topk:' + str(update_topk))
            print('num_modules_read_input:'+str(num_modules_read_input))
            print('inp_heads:' + str(inp_heads))
            print('device:' + str(device))
            print('share_comm:' + str(share_comm))
            print('share_inp:' + str(share_inp))
            print('attention_out:' + str(attention_out))
            print('version:' + str(version))
            print('step_att:' + str(step_att))

            self.myrnn = SCOFFv2('cuda', ntokens, nhid, num_blocks, update_topk, num_templates = n_templates, rnn_cell = 'GRU',
                n_layers = 1, bidirectional = False, num_rules = num_rules, rule_time_steps = rule_time_steps, perm_inv = True, application_option = application_option,
                version=version, attention_out=attention_out, rule_dim = rule_dim, step_att=step_att, dropout=dropout, rule_selection = rule_selection)

        elif algo in ['GRU','LSTM']:
            print("Using Baseline RNN")
            self.myrnn = nn.GRU(ntokens, nhid)

        elif algo == 'RIM':
            self.myrnn = RIMv2('cuda', ntokens, nhid, num_blocks, update_topk, rnn_cell = 'GRU', n_layers = 1,
                bidirectional = False, num_rules = num_rules, rule_time_steps = rule_time_steps, application_option = application_option,
                version=version, attention_out=attention_out, step_att=step_att, rule_dim = rule_dim, dropout=dropout, rule_selection = rule_selection)

        #elif algo == 'GWT':
        #    #self.lstm =  GWT('cuda', num_inputs, 500, num_units, num_units, memorytopk=memorytopk, memory_slots=memory_slots,
        #    #              num_memory_heads=num_memory_heads, memory_head_size=memory_head_size, rnn_cell = 'GRU',
        #    #              version=2, attention_out=att_out, step_att=False)

        #    self.myrnn = GWT('cuda', ntokens, nhid, num_blocks, update_topk, rnn_cell = 'GRU', n_layers = 1,
        #        bidirectional = False, version=version, attention_out=attention_out, step_att=step_att, dropout=dropout,
        #        memorytopk=memory_topk, memory_slots=memory_slots,  num_memory_heads=num_memory_heads, memory_head_size=memory_head_size)
        #        #num_rules = num_rules, rule_time_steps = rule_time_steps, application_option = application_option,
        #        #version=version, attention_out=attention_out, step_att=step_att, dropout=dropout, rule_selection = rule_selection)


        else:
            raise ValueError('Algo format {} not recognized.'.format(algo))

        self.nhid = nhid
        self.algo = algo

    def forward(self, inp, h):
        assert len(h.shape) == 3
        assert len(inp.shape) == 3
        hidden = (h, h)
        entropy = 0
        if self.algo in ['GRU', 'LSTM']:
            ob, hb = self.myrnn(inp, hidden[0])
        else:
            ob, hidden, entropy = self.myrnn(inp, hidden)
            hb = hidden[0]
        return ob,hb, None, None, None, entropy


if __name__ == "__main__":
    nhid = 600
    ntokens = 10

    blocks = BlockWrapper(ntokens, nhid, n_out=nhid)
    gru = torch.nn.GRU(ntokens, nhid).cuda()

    x = torch.randn(50, 64, 10).cuda()

    h0 = torch.randn(1, 64, nhid).cuda()
    h0_blocks = torch.randn(1, 64, nhid*2).cuda()

    og, hg = gru(x, h0)
    print('gru of x: o,h', og.shape, hg.shape)

    ob, hb = blocks(x, h0_blocks)
    print('block res: o,h', ob.shape, hb.shape)



