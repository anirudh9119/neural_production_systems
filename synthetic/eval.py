import torch
import argparse
import numpy as np
from data import ArithmeticData
from torch.utils.data import DataLoader
from model import ArithmeticModel
import torch.nn as nn
from tqdm import tqdm
from utilities.rule_stats import get_stats
import random
import os

def none_or_str(value):
    if value == 'None':
        return None
    return value

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--noise', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--tied', default=False, action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--cudnn', action='store_true',
                    help='use cudnn optimized version. i.e. use RNN instead of RNNCell with for loop')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='sparse_factor_graphs',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--resume', type=str, default=None,
                    help='if specified with the 1-indexed global epoch, loads the checkpoint and resumes training')
parser.add_argument('--algo', type=str, choices=('rim', 'lstm', 'SCOFF'))
parser.add_argument('--num_blocks', type=int, default=6)
parser.add_argument('--nhid', type=int, default=300)
parser.add_argument('--topk', type=int, default=4)
parser.add_argument('--block_dilation', nargs='+', type=int, default=-1)
parser.add_argument('--layer_dilation', nargs='+', type=int, default=-1)
parser.add_argument('--train_len', type=int, default=500)
parser.add_argument('--test_len', type=int, default=1000)
parser.add_argument('--read_input', type=int, default=2)
parser.add_argument('--memory_slot', type=int, default=4)
parser.add_argument('--memory_heads', type=int, default=4)
parser.add_argument('--memory_head_size', type=int, default=16)
parser.add_argument('--gate_style', type=none_or_str, default=None)
parser.add_argument('--use_inactive', action='store_true',
                    help='Use inactive blocks for higher level representations too')
parser.add_argument('--blocked_grad', action='store_true',
                    help='Block Gradients through inactive blocks')
parser.add_argument('--scheduler', action='store_true',
                    help='Scheduler for Learning Rate')
parser.add_argument('--adaptivesoftmax', action='store_true',
                    help='use adaptive softmax during hidden state to output logits.'
                         'it uses less memory by approximating softmax of large vocabulary.')
parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                    help='cutoff values for adaptive softmax. list of integers.'
                         'optimal values are based on word frequencey and vocabulary size of the dataset.')
parser.add_argument('--use_attention', action ='store_true')
#parser.add_argument('--name', type=str, default=None,
#                    help='name for this experiment. generates folder with the name if specified.')

## Rule Network Params
parser.add_argument('--use_rules', action ='store_true')
parser.add_argument('--rule_time_steps', type = int, default = 1)
parser.add_argument('--num_rules', type = int, default = 4) 
parser.add_argument('--rule_emb_dim', type = int, default = 64)
parser.add_argument('--rule_query_dim', type = int, default = 32)
parser.add_argument('--rule_value_dim', type = int, default = 64)
parser.add_argument('--rule_key_dim', type = int, default = 32)
parser.add_argument('--rule_heads', type = int, default = 4)


parser.add_argument('--comm', type = str2bool, default = True)
parser.add_argument('--grad', type = str, default = "yes")
parser.add_argument('--transformer', type = str, default = "yes")
parser.add_argument('--application_option', type = str, default = '3')
parser.add_argument('--training_interval', type = int, default = 2)
parser.add_argument('--alternate_training', type = str, default = "yes")

parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--split', type = str, default = 'mcd1')
parser.add_argument('--perm_inv', type=str2bool, default=False)
parser.add_argument('--n_templates', type=int, default=2)
parser.add_argument('--gumble_anneal_rate', type = float, default = 0.00003)
parser.add_argument('--use_entropy', type = str2bool, default = True)
parser.add_argument('--use_biases', type = str2bool, default = True)
parser.add_argument('--timesteps', type = int, default = 20)

args = parser.parse_args()


def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)

import os



def operation_to_rule_train(operations, rule_selections, vault, inverse_vault):
	operations = operations.cpu().numpy()
	#operations = torch.argmax(operations, dim = 2).cpu().numpy()
	#print(operations)
	t = 0
	for b in range(operations.shape[0]):
		if rule_selections[t][b] not in inverse_vault:
			inverse_vault[rule_selections[t][b]] = {'addition':0, 'subtraction': 0, 'multiplication':0}
		if operations[b,0] == 1:
			if rule_selections[t][b] not in vault['addition']:
				vault['addition'].append(rule_selections[t][b])
			inverse_vault[rule_selections[t][b]]['addition'] += 1
		elif operations[b, 1] == 1:
			if rule_selections[t][b] not in vault['subtraction']:
				vault['subtraction'].append(rule_selections[t][b])
			inverse_vault[rule_selections[t][b]]['subtraction'] += 1
		elif operations[b,2] == 1:
			if rule_selections[t][b] not in vault['multiplication']:
				vault['multiplication'].append(rule_selections[t][b])
			inverse_vault[rule_selections[t][b]]['multiplication'] += 1


	return vault, inverse_vault


def operation_to_rule(operations, rule_selections, vault, inverse_vault):
	operations = operations.cpu().numpy()
	#operations = torch.argmax(operations, dim = 2).cpu().numpy()
	#print(operations)
	for b in range(operations.shape[0]):
		for t in range(operations.shape[1]):
			if rule_selections[t][b] not in inverse_vault:
				inverse_vault[rule_selections[t][b]] = {'addition':0, 'subtraction': 0, 'multiplication':0}
			if operations[b,t, 0] == 1:
				if rule_selections[t][b] not in vault['addition']:
					vault['addition'].append(rule_selections[t][b])
				inverse_vault[rule_selections[t][b]]['addition'] += 1
			elif operations[b,t, 1] == 1:
				if rule_selections[t][b] not in vault['subtraction']:
					vault['subtraction'].append(rule_selections[t][b])
				inverse_vault[rule_selections[t][b]]['subtraction'] += 1
			elif operations[b,t,2] == 1:
				if rule_selections[t][b] not in vault['multiplication']:
					vault['multiplication'].append(rule_selections[t][b])
				inverse_vault[rule_selections[t][b]]['multiplication'] += 1


	return vault, inverse_vault
best_eval_mse = 10

def eval_epoch(epoch):
	global best_eval_mse
	model.eval()
	correct = 0
	total = 0
	num_examples = 0
	variable_rule = {}
	inverse_vault = {}
	correct_sentences = 1
	total_sentences = 0
	loss = 0

	with torch.no_grad():
		vault = {'addition':[], 'subtraction':[], 'multiplication': []}
		for i, data in enumerate(val_dataloader):
			if args.num_rules > 0:
				model.rule_network.reset_activations()
			data = data.transpose(1, 2).float().to(device)
			#inp, target = data[:, :, :-1], data[:, :, -1]
			inps, operations, targets = data[:, :, 0:1], data[:, :, 1:-1], data[:, :, -1].unsqueeze(-1)
			cur_inps = inps
			prev_inps = torch.cat((torch.zeros(targets.size(0), targets.size(1), 1).to(targets.device),targets[:, :, :-1]), dim = -1)
			cur_targets = targets

			for t in range(cur_inps.size(1)):
				out = model(prev_inps[:, t, :], cur_inps[:, t, :], operations[:, t, :])
				loss += objective(out, cur_targets[:, t, :])
				if t < cur_inps.size(1) - 1:
					prev_inps[:,t + 1, :] = out
			#if i == 0:
			#	print(prev_inps[0, :, :])
			#	print(cur_targets[0, :, :])
			num_examples += 1
			if args.num_rules > 0:
				rule_selections = model.rule_network.rule_activation
				variable_selections = model.rule_network.variable_activation
				variable_rule = get_stats(rule_selections, variable_selections, variable_rule, args.application_option, args.num_rules, args.num_blocks)
				vault, inverse_vault = operation_to_rule(operations, rule_selections, vault, inverse_vault)
			
			total_sentences += 1





			
	#print(model.encoder.rimcell[0].bc_lst[0].iatt_log)
	print('eval_mse:'+str(loss/total_sentences))
	
	eval_mse = loss / total_sentences
	
	#if eval_mse < best_eval_mse:
	#	best_eval_mse = eval_mse
	#	torch.save(model.state_dict(), args.save_dir + '/model_best.pt')
	
	print('eval stats')
	for v in variable_rule:
		print(v, end = ' : ')
		print(variable_rule[v])
	for v in vault:
		print(v, end = ' : ')
		print(vault[v])
	for v in inverse_vault:
		print(v, end = ' : ')
		print(inverse_vault[v])

	return eval_mse


def train_epoch(epoch):
	global best_eval_mse
	loss_ = 0
	model.train()
	correct = 0
	total = 0
	num_examples = 0
	vault = {'addition':[], 'subtraction':[], 'multiplication':[]}
	inverse_vault = {}
	variable_rule = {}
	for i, data in tqdm(enumerate(train_dataloader)):
		if args.num_rules > 0:
			model.rule_network.reset_activations()
		data = data.transpose(1, 2).float().to(device)
		inps, operations, targets = data[:, :, 0:1], data[:, :, 1:-1], data[:, :, -1].unsqueeze(-1)
		#print(inps)
		#print(operations)
		#print(targets)

		cur_inps = inps
		prev_inps = torch.cat((torch.zeros(targets.size(0), 1, targets.size(2)).to(targets.device),targets[:, :-1, :]), dim = 1)
		cur_targets = targets


		cur_operations = operations.reshape(-1,3)
		cur_inps = cur_inps.reshape(-1, 1)
		prev_inps = prev_inps.reshape(-1, 1)
		cur_targets = cur_targets.reshape(-1,1)



		#for j in range(cur_operations.size(0)):
		#	print(cur_operations[j], end = ' ')
		#	print(prev_inps[j], end = ' ')
		#	print(cur_inps[j], end = ' ')
		#	print(cur_targets[j])

		out = model(prev_inps, cur_inps, cur_operations)

		
		loss = objective(out, cur_targets)
		model.zero_grad()
		(loss).backward()
		loss_ += loss
		num_examples += 1
		
		if args.num_rules > 0:
			rule_selections = model.rule_network.rule_activation
			variable_selections = model.rule_network.variable_activation
			variable_rule = get_stats(rule_selections, variable_selections, variable_rule, args.application_option, args.num_rules, args.num_blocks)
			vault, inverse_vault = operation_to_rule_train(cur_operations, rule_selections, vault, inverse_vault)
			

		optimizer.step()

		
		if i % args.log_interval == 1 and i != 1:
			
			print('epoch:' + str(epoch), end = ' ')
			print('loss: '+str(loss_/num_examples), end = ' ')
			
			loss_ = 0
			correct = 0
			total = 0
			num_examples = 0
			
			print('train stats')
			for v in variable_rule:
				print(v, end = ' : ')
				print(variable_rule[v])
			for v in vault:
				print(v, end = ' : ')
				print(vault[v])
			for v in inverse_vault:
				print(v, end = ' : ')
				print(inverse_vault[v])
			inverse_vault = {}
			vault = {'addition':[], 'subtraction':[], 'multiplication':[]}

			eval_epoch(epoch)
			print('best_eval_mse:' + str(best_eval_mse))
			model.train()
	torch.save(model.state_dict(), args.save_dir + '/model_latest.pt')
#for epoch in range(1, args.epochs):
#	train_epoch(epoch)

#dirs = os.listdir('./')
#dirs = [d for d in dirs if d.startswith('algo')]

'''rule_dim_ = {
			 '16': {10:[],20:[],30:[], 40:[], 50:[]},
			 '40':{10:[],20:[],30:[], 40:[], 50:[]},
			 '32':{10:[],20:[],30:[], 40:[], 50:[]},
			 '48':{10:[],20:[],30:[], 40:[], 50:[]},
			 '64':{10:[],20:[],30:[], 40:[], 50:[]},
			 '128':{10:[],20:[],30:[], 40:[], 50:[]}
			 }
'''
_num_rules = [0, 3]
_seed = [0,1,2,4,5]
_rule_dim = [40, 32, 48, 64, 128]

dirs = []

for r in _num_rules:
	for s in _seed:
		for d in _rule_dim:
			if r == 0:
				if d == 40:
					_dir_name = str(r) + '-' + str(d) + '-' + '100' + '-' + str(s)
					dirs.append(_dir_name)
			else:
				if d == 40:
					continue
				_dir_name = str(r) + '-' + str(d) + '-' + '100' + '-' + str(s) + '-False'
				dirs.append(_dir_name)



dirs = [args.save_dir]

res = {}

for d in dirs:
	for t in [10,20,30,40,50]:
		args.rule_emb_dim = int(d.split('-')[1])
		args.num_rules = int(d.split('-')[0])

		print('RULE DIM:' + str(args.rule_emb_dim))
		print('NUM RULES:' + str(args.num_rules))

		val_dataset = ArithmeticData(2000, t)


		val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size,
				shuffle = True, num_workers = 4)

		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

		model = ArithmeticModel(args, 10).to(device)
		model.load_state_dict(torch.load(d + '/model_best.pt'))

		if args.use_biases and args.num_rules > 0:
			model.rule_network.use_biases = True
		elif args.num_rules > 0:
			model.rule_network.use_biases = False

		optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
		objective = nn.MSELoss()
		mse = eval_epoch(0)
		with open(d + '/' + str(t) + '.txt', 'w') as f:
			f.write(str(mse))
		#rule_dim_[str(args.rule_emb_dim)][t].append(round(mse.item(),3))
		res[t] = mse

#print(rule_dim_)
print()
print('FINAL RESULTS ACROSS VARIOUS SEQUENCE LENGTHS')
print('\tSEQUENCE LENGTH\t|\tMSE')
for r in res:
	print('\t      '+str(r)+'        |\t'+str(round(res[r].cpu().item(), 4)))
#for r in rule_dim_:
#	for t in rule_dim_[r]:
#		print(r, end = ',')
#		print(t, end = ':')
#		for k in rule_dim_[r][t]:
			
#			print(k, end = ' , ')
#		print()




