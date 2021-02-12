"""Main entry point of the code"""
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import random

from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader

from argument_parser import argument_parser
from dataset import LoadDataset
from EncoderDecoder import Model, MNISTModel
from model_components import GruState
from utils.util import set_seed, make_dir
from logbook.logbook import LogBook
from box import Box
from argparse import Namespace
from utilities.rule_stats import get_stats
import cv2
from crl.dataset import get_dataloaders as crl_dataloaders


import seaborn as sns
sns.set()
#sns.set(style="whitegrid")

sns.set_style('darkgrid')

set_seed(0)

bce_fn = torch.nn.BCELoss(reduction='none')


def repackage_hidden(h, args):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    hidden = []
    for i in range(1):
        if isinstance(h[i], torch.Tensor):
            hidden.append(h[i].detach())
        else:
            hidden.append(tuple((h[i][0].detach(), h[i][1].detach())))
    return hidden

def bootstrap(model_dir):#, prefix="probe_"):
    """Code to bootstrap probing methods"""
    model_ = model_dir + "model"
    path_to_load_config = model_ + "/args"
    #dir_to_load_model = f"logs/{config_id}/model"
    args = Box(torch.load(path_to_load_config)["args"])
    #args.batch_size = 200

    #args.id = f"{prefix}{args.id}"
    #args.folder_log = f"./logs/{args.id}"

    return args#, logbook, dir_to_load_model

def setup_model(args, model_dir):
    """Method to setup the model"""
    #model.rnn_.gru.RNNModelRules.bc_lst[0].selected_rules
    model = MNISTModel(args)
    # Find the last checkpointed model and resume from that
    model_dir = model_dir + "model"
    print("++++++++++++++++", model_dir)
    latest_model_idx = max(
        [int(model_idx) for model_idx in listdir(model_dir) if model_idx != "args"]
    )
    latest_model_idx = min(latest_model_idx, 50)
    args.path_to_load_model = f"{model_dir}/{latest_model_idx}"
    args.checkpoint = {"epoch": latest_model_idx}

    shape_offset = {}

    model.load_state_dict(torch.load(args.path_to_load_model.strip()), shape_offset)
    return model

    #model = torch.load(args.path_to_load_model.strip())
    #for path_to_load_model in args.path_to_load_model.split(","):
    #    # logbook.write_message_logs(message=f"Loading model from {path_to_load_model}")
    #    _, shape_offset = model.load_state_dict(torch.load(path_to_load_model.strip()),
    #                                         shape_offset)
    '''
    components_to_load = set(args.components_to_load.split("_"))
    total_components = set(["encoders", "decoders", "rules", "blocks"])
    components_to_reinit = [component for component in total_components
                            if component not in components_to_load]
    for component in components_to_reinit:
        if component == "blocks":
            # logbook.write_message_logs(message="Reinit Blocks")
            model.rnn_.gru.myrnn.init_blocks()
        elif component == "rules":
            # logbook.write_message_logs(message="Reinit Rules")
            model.rnn_.gru.myrnn.init_rules()
        elif component == "encoders":
            # logbook.write_message_logs(message="Reinit Encoders")
            model.init_encoders()
        elif component == "decoders":
            # logbook.write_message_logs(message="Reinit Decoders")
            model.init_decoders()
   '''

def rule_transform_stats(rule_selections, transforms, rule_to_transform, args):
    for b in range(rule_selections[0].shape[0]):
        for t in range(len(rule_selections)):
            if transforms[b][t] not in rule_to_transform:
                rule_to_transform[transforms[b][t]] = {r:0 for r in range(args.num_rules)}
            rule_to_transform[transforms[b][t]][rule_selections[t][b]] += 1
    return rule_to_transform


def get_model_output_and_bce(model, data_loader, args, observation_images = None, put_text = True, label = 'baseline'):
    assert args.device is not None, "Please specify args.device!"
    # Get outputs
    #bces = np.zeros((args.n_past+args.n_future, len(data_loader.dataset), args.n_trajectories))

    # import pdb; pdb.set_trace()
    rule_to_transform = {}
    rule_variable = {}
    do_observation = False
    if observation_images is None:
        do_observation = True
        observation_images = []
        obs_inp_transforms = []
    for d, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if args.num_rules > 0:
            model.rule_network.reset_activations()
        inp_data = data[0].to(args.device)
        if do_observation:
            if random.uniform(0, 1) > 0.7:
                observation_images.append(inp_data[0, 0].permute(1,2,0).cpu().numpy())
                obs_inp_transforms.append(data[2].cpu().numpy())

        #target = target[:,0]
        inp_transforms = data[2].to(args.device)
        loss = 0
        kl_loss = 0

        hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
        hidden = GruState(hidden)
        for frame in range(inp_data.size(1) - 1):
        	#output, hidden, extra_loss, block_mask, _, _ = model(inp_data[:, frame, :, :, :], hidden, inp_transforms[:, frame, :])
        	output = model(inp_data[:, frame, :, :, :], None, inp_transforms[:,frame, :])
        	target = inp_data[:, frame + 1, :, :, :]
        	#print(target.shape)
        	#print(output.shape)
        	#loss = loss_fn(output, target)
        	#losses.append(loss.cpu().detach().numpy())
        #losses.append(loss.cpu().detach().numpy())
        #batch += 1
        rule_operation_to_mask = {}
        if args.num_rules > 0:
        	rule_selections = model.rule_network.rule_activation
        	variable_selections = model.rule_network.variable_activation
        	rule_probs = model.rule_network.rule_probabilities
        	if d % 100 == 0:
        		print(rule_probs)
        	rule_variable = get_stats(rule_selections, variable_selections, rule_variable, args.application_option, args.num_rules, args.num_blocks)
        	#print(len(rule_selections))
        	#print(len(data[1][0]))
        	rule_to_transform = rule_transform_stats(rule_selections, data[1], rule_to_transform, args)
        else:
            rule_operation_to_mask = {'translate_up': torch.tensor([0., 0, 1., 0.]).cuda(),
                                        'translate_down': torch.tensor([0., 0., 0., 1.]).cuda(),
                                        'rotate_right': torch.tensor([1., 0., 0., 0.]).cuda(),
                                        'rotate_left':torch.tensor([0., 1., 0., 0.]).cuda()}
    # Make targets
    
    index_to_name = []
    c= 0 
    if args.num_rules > 0:
        for r in rule_to_transform:
            print(r, end = ' : ')
            print(rule_to_transform[r])
            if r not in rule_operation_to_mask:
                index_to_name.append([r.split('_')[0], r.split('_')[1]])
                
                rule_operation_to_mask[r] = torch.tensor([0, 0, 0, 0]).float().to(args.device)
            for t in rule_to_transform[r]:
                if rule_to_transform[r][t] > 0 :
                    index_to_name[-1].append(str(t))
                    rule_operation_to_mask[r][t] = 1

    rat_image = torch.from_numpy(observation_images[21]).to(args.device).permute(2, 0,1)
    rat_image = rat_image.unsqueeze(0)
    show_images = [rat_image.squeeze(0).permute(1, 2, 0).cpu().numpy()]
    whitegrid = np.ones((show_images[-1].shape[0], 40, show_images[-1].shape[2])) * 255.0
    whitegrid = cv2.arrowedLine(whitegrid, (0, whitegrid.shape[0]//2), (whitegrid.shape[1] - 2, whitegrid.shape[0]//2), 
                                     (0,0,0), 1)
    height = show_images[-1].shape[0]
    width = show_images[-1].shape[1]

    show_images.append(whitegrid)
    text_images = []
    print(rule_operation_to_mask)
    operation_to_one_hot = {0: torch.tensor([1.,0., 0., 0.]).cuda(),
                                1: torch.tensor([0.,1., 0., 0.]).cuda(),
                                2: torch.tensor([0.,0., 1., 0.]).cuda(),
                                3: torch.tensor([0.,0., 0., 1.]).cuda()
                                }
    
    operation_to_name = {0:"rotate_right", 1: "rotate_left", 2:"translate_up",3:"translate_down"}
    operations = [0,2]
    #first_operation = list(rule_operation_to_mask.keys())[operations[0]]
    operation_to_paper_name = {'translate_up':[0.20, 'Translate Up'],
                                'translate_down': [0.35, 'Translate Down'],
                                'rotate_left': [0.20, 'Rotate Left'],
                                'rotate_right': [0.25, 'Rotate Right']}
    names = []

    with torch.no_grad():
        for t in tqdm(range(len(operations))):
            
            operation = operations[t]
            operation = operation_to_name[operation]
            operation_vector = torch.tensor([0.5, 0.0, 0.0, 0.0]).cuda()#rule_operation_to_mask[operation].unsqueeze(0)
            print(operation)
            names.append(operation)
            #text_image = np.zeros((50, 64, 1))
    
            #text_image = cv2.putText(text_image, index_to_name[operations[t]][0], (10, 10), cv2.FONT_HERSHEY_SIMPLEX ,
            #    0.35, (255,255,255))
            #text_image = cv2.putText(text_image, index_to_name[operations[t]][1], (15, 20), cv2.FONT_HERSHEY_SIMPLEX ,
            #    0.35, (255,255,255))
            #text_image = cv2.putText(text_image, 'rule:' + index_to_name[operations[t]][2], (15, 30), cv2.FONT_HERSHEY_SIMPLEX ,
            #    0.35, (255,255,255))
            #show_images[-1] = np.concatenate((show_images[-1], text_image), axis = 0)
            #operation_mask = rule_operation_to_mask[operation].unsqueeze(0)
            rat_image = model(rat_image, None, operation_vector.unsqueeze(0))
            rat_image_ = rat_image.squeeze(0).permute(1,2,0).cpu().numpy()
            whitegrid = np.ones((rat_image_.shape[0], 40, rat_image_.shape[2])) * 255.0
            whitegrid = cv2.arrowedLine(whitegrid, (0, whitegrid.shape[0]//2), (whitegrid.shape[1] - 2, whitegrid.shape[0]//2), 
                                     (0,0,0), 1)
            show_images.append(rat_image_)
            show_images.append(whitegrid)

    #text_image = np.zeros((50, 64, 1))
    #show_images[-1] = np.concatenate((show_images[-1], text_image), axis = 0)
    show_images = np.concatenate(show_images[:-1], axis = 1)
    whiteboard = np.ones((20, show_images.shape[1], show_images.shape[2])) * 255.0
    if put_text:
        for i, n in enumerate(names):
            n_ = operation_to_paper_name[n]

            whiteboard = cv2.putText(whiteboard, n_[1], (int(width * (i + 1) + 40 * i - n_[0] * width), 10), cv2.FONT_HERSHEY_SIMPLEX , 0.35, (0,0,0))

    show_images = np.concatenate((whiteboard, show_images), axis = 0) 
    left_margin = np.ones((show_images.shape[0], 50, show_images.shape[2])) * 255.0
    left_margin = cv2.putText(left_margin, label, (0,50), cv2.FONT_HERSHEY_SIMPLEX , 0.35, (0,0,0))
    show_images = np.concatenate((left_margin, show_images), axis = 1)
    
    #text_image = np.zeros((30, 64, 1))
    
    #text_image = cv2.putText(text_image, "nothing", (10, 15), cv2.FONT_HERSHEY_SIMPLEX ,
    #            0.3, (255,255,255))
    #text_images.append(text_images)

    #text_images = np.concatenate(text_images, axis = 1)

    #show_images = np.concatenate((show_images, text_images), axis = 0)

    show_image = observation_images[1]
    cv2.imshow('im', show_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






    #targets = torch.squeeze(gt)[:, args.n_past-args.show_n_from_past:args.n_past+args.show_n_in_future].unsqueeze(-1)

    # Make outputs
    #o = 1 - torch.squeeze(torch.stack(outputs)[args.n_past-args.show_n_from_past:args.n_past+args.show_n_in_future]).transpose(0, 1).unsqueeze(-1)  # [B, T, imsize, imsize, 1]

    # Superimpose outputs on targets
    #o[targets==1] = o[targets==1]*0.8 + targets[targets==1]*0.2
    #o = o.repeat(1, 1, 1, 1, 3)
    #o[:, :, :, :, 0][targets[:, :, :, :, 0]==1] = 0

    #if args.num_rules > 0:
    #    selected_rules = torch.stack(selected_rules, dim = 1)
    #    selected_variables = torch.stack(selected_variables, dim = 1)

    #    selected_rules = selected_rules[:, args.n_past-args.show_n_from_past:args.n_past+args.show_n_in_future, :, :, :]
    #    selected_variables = selected_variables[:, args.n_past-args.show_n_from_past:args.n_past+args.show_n_in_future, :, :, :]

    #    return o, bces, selected_rules, selected_variables
    #return o, bces, None, None
    return show_images, observation_images

def main_one_model():
    """Function to run the experiment"""
    args = argument_parser()
    args.plot_save_dir = "./"
    args.save_n_outputs = 10
    args.show_n_from_past = 6
    args.show_n_in_future = 15
    args.n_past = 15
    args.n_future = 35
    args.n_trajectories = 4
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logbook = LogBook(config=args)

    print(args)

    if not args.should_resume:
        # New Experiment
        make_dir(f"{args.folder_log}/model")
        logbook.write_message_logs(message=f"Saving args to {args.folder_log}/model/args")
        torch.save({"args": vars(args)}, f"{args.folder_log}/model/args")

    # DATASET
    args.directory = './'
    #args.train_dataset = 'balls3curtain64.h5'
    dataset = LoadDataset(mode="test",
                          length=args.sequence_length,
                          directory=args.directory,
                          dataset=args.train_dataset)
    test_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # MODEL
    model = setup_model(args=args)
    model = model.to(args.device)
    model.eval()

    o, bces = get_model_output_and_bce(model, test_loader, args)

    # Save grid of images
    imsize = o.shape[2]
    o = o[:args.save_n_outputs]
    grid = torchvision.utils.make_grid(o.view(-1, *o.shape[2:]).permute(0, 3, 1, 2), nrow=args.show_n_from_past+args.show_n_in_future)
    grid = torch.cat([grid[:, :, :(imsize+2)*args.show_n_from_past], torch.zeros(grid.shape[0], grid.shape[1], 10), grid[:, :, (imsize+2)*args.show_n_from_past:]], dim=-1)
    torchvision.utils.save_image(grid, os.path.join(args.plot_save_dir, "bouncing_balls_vis.png"))

    #grid_rules = torchvision.utils.make_grid(rules.view(-1, *rules.shape[2:]).permute(0, 3, 1, 2), nrow=args.show_n_from_past+args.show_n_in_future)
    #grid_rules = torch.cat([grid[:, :, :(imsize+2)*args.show_n_from_past], torch.zeros(grid.shape[0], grid.shape[1], 10), grid[:, :, (imsize+2)*args.show_n_from_past:]], dim=-1)
    #torchvision.utils.save_image(grid_rules, os.path.join(args.plot_save_dir, "bouncing_balls_vis_rules.png"))


    # Choose trajectory with best BCE, then average across batch
    # bces [T, len(dataset), traj]
    bces = bces.min(-1)
    # Make BCE plot
    plt.errorbar(np.arange(args.n_past), bces[:args.n_past].mean(-1), bces[:args.n_past].std(-1), color='C0')
    plt.errorbar(np.arange(args.n_past, args.n_past+args.n_future), bces[args.n_past:].mean(-1), bces[args.n_past:].std(-1), color='C1')
    plt.xlabel("Time steps")
    plt.ylabel("Binary Cross Entropy")
    plt.savefig(os.path.join(args.plot_save_dir, "bce_plots.png"), bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()


def plot_for_multiple_models(model_dir, data_loader, args=None, observation_images = None, put_text = True, label = 'Baseline'):
        bargs=bootstrap(model_dir)
        args = Namespace(**bargs)

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #import ipdb
        #ipdb.set_trace()
        model = setup_model(args=args, model_dir=model_dir)
        model = model.to(args.device)
        model.eval()
        image, obs_images = get_model_output_and_bce(model, data_loader, args, observation_images = observation_images, put_text = put_text, label = label)
        return image, obs_images

folder_name = 'logs/'

args = argument_parser()
model_name_ = args.something
print(model_name_)
model_dir = 'logs/CRL_RIM-100_1_num_rules_4_rule_time_steps_1-5-False-False/'

print(args.train_dataset)

_, test_loader, _ = crl_dataloaders(num_transforms = 4, transform_length = 4, batch_size = args.batch_size, color = False, shuffle = False) 
image_1, observation_images = plot_for_multiple_models(model_dir, test_loader, args = args)


#args = argument_parser()
#model_name_ = args.something
#print(model_name_)
#model_dir = 'logs/CRL_RIM-100_1_num_rules_4_rule_time_steps_1-5-False-False/'

#print(args.train_dataset)
#image_2,_ = plot_for_multiple_models(model_dir, test_loader, args = args, observation_images = observation_images, put_text = False, label = 'Ours')

#final_image = np.concatenate((image_1, image_2), axis = 0)

#cv2.imshow('image', final_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


