"""Main entry point of the code"""
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader

from argument_parser import argument_parser
from dataset import LoadDataset
from EncoderDecoder import Model
from model_components import GruState
from utils.util import set_seed, make_dir
from logbook.logbook import LogBook
from box import Box
from argparse import Namespace
import cv2

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
    model = Model(args)
    # Find the last checkpointed model and resume from that
    model_dir = model_dir + "model"
    print("++++++++++++++++", model_dir)
    latest_model_idx = max(
        [int(model_idx) for model_idx in listdir(model_dir) if model_idx != "args"]
    )
    latest_model_idx = min(latest_model_idx, 60)
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


def get_model_output_and_bce(model, data_loader, args):
    assert args.device is not None, "Please specify args.device!"
    # Get outputs
    bces = np.zeros((args.n_past+args.n_future, len(data_loader.dataset), args.n_trajectories))

    # import pdb; pdb.set_trace()
    for d, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        for trajectory in tqdm(range(args.n_trajectories)):
            hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
            hidden = GruState(hidden)
            if args.num_rules > 0:
                model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.reset_activations()
            if trajectory == 0 and d == 0:
                save_outputs = True
                outputs = []
                selected_rules = []
                selected_variables = []

                gt = data.clone()
            else:
                save_outputs = False
            # Data
            data = data.to(args.device) # [B, T, 1, 64, 64]
            # Hidden
            # hidden.h = repackage_hidden(hidden.h, args)
            # Ground truth
            for frame in range(args.n_past):
                target = data[:, frame+1, :, :, :]
                output, hidden, _, _, _, _ = model(data[:, frame, :, :, :], hidden)
                bces[frame, d*args.batch_size:(d+1)*args.batch_size, trajectory] = bce_fn(output, target).mean([-1, -2, -3]).data.cpu().numpy()

                if args.num_rules > 0:
                    rules = model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.rule_activation[-1]
                    variables = model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.variable_activation[-1]
                    print(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.rule_probabilities[-1][0, :])
                    print(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.variable_probabilities[-1][0, :])
                    #print(rules)
                #print(variables)

                if save_outputs:
                    outputs.append(output.detach().cpu())
                    if args.num_rules > 0:
                        rule_grid = torch.zeros(output.size(0), 20 * (args.num_rules+1), 1 * 64, 3)
                        variable_grid = torch.zeros(output.size(0), 20 * args.num_blocks, 1 * 64, 3)
                        rule_grid.fill_(0)
                        variable_grid.fill_(0)
                        for b in range(rules.shape[0]):
                            #print(rules.shape)
                            #print(b)
                            rule_grid[b, rules[b].astype(np.int) * 20 : rules[b].astype(np.int) * 20 + 20, :, 0] = 125
                            variable_grid[b, variables[b, 0] * 20 : variables[b, 0] * 20 + 20, :,0] = 125
                            variable_grid[b, variables[b, 1] * 20 : variables[b, 1] * 20 + 20, :,0] = 125

                        selected_rules.append(rule_grid)
                        selected_variables.append(variable_grid)

            # Roll out
            for frame in range(args.n_past, args.n_past+args.n_future):
                output, hidden, _, _, _, _ = model(output, hidden)
                target = data[:, frame+1, :, :, :]
                bces[frame, d*args.batch_size:(d+1)*args.batch_size, trajectory] = bce_fn(output, target).mean([-1, -2, -3]).data.cpu().numpy()

                if args.num_rules > 0:
                    rules = model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.rule_activation[-1]
                    variables = model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.variable_activation[-1]
                    print(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.rule_probabilities[-1][0, :])
                    print(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.variable_probabilities[-1][0, :])
                if save_outputs:
                    outputs.append(output.detach().cpu()) # [T, B, 1, imsize, imsize]
                    if args.num_rules > 0:
                        rule_grid = torch.zeros(output.size(0), 20 * (args.num_rules+1), 1 * 64, 3)
                        variable_grid = torch.zeros(output.size(0), 20 * args.num_blocks, 1 * 64, 3)
                        rule_grid.fill_(0)
                        variable_grid.fill_(0)
                        for b in range(rules.shape[0]):
                            rule_grid[b, rules[b] * 20 : rules[b] * 20 + 20, :, 0] = 125
                            variable_grid[b, variables[b, 0] * 20 : variables[b, 0] * 20 + 20, :,0] = 125
                            variable_grid[b, variables[b, 1] * 20 : variables[b, 1] * 20 + 20, :,0] = 125


                        selected_rules.append(rule_grid)
                        selected_variables.append(variable_grid)
        if d > 20:
            break

    # Make targets
    targets = torch.squeeze(gt)[:, args.n_past-args.show_n_from_past:args.n_past+args.show_n_in_future].unsqueeze(-1)

    # Make outputs
    o = 1 - torch.squeeze(torch.stack(outputs)[args.n_past-args.show_n_from_past:args.n_past+args.show_n_in_future]).transpose(0, 1).unsqueeze(-1)  # [B, T, imsize, imsize, 1]

    # Superimpose outputs on targets
    o[targets==1] = o[targets==1]*0.8 + targets[targets==1]*0.2
    o = o.repeat(1, 1, 1, 1, 3)
    o[:, :, :, :, 0][targets[:, :, :, :, 0]==1] = 0

    if args.num_rules > 0:
        selected_rules = torch.stack(selected_rules, dim = 1)
        selected_variables = torch.stack(selected_variables, dim = 1)

        selected_rules = selected_rules[:, args.n_past-args.show_n_from_past:args.n_past+args.show_n_in_future, :, :, :]
        selected_variables = selected_variables[:, args.n_past-args.show_n_from_past:args.n_past+args.show_n_in_future, :, :, :]

        return o, bces, selected_rules, selected_variables
    return o, bces, None, None


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


def plot_for_multiple_models(models_list, labels_list, data_loader, something, args=None):

    name_cat = something
    if args is None:
        args = argument_parser()
        args.plot_save_dir = "./"
        args.save_n_outputs = 10
        args.show_n_from_past = 6
        args.show_n_in_future = 15
        args.n_past = 10
        args.n_future = 35
        args.n_trajectories = 4
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cpal = sns.color_palette(n_colors=len(models_list))
    # Get outputs and bces
    outputs_m = []
    bces_m = []
    rules_m = []
    variables_m = []
    algos = ['GRU', 'blocks', 'Rules', 'SCOFF', 'RIM']
    for alog_num, (label, model_) in tqdm(enumerate(zip(labels_list, models_list)), total=len(models_list)):
        print(label)
        #args.algo = algos[alog_num]
        print("Anirudh ", model_)
        print(args)
        bargs=bootstrap(model_)
        args = Namespace(**bargs)
        '''
        args.memorytopk=3
        if "attention" in model_ :
            print(args.attention_out)
        else:
            if "4Att" in model_:
                args.attention_out = 340
            else:
                args.attention_out = 512
        if "ver" in model_:
            print("Version is", args.version)
        else:
            if "V0" in model_:
                args.version=0
            else:
                args.version=1
        if "com" in model_:
            print(args.do_comm)
        else:
            args.do_comm=True
        '''
        args.plot_save_dir = "/home/anirudh/iclr2021/modular_central/bouncing_balls/plots/"
        args.save_n_outputs = 10
        args.show_n_from_past = 6
        args.show_n_in_future = 30
        args.n_past = 15
        args.n_future = 30
        args.n_trajectories = 4
        args.rule_selection = 'argmax'
        #args.num_rules = 15
        #args.rule_time_steps = 1
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #import ipdb
        #ipdb.set_trace()
        model = setup_model(args=args, model_dir=model_)
        model = model.to(args.device)
        model.eval()
        o, bces, ru, va= get_model_output_and_bce(model, data_loader, args)
        outputs_m.append(o)
        bces_m.append(bces)
        if args.num_rules > 0:
            rules_m.append(ru)
            variables_m.append(va)

    # Save grid of images
    # o [B, T, 64, 64, 3]
    imsize = 64
    imsize_r = 64
    skip_image=1
    FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, LINE_TYPE = cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA
    times = [torch.from_numpy(np.ones((3, 64, 64))).float() for _ in range(args.show_n_from_past)] + [torch.from_numpy(cv2.putText(np.ones((64, 64, 3)), f"+{i}", (15, 62), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, LINE_TYPE)).float().permute(2, 0, 1) for i in range(1, 1+args.show_n_in_future, skip_image)]
    outputs = [o[:args.save_n_outputs] for o in outputs_m]
    outputs = torch.stack(outputs_m, dim=1)   # [B, m, T, 64, 64, 3]
    if len(rules_m) > 0:
        rules_m = torch.stack(rules_m, dim = 1)
        variables_m = torch.stack(variables_m, dim = 1)


    # Model names
    widths = []
    for label in labels_list:
        (label_width, label_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        widths.append(label_width)

    max_width = max(widths)
    label_images = [torch.from_numpy(np.ones((3, 64, max_width))).float()]

    for width, label in zip(widths, labels_list):
        label_images.append(torch.from_numpy(cv2.putText(np.ones((64, max_width, 3)), label, (max_width-width, 40), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, LINE_TYPE)).float().permute(2, 0, 1))

    labels_grid = torchvision.utils.make_grid(label_images, nrow=1, pad_value=1)


    for i, output in enumerate(outputs):
        images = times + [o for o in output.view(-1, *output.shape[2:]).permute(0, 3, 1, 2)]   # [m*T, 3, 64, 64]
        show_n_in_future = np.int(args.show_n_in_future/skip_image)
        images_grid = torchvision.utils.make_grid(images, nrow=args.show_n_from_past+ show_n_in_future)
        images_grid = torch.cat([images_grid[:, :, :(imsize+2)*args.show_n_from_past], torch.zeros(images_grid.shape[0], images_grid.shape[1], 10), images_grid[:, :, (imsize+2)*args.show_n_from_past:]], dim=-1)
        concat_name = "plots/" + name_cat
        grid = torch.cat([labels_grid, images_grid], dim=2)[:, 50:, :]
        if len(rules_m) > 0:
            rules_ = [o for o in rules_m[i].view(-1, *rules_m[i].shape[2:]).permute(0, 3, 1, 2)]
            variables_ = [o for o in variables_m[i].view(-1, *variables_m[i].shape[2:]).permute(0, 3, 1, 2)]
            rules_grid = torchvision.utils.make_grid(rules_, nrow=args.show_n_from_past+ show_n_in_future)
            rules_grid = torch.cat([rules_grid[:, :, :(imsize_r+2)*args.show_n_from_past], torch.zeros(rules_grid.shape[0], rules_grid.shape[1], 10), rules_grid[:, :, (imsize_r+2)*args.show_n_from_past:]], dim=-1)

            variables_grid = torchvision.utils.make_grid(variables_, nrow=args.show_n_from_past+ show_n_in_future)
            variables_grid = torch.cat([variables_grid[:, :, :(imsize_r+2)*args.show_n_from_past], torch.zeros(variables_grid.shape[0], variables_grid.shape[1], 10), variables_grid[:, :, (imsize_r+2)*args.show_n_from_past:]], dim=-1)
            torchvision.utils.save_image(rules_grid, os.path.join(args.plot_save_dir, concat_name + "_bballs_schema_vis_rules_{}.png".format(i)))
            torchvision.utils.save_image(variables_grid, os.path.join(args.plot_save_dir, concat_name + "_bballs_schema_vis_variables_{}.png".format(i)))




        torchvision.utils.save_image(grid, os.path.join(args.plot_save_dir, concat_name + "_bballs_schema_vis_{}.png".format(i)))

    # Choose trajectory with best BCE, then average across batch
    # bces [m, T, len(dataset), traj]
    bces_m_min = [bces.min(-1) for bces in bces_m]


    fig = plt.figure()
    ax = plt.subplot(111)


    idx=0
    # Make BCE plot
    for label, bces in zip(labels_list, bces_m_min):
         #x = np.arange(args.n_past)
         #y = bces[:args.n_past].mean(-1)
         #error = bces[:args.n_past].std(-1)
         #ax.plot(x, y, alpha=0.7, label=f"{label}_gt")
         #ax.fill_between(x, y-error, y+error, alpha=0.7)
         x = np.arange(args.n_past, args.n_past+args.n_future)
         y = bces[args.n_past:].mean(-1)
         error = bces[args.n_past:].std(-1)
         #plt.plot(x, y, alpha=0.7, label=f"{label}_rollout", c=cpal[idx])
         ax.plot(x, y, alpha=0.7, label=f"{label}", c=cpal[idx], linewidth=3)
         idx=idx+1

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                       box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                     fancybox=True, shadow=True, ncol=4)

    #plt.legend(loc=(1.05, 0.5))
    #plt.tight_layout()

    plt.xlabel("Time steps", fontsize=20)
    plt.ylabel("Binary Cross Entropy", fontsize=20)
    name_cat_ = "plots/bce_plots_" + name_cat + "_schema.pdf"
    plt.savefig(os.path.join(args.plot_save_dir, name_cat_), bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()



folder_name = '/home/anirudh/iclr2021/modular_central/bouncing_balls/logs/'

args = argument_parser()
model_name_ = args.something
print(model_name_)



#RIMs_v0_128 == RIMs_v2_128 == RIMs_v1 >> RIMs_v0_256/RIMsv2_256 >> GRU_256!
#Sch_v0_256, Sch_v2_256 performs bad!
#Sch_v1 performs better than v0_256 but still worse than v0_128, v2_128
if model_name_ == '4Balls':
    models_list=[
        folder_name + 'GRU_256_4Balls_0.0001/',
        #folder_name + 'RIMs_400_4_4_4Balls_0.0001_inp_heads_1_templates__enc_1_ver_0_com_True_drop_0.0_att_128/',
        #folder_name + 'RIMs_400_4_4_4Balls_0.0001_inp_heads_1_templates__enc_1_ver_0_com_True_drop_0.0_att_256/',
        #folder_name + 'RIMs_400_4_4_4Balls_0.0001_inp_heads_1_templates__enc_1_ver_1_com_True_drop_0.0_att_256/',
        folder_name + 'RIMs_400_4_4_4Balls_0.0001_inp_heads_1_templates__enc_1_ver_2_com_True_drop_0.0_att_128/',
        #folder_name + 'RIMs_400_4_4_4Balls_0.0001_inp_heads_1_templates__enc_1_ver_2_com_True_drop_0.0_att_256/',
        folder_name + 'Schema_400_4_4_4Balls_0.0001_inp_heads_1_templates_2_enc_1_ver_0_com_True_drop_0.0_att_128/',
        #folder_name + 'Schema_400_4_4_4Balls_0.0001_inp_heads_1_templates_2_enc_1_ver_0_com_True_drop_0.0_att_256/',
        #folder_name + 'Schema_400_4_4_4Balls_0.0001_inp_heads_1_templates_2_enc_1_ver_1_com_True_drop_0.0_att_256/',
        #folder_name + 'Schema_400_4_4_4Balls_0.0001_inp_heads_1_templates_2_enc_1_ver_2_com_True_drop_0.0_att_85/',
        folder_name + 'Schema_400_4_4_4Balls_0.0001_inp_heads_1_templates_2_enc_1_ver_2_com_True_drop_0.0_att_128/',
        #folder_name + 'Schema_400_4_4_4Balls_0.0001_inp_heads_1_templates_2_enc_1_ver_2_com_True_drop_0.0_att_256/',
        folder_name + 'Schema_400_4_4_4Balls_0.0001_inp_heads_1_templates_4_enc_1_ver_0_com_True_drop_0.0_att_128/',
        folder_name + 'Schema_400_4_4_4Balls_0.0001_inp_heads_1_templates_4_enc_1_ver_2_com_True_drop_0.0_att_128/',
    ]
    labels_list= [
         #'GRU_128',
         'GRU_256',
         #'RIMs_v0_128',
         #'RIMs_v0_256',
         #'RIMs_v1',
         'RIMs_v2_128',
         #'RIMs_v2_256',
         '2Sch_v0_128',
         #'Sch_v0_256',
         #'Sch_v1',
         #'Sch_v2_85',
         '2Sch_v2_128',
         '4Sch_v0_128',
         '4Sch_v2_128',
         #'Sch_v2_256',
    ]



if model_name_ == '2Balls':
    models_list=[
        'logs/' + 'SchemaRule_RIM_128_2_2_Balls4__0.0001_inp_heads_1_templates_2_enc_1_ver_1_com_True_Sharing_num_rules2_rule_time_steps1_att_out32_rule_selectiongumble/'

    ]
    labels_list= [
        #'LSTM (256)',
        #'RIMs',
        #'SOFF (2 Sch)',
        #'SOFF (4 Sch)',
        #'SOFF (6 Sch)',
        #'LSTM_400'
        'RIM_128_32_2_1',
    ]

#2Sch_v2_128 == 4Sch_v2_128 > RIMs_v2_128 > RIMs_v0_128 > GRU_256!

if model_name_ == 'Curtain':
    models_list=[
        #folder_name + 'GRU_128_Curtain_0.0001/',
        folder_name + 'GRU_256_Curtain_0.0001/',
        #folder_name + 'GRU_512_Curtain_0.0001/',
        folder_name + 'RIMs_400_4_4_4Balls_0.0001_inp_heads_1_templates__enc_1_ver_0_com_True_drop_0.0_att_128/',
        folder_name + 'RIMs_400_4_4_4Balls_0.0001_inp_heads_1_templates__enc_1_ver_2_com_True_drop_0.0_att_128/',
        #folder_name + 'Schema_400_4_4_Curtain_0.0001_inp_heads_1_templates_2_enc_1_ver_2_com_True_drop_0.0_att_85/',
        folder_name + 'Schema_400_4_4_Curtain_0.0001_inp_heads_1_templates_2_enc_1_ver_2_com_True_drop_0.0_att_128/',
        #folder_name + 'Schema_400_4_4_Curtain_0.0001_inp_heads_1_templates_2_enc_1_ver_2_com_True_drop_0.0_att_256/',
        #folder_name + 'Schema_400_4_4_Curtain_0.0001_inp_heads_1_templates_4_enc_1_ver_2_com_True_drop_0.0_att_85/',
        folder_name + 'Schema_400_4_4_Curtain_0.0001_inp_heads_1_templates_4_enc_1_ver_2_com_True_drop_0.0_att_128/',
        #folder_name + 'Schema_400_4_4_Curtain_0.0001_inp_heads_1_templates_4_enc_1_ver_2_com_True_drop_0.0_att_256/',
        #folder_name + 'Schema_400_5_3_Curtain_0.0001_inp_heads_1_templates_2_enc_1_ver_1_com_True_drop_0.0_att_256/',
        #folder_name + 'Schema_400_5_3_Curtain_0.0001_inp_heads_1_templates_4_enc_1_ver_1_com_True_drop_0.0_att_256/',
        #folder_name + 'Schema_400_5_3_Curtain_0.0001_inp_heads_1_templates_4_enc_1_ver_0_com_True_drop_0.0_att_85/',
        #folder_name + 'Schema_400_5_3_Curtain_0.0001_inp_heads_1_templates_4_enc_1_ver_0_com_True_drop_0.0_att_128/',
        #folder_name + 'Schema_400_5_3_Curtain_0.0001_inp_heads_1_templates_4_enc_1_ver_0_com_True_drop_0.0_att_256/',
    ]
    labels_list= [
        'GRU_256',
        'RIMs_v0_128',
        'RIMs_v2_128',
        '2Sch_v2_128',
        '4Sch_v2_128',
        #'GRU_256',
        #'GRU_512',
        #'Schema_2_85',
        #'Schema_2_128',
        #'Schema_2_256',
        #'Schema_4_85',
        #'Schema_4_128',
        #'Schema_4_256',
        #'Sch_2_v2',
        #'Sch_4_v2',
        #'Sch_2_v1',
        #'Sch_4_v1',
        #'Sch_4_v0',
    ]

#TestBlocks_510_6_4_678Balls_0.0007_inp_heads_1_templates_2_enc_6

if model_name_ == '678Balls':
    models_list=[
        #'bb_models/' + 'WithoutDropGRU_256_678Balls_0.0001/',
        #'bb_models/' + 'RIMs_510_6_4_678Balls_0.0001_inp_heads_1_enc_1_version_1/',
        #folder_name + 'SchemaBlocks_510_6_4_678Balls_0.0001_inp_heads_1_templates_2_enc_1_ver_1_com_True/',
        #folder_name + 'SchemaBlocks_510_6_4_678Balls_0.0001_inp_heads_1_templates_4_enc_1_ver_1_com_True/',
        #folder_name + 'SchemaBlocks_510_6_4_678Balls_0.0001_inp_heads_1_templates_6_enc_1_ver_1_com_True/',

        'logs/' + 'RuleBlocks_500_6_4_balls678mass64.h5_0.0001_inp_heads_1_templates_2_enc_1_ver_1_com_True_Sharing_rules_15_rule_time_steps_2/'
    ]
    labels_list= [
                   #'LSTM (256)',
                   #'RIMs',
                   #'SOFF (2 Sch)',
                   #'SOFF (4 Sch)',
                   #'SOFF (6 Sch)',
                   'Rules (15 rules)',
    ]



args.directory = './'
print(args.train_dataset)
dataset = LoadDataset(mode="test",
                      length=args.sequence_length,
                      directory=args.directory,
                      dataset=args.train_dataset)
test_loader = DataLoader(dataset, batch_size=args.batch_size,
                     shuffle=False, num_workers=0)
plot_for_multiple_models(models_list, labels_list, test_loader, model_name_)
