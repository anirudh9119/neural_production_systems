"""Main entry point of the code"""
from __future__ import print_function

from time import time

import matplotlib
matplotlib.use("PS")
import matplotlib.pyplot as plt
import numpy as np
import torch

import bootstrap
#from model_components import GruState
from utils.util import set_seed
from argument_parser import argument_parser

from utilities.rule_stats import get_stats

loss_fn = torch.nn.BCELoss()
def rule_transform_stats(rule_selections, transforms, rule_to_transform, args):
    for b in range(rule_selections[0].shape[0]):
        for t in range(len(rule_selections)):
            if transforms[b][t] not in rule_to_transform:
                rule_to_transform[transforms[b][t]] = {r:0 for r in range(args.num_rules)}
            rule_to_transform[transforms[b][t]][rule_selections[t][b]] += 1
    return rule_to_transform



def train_loop(model, train_loader, optimizer, epoch, logbook,
               train_batch_idx, args):
    """Function to train the model"""
    model.train()
    #hidden = model.rnn_.gru.myrnn.init_hidden(args.batch_size)[0].squeeze().to(args.device)
    #enc_dec_parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())

    #optimizer1 = torch.optim.Adam(enc_dec_parameters, lr = args.lr)
    #optimizer2 = torch.optim.Adam(model.parameters(), lr = args.lr)

    hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
    #hidden = GruState(hidden)
    rule_variable = {}
    rule_probability = None
    rule_to_transform = {}
    for batch_idx, data in enumerate(train_loader):
        if args.num_rules > 0:
            model.rule_network.reset_activations()
        if args.batch_frequency_to_log_heatmaps > 0 and \
                train_batch_idx % args.batch_frequency_to_log_heatmaps == 0:
            should_log_heatmap = True
        else:
            should_log_heatmap = False
        start_time = time()
        inp_data = data[0].to(args.device)

        inp_transforms = data[2].to(args.device)
        


        #hidden.h = hidden.h.detach()
        optimizer.zero_grad()
        #loss = 0
        #losses = []
        #kl_loss = 0
        #activation_frequency_list = []
        inp_data_ = inp_data
        inp_data = inp_data_[:, :-1]
        tar_data = inp_data_[:, 1:]

        
        b, t, c, w, h = inp_data.size()
        inp_data = inp_data.reshape(b * t, c, w, h)
        tar_data = tar_data.reshape(b * t, c, w, h)
        inp_transforms = inp_transforms.reshape(b * t, -1)
        

        #for frame in range(inp_data.size(1) - 1):
        #    output, hidden, extra_loss, block_mask, _, _ = model(inp_data[:, frame, :, :, :], hidden, inp_transforms[:, frame, :])
        #    target = inp_data[:, frame + 1, :, :, :]

        #    loss += loss_fn(output, target)
        #    losses.append(loss.cpu().detach().numpy())

        out = model(inp_data, None, inp_transforms)
        #out_1 = out[:, 0, :, :].unsqueeze(1) # inp reconstruction
        #out_2 = out[:, 1, :, :].unsqueeze(1) # transform
        loss = loss_fn(out, tar_data)


        entire_trasnform_list = []
        for t in data[1]:
            entire_trasnform_list.extend(t)

        if args.num_rules > 0:
            rule_selections = model.rule_network.rule_activation
            variable_selections = model.rule_network.variable_activation
            rule_variable = get_stats(rule_selections, variable_selections, rule_variable, args.application_option, args.num_rules, args.num_blocks)
            rule_to_transform = rule_transform_stats(rule_selections, entire_trasnform_list, rule_to_transform, args)
        
        #if should_log_heatmap:
        #    logbook.write_image(
        #        img=plt.imshow(block_rules_correlation_matrix,
        #                       cmap='hot', interpolation='nearest'),
        #        mode="train",
        #        step=train_batch_idx,
        #        caption=f"0_block_rules_correlation_matrix"
        #    )
        

        (loss).backward()
        total_norm = 0
        #for p in model.rule_network.dummy_rule_selector.parameters():
        #    if p.grad is not None:
        #        param_norm = p.grad.data.norm(2)
        #        total_norm += param_norm.item() ** 2
        #total_norm_1 = total_norm ** (1. / 2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if batch_idx % 100 == 0 and batch_idx != 0:
             #print(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.rule_probabilities[-1])
             #print(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.variable_probabilities[-1])
             if args.num_rules > 0:
                for v in rule_variable:
                    print(v, end = ' : ')
                    print(rule_variable[v])
                for r in rule_to_transform:
                    print(r, end = ' : ')
                    print(rule_to_transform[r])


        train_batch_idx += 1
        metrics = {
            "loss": loss.cpu().item(),
            # "kl_loss": kl_loss.cpu().item(),
            "mode": "train",
            "batch_idx": train_batch_idx,
            "epoch": epoch,
            "time_taken": time() - start_time,
            #'selector_norm': str(total_norm_1)
        }
        #logbook.write_metric_logs(metrics=metrics)

        #print("Train loss is: ", loss)

    return train_batch_idx


@torch.no_grad()
def test_loop(model, test_loader, epoch, transfer_loader, logbook,
              train_batch_idx, args):
    model.eval()
    batch = 0
    losses = []
    activation_frequency_list = []
    start_time = time()
    rule_variable = {}
    rule_to_transform = {}
    for i, data in enumerate(test_loader):
        if args.num_rules > 0:
            model.rule_network.reset_activations()
        inp_data = data[0].to(args.device)
        #target = target[:,0]
        inp_transforms = data[2].to(args.device)
        loss = 0
        kl_loss = 0

        hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
        #hidden = GruState(hidden)
        for frame in range(inp_data.size(1) - 1):
            #output, hidden, extra_loss, block_mask, _, _ = model(inp_data[:, frame, :, :, :], hidden, inp_transforms[:, frame, :])
            output = model(inp_data[:, frame, :, :, :], None, inp_transforms[:,frame, :])
            target = inp_data[:, frame + 1, :, :, :]
            #print(target.shape)
            #print(output.shape)
            loss = loss_fn(output, target)
            losses.append(loss.cpu().detach().numpy())
        losses.append(loss.cpu().detach().numpy())
        batch += 1
        if args.num_rules > 0:
            rule_selections = model.rule_network.rule_activation
            variable_selections = model.rule_network.variable_activation
            rule_probs = model.rule_network.rule_probabilities
            if i % 100 == 0:
                print(rule_probs)
            rule_variable = get_stats(rule_selections, variable_selections, rule_variable, args.application_option, args.num_rules, args.num_blocks)
            #print(len(rule_selections))
            #print(len(data[1][0]))
            rule_to_transform = rule_transform_stats(rule_selections, data[1], rule_to_transform, args)

        print("Test loss is: ", loss)


    logbook.write_metric_logs(metrics={
        "loss": np.sum(np.array(losses)).item(),
        "mode": "test",
        "epoch": epoch,
        "batch_idx": train_batch_idx,
        "time_taken": time() - start_time,
    })
    if args.num_rules > 0:
        print(rule_probs)

        for v in rule_variable:
            print(v, end = ' : ')
            print(rule_variable[v])
        for r in rule_to_transform:
            print(r, end = ' : ')
            print(rule_to_transform[r])

    """batch = 0
    losses = []
    activation_frequency_list = []
    start_time = time()

    for data in transfer_loader:
        inp_data = data[0].to(args.device)
        inp_transforms = data[2].to(args.device)
        #target = target[:, 0]
        loss = 0
        kl_loss = 0
        hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
        #hidden = GruState(hidden)

        
        for frame in range(inp_data.size(1) - 1):
            #output, hidden, extra_loss, block_mask, _, _ = model(inp_data[:, frame, :, :, :], hidden, inp_transforms[:, frame, :])
            output = model(inp_data[:, frame, :, :, :], None, inp_transforms[:, frame, :])    
            target = inp_data[:, frame + 1, :, :, :]
            #print(target.shape)
            #print(output.shape)
            loss = loss_fn(output, target)
            losses.append(loss.cpu().detach().numpy())
        batch += 1
        print("Transfer loss is: ", loss)

    logbook.write_metric_logs(metrics={
        "loss": np.sum(np.array(losses)).item(),
        "mode": "transfer",
        "epoch": epoch,
        "batch_idx": train_batch_idx,
        "time_taken": time() - start_time,
    })
    if args.should_save_csv:
        np.savetxt(args.folder_log + 'losses_' +
                   str(epoch) + '.csv', np.array(losses), delimiter=',')"""


if __name__ == '__main__':
    bootstrap.main(train_loop=train_loop,
                   test_loop=test_loop)
