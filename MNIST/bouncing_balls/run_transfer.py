"""Main entry point of the code"""
from __future__ import print_function

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from EncoderDecoder import Model
from model_components import GruState
from argument_parser import argument_parser
from dataset import get_dataloaders
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from box import Box

import os
from os import listdir
from os.path import isfile, join
from utilities.rule_stats import get_stats

set_seed(0)

loss_fn = torch.nn.BCELoss()


def repackage_hidden(ten_):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(ten_, torch.Tensor):
        return ten_.detach()
    else:
        return tuple(repackage_hidden(v) for v in ten_)


def train(model, train_loader, optimizer, epoch, logbook,
          train_batch_idx, args):
    """Function to train the model"""
    model.train()
    gamma = 0.5
    hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
    hidden = GruState(hidden)
    rule_variable = {}
    rule_probability = None
    for batch_idx, data in enumerate(train_loader):
        #import ipdb
        #ipdb.set_trace()
        model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].memory = model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].relational_memory._init_state(torch.ones([args.batch_size, args.hidden_size]))
        #model.myrnn.rimcell[0].bc_lst[0].memory = model.myrnn.rimcell[0].bc_lst[0].relational_memory._init_state(torch.ones([args.batch_size, args.hidden_dim]))

        if args.batch_frequency_to_log_heatmaps > 0 and \
                train_batch_idx % args.batch_frequency_to_log_heatmaps == 0:
            should_log_heatmap = True
        else:
            should_log_heatmap = False
        if args.num_rules > 0:
            model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.reset_activations()
        start_time = time()
        data = data.to(args.device)
        hidden.h = hidden.h.detach()
        optimizer.zero_grad()
        loss = 0
        entropy = 0
        for frame in range(49):
            output, hidden, extra_loss, block_mask, _, entropy_ = model(data[:, frame, :, :, :], hidden)
            entropy += entropy_
            if args.num_rules > 0:
                rule_selections = model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.rule_activation
                variable_selections = model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.variable_activation
                rule_variable = get_stats(rule_selections, variable_selections, rule_variable, args.application_option, args.num_rules, args.num_blocks)

            if should_log_heatmap:
                if frame % args.frame_frequency_to_log_heatmaps == 0:
                    logbook.write_image(
                        img=plt.imshow(block_rules_correlation_matrix,
                                       cmap='hot', interpolation='nearest'),
                        mode="train",
                        step=train_batch_idx,
                        caption=f"{frame}_block_rules_correlation_matrix"
                    )

            target = data[:, frame + 1, :, :, :]
            loss += loss_fn(output, target)

        if args.use_entropy:
            (loss - entropy).backward()
        else:
            (loss).backward()
        total_norm_1 = 0
        total_norm_2 = 0

        if batch_idx % 100 == 0 and batch_idx != 0:
            #print(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.rule_probabilities[-1])
            #print(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.variable_probabilities[-1])
            if args.num_rules > 0:
                for v in rule_variable:
                    print(v, end = ' : ')
                    print(rule_variable[v])
            rule_variable = {}
        if batch_idx % 100 == 1 and args.num_rules > 0:
            model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.gumble_temperature = np.maximum(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.gumble_temperature*np.exp(-0.00003*batch_idx),0.5)
            print('dropped gumble temperature to' + str(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.gumble_temperature))

        #print(model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.rule_embeddings.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if args.num_rules > 0:
            total_norm = 0
            for p in model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.transformer.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm_1 = total_norm ** (1. / 2)
            total_norm = 0
            for p in model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.selecter_2.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm_2 = total_norm ** (1. / 2)

            total_norm = 0
            for p in model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.selecter.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm_3 = total_norm ** (1. / 2)

        """for name, param in model.state_dict().items():
             print(name, param.size())"""
        """ct=0
        for name, param in model.named_parameters():
             if param.requires_grad:
                print("It has gradient", name)
                ct+=1
             else:
               print("it has no gradient", name)
        print(ct)
        ct=0
        total_norm = 0
        for name, p in model.named_parameters():
            if p.grad == None:
                print("something", name)
            else:
                ct+=1
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        print(ct)"""

        optimizer.step()


        train_batch_idx += 1
        metrics = {
            "loss": loss.cpu().item(),
            "mode": "train",
            "batch_idx": train_batch_idx,
            "epoch": epoch,
            "time_taken": time() - start_time,
            #"rule_selector_norm": total_norm_1,
            #"variable_selecter_norm": total_norm_2,
            #"selecter_norm": total_norm_3
        }
        logbook.write_metric_logs(metrics=metrics)

        print("Train loss is: ", loss)
    '''
    try:
        print('rule_probability:')
        print(rule_probability[-1])
        print('variable_probaility:')
        print(variable_probaility[-1])
    except:
        pass
    '''
    return train_batch_idx


@torch.no_grad()
def test(model, test_loader, epoch, transfer_loader, logbook,
         train_batch_idx, args):
    model.eval()
    batch = 0
    losses = []
    start_time = time()

    for data in test_loader:
        data = data.to(args.device)
        loss = 0
        ### Rollout a single trajectory for all frames, using the previous
        if args.should_save_csv and batch == 0:
            for trajectory_to_save in range(4):
                hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
                    #.squeeze().to(args.device)
                hidden = GruState(hidden)
                for frame in range(25):
                    output, hidden, _, _, _, _ = model(data[:, frame, :, :, :], hidden)
                    target = data[:, frame + 1, :, :, :]

                    np.savetxt(f"{args.folder_log}ROP_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')
                for frame in range(25, 49):
                    output, hidden, _, _, _, _  = model(output, hidden)
                    np.savetxt(f"{args.folder_log}ROP_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')

        ### Save all frames from the first 9 trajectories
        hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
        hidden = GruState(hidden)

        for frame in range(49):
            output, hidden, extra_loss, block_mask, _, _ = model(data[:, frame, :, :, :], hidden)
            target = data[:, frame + 1, :, :, :]
            loss = loss_fn(output, target)
            losses.append(loss.cpu().detach().numpy())
        batch += 1
        print("Test loss is: ", loss)

    logbook.write_metric_logs(metrics={
        "loss": np.sum(np.array(losses)).item(),
        "mode": "test",
        "epoch": epoch,
        "batch_idx": train_batch_idx,
        "time_taken": time() - start_time,
    })

    batch = 0
    losses = []
    start_time = time()

    for data in transfer_loader:
        data = data.to(args.device)
        loss = 0
        ### Rollout a single trajectory for all frames, using the previous
        if args.should_save_csv and batch == 0:
            for trajectory_to_save in range(9):
                hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
                hidden = GruState(hidden)
                for frame in range(25):
                    output, hidden, _, _, _, _  = model(data[:, frame, :, :, :], hidden)
                    target = data[:, frame + 1, :, :, :]
                    np.savetxt(f"{args.folder_log}ROPT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROTT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')

                for frame in range(25, 49):
                    output, hidden, _, _, _, _ = model(output, hidden)
                    target = data[:, frame + 1, :, :, :]
                    np.savetxt(f"{args.folder_log}ROPT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROTT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')

        hidden = torch.zeros(1, args.batch_size, args.hidden_size).squeeze().to(args.device)
        hidden = GruState(hidden)

        for frame in range(49):
            output, hidden, extra_loss, block_mask, _, _ = model(data[:, frame, :, :, :], hidden)
            target = data[:, frame + 1, :, :, :]
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
                   str(epoch) + '.csv', np.array(losses), delimiter=',')


def main():
    """Function to run the experiment"""
    args = argument_parser()
    print(args)
    logbook = LogBook(config=args)

    if not args.should_resume:
        # New Experiment
        make_dir(f"{args.folder_log}/model")
        logbook.write_message_logs(message=f"Saving args to {args.folder_log}/model/args")
        torch.save({"args": vars(args)}, f"{args.folder_log}/model/args")

    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    model = setup_model(args=args, logbook=logbook)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.directory = '/home/anirudh/iclr2021/modular_central/bouncing_balls/'
    train_loader, test_loader, transfer_loader = get_dataloaders(args)

    train_batch_idx = 0

    start_epoch = 1
    if args.should_resume:
        start_epoch = args.checkpoint["epoch"] + 1
        logbook.write_message_logs(message=f"Resuming experiment id: {args.id}, from epoch: {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):

        train_batch_idx = train(model=model,
                                train_loader=train_loader,
                                optimizer=optimizer,
                                epoch=epoch,
                                logbook=logbook,
                                train_batch_idx=train_batch_idx,
                                args=args)
        if epoch % 5 == 0 and epoch != 0 and args.num_rules > 0:
            try:
                model.rnn_.gru.myrnn.rimcell[0].bc_lst[0].rule_network.reset_bias()
                print('rule bias reset')
            except:
                pass
        if epoch%50==0:
             print("Epoch number", epoch)
             test(model=model,
                 test_loader=test_loader,
                 epoch=epoch,
                 transfer_loader=transfer_loader,
                 logbook=logbook,
                 train_batch_idx=train_batch_idx,
                 args=args)

        if (args.model_persist_frequency > 0 and epoch % args.model_persist_frequency == 0) or epoch == 5:
            logbook.write_message_logs(message=f"Saving model to {args.folder_log}/model/{epoch}")
            torch.save(model.state_dict(), f"{args.folder_log}/model/{epoch}")


def setup_model(args, logbook):
    """Method to setup the model"""

    model = Model(args)
    if args.should_resume:
        # Find the last checkpointed model and resume from that
        model_dir = f"{args.folder_log}/model"
        latest_model_idx = max(
            [int(model_idx) for model_idx in listdir(model_dir)
             if model_idx != "args"]
        )
        args.path_to_load_model = f"{model_dir}/{latest_model_idx}"
        args.checkpoint = {"epoch": latest_model_idx}

    if args.path_to_load_model != "":

        shape_offset = {}
        for path_to_load_model in args.path_to_load_model.split(","):
            logbook.write_message_logs(message=f"Loading model from {path_to_load_model}")
            _, shape_offset = model.load_state_dict(torch.load(path_to_load_model.strip()),
                                                 shape_offset)


        if not args.should_resume:
            components_to_load = set(args.components_to_load.split("_"))
            total_components = set(["encoders", "decoders", "rules", "blocks"])
            components_to_reinit = [component for component in total_components
                                    if component not in components_to_load]
            for component in components_to_reinit:
                if component == "blocks":
                    logbook.write_message_logs(message="Reinit Blocks")
                    model.rnn_.gru.myrnn.init_blocks()
                elif component == "encoders":
                    logbook.write_message_logs(message="Reinit Encoders")
                    model.init_encoders()
                elif component == "decoders":
                    logbook.write_message_logs(message="Reinit Decoders")
                    model.init_decoders()



    else:
        model = Model(args)

    model = model.to(args.device)

    return model


if __name__ == '__main__':
    main()
