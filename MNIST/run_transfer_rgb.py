"""Main entry point of the code"""
from __future__ import print_function

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from EncoderDecoderRGB import Model
from model_components import GruState
from argument_parser import argument_parser
from dataset import get_dataloaders, get_rgb_dataloaders
from logbook.logbook import LogBook
from utils.util import set_seed, make_dir
from box import Box

import os
from os import listdir
from os.path import isfile, join

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
    hidden = torch.zeros(1, args.batch_size, args.hidden_dim).squeeze().to(args.device)
    hidden = GruState(hidden)
    for batch_idx, (data,data2) in enumerate(train_loader):
        if args.batch_frequency_to_log_heatmaps > 0 and \
                train_batch_idx % args.batch_frequency_to_log_heatmaps == 0:
            should_log_heatmap = True
        else:
            should_log_heatmap = False

        start_time = time()
        data = data.to(args.device)
        data2 = data2.to(args.device)
        hidden.h = hidden.h.detach()
        optimizer.zero_grad()
        loss = 0
        for frame in range(49):
            output, hidden, extra_loss, block_mask = model(data[:, frame, :, :, :], hidden)

            if should_log_heatmap:
                if frame % args.frame_frequency_to_log_heatmaps == 0:
                    logbook.write_image(
                        img=plt.imshow(block_rules_correlation_matrix,
                                       cmap='hot', interpolation='nearest'),
                        mode="train",
                        step=train_batch_idx,
                        caption=f"{frame}_block_rules_correlation_matrix"
                    )
            target = data2[:,frame + 1, :, :, :]
            loss += loss_fn(output, target)

        (loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_batch_idx += 1
        metrics = {
            "loss": loss.cpu().item(),
            "mode": "train",
            "batch_idx": train_batch_idx,
            "epoch": epoch,
            "time_taken": time() - start_time,
        }
        logbook.write_metric_logs(metrics=metrics)

        print("Train loss is: ", loss)

    return train_batch_idx


@torch.no_grad()
def test(model, test_loader, epoch, transfer_loader, logbook,
         train_batch_idx, args):
    model.eval()
    batch = 0
    losses = []
    start_time = time()

    for (data,data2) in test_loader:
        data = data.to(args.device)
        data2 = data2.to(args.device)
        loss = 0
        ### Rollout a single trajectory for all frames, using the previous
        if args.should_save_csv and batch == 0:
            for trajectory_to_save in range(4):
                hidden = torch.zeros(1, args.batch_size, args.hidden_dim).squeeze().to(args.device)
                hidden = GruState(hidden)
                for frame in range(25):
                    output, hidden, _, _ = model(data[:, frame, :, :, :], hidden)
                    target2 = data2[:,frame+1,:,:,:]
                    np.savetxt(f"{args.folder_log}ROP_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target2[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')
                for frame in range(25, 49):
                    output, hidden, _, _  = model(output, hidden)
                    target2 = data2[:,frame+1,:,:,:]
                    np.savetxt(f"{args.folder_log}ROP_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target2[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')

        ### Save all frames from the first 9 trajectories
        hidden =torch.zeros(1, args.batch_size, args.hidden_dim).squeeze().to(args.device)
        hidden = GruState(hidden)

        for frame in range(49):
            output, hidden, extra_loss, block_mask = model(data[:, frame, :, :, :], hidden)
            target2 = data2[:,frame+1,:,:,:]
            loss = loss_fn(output, target2)
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

    for (data, data2) in transfer_loader:
        data = data.to(args.device)
        data2 = data2.to(args.device)
        loss = 0
        ### Rollout a single trajectory for all frames, using the previous
        if args.should_save_csv and batch == 0:
            for trajectory_to_save in range(9):
                hidden = torch.zeros(1, args.batch_size, args.hidden_dim).squeeze().to(args.device)
                hidden = GruState(hidden)
                for frame in range(25):
                    output, hidden, _, _  = model(data[:, frame, :, :, :], hidden)
                    target = data2[:, frame + 1, :, :, :]
                    np.savetxt(f"{args.folder_log}ROPT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROTT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')

                for frame in range(25, 49):
                    output, hidden, _, _ = model(output, hidden)
                    target = data2[:, frame + 1, :, :, :]
                    np.savetxt(f"{args.folder_log}ROPT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               output[trajectory_to_save].cpu()
                               .detach().numpy().flatten(), delimiter=',')
                    np.savetxt(f"{args.folder_log}ROTT_{epoch}_"
                               f"{trajectory_to_save}_{frame}.csv",
                               target[trajectory_to_save].cpu()
                               .numpy().flatten(), delimiter=',')

        hidden = torch.zeros(1, args.batch_size, args.hidden_dim).squeeze().to(args.device)
        hidden = GruState(hidden)

        for frame in range(49):
            output, hidden, extra_loss, block_mask = model(data[:, frame, :, :, :], hidden)
            target = data2[:, frame + 1, :, :, :]
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

    args.directory = './'
    train_loader, test_loader, transfer_loader = get_rgb_dataloaders(args)#get_dataloaders(args)

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
        if epoch%50==0:
             print("Epoch number", epoch)
             test(model=model,
                 test_loader=test_loader,
                 epoch=epoch,
                 transfer_loader=transfer_loader,
                 logbook=logbook,
                 train_batch_idx=train_batch_idx,
                 args=args)

        if args.model_persist_frequency > 0 and epoch % args.model_persist_frequency == 0:
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
