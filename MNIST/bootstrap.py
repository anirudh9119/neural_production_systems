"""Main entry point of the code"""
from __future__ import print_function

from os import listdir

import torch
import random
import numpy as np
import os
from EncoderDecoder import MNISTModel
from argument_parser import argument_parser
from crl.dataset import get_dataloaders as crl_dataloaders
from logbook.logbook import LogBook
from utils.util import make_dir


def repackage_hidden(ten_):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(ten_, torch.Tensor):
        return ten_.detach()
    else:
        return tuple(repackage_hidden(v) for v in ten_)


def main(train_loop, test_loop):
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
    if args.num_rules > 0:
        model.rule_network.share_key_value = args.share_key_value

    # logbook.watch_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.directory = '.'

    train_loader, test_loader, transfer_loader = crl_dataloaders(num_transforms = args.num_transforms, transform_length = args.transform_length,  batch_size = args.batch_size, color = args.color)

    train_batch_idx = 0

    start_epoch = 1
    if args.should_resume:
        start_epoch = args.checkpoint["epoch"] + 1
        logbook.write_message_logs(message=f"Resuming experiment id: {args.id}, from epoch: {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        test_loop(model=model,
                  test_loader=test_loader,
                  epoch=epoch,
                  transfer_loader=transfer_loader,
                  logbook=logbook,
                  train_batch_idx=train_batch_idx,
                  args=args)
        train_batch_idx = train_loop(model=model,
                                     train_loader=train_loader,
                                     optimizer=optimizer,
                                     epoch=epoch,
                                     logbook=logbook,
                                     train_batch_idx=train_batch_idx,
                                     args=args)
        print("Epoch number", epoch)

        if args.model_persist_frequency > 0 and epoch % args.model_persist_frequency == 0:
            logbook.write_message_logs(message=f"Saving model to {args.folder_log}/model/{epoch}")
            torch.save(model.state_dict(), f"{args.folder_log}/model/{epoch}")


def setup_model(args, logbook):
    """Method to setup the model"""
    print('Setting seed to ' + str(args.seed))
    
    #seed = args.seed
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    model = MNISTModel(args)


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
                elif component == "rules":
                    logbook.write_message_logs(message="Reinit Rules")
                    model.rnn_.gru.myrnn.init_rules()
                elif component == "encoders":
                    logbook.write_message_logs(message="Reinit Encoders")
                    model.init_encoders()
                elif component == "decoders":
                    logbook.write_message_logs(message="Reinit Decoders")
                    model.init_decoders()


    else:
        model = MNISTModel(args)

    model = model.to(args.device)

    return model

