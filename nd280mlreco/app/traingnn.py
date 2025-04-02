## python imports
import os
import glob

import torch.optim.optimizer

## nd20mlreco imports
from nd280mlreco import dataloader

## torch imports
import torch
from torch.nn import Linear, ReLU, Sigmoid
from torch_geometric.nn import Sequential, GCNConv
from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss
from torch_geometric.nn.conv import PointNetConv
from torcheval.metrics import BinaryAccuracy
from torch.utils.tensorboard import SummaryWriter

## torch_geometric imports
from torch_geometric.transforms import RadiusGraph
from torch_geometric import data as tg_data
from torch_geometric import utils as tg_utils
from torch_geometric.loader import DataLoader

## other stuff
from datetime import datetime
import networkx as nx
import math as m
from matplotlib import pyplot as plt
import typing
import argparse

def make_datasets(
        input_files:typing.List[str],
        max_radius:float,
        validation_split:float = 0.7,
    ) -> typing.Tuple[dataloader.HATDataset]:
    """ Greate a dataset from a list of input files

    :param input_files: The list of files to process into a dataset
    :type input_files: typing.List[str]
    :param max_radius: The radius to use for the ball query when building the input graph (i.e. the radius within which nodes will be connected to each other)
    :type max_radius: float
    :param validation_split: The data:validation ratio, defaults to 0.7
    :type validation_split: float, optional
    :return: tuple containing training dataset, validation dataset
    :rtype: typing.Tuple[dataloader.HATDataset]
    """
        
    ## index to split the file names according to the VALIDATION_SPLIT ratio
    n_train = m.floor(validation_split* len(input_files))
    train_files = input_files[:n_train]
    validation_files = input_files[n_train:]

    ## make lists of processed file names
    processed_train_files = []
    processed_validation_files = []

    for i, input_file in enumerate(input_files):
        file_name = os.path.split(input_file)[-1] 

        if i < n_train:
            processed_train_files.append(file_name)
        else:
            processed_validation_files.append(file_name)

    ## check that the size of the lists matches between raw and processed files
    assert(len(processed_train_files) == len(train_files))
    assert(len(processed_validation_files) == len(validation_files))

    print(f"N Train files: {len(train_files)}")
    print(f"N Validation files: {len(validation_files)}")

    ## make the training data set
    train_set = dataloader.HATDataset(
        raw_filenames = train_files, 
        processed_file_names = processed_train_files,
        root = "/home/ewan/ND280-ML/nd280-ml-reco/processed_files/train",
        pre_transform = RadiusGraph(max_radius)
        )

    train_set.process()

    ## make the validation dataset
    validation_set = dataloader.HATDataset(
        raw_filenames = validation_files, 
        processed_file_names = processed_validation_files,
        root = "/home/ewan/ND280-ML/nd280-ml-reco/processed_files/validation",
        pre_transform = RadiusGraph(max_radius)
        )

    validation_set.process()

    return train_set, validation_set

def plot_input(data:dataloader.HATDataset) -> None:
    """ Plots the "input" graph from a data object 

    :param data: The data object to plot from
    :type data: dataloader.HATDataset
    """

    draw_options = {
        'node_color': 'black',
        'width': 0.2,
    }
    
    g = tg_utils.to_networkx(data, to_undirected=True, remove_self_loops=True)
    plt.clf()
    nx.draw(g, data.pos[:, :-1], node_size = data.x / 20.0, **draw_options)
    plt.title("Example data graph")

def plot_truth(data:dataloader.HATDataset) -> None:
    """ Plots the "label" graph from a data object

    :param data: The data object to plot from
    :type data: dataloader.HATDataset
    """
    
    plt.clf()

    label = tg_data.Data(
        edge_index = data.edge_label_index, 
        pos = data.pos,
        x = data.y
    ) 

    g = tg_utils.to_networkx(label, to_undirected=True, remove_self_loops=True)
    plt.clf()

    node_colours = []
    for i in range(len(label.x)):
        if label.x[i][0] and label.x[i][1]:
            node_colours.append('m')
        else:
            if label.x[i][0]:
                node_colours.append('b')
            elif label.x[i][1]:
                node_colours.append('r')
            else:
                node_colours.append('k')

    nx.draw(g, label.pos[:, :-1], node_size = data.x / 20.0, width = 1, node_color = node_colours)

def plot_prediction(data:dataloader.HATDataset, logits, thresh:float = 0.5):
    """ Plot predicted labels on top of the input graph that was used to make the predictions 

    :param data: The data object that was given to the model
    :type data: dataloader.HATDataset
    :param logits: The logits outputted by the model
    :type logits: Array Like
    :param thresh: The threshold to apply to the logits to get a binary prediction
    :type thresh: float
    """

    plt.clf()

    g = tg_utils.to_networkx(data, to_undirected=True, remove_self_loops=True)
    
    node_colours = []
    for i in range(len(logits)):
        if logits[i][0] > thresh and logits[i][1] > thresh:
            node_colours.append('m')
        else:
            if logits[i][0] > thresh:
                node_colours.append('b')
            elif logits[i][1] > thresh:
                node_colours.append('r')
            else:
                node_colours.append('k')

    nx.draw(g, data.pos[:, :-1], node_size = data.x / 20.0, width = 1, node_color = node_colours)


def build_pointnet_model(n_features:int, n_classes:int, internal_nodes:int, n_layers:int) -> torch.nn.Module:
    """ Construct a pointnet++ model (largely based on https://pytorch-geometric.readthedocs.io/en/latest/tutorial/point_cloud.html?highlight=pointnet)

    :param n_features: number of input features in each node of the input graph
    :type n_features: int
    :param n_classes: number possible node classes that the model can predict
    :type n_classes: int
    :param internal_nodes: number of nodes to use in hidden layers
    :type internal_nodes: int
    :param n_layers: number of PointNetConv layars (see https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PointNetConv.html)
    :type n_layers: int
    :return: The model
    :rtype: torch.nn.Module
    """

    local_nn_0 = Sequential("x",
                    [(Linear(n_features + 3, internal_nodes), "x -> x"),
                    ReLU(),
                    Linear(internal_nodes, internal_nodes)],
                )

    local_nn_1 = Sequential("x",
                    [(Linear(internal_nodes+3, internal_nodes), "x -> x"),
                    ReLU(),
                    Linear(internal_nodes, internal_nodes)],
                )

    global_nn = Sequential("x",
                    [(Linear(internal_nodes, internal_nodes), "x -> x"),
                    ReLU(),
                    Linear(internal_nodes, internal_nodes)],
                )

    layers = [ 
        (PointNetConv(local_nn_1, global_nn), 'x, pos, edge_index -> x') 
        for i in range(n_layers) 
    ]

    model = Sequential(
        'x, pos, edge_index', 
        [
            (PointNetConv( 
                local_nn_0,
                global_nn,
            ), 'x, pos, edge_index -> x'),

            *layers,

            (ReLU(inplace=True), "x -> x"),
            (Linear(internal_nodes, n_classes), "x -> x"),
            (Sigmoid(), "x -> x")
        ]
    )

    model.compile()

    return model


def train_one_epoch(
        model:torch.nn.Module, 
        optimizer:torch.optim.Optimizer,
        device:torch.device,
        train_loader:DataLoader, 
        epoch_index:int, 
        tb_writer:SummaryWriter=None, 
        print_freq:int=None
    ) -> None:

    """ Train a model for a single epoch

    :param model: The model to be trained
    :type model: torch.nn.Module
    :param optimizer: The optimiser to use to train the model
    :type optimizer: torch.optim.Optimizer
    :param device: The device to use for crunchin' the numbers
    :type device: torch.device
    :param train_loader: the dataloader that provides the training data
    :type train_loader: DataLoader
    :param epoch_index: the index of this epoch (used for bookkeeping purposes)
    :type epoch_index: int
    :param tb_writer: Summary writer, epoch metrics will be written to this if provided, defaults to None
    :type tb_writer: SummaryWriter, optional
    :param print_freq: frequency to print out loss and accuracy, defaults to None
    :type print_freq: int, optional
    :return: The final loss
    :rtype: float
    """

    running_loss = 0.0
    bin_accuracy = BinaryAccuracy()

    n_batches = 0

    for i, data in enumerate(train_loader):

        data.to(device)
        
        optimizer.zero_grad()

        # Make predictions for this batch
        feat = data.x.to(torch.float32) / 1000.0
        pos = data.pos.to(torch.float32) / 2500.0

        outputs = model(feat, pos, data.edge_index)

        # Compute the loss and its gradients
        loss = CrossEntropyLoss()(outputs, data.y.to(torch.float32))
        bin_accuracy.update(outputs.flatten(), data.y.to(torch.float32).flatten())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        if print_freq is not None and i %print_freq == 0:
            print(f"    - step {i}: loss: {loss.item()}, acc: {bin_accuracy.compute()}")

        n_batches += 1

    last_loss = running_loss / n_batches # loss per batch
    print(f'  Train loss: {last_loss}: accuracy: {bin_accuracy.compute()}')
    tb_x = epoch_index * len(train_loader) + n_batches + 1

    if tb_writer is not None:
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        tb_writer.add_scalar('binary_accuracy/train', bin_accuracy.compute(), tb_x)

    return last_loss 


def validate_one_epoch(
        model:torch.nn.Module, 
        device:torch.device,
        val_loader:DataLoader, 
        epoch_index:int, 
        tb_writer:SummaryWriter=None
    ) -> float:
    """ Run validation for a single epoch

    :param model: The model to validate
    :type model: torch.nn.Module
    :param device: The device to use for crunchin' the numbers
    :type device: torch.device
    :param val_loader: the dataloader that provides the validation data
    :type val_loader: DataLoader
    :param epoch_index: the index of this epoch (used for bookkeeping)
    :type epoch_index: int
    :param tb_writer: Summary writer, validation metrics will be written to this if provided, defaults to None
    :type tb_writer: SummaryWriter, optional
    :return: The final loss
    :rtype: float
    """

    running_loss = 0.0
    bin_accuracy = BinaryAccuracy()
    
    for i, data_val in enumerate(val_loader):

        data_val.to(device)

        # Make predictions for this batch
        outputs = model(data_val.x.to(torch.float32) / 1000.0, data_val.pos.to(torch.float32) / 2500.0, data_val.edge_index)

        # Compute the loss and its gradients
        loss = CrossEntropyLoss()(outputs, data_val.y.to(torch.float32))
        bin_accuracy.update(outputs.flatten(), data_val.y.to(torch.float32).flatten())

        # Gather data and report
        running_loss += loss.item()

    last_loss = running_loss / i # loss per batch
    print(f'  Validation loss: {last_loss}: accuracy: {bin_accuracy.compute()}')
    tb_x = epoch_index * len(val_loader) + i + 1

    if tb_writer is not None:
        tb_writer.add_scalar('Loss/validation', last_loss, tb_x)
        tb_writer.add_scalar('binary_accuracy/validation', bin_accuracy.compute(), tb_x)
    
    return last_loss

def run():

    parser = argparse.ArgumentParser(description="Train a graph neural network to classify hits in a HAT dataset.")
    parser.add_argument("--input-files", "-i", required=True, type=str, action="append", nargs='+', help="The processed gnn files that make up the dataset")
    parser.add_argument("--output-dir", "-o", required=True, type=str, help="The directory to output anything that falls out of the training process")
    parser.add_argument("--epochs", "-e", required=True, type=int, help="The number of epochs to train for")
    parser.add_argument("--start-from", "-s", required=False, type=str, default=None, help="Checkpoint to start the training from")
    parser.add_argument("--print-freq", "-p", required=False, type=int, default=None, help="print loss and validation info every n batches when training")
    parser.add_argument("--save-examples", required=False, type=bool, default=False, help="Whether or not to save some example plots")
    parser.add_argument("--val-split", "-v", required=False, type=float, default=0.7, help="The dat:validation split to use when dividing the dataset")
    parser.add_argument("--max-radius", "-r", required=False, type=float, default=50.0, help="The radius to use when performing the ball query when building the graph dataset")

    args = parser.parse_args()

    EXAMPLE_INDEX = 0
    
    print( args.input_files[0])
    input_files = args.input_files[0]
    input_files.sort()

    print(input_files)

    ## make the datasets
    train_set, validation_set = make_datasets(input_files, args.max_radius, args.val_split)


    if args.save_examples:
        example = train_set.get(EXAMPLE_INDEX)

        plot_input(example)
        plt.savefig( os.path.join(args.output_dir, f"data_graph_example.png") )

        plot_truth(example)
        plt.title("Example label graph")
        plt.savefig( os.path.join(args.output_dir, f"label_graph_example.png") )


    model = build_pointnet_model(1, 4, 150, 3)
    optimizer = Adam(model.parameters(), lr = 0.001)

    train_loader = DataLoader(train_set, batch_size=64)
    val_loader = DataLoader(validation_set)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir = 'runs/GCN_trainer_{}'.format(timestamp))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_epoch = -1

    ## if user specified, start from a previous checkpoint
    if args.start_from is not None:
        checkpoint = torch.load(args.start_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']


    final_epoch = 0
    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(start_epoch + 1, start_epoch + args.epochs):

        print(f'#####  Epoch {epoch} #####')
        
        final_epoch = epoch
        final_train_loss = train_one_epoch(model, optimizer, device, train_loader, epoch, writer, print_freq = args.print_freq)
        final_val_loss = validate_one_epoch(model, device, val_loader, epoch, writer)

    ## save the trained model

    if not os.path.exists( os.path.join(args.output_dir, f"checkpoints/") ):
        os.makedirs(os.path.join(args.output_dir, f"checkpoints/") )

    torch.save(
        {
            'epoch': final_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
        }, os.path.join(args.output_dir, f"checkpoints/{timestamp}"))

    if args.save_examples:
        VAL_EXAMPLE_INDEX = 0

        outputs = model(
            validation_set[VAL_EXAMPLE_INDEX].x.to(torch.float32), 
            validation_set[VAL_EXAMPLE_INDEX].pos.to(torch.float32), 
            validation_set[VAL_EXAMPLE_INDEX].edge_index
        )

        plot_prediction(validation_set[VAL_EXAMPLE_INDEX], outputs)
        plt.savefig( os.path.join(args.output_dir, f"prediction_example.png") )

if __name__ == "__main__":
    run()

