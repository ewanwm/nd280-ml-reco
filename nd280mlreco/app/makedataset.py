import torch
import torch_geometric

import nd280mlreco
from nd280mlreco import datamaker

import argparse
import sys

import logging

def run():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger('fsspec').setLevel(logging.WARN)

    ## Parse user arguments
    parser = argparse.ArgumentParser(description="Build a graph dataset from a file created by MLTTreeMaker in the ND280Software.")
    parser.add_argument("--input-file", "-i", required=True, type=str, help="The ROOT file to create a dataset from. Should be the output of MLTTreeMaker")
    parser.add_argument("--output-dir", "-o", required=True, type=str, help="The directory to output the dataset files to *WARNING* There may be a LOT of them")
    parser.add_argument("--start-index", required=False, type=int, default=0, help="The event index to start from")
    parser.add_argument("--stop-index", required=False, type=int, default=99999999, help="The event index to stop at")

    args = parser.parse_args()
    print(parser)

    ## Construct the datamaker object
    hat_datamaker = datamaker.HATDataMaker(
        filenames = [args.input_file],
        processed_file_path = args.output_dir,
        start = args.start_index,
        stop = args.stop_index,
        log_level=logging.INFO
    )

    ## run it
    hat_datamaker.process()