import typing
from collections import namedtuple
import os

import pdg
import uproot
import numpy as np

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.explain import Explanation
from torch_geometric import utils as tg_utils

import time
import networkx as nx
import matplotlib.pyplot as plt

pdg_api = pdg.connect()

Hit = namedtuple("Hit", ["t", "y", "z", "charge", "hit_id"])
Hit.__doc__ = '''\
A hit in the detector

t - The time of the hit
y - The y position of the hit (up and down)
z - The z position of the hit (along the beam direction)
hit_id - Should be a unique identifier for this hit
'''

# Print iterations progress (stolen from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters)
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    '''
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

class Track:
    ''' Hold information about a track, including the Hit objects that make it up. '''
    
    def __init__(self, track_id:int, pid:int, parent_id:int):
        self._id = track_id
        self._pid:int = pid
        self._parent_id:int = pid
        
        self._hits = []
        #self._particle = pdg_api.get_particle_by_mcid(pid)

    def get_hits(self) -> list[Hit]:
        return self._hits

    def add_hit(self, hit:Hit) -> None:
        self._hits.append(hit)

    def get_id(self) -> int:
        return self._id

    def get_pid(self) -> int:
        return self._pid

    def get_particle_type(self) -> str:
        return self._particle

    def get_parent_id(self) -> int:
        return self._parent_id

class Event:
    ''' Hold information about an event, including all tracks that make it up. '''
    
    def __init__(self, event_id:int):
        self._id:int = event_id

        ## mapping between track IDs and the track objects themselves
        self._tracks:typing.Dict[int,Track] = {}

        self._hits = []

    def print(self) -> None:
        print(f'### Event ID {self._id} ###')
        print(f'  - has {len(self._hits)} hits')
        print(f'  - has {len(self._tracks)} tracks:')
        for track in self._tracks.values():
            print(f'    -> ID: {track.get_id()} :: PDG: {track.get_pid()} :: Parent ID: {track.get_parent_id()} :: has {len(track.get_hits())} hits')

    def get_id(self) -> int:
        ''' Get the ID of this event. '''
        
        return self._id

    def add_track(self, track:Track):
        self._tracks[track.get_id()] = track

    def add_hit(self, hit:Hit, track_id_list:typing.List[int]) -> None:
        ''' Add a hit to a track within this event. '''

        ## if this hit has not been seen before, add it to our list
        if not hit in self._hits:
            self._hits.append(hit)

        for track_id in track_id_list:
            ## check that the track actually exists in this event object
            ## might be the case that the track was one that was discarded when making the treemaker
            if not track_id in self._tracks:
                return
                
                #print(f'WARNING: track_id {track_id} does not exist in this event, seems structure not set up properly')
                #print(f'         Happened when trying to add hit {hit}')
                #raise ValueError(f'track_id {track_id} does not exist in this event, seems structure not set up properly')

            else:
                self._tracks[track_id].add_hit(hit)

    def build_input_graph(self, max_dist:float) -> Data:

        node_features = []
        node_positions = []
        edge_indices = [[],[]]

        max_dist_sq = max_dist * max_dist

        ## Build the node feature vectors
        for hit in self._hits:
            node_positions.append( [ hit[0], hit[1], hit[2]] )
            node_features.append( [ hit[3] ] )

        for hit0_id in range(len(self._hits)):
            hit0 = self._hits[hit0_id]
            
            for hit1_id in range(len(self._hits)):
                hit1 = self._hits[hit1_id]

                dist_sq = (hit0[1] - hit1[1]) * (hit0[1] - hit1[1]) + (hit0[2] - hit1[2]) * (hit0[2] - hit1[2]) 

                ## if the hits are closer than the max distance, connect them with an edge
                if dist_sq < max_dist_sq:
                    
                    edge_indices[0].append(hit0[4])
                    edge_indices[1].append(hit1[4])

        edge_index_tensor = torch.tensor(edge_indices)
        node_feature_tensor = torch.tensor(node_features)
        node_position_tensor = torch.tensor(node_positions)

        return Data( edge_index = edge_index_tensor, node_attr = node_feature_tensor, pos = node_position_tensor )

    def build_label_graph(self) -> Data:
        ''' Builds a "label graph" for this event. i.e. the target for the GNN

        The truth information is encoded in the graph as follows:

        The nodes represent hits, the same hits as in the input graph
        Edges represent that two hits belong to a track
        The edges have features associated with them which represent a one-hot encoding of the PDG Id of the track connecting those two hits

        '''

        ## Build the node position vectors
        node_positions = []
        for hit in self._hits:
            node_positions.append( [ hit[0], hit[1], hit[2]] )
        
        edge_labels = []
        edge_indices = [[],[]]
        
        for track in self._tracks.values():

            ## assume hits are ordered by the time that they actually happened?
            for hit0, hit1 in zip(track.get_hits()[1:], track.get_hits()[:-1]):
                edge_indices[0].append(hit0[4])
                edge_indices[1].append(hit1[4])
                
                ## one hot encoding for muonness or electronness
                edge_labels.append(
                    [
                        int(abs(track.get_pid()) == 13),
                        int(abs(track.get_pid()) == 11)
                    ]
                )

        edge_index_tensor = torch.tensor(edge_indices)
        edge_label_tensor = torch.tensor(edge_labels)
        node_position_tensor = torch.tensor(node_positions)

        data =  Data( edge_index = edge_index_tensor, edge_attr = edge_label_tensor, pos = node_position_tensor )

        data.num_nodes = len(self._hits)

        return data


class HATDataMaker:
    
    def __init__(self, filenames:list[str], processed_file_path:str):
        # keep copies of the raw and processed file names
        self._raw_filenames:list[str] = list(filenames)
        self._processed_filenames:list[str] = []

        self._processed_file_path:str = processed_file_path

    @property
    def raw_file_names(self) -> list[str]:
        return self._raw_filenames

    @property
    def processed_file_names(self) -> list[str]:
        return self._processed_filenames

    def process(self) -> None:
        for file_name in self._raw_filenames:
            
            print(f'Processing file {file_name}')
            
            with uproot.open(file_name) as file:
                
                ## get the hit and track TTrees for this file
                file_hits = file["hatdigits"]
                file_tracks = file["hattracks"]

                print("Hit branches: ", file_hits.keys())
                print("Track branches: ", file_tracks.keys())

                # get the track and hit info in a more useful format
                track_df = file_tracks.arrays(
                    ["event", "track", "pdg", "parent", "nhits"], 
                    library="pd"
                )
                hit_df = file_hits.arrays(
                    ["event", "trkid", "time", "y", "z", "qmax", "tmax", "fwhm"], 
                    library="pd"
                )

                print(hit_df)
                print(track_df)
                

                ## for iterating through the above dataframes
                track_iterator = track_df.iterrows()
                hit_iterator = hit_df.iterrows()

                ## initial values
                event_id = 0
                track = next(track_iterator, None)[1]
                hit = next(hit_iterator, None)[1]

                # assume that the last event ID tells us the number of events
                n_events = file_tracks["event"].array()[-1]
                print (f'Has {n_events} events')
                
                while event_id < 10: #True:
                    if (hit is None) and (track is None):
                        break

                    ## make the event
                    event = Event(event_id)

                    ## print out progress bar every few events
                    if(event_id % (int(n_events/100)) == 0 or event_id == n_events - 1):
                        printProgressBar(event_id, n_events - 1)

                    
                    ## first fill up the structure of the event
                    while True:
                        
                        event.add_track(
                            Track(
                                track["track"], 
                                track["pdg"],
                                track["parent"]
                            )
                        )
                        
                        track = next(track_iterator)[1]
                        
                        ## check if we've moved to a new event
                        if track["event"] != event_id:
                            break

                    ## now fill those tracks with hits
                    hit_id = 0

                    while True:
                        
                        event.add_hit(
                            Hit(
                                hit["time"], 
                                hit["y"], 
                                hit["z"], 
                                hit["qmax"],
                                hit_id
                            ), 
                            hit["trkid"]
                        )
                        
                        hit = next(hit_iterator)[1]

                        ## check if we've moved to a new event
                        if hit["event"] != event_id:
                            ## should move to next event
                            event_id = hit["event"]
                            break

                        self._save_graphs(event, event_id)
                        hit_id += 1

    def _save_graphs(self, event:Event, event_id:int) -> None:
        
        data_inputs = event.build_input_graph(250)

        draw_options = {
            'node_color': 'black',
            'node_size': 20,
            'width': 1,
        }

        g = tg_utils.to_networkx(data_inputs, to_undirected=True, remove_self_loops=True)
        nx.draw(g, **draw_options)
        plt.savefig( os.path.join(self._processed_file_path, f"data_graph_{event_id}.png") )
        plt.clf
        
        data_labels = event.build_label_graph()
        
        g = tg_utils.to_networkx( data_labels, to_undirected=True, remove_self_loops=True )
        nx.draw(g, **draw_options)
        plt.savefig( os.path.join(self._processed_file_path, f"data_label_graph{event_id}.png") )
        
        torch.save(data_labels, os.path.join(self._processed_file_path, f'labels_HAT_{event_id}.pt'))
        torch.save(data_inputs, os.path.join(self._processed_file_path, f'data_HAT_{event_id}.pt'))
        