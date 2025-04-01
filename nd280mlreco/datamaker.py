import typing
from collections import namedtuple
from collections.abc import Callable
from typing import NamedTuple, Any
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

from itertools import groupby, product

import logging

pdg_api = pdg.connect()

class Hit(NamedTuple):
    '''
    A hit in the detector

    attributes:
        t - The time of the hit
        y - The y position of the hit (up and down)
        z - The z position of the hit (along the beam direction)
        row - The row of the pad hit in the HAT
        col - The column of the pad hit in the HAT
        charge - The charge deposited in the pad
        id - Should be a unique identifier for this hit
        tracks - The IDs of the tracks that contributed to this hit
    '''

    t:float
    y:float
    z:float
    row:int
    col:int
    charge:float
    id:int
    tracks:list[int]

    def __hash__(self):
        return hash(self.t) * hash(self.y) * hash(self.z) * hash(self.row) * hash(self.col) * hash(self.charge) * hash(self.id)

class HATCluster:
    ''' Describes a collection of hits in a HAT
    '''
    def __init__(self, hits:list[Hit], cluster_id:int):
        self.charge = self._get_charge_from_hits(hits)
        self.t = self._get_time_from_hits(hits)
        self.y = self._get_y_from_hits(hits)
        self.z = self._get_z_from_hits(hits)
        self.tracks = self._get_tracks_from_hits(hits)
        self.n_hits = len(hits)
        self.id = cluster_id

    def _get_charge_from_hits(self, hits:list[Hit]):
        c = 0.0

        for hit in hits:
            c += hit.charge

        return c

    def _get_time_from_hits(self, hits:list[Hit]):
        if(self.charge is None):
            raise ValueError("Must do get_charge_from_hits() first")
        
        t = 0.0
        for hit in hits:
            t += hit.t * hit.charge / self.charge

        return t
    
    def _get_y_from_hits(self, hits:list[Hit]):
        if(self.charge is None):
            raise ValueError("Must do get_charge_from_hits() first")
        
        y= 0.0
        for hit in hits:
            y += hit.y * hit.charge / self.charge
        
        return y
    
    def _get_z_from_hits(self, hits:list[Hit]):
        if(self.charge is None):
            raise ValueError("Must do get_charge_from_hits() first")
        
        z = 0.0
        for hit in hits:
            z += hit.z * hit.charge / self.charge
        
        return z
    
    def _get_tracks_from_hits(self, hits:list[Hit]):
        tracks = set()

        for hit in hits:
            for track in hit.tracks:
                tracks.add(track)

        return tracks

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

def HATManhattan(hit1:Hit, hit2:Hit, use_time=False):
    ''' Get the Manhattan distance between two HAT hits.

    This is abs(hit1["col"] - hit2["col"]) + abs(hit1["row"] - hit2["row"]) 
    '''

    ret = abs(int(hit1.row) - int(hit2.row)) + abs(int(hit1.col) - int(hit2.col))
    if use_time:
        ret += abs(int(hit1.t) - int(hit2.t))

    return ret

class Track:
    ''' Hold information about a track, including the Hit objects that make it up. '''
    
    def __init__(self, track_id:int, pid:int, parent_id:int):
        self._id = track_id
        self._pid:int = pid
        self._parent_id:int = pid
        
        self._hits = []
        self._clusters = []
        #self._particle = pdg_api.get_particle_by_mcid(pid)

    def get_hits(self) -> list[Hit]:
        return self._hits

    def add_hit(self, hit:Hit) -> None:
        self._hits.append(hit)

    def get_clusters(self) -> list[HATCluster]:   
        return self._clusters

    def add_cluster(self, cluster:HATCluster) -> None:

        if self._clusters is None:
            self._clusters = []

        self._clusters.append(cluster)

    def get_id(self) -> int:
        return self._id

    def get_pid(self) -> int:
        return self._pid

    def get_particle_type(self) -> str:
        return self._particle

    def get_parent_id(self) -> int:
        return self._parent_id
    
    def print(self, log_cmd:Callable=print) -> int:
        log_cmd(f'### track {self._id} ###')
        log_cmd(f'  - PID: {self._pid}')
        log_cmd(f'  - parent ID: {self._parent_id}')
        log_cmd(f'  - has {len(self._hits)} hits:')
        for hit in self._hits:
            log_cmd("    ->",hit)

class Event:
    ''' Hold information about an event, including all tracks that make it up. '''
    
    def __init__(self, event_id:int, logger:logging.Logger):
        self._id:int = event_id

        ## mapping between track IDs and the track objects themselves
        self._tracks:typing.Dict[int,Track] = {}

        self._hits = []
        self._clusters = None

        self._logger = logger

    def print(self, log_cmd:Callable=print) -> None:
        log_cmd(f'### Event ID {self._id} ###')
        log_cmd(f'  - has {len(self._hits)} hits')
        log_cmd(f'  - has {len(self._tracks)} tracks:')
        for track in self._tracks.values():
            log_cmd(f'    -> ID: {track.get_id()} :: PDG: {track.get_pid()} :: Parent ID: {track.get_parent_id()} :: has {len(track.get_hits())} hits')

    def get_id(self) -> int:
        ''' Get the ID of this event. '''
        
        return self._id

    def add_track(self, track:Track):
        self._tracks[track.get_id()] = track

    def get_tracks(self) -> list[Hit]:
        return self._hits

    def get_clusters(self) -> list[HATCluster]:
        return self._clusters

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

    def build_clusters(self, max_manhattan_dist=1, min_charge_thresh=0.0) -> None:
        ''' Construct clusters of hits.
        
        Connects all cells which are connected with a Manhattan distance less than the max specified distance
        '''

        self._clusters = []

        # Group Adjacent Coordinates
        # Using product() + groupby() + list comprehension
        man_tups = [sub for sub in product(self._hits, repeat = 2)
                                                if HATManhattan(*sub, use_time=True) <= max_manhattan_dist]
        
        res_dict = {ele: {ele} for ele in self._hits}
        for tup1, tup2 in man_tups:
            for tup3 in res_dict[tup2]:
                res_dict[tup1] |= res_dict[tup3]
                res_dict[tup3] = res_dict[tup1]
        
        clusters = [[*next(val)] for key, val in groupby(
                sorted(res_dict.values(), key = id), id)]
        
        self._logger.debug("The found clusters: ")

        good_clusters:int = 0
        for cluster_hits in clusters:
            self._logger.debug(f"  {[f'({c.t}, {c.col}, {c.row} :: {c.charge}) ' for c in cluster_hits]}")
            cluster_obj = HATCluster(cluster_hits, -999)

            if cluster_obj.charge >= min_charge_thresh:
                cluster_obj.id = good_clusters
                self._clusters.append(cluster_obj)

                for track_id in cluster_obj.tracks:
                    if track_id in self._tracks:
                        self._tracks[track_id].add_cluster(cluster_obj)

                good_clusters += 1


    def build_clusters_time_slices(self, max_manhattan_dist=1) -> None:
        ''' Construct clusters of hits which occur at the same time, in connected clusters of pads
        
        .. warning::
            This will change the order of hits in the event
        '''

        ## Sort the hits in this track by the hit time
        ## *WARNING* This *will* change the order of the hits inside the track object
        self._hits.sort(key=lambda hit: hit[0])

        hit_iter = iter(self._hits)
        hit = next(hit_iter, None)
        hits = []
        cluster_time = hit[0]

        while True:
            
            if hit is None:
                break
            
            ## if we've hit a new time step, make the clusters then move on
            if hit[0] != cluster_time:
                
                # Group Adjacent Coordinates
                # Using product() + groupby() + list comprehension
                man_tups = [sub for sub in product(hits, repeat = 2)
                                                        if HATManhattan(*sub) <= max_manhattan_dist]
                
                res_dict = {ele: {ele} for ele in hits}
                for tup1, tup2 in man_tups:
                    for tup3 in res_dict[tup2]:
                        res_dict[tup1] |= res_dict[tup3]
                        res_dict[tup3] = res_dict[tup1]
                
                clusters = [[*next(val)] for key, val in groupby(
                        sorted(res_dict.values(), key = id), id)]
                
                self._logger.debug("The found clusters: ")
                for cluster_hits in clusters:
                    self._logger.debug("  ", [f'({c[3]}, {c[4]}) ' for c in cluster_hits])
                    cluster_obj = HATCluster(cluster_hits)
                    self._clusters.append(cluster_obj)

                    for track_id in cluster_obj.tracks:
                        if track_id in self._tracks:
                            self._tracks[track_id].add_cluster(cluster_obj)

                ## reset stuff
                cluster_time = hit[0]
                hits = []
                continue

            ## otherwise we add the hit to the list of hits to consider
            hits.append(hit)

            hit = next(hit_iter, None)

    def build_input_graph(self) -> Data:

        node_features = []
        node_positions = []
        edge_indices = [[],[]]

        # If clusters haven't been build, use hits
        if self._clusters is None:
            clusters_or_hits = self._hits
        else:
            clusters_or_hits = self._clusters

        ## Build the node feature vectors
        for cluster in clusters_or_hits:
            node_positions.append( [ cluster.t, cluster.y, cluster.z] )
            node_features.append( [ cluster.charge ] )

        edge_index_tensor = torch.tensor(edge_indices)
        node_feature_tensor = torch.tensor(node_features)
        node_position_tensor = torch.tensor(node_positions)

        label_graph = self.build_label_graph()

        data = Data( 
            edge_index = edge_index_tensor, 
            x = node_feature_tensor, 
            pos = node_position_tensor, 
            edge_label_index = label_graph.edge_index, 
            y = label_graph.x 
        )

        return data

    def plot_event(self, event_id:int, path="") -> None:

        plt.clf()

        plt.subplot(211)

        plt.xlim(-2800,-800)
        plt.ylim(450, 1200)

        plt.title("Hits")

        for hit in self._hits:

            colour = "ko"
            plt.plot(hit.z, hit.y, colour, ms=0.001 * hit.charge)


        for track in self._tracks.values():
            for hit in track.get_hits():

                colour = "ko"
                if int(abs(track.get_pid()) == 13):
                    colour = "bo"
                elif int(abs(track.get_pid()) == 11):
                    colour = "ro"

                plt.plot(hit.z, hit.y, colour, ms=0.001 * hit.charge)


        plt.subplot(212)

        plt.xlim(-2800,-800)
        plt.ylim(450, 1200)

        plt.title("Clusters")

        ## need to check if clusters have been made
        clusters_or_hits = []
        use_clusters = False
        if self._clusters is None:
            clusters_or_hits = self._hits
        else:
            clusters_or_hits = self._clusters
            use_clusters = True

        for cluster in clusters_or_hits:

            colour = "ko"
            plt.plot(cluster.z, cluster.y, colour, ms=0.001 * cluster.charge)


        for track in self._tracks.values():

            track_clusters_or_hits = []
            if use_clusters:
                track_clusters_or_hits = track.get_clusters()
            else:
                track_clusters_or_hits = track.get_hits()

            for cluster in track_clusters_or_hits:

                colour = "ko"
                if int(abs(track.get_pid()) == 13):
                    colour = "bo"
                elif int(abs(track.get_pid()) == 11):
                    colour = "ro"

                plt.plot(cluster.z, cluster.y, colour, ms=0.001 * cluster.charge)

        plt.savefig(os.path.join(path, f'event_{event_id}.png'), dpi=500)

    def build_label_graph(self) -> Data:
        ''' Builds a "label graph" for this event. i.e. the target for the GNN

        The truth information is encoded in the graph as follows:

        The nodes represent hits, the same hits as in the input graph
        Edges represent that two hits belong to a track
        The edges have features associated with them which represent a one-hot encoding of the PDG Id of the track connecting those two hits

        '''

        self._logger.debug(f"Building label graph for event {self._id}")

        # if clusters haven't been constructed, just use hits
        use_clusters = False
        clusters_or_hits = []
        if self._clusters is None:
            clusters_or_hits = self._hits
        else:
            clusters_or_hits = self._clusters
            use_clusters = True

        ## Build the node position vectors
        node_positions = []
        node_labels = []
        for cluster in clusters_or_hits:
            node_positions.append( [ cluster.t, cluster.y, cluster.z] )
            
            node_labels.append(
                    [
                        0,
                        0
                    ]
                )
        
        edge_labels = []
        edge_indices = [[],[]]
        
        for track in self._tracks.values():

            self._logger.debug(f"  track {track._id} has clusters:")

            track_clusters_or_hits = []

            if use_clusters:
                track_clusters_or_hits = track.get_clusters()
            else:
                track_clusters_or_hits = track.get_hits()

            ## Sort the clusters in this track by the cluster time
            track_clusters_or_hits.sort(key=lambda cluster: cluster.t)

            for cluster in track_clusters_or_hits:
                node_labels[cluster.id] = [
                    int(abs(track.get_pid()) == 13 or node_labels[cluster.id][0]) ,
                    int(abs(track.get_pid()) == 11 or node_labels[cluster.id][1])
                ]

                edge_indices[0].append(cluster.id)
                edge_indices[1].append(cluster.id)
        
        edge_index_tensor = torch.tensor(edge_indices)
        #edge_label_tensor = torch.tensor(edge_labels)
        node_label_tensor = torch.tensor(node_labels)
        node_position_tensor = torch.tensor(node_positions)

        data = Data( edge_index = edge_index_tensor, pos = node_position_tensor, x = node_label_tensor )

        data.num_nodes = len(clusters_or_hits)

        return data


class HATDataMaker:
    ''' Creates ML data files for the HAT from a ML TTree made witht the MLTTreeMaker app in ND280 detResponseSim 
    '''
    
    def __init__(self, filenames:list[str], processed_file_path:str, start=None, stop=None, log_level:int=logging.INFO):
        # keep copies of the raw and processed file names
        self._raw_filenames:list[str] = list(filenames)
        self._processed_filenames:list[str] = []

        self._processed_file_path:str = processed_file_path

        self._process_start:int = start
        self._process_stop:int = stop

        self._logger = logging.getLogger("HATDataMaker")
        self._logger.setLevel(log_level)

    @property
    def raw_file_names(self) -> list[str]:
        ''' The name of the raw input root files
        '''
        return self._raw_filenames

    @property
    def processed_file_names(self) -> list[str]:
        return self._processed_filenames

    def process(self) -> None:
        ''' "process" the input raw root files and turn it into lovely machine learning input files
        '''
        for file_name in self._raw_filenames:
            
            self._logger.info(f'Processing file {file_name}')
            
            with uproot.open(file_name) as file:
                
                ## get the hit and track TTrees for this file
                file_hits = file["hatdigits"]
                file_tracks = file["hattracks"]

                self._logger.info(f"Hit branches:   {file_hits.keys()}")
                self._logger.info(f"Track branches: {file_tracks.keys()}")

                # get the track and hit info in a more useful format
                track_iterator = file_tracks.iterate(
                    ["event", "track", "pdg", "parent", "nhits"], 
                    library="pd",
                    step_size = 1
                )
                hit_iterator = file_hits.iterate(
                    ["event", "trkid", "time", "y", "z", "row", "col", "qmax", "tmax", "fwhm"], 
                    library="pd",
                    step_size = 1
                )

                ## initial values
                event_id = 0
                new_event_id = None
                track = next(track_iterator, None)
                hit = next(hit_iterator, None)

                # assume that the last event ID tells us the number of events
                n_events = file_tracks["event"].array()[-1]
                self._logger.info(f'Has {n_events} events')
                
                while True:
                    if (hit is None) and (track is None):
                        break

                    if((self._process_stop is not None) & (event_id > self._process_stop)):
                        break

                    ## make the event
                    event = Event(event_id, logger=self._logger)

                    ## print out progress bar every few events
                    if(event_id % (int(n_events/100)) == 0 or event_id == n_events - 1):
                        printProgressBar(event_id, n_events - 1)

                    
                    ## first fill up the structure of the event
                    while True:
                        
                        if((self._process_start is not None) & (event_id >= self._process_start)):
                            event.add_track(
                                Track(
                                    track["track"].values[0], 
                                    track["pdg"].values[0],
                                    track["parent"].values[0]
                                )
                            )
                        
                        track = next(track_iterator, None)
                        
                        ## check if we've moved to a new event
                        if track["event"].values[0] != event_id or track is None:
                            break

                    ## now fill those tracks with hits
                    hit_id = 0

                    while True:

                        if((self._process_start is not None) & (event_id >= self._process_start)):
                            event.add_hit(
                                Hit(
                                    hit["time"].values[0], 
                                    hit["y"].values[0], 
                                    hit["z"].values[0], 
                                    hit["row"].values[0],
                                    hit["col"].values[0],
                                    hit["qmax"].values[0],
                                    hit_id,
                                    hit["trkid"].values[0].to_list()
                                ), 
                                hit["trkid"].values[0].to_list()
                            )
                            
                        hit = next(hit_iterator, None)

                        ## check if we've moved to a new event
                        if hit["event"].values[0] != event_id or hit is None:
                            ## should move to next event
                            new_event_id = hit["event"].values[0]
                            break

                        hit_id += 1
                
                    if((self._process_start is not None) & (event_id >= self._process_start)):
                        event.build_clusters(0)
                        event.print(self._logger.debug)
                        self._save_graphs(event, event_id, make_plots=False)

                    event_id = new_event_id

    def _save_graphs(self, event:Event, event_id:int, make_plots=False) -> None:

        data_inputs = event.build_input_graph()
        
        data_file_name = os.path.join(self._processed_file_path, f'data_HAT_{event_id}.pt')
        
        torch.save(data_inputs, data_file_name)

        self.processed_file_names.append(data_file_name)

        # optionally save some plots of the graphs
        if make_plots:

            draw_options = {
                'node_color': 'black',
                'node_size': 20,
                'width': 1,
            }
            
            g = tg_utils.to_networkx( data_labels, to_undirected=True, remove_self_loops=True )
            plt.clf()
            nx.draw(g, data_labels.pos[:, 1:], **draw_options)
            plt.savefig( os.path.join(self._processed_file_path, f"data_label_graph{event_id}.png") )

            ## need to separately build the label graph here to plot it    
            data_labels = event.build_label_graph()
    
            g = tg_utils.to_networkx(data_inputs, to_undirected=True, remove_self_loops=True)
            plt.clf()
            nx.draw(g, data_inputs.pos[:, 1:], **draw_options)
            plt.savefig( os.path.join(self._processed_file_path, f"data_graph_{event_id}.png") )
        