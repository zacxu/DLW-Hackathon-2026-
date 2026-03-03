Install dependencies

Download Singapore’s road network (driving / walking / biking) from OpenStreetMap via OSMnx

Add edge speeds + travel times (so “optimal” can mean fastest not just shortest)

Pick a manageable subgraph (Singapore is big; RL on the full graph is possible but slow)


"""
Road-graph RL for Singapore (Option A)
- Road network from OpenStreetMap via OSMnx
- Gymnasium environment
- Tabular Q-learning

Notes:
- For realistic "fastest route", reward uses edge travel_time (seconds).
- For bigger areas, tabular Q-learning will get slow. Start small.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

import osmnx as ox
import networkx as nx

import gymnasium as gym
from gymnasium import spaces


# ----------------------------
# 1) Download + prepare graph
# ----------------------------

@dataclass
class GraphConfig:
    place: str = "Singapore"
    network_type: str = "drive"  # "drive", "walk", "bike"
    # Subgraph selection method:
    # We'll clip by a center point + radius (meters) to keep the state space manageable.
    center_lat: float = 1.3048   # around Orchard / Dhoby Ghaut-ish
    center_lon: float = 103.8318
    dist_m: int = 2500           # radius meters for subgraph
    simplify: bool = True


def load_singapore_graph(cfg: GraphConfig) -> nx.MultiDiGraph:
    # Download full place graph
    G = ox.graph_from_place(
        cfg.place,
        network_type=cfg.network_type,
        simplify=cfg.simplify
    )

    # Add speeds + travel times on edges
    # OSM often lacks maxspeed; OSMnx imputes by highway type means when missing.
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Clip to a smaller subgraph around a center point to make RL tractable
    G_sub = ox.graph_from_point(
        (cfg.center_lat, cfg.center_lon),
        dist=cfg.dist_m,
        network_type=cfg.network_type,
        simplify=cfg.simplify
    )
    G_sub = ox.add_edge_speeds(G_sub)
    G_sub = ox.add_edge_travel_times(G_sub)

    # Largest strongly connected component helps avoid dead-ends in directed graphs (drive)
    # For walking networks, you might prefer weakly connected.
    if cfg.network_type == "drive":
        comps = list(nx.strongly_connected_components(G_sub))
        largest = max(comps, key=len)
        G_sub = G_sub.subgraph(largest).copy()
    else:
        comps = list(nx.connected_components(ox.utils_graph.get_undirected(G_sub)))
        largest = max(comps, key=len)
        G_sub = G_sub.subgraph(largest).copy()

    return G_sub
