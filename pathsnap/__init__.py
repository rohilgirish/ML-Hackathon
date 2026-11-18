"""PathSnap: Map-Matching and Route Reconstruction package.

Module 1 (M1): Problem framing, data I/O, geographical projections, road network
representation, label alignment, candidate generation, and feature extraction.
"""

from . import config
from .data_io import Trip, clean_track, list_trips, load_route, load_trip
from .dataset import build_partition, build_trip_table, label_trip
from .features import FEATURES, PreparedTrip, build_features, generate_candidates
from .geo import LocalProjection, haversine_m
from .labeling import align_to_route
from .network import RoadNetwork

__all__ = [
    "config",
    "Trip",
    "clean_track",
    "list_trips",
    "load_trip",
    "load_route",
    "LocalProjection",
    "haversine_m",
    "RoadNetwork",
    "align_to_route",
    "FEATURES",
    "PreparedTrip",
    "generate_candidates",
    "build_features",
    "label_trip",
    "build_trip_table",
    "build_partition",
]
