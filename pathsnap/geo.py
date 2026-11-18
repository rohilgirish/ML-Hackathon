"""Local metric projection.

Each trip covers at most a few tens of kilometres, so a local equirectangular projection
centred on the trip (x east, y north, metres) has distance error well below GPS noise
(< 0.1 % over 50 km). It is fast, vectorised and needs no external CRS database.
"""
import numpy as np

R_EARTH = 6_371_008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.radians(1.0) * R_EARTH * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * R_EARTH

    @classmethod
    def for_points(cls, lonlat: np.ndarray) -> "LocalProjection":
        return cls(np.median(lonlat[:, 0]), np.median(lonlat[:, 1]))

    def forward(self, lon, lat) -> np.ndarray:
        return np.c_[(np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky]

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        return np.c_[xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0]


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))
