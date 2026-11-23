from collections import OrderedDict
from itertools import count

import numpy as np


class CentroidTracker:
    def __init__(self, max_lost: int = 20, max_distance: float = 75.0):
        self.max_lost = max_lost
        self.max_distance = max_distance
        self.next_id = count(1)
        self.objects = OrderedDict()
        self.lost = OrderedDict()

    def register(self, centroid):
        identifier = next(self.next_id)
        self.objects[identifier] = centroid
        self.lost[identifier] = 0
        return identifier

    def deregister(self, identifier):
        self.objects.pop(identifier, None)
        self.lost.pop(identifier, None)

    def update(self, detections):
        if len(detections) == 0:
            for identifier in list(self.objects):
                self.lost[identifier] += 1
                if self.lost[identifier] > self.max_lost:
                    self.deregister(identifier)
            return self.objects

        detections = np.asarray(detections)
        centroids = detections[:, :2]
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(tuple(centroid))
            return self.objects

        object_ids = list(self.objects)
        object_centroids = list(self.objects.values())
        distances = np.linalg.norm(
            np.asarray(object_centroids)[:, None, :] - centroids[None, :, :],
            axis=2,
        )

        rows = distances.min(axis=1).argsort()
        columns = distances.argmin(axis=1)[rows]
        used_rows, used_columns = set(), set()

        for row, column in zip(rows, columns):
            if row in used_rows or column in used_columns:
                continue
            if distances[row, column] > self.max_distance:
                continue
            identifier = object_ids[row]
            self.objects[identifier] = tuple(centroids[column])
            self.lost[identifier] = 0
            used_rows.add(row)
            used_columns.add(column)

        for row in set(range(len(object_ids))).difference(used_rows):
            identifier = object_ids[row]
            self.lost[identifier] += 1
            if self.lost[identifier] > self.max_lost:
                self.deregister(identifier)

        for column in set(range(len(centroids))).difference(used_columns):
            self.register(tuple(centroids[column]))
        return self.objects

