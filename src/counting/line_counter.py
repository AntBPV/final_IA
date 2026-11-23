from collections import OrderedDict


class LineCounter:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.entries = 0
        self.exits = 0
        self.occupancy = 0
        self.previous_side = OrderedDict()

    @staticmethod
    def _cross_product(start, end, point):
        return (end[0] - start[0]) * (point[1] - start[1]) - (
            end[1] - start[1]
        ) * (point[0] - start[0])

    def update(self, tracked_objects):
        for identifier, centroid in tracked_objects.items():
            side = self._cross_product(self.start, self.end, centroid)
            current = 1 if side >= 0 else -1
            previous = self.previous_side.get(identifier)
            if previous is not None and previous != current:
                if previous < 0 and current > 0:
                    self.entries += 1
                else:
                    self.exits += 1
            self.previous_side[identifier] = current
        self.occupancy = max(0, self.entries - self.exits)
        return self.entries, self.exits, self.occupancy

    def reset(self):
        self.entries = 0
        self.exits = 0
        self.occupancy = 0
        self.previous_side.clear()

