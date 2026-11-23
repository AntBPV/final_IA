from src.tracking.centroid_tracker import CentroidTracker


def test_tracker_keeps_same_id_for_close_object():
    tracker = CentroidTracker(max_distance=50)
    first = tracker.update([(10, 10)])
    identifier = next(iter(first))
    second = tracker.update([(12, 11)])
    assert identifier in second
