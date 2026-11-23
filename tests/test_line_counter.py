from src.counting.line_counter import LineCounter


def test_line_counter_counts_entry_and_exit():
    counter = LineCounter((0, 100), (200, 100))
    counter.update({1: (100, 80)})
    counter.update({1: (100, 120)})
    assert counter.entries == 1
    assert counter.exits == 0
    assert counter.occupancy == 1


def test_reset_clears_totals():
    counter = LineCounter((0, 100), (200, 100))
    counter.entries = 3
    counter.exits = 1
    counter.reset()
    assert counter.entries == 0
    assert counter.exits == 0
    assert counter.occupancy == 0

