import json
import math
import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


def _to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, (pd.Index, np.ndarray)):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def branch_count(branches):
    if branches is None:
        return 0
    if hasattr(branches, 'empty') and branches.empty:
        return 0
    if hasattr(branches, 'index'):
        return len(branches.index)
    return 0


class AlgorithmTracker:
    def __init__(self, location, path_results, enabled=True):
        self.location = location
        self.enabled = enabled
        self.start_time = time.perf_counter()
        self.path_tracking = os.path.join(path_results, 'algorithm_tracking')
        self.path_file = os.path.join(self.path_tracking, f'{location}_tracking.jsonl')

        if self.enabled:
            os.makedirs(self.path_tracking, exist_ok=True)
            with open(self.path_file, 'w', encoding='utf-8') as file:
                file.write('')

    def event(self, iteration=None, phase=None, method=None, event=None,
              before=None, after=None, created=None, removed=None,
              runtime_s=None, details=None):
        if not self.enabled:
            return

        record = {
            'location': self.location,
            'time_since_location_start_s': time.perf_counter() - self.start_time,
            'iteration': iteration,
            'phase': phase,
            'method': method,
            'event': event,
            'before': before,
            'after': after,
            'created': created,
            'removed': removed,
            'runtime_s': runtime_s,
            'details': details or {},
        }

        with open(self.path_file, 'a', encoding='utf-8') as file:
            file.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + '\n')

    @contextmanager
    def time_block(self, iteration=None, phase=None, method=None, event=None, details=None):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.event(
                iteration=iteration,
                phase=phase,
                method=method,
                event=event,
                runtime_s=time.perf_counter() - start,
                details=details,
            )


def get_tracker(data):
    if isinstance(data, dict):
        return data.get('tracker')
    return None


def track_event(data, *args, **kwargs):
    tracker = get_tracker(data)
    if tracker is not None:
        tracker.event(*args, **kwargs)
