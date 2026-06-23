import json
import math
import os
import inspect
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


def get_benchmark_branches(branches, benchmark_info):
    if branches is None or benchmark_info is None:
        return pd.DataFrame()
    if not hasattr(branches, 'empty') or branches.empty:
        return pd.DataFrame()
    if 'current_commodity' not in branches.columns or 'current_node' not in branches.columns:
        return pd.DataFrame()

    commodities = benchmark_info[0]
    locations = benchmark_info[2]
    combinations = {(c, l) for c in commodities for l in locations}
    benchmark_branches = []

    for commodity, location in combinations:
        branch = branches[
            (branches['current_commodity'] == commodity)
            & (branches['current_node'] == location)
        ]
        if not branch.empty:
            benchmark_branches.append(branch)

    if not benchmark_branches:
        return pd.DataFrame()
    return pd.concat(benchmark_branches)


def is_enabled(value):
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes', 'y'}
    return bool(value)


def _get_source_location(stack_depth):
    frame = inspect.currentframe()
    try:
        for _ in range(stack_depth):
            if frame is None:
                return None
            frame = frame.f_back
        if frame is None:
            return None
        return {
            'file': os.path.abspath(frame.f_code.co_filename),
            'line': frame.f_lineno,
        }
    finally:
        del frame


def track_benchmark_removal(data, configuration, before_branches, after_branches,
                            iteration=None, phase=None, method=None, code=None,
                            details=None, source_stack_depth=2):
    if not is_enabled(configuration.get('print_benchmark_info', False)):
        return
    if not isinstance(data, dict):
        return
    if data.get('_benchmark_removal_reported'):
        return

    benchmark_info = data.get('benchmark_info')
    before_benchmark = get_benchmark_branches(before_branches, benchmark_info)
    if before_benchmark.empty:
        return

    after_benchmark = get_benchmark_branches(after_branches, benchmark_info)
    if not after_benchmark.empty:
        return

    data['_benchmark_removal_reported'] = True

    source = _get_source_location(source_stack_depth)
    location = data.get('location_index', data.get('k'))
    print('Benchmark removed from branches')
    print('Location: ' + str(location))
    print('Iteration: ' + str(iteration))
    print('Phase: ' + str(phase))
    print('Method: ' + str(method))
    print('Code: ' + str(code))
    if source is not None:
        print('Source:')
        print('  File "' + source['file'] + '", line ' + str(source['line']) + ', in ' + str(method))
        print('  ' + source['file'] + ':' + str(source['line']))
    if details:
        print('Details: ' + str(details))
    print('Benchmark branches before removal:')
    print(before_benchmark[['current_commodity', 'current_node', 'current_total_costs']])

    tracker = get_tracker(data)
    if tracker is not None:
        tracker.event(iteration=iteration, phase=phase, method=method,
                      event='benchmark_removed', before=branch_count(before_branches),
                      after=branch_count(after_branches),
                      removed=branch_count(before_branches) - branch_count(after_branches),
                      details={
                          'code': code,
                          'source': source,
                          'benchmark_rows_before': branch_count(before_benchmark),
                          **(details or {}),
                      })
