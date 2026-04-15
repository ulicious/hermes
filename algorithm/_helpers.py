import random
import shapely
import os
import time
import functools
import tracemalloc
import psutil
import threading

import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime

LOG_FILE = "memory_calls.log"
PROCESS = psutil.Process(os.getpid())
ENABLE_MEMORY_PROFILE = True

# Intervall zur Peak-Messung innerhalb der Funktion
INTERVAL = 0.1  # Sekunden


def plot_geometry_list(lines, same_colors=False, wait_plot=False, ax=None):

    """
    method to plot shapely.geometry.LineString

    @param lines: list of LineStrings
    @param same_colors: boolean if same colors are applied to all lines
    @param wait_plot: boolean if figure should be plotted immediately
    @param ax: matplotlib axis if axis defined outside of method should be used
    """

    def generate_random_colors(n):
        colors = []
        for _ in range(n):
            # Generate random RGB values
            r = random.random()
            g = random.random()
            b = random.random()
            # Append the color in hexadecimal format
            colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
        return colors

    if same_colors:
        c = generate_random_colors(1)[0]
    else:
        c = generate_random_colors(len(lines))

    gdf = gpd.GeoDataFrame(geometry=lines)

    if ax is not None:
        gdf.plot(color=c, ax=ax)
    else:
        gdf.plot(color=c)

    if not wait_plot:
        plt.show()


def plot_geometry(geometry):
    """
    method to plot any shapely.geometry object

    @param geometry: shapely.geometry object
    """

    def generate_random_colors(n):
        colors = []
        for _ in range(n):
            # Generate random RGB values
            r = random.random()
            g = random.random()
            b = random.random()
            # Append the color in hexadecimal format
            colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
        return colors

    c = generate_random_colors(len([geometry]))

    gdf = gpd.GeoDataFrame(geometry=[geometry])
    gdf.plot(color=c)
    plt.show()


def plot_subgraphs(graph_data, subgraph_nodes):

    """
    method to plot subgraphs of a graph. Mainly used for debugging of network issues

    @param graph_data: DataFrame with graph information (nodes, lines)
    @param subgraph_nodes: nodes of different subgraphs which seem disconnected from graph
    """

    fig, ax = plt.subplots()

    for nodes in subgraph_nodes:
        edge_index = graph_data[(graph_data['node_start'].isin(nodes)) | graph_data['node_end'].isin(nodes)].index
        lines = graph_data.loc[edge_index, 'line'].apply(shapely.wkt.loads).tolist()

        plot_geometry_list(lines, same_colors=True, wait_plot=True, ax=ax)

    plt.show()

def log_line(text: str):
    """Schreibt Log-Eintrag in Datei"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def format_mb(num_bytes: int) -> str:
    """Bytes in MB umrechnen"""
    return f"{num_bytes / 1024 / 1024:.2f} MB"

def memory_profile(func):
    """Decorator für Peak-RAM-Messung pro Funktionsaufruf, multiprocessing-kompatibel"""
    if not ENABLE_MEMORY_PROFILE:
        return func  # Profiling deaktiviert

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pid = os.getpid()
        log_line(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[START] PID={pid} {func.__module__}.{func.__qualname__}"
        )
        start_time = time.perf_counter()
        peak_rss = psutil.Process(pid).memory_info().rss

        result = None

        def target():
            nonlocal result
            result = func(*args, **kwargs)

        # Funktion in Thread ausführen, um Peak während Laufzeit zu tracken
        thread = threading.Thread(target=target)
        thread.start()

        process = psutil.Process(pid)
        while thread.is_alive():
            rss = process.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss
            time.sleep(INTERVAL)

        thread.join()
        end_time = time.perf_counter()

        log_line(
            f"[END]   PID={pid} {func.__module__}.{func.__qualname__} | "
            f"Peak RSS={format_mb(peak_rss)} | Dauer={end_time - start_time:.3f}s"
        )
        log_line("-" * 100)
        return result

    return wrapper


import os
import gc
import psutil
import numpy as np
import pandas as pd

_PROCESS = psutil.Process(os.getpid())


def _bytes_to_gb(num_bytes):
    return num_bytes / 1024 ** 3


def get_size_gb(obj):
    """
    Return memory usage in GB for common object types.
    """
    if obj is None:
        return 0.0

    if isinstance(obj, np.ndarray):
        return _bytes_to_gb(obj.nbytes)

    if isinstance(obj, pd.DataFrame):
        return _bytes_to_gb(obj.memory_usage(index=True, deep=True).sum())

    if isinstance(obj, pd.Series):
        return _bytes_to_gb(obj.memory_usage(index=True, deep=True))

    if isinstance(obj, (list, tuple)):
        total = 0
        for item in obj:
            if isinstance(item, np.ndarray):
                total += item.nbytes
            elif isinstance(item, pd.DataFrame):
                total += item.memory_usage(index=True, deep=True).sum()
            elif isinstance(item, pd.Series):
                total += item.memory_usage(index=True, deep=True)
            else:
                # fallback for simple Python objects
                try:
                    import sys
                    total += sys.getsizeof(item)
                except Exception:
                    pass
        return _bytes_to_gb(total)

    if isinstance(obj, dict):
        total = 0
        for k, v in obj.items():
            try:
                import sys
                total += sys.getsizeof(k)
            except Exception:
                pass

            if isinstance(v, np.ndarray):
                total += v.nbytes
            elif isinstance(v, pd.DataFrame):
                total += v.memory_usage(index=True, deep=True).sum()
            elif isinstance(v, pd.Series):
                total += v.memory_usage(index=True, deep=True)
            else:
                try:
                    import sys
                    total += sys.getsizeof(v)
                except Exception:
                    pass
        return _bytes_to_gb(total)

    # fallback
    try:
        import sys
        return _bytes_to_gb(sys.getsizeof(obj))
    except Exception:
        return -1.0


def log_mem(name, obj=None, extra=None, run_gc=False):
    """
    Log process RSS and optionally object memory.
    """
    if run_gc:
        gc.collect()

    rss_gb = _bytes_to_gb(_PROCESS.memory_info().rss)

    parts = [f"[MEM] {name}", f"rss={rss_gb:.3f} GB"]

    if obj is not None:
        obj_gb = get_size_gb(obj)
        parts.append(f"obj={obj_gb:.3f} GB")
        parts.append(f"type={type(obj).__name__}")

        if isinstance(obj, np.ndarray):
            parts.append(f"shape={obj.shape}")
            parts.append(f"dtype={obj.dtype}")

        elif isinstance(obj, pd.DataFrame):
            parts.append(f"shape={obj.shape}")

        elif isinstance(obj, pd.Series):
            parts.append(f"len={len(obj)}")
            parts.append(f"dtype={obj.dtype}")

        elif isinstance(obj, (list, tuple, dict)):
            parts.append(f"len={len(obj)}")

    if extra is not None:
        parts.append(str(extra))

    print(" | ".join(parts))


def stack_dropna_fast(df):
    values = df.to_numpy()
    mask = ~pd.isna(values)
    row_inds, col_inds = np.where(mask)

    return pd.Series(
        values[row_inds, col_inds],
        index=pd.MultiIndex.from_arrays(
            [
                df.index.to_numpy()[row_inds],
                df.columns.to_numpy()[col_inds]
            ]
        )
    )

def transpose_stack_dropna_fast(df):
    values_t = df.to_numpy().T
    mask = ~pd.isna(values_t)
    row_inds, col_inds = np.where(mask)

    return pd.Series(
        values_t[row_inds, col_inds],
        index=pd.MultiIndex.from_arrays(
            [
                df.columns.to_numpy()[row_inds],
                df.index.to_numpy()[col_inds]
            ]
        )
    )
