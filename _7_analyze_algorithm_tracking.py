import json
import os

import pandas as pd

from data_processing.configuration import load_algorithm_configuration


def load_configuration():
    return load_algorithm_configuration()


def get_default_tracking_folder(config_file):
    path_results = os.path.join(config_file['project_folder_path'], 'results')
    return os.path.join(path_results, 'algorithm_tracking')


def get_output_base(path_tracking_file):
    filename = os.path.basename(path_tracking_file)
    if filename.endswith('_tracking.jsonl'):
        location = filename[:-len('_tracking.jsonl')]
    else:
        location = os.path.splitext(filename)[0]
    path_folder = os.path.dirname(path_tracking_file)
    return location, os.path.join(path_folder, f'{location}_tracking_analysis')


def get_tracking_files(path_tracking):
    if not os.path.exists(path_tracking):
        raise FileNotFoundError(f'Tracking folder not found: {path_tracking}')

    tracking_files = []
    for filename in os.listdir(path_tracking):
        if filename.endswith('_tracking.jsonl'):
            tracking_files.append(os.path.join(path_tracking, filename))

    tracking_files.sort(key=lambda path: os.path.basename(path))
    return tracking_files


def load_tracking_file(path_input):
    if not os.path.exists(path_input):
        raise FileNotFoundError(
            f'Tracking file not found: {path_input}\n'
            f'Run the heuristic for this location first or choose another LOCATION.'
        )

    records = []
    with open(path_input, encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSON in {path_input}, line {line_number}: {error}') from error

    if not records:
        raise ValueError(f'Tracking file is empty: {path_input}')

    events = pd.DataFrame(records)
    events['details_json'] = events['details'].apply(lambda value: json.dumps(value, ensure_ascii=False))
    return events


def detail_value(details, key, default=None):
    if isinstance(details, dict):
        return details.get(key, default)
    return default


def make_iteration_summary(events):
    iteration_end = events[(events['phase'] == 'iteration') & (events['event'] == 'end')].copy()
    if iteration_end.empty:
        return pd.DataFrame()

    iteration_end['benchmark_old'] = iteration_end['details'].apply(lambda value: detail_value(value, 'benchmark_old'))
    iteration_end['benchmark'] = iteration_end['details'].apply(lambda value: detail_value(value, 'benchmark'))
    iteration_end['removed_in_iteration'] = iteration_end['before'] - iteration_end['after']

    return iteration_end[[
        'iteration', 'runtime_s', 'before', 'after', 'removed_in_iteration', 'benchmark_old', 'benchmark'
    ]].sort_values('iteration')


def make_runtime_summary(events):
    runtime_events = events[events['runtime_s'].notna()].copy()
    if runtime_events.empty:
        return pd.DataFrame()

    return runtime_events.groupby(['phase', 'method', 'event'], dropna=False).agg(
        calls=('runtime_s', 'size'),
        total_runtime_s=('runtime_s', 'sum'),
        mean_runtime_s=('runtime_s', 'mean'),
        max_runtime_s=('runtime_s', 'max'),
    ).reset_index().sort_values('total_runtime_s', ascending=False)


def make_filter_summary(events):
    removed_events = events[events['removed'].fillna(0) > 0].copy()
    if removed_events.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = removed_events.groupby(['phase', 'method', 'event'], dropna=False).agg(
        calls=('removed', 'size'),
        total_removed=('removed', 'sum'),
        max_removed=('removed', 'max'),
        total_runtime_s=('runtime_s', 'sum'),
        mean_runtime_s=('runtime_s', 'mean'),
        max_runtime_s=('runtime_s', 'max'),
    ).reset_index().sort_values('total_removed', ascending=False)

    largest_events = removed_events[[
        'time_since_location_start_s', 'iteration', 'phase', 'method', 'event',
        'runtime_s', 'before', 'after', 'removed', 'details_json'
    ]].sort_values('removed', ascending=False)

    return summary, largest_events


def make_creation_summary(events):
    created_events = events[events['created'].fillna(0) > 0].copy()
    if created_events.empty:
        return pd.DataFrame()

    return created_events.groupby(['phase', 'method', 'event'], dropna=False).agg(
        calls=('created', 'size'),
        total_created=('created', 'sum'),
        max_created=('created', 'max'),
        total_runtime_s=('runtime_s', 'sum'),
        mean_runtime_s=('runtime_s', 'mean'),
        max_runtime_s=('runtime_s', 'max'),
    ).reset_index().sort_values('total_created', ascending=False)


def make_chronological_measurements(events):
    """Return all events that contain branch counts or measured block runtimes."""
    measurement_columns = ['runtime_s', 'before', 'after', 'created', 'removed']
    measurements = events[events[measurement_columns].notna().any(axis=1)].copy()
    if measurements.empty:
        return pd.DataFrame()

    measurements['details_short'] = measurements['details_json'].str.slice(0, 220)
    return measurements[[
        'time_since_location_start_s', 'iteration', 'phase', 'method', 'event',
        'runtime_s', 'before', 'after', 'created', 'removed', 'details_short'
    ]].sort_values('time_since_location_start_s')


def make_location_summary(events):
    location_end = events[(events['phase'] == 'location') & (events['event'] == 'end')]
    total_runtime = None
    final_details = {}
    if not location_end.empty:
        last_end = location_end.iloc[-1]
        total_runtime = last_end['runtime_s']
        final_details = last_end['details'] if isinstance(last_end['details'], dict) else {}
    else:
        total_runtime = events['time_since_location_start_s'].max()

    return {
        'total_runtime_s': total_runtime,
        'initial_benchmark': final_details.get('initial_benchmark'),
        'solution': final_details.get('solution'),
        'total_branches_created': final_details.get('total_branches_created'),
        'new_approach_branches_created': final_details.get('new_approach_branches_created'),
        'final_solution_exists': final_details.get('final_solution_exists'),
    }


def format_table(df, columns=None, max_rows=None):
    if df.empty:
        return '  none\n'

    view = df.copy()
    if columns is not None:
        view = view[columns]
    if max_rows is not None:
        view = view.head(max_rows)

    return view.to_string(index=False) + '\n'


def write_text_summary(path_output, location, events, location_summary,
                       iteration_summary, runtime_summary, filter_summary,
                       largest_filter_events, creation_summary, chronological_measurements):
    lines = []
    lines.append(f'Algorithm tracking analysis for location {location}')
    lines.append('=' * 80)
    lines.append('')

    lines.append('Run summary')
    lines.append('-' * 80)
    for key, value in location_summary.items():
        lines.append(f'{key}: {value}')
    lines.append(f'tracking events: {len(events)}')
    lines.append('')

    lines.append('Iteration overview')
    lines.append('-' * 80)
    lines.append(format_table(iteration_summary))

    lines.append('Runtime hotspots')
    lines.append('-' * 80)
    lines.append(format_table(
        runtime_summary,
        columns=['phase', 'method', 'event', 'calls', 'total_runtime_s', 'mean_runtime_s', 'max_runtime_s'],
        max_rows=TOP_EVENTS,
    ))

    lines.append('Branch removals by filter')
    lines.append('-' * 80)
    lines.append(format_table(
        filter_summary,
        columns=['phase', 'method', 'event', 'calls', 'total_removed', 'max_removed',
                 'total_runtime_s', 'mean_runtime_s', 'max_runtime_s'],
        max_rows=TOP_EVENTS,
    ))

    lines.append('Largest single removal events')
    lines.append('-' * 80)
    lines.append(format_table(
        largest_filter_events,
        columns=['time_since_location_start_s', 'iteration', 'phase', 'method', 'event',
                 'runtime_s', 'before', 'after', 'removed'],
        max_rows=TOP_EVENTS,
    ))

    lines.append('Branch creation events')
    lines.append('-' * 80)
    lines.append(format_table(
        creation_summary,
        columns=['phase', 'method', 'event', 'calls', 'total_created', 'max_created',
                 'total_runtime_s', 'mean_runtime_s', 'max_runtime_s'],
        max_rows=TOP_EVENTS,
    ))

    lines.append('')
    lines.append('=' * 80)
    lines.append('Chronological Measurements')
    lines.append('=' * 80)
    lines.append('All measured branch/time events in exact chronological order.')
    lines.append('runtime_s is the duration of the measured block; time_since_location_start_s is only the timestamp.')
    lines.append('')
    lines.append(format_table(
        chronological_measurements,
        columns=['time_since_location_start_s', 'iteration', 'phase', 'method', 'event',
                 'runtime_s', 'before', 'after', 'created', 'removed', 'details_short'],
    ))

    with open(path_output, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))


def write_csv_outputs(path_output_base, events, iteration_summary, runtime_summary,
                      filter_summary, largest_filter_events, creation_summary,
                      chronological_measurements):
    export_events = events.copy()
    export_events.drop(columns=['details'], inplace=True)
    export_events.to_csv(path_output_base + '_events.csv', index=False, encoding='utf-8')
    iteration_summary.to_csv(path_output_base + '_iterations.csv', index=False, encoding='utf-8')
    runtime_summary.to_csv(path_output_base + '_runtime.csv', index=False, encoding='utf-8')
    filter_summary.to_csv(path_output_base + '_filters.csv', index=False, encoding='utf-8')
    largest_filter_events.to_csv(path_output_base + '_largest_filter_events.csv', index=False, encoding='utf-8')
    creation_summary.to_csv(path_output_base + '_creations.csv', index=False, encoding='utf-8')
    chronological_measurements.to_csv(path_output_base + '_chronological_measurements.csv',
                                      index=False, encoding='utf-8')


def analyze_tracking_file(path_input):
    location, path_output_base = get_output_base(path_input)
    events = load_tracking_file(path_input)
    location_summary = make_location_summary(events)
    iteration_summary = make_iteration_summary(events)
    runtime_summary = make_runtime_summary(events)
    filter_summary, largest_filter_events = make_filter_summary(events)
    creation_summary = make_creation_summary(events)
    chronological_measurements = make_chronological_measurements(events)

    path_text_output = path_output_base + '_summary.txt'
    write_text_summary(path_text_output, location, events, location_summary,
                       iteration_summary, runtime_summary, filter_summary,
                       largest_filter_events, creation_summary, chronological_measurements)
    if False:
        write_csv_outputs(path_output_base, events, iteration_summary, runtime_summary,
                          filter_summary, largest_filter_events, creation_summary,
                          chronological_measurements)
    return path_text_output


def main():
    config_file = load_configuration()

    path_tracking = TRACKING_FOLDER or get_default_tracking_folder(config_file)
    path_tracking = os.path.abspath(path_tracking)
    tracking_files = get_tracking_files(path_tracking)

    if not tracking_files:
        print(f'No tracking files found in: {path_tracking}')
        return

    print(f'Analyze {len(tracking_files)} tracking files from:')
    print(path_tracking)

    written_files = []
    failed_files = []
    for path_input in tracking_files:
        try:
            path_text_output = analyze_tracking_file(path_input)
            written_files.append(path_text_output)
            print(f'Wrote tracking analysis: {path_text_output}')
        except Exception as error:
            failed_files.append((path_input, error))
            print(f'Failed tracking analysis: {path_input}')
            print(f'  {error}')

    print('')
    print(f'Finished tracking analysis: {len(written_files)} written, {len(failed_files)} failed.')
    if failed_files:
        print('Failed files:')
        for path_input, error in failed_files:
            print(f'  {path_input}: {error}')


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Set the folder containing all *_tracking.jsonl files here.
# If set to None, the configured project results/algorithm_tracking folder is used.
TRACKING_FOLDER = None

# Number of largest single filter/removal events shown in the text summary.
TOP_EVENTS = 20


if __name__ == '__main__':
    main()
