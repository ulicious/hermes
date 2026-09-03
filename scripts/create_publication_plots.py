import os
import sys


PROJECT_FOLDER = r'C:\Users\mt5285\Documents\Transportmodell'
CASE_STUDIES = []
COMPARISON_CASE_STUDIES = []
DISTRIBUTION_CASE_STUDIES = []
SUPPLY_CURVE_COMPARISON_CASE_STUDIES = []
SUPPLY_CURVE_COMPARISON_COUNTRIES = []


REPOSITORY_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_FOLDER not in sys.path:
    sys.path.insert(0, REPOSITORY_FOLDER)

from plotting.get_figures_publication import (
    create_publication_plot_1,
    create_publication_plot_2,
    create_publication_plot_3,
    create_publication_plot_4,
)


def _create_supply_curve_comparison_sources(
    comparison_case_studies,
    countries,
    plots_directory,
):
    import pandas as pd
    import shapely

    from data_processing.configuration import (
        load_algorithm_configuration,
        load_plotting_configuration,
    )
    from plotting.get_figures import (
        DEFAULT_PLOT_BOUNDARIES,
        configure_plot_output,
        get_configured_colormap,
        get_plot_color_config,
    )
    from plotting.helpers_plotting import plot_comparison_plot

    general_config = load_algorithm_configuration()
    general_config['project_folder_path'] = PROJECT_FOLDER
    plotting_config = load_plotting_configuration(general_config)
    plotting_config = dict(plotting_config)
    vector_filetypes = plotting_config.get('vector_filetype', ['svg'])
    if isinstance(vector_filetypes, str):
        vector_filetypes = [vector_filetypes]
    vector_filetypes = list(vector_filetypes)
    if 'svg' not in [str(filetype).lower().lstrip('.') for filetype in vector_filetypes]:
        vector_filetypes.append('svg')
    plotting_config['vector_filetype'] = vector_filetypes
    configure_plot_output(plotting_config)

    processed_results_directory = os.path.join(
        PROJECT_FOLDER,
        'results',
        'processed_results',
    )
    production_costs_path = os.path.join(
        PROJECT_FOLDER,
        'start_destination_combinations.csv',
    )
    if os.path.isfile(production_costs_path):
        production_costs = pd.read_csv(production_costs_path, index_col=0)
        if 'geometry' in production_costs.columns:
            production_costs['geometry'] = production_costs['geometry'].apply(
                shapely.wkt.loads
            )
    else:
        production_costs = pd.DataFrame()

    plot_colors = get_plot_color_config(plotting_config)
    for country in countries:
        plot_comparison_plot(
            'supply_curves',
            comparison_case_studies,
            processed_results_directory,
            plots_directory,
            plotting_config,
            production_costs,
            get_configured_colormap(plotting_config),
            DEFAULT_PLOT_BOUNDARIES,
            color_dictionary=plot_colors['commodity_colors'],
            nice_name_dictionary=dict(plotting_config['nice_name_dictionary']),
            country=country,
            distance_between=1,
            subplot_height=4,
        )


def main():
    if not isinstance(CASE_STUDIES, list):
        raise TypeError('CASE_STUDIES must be a list.')
    if not isinstance(COMPARISON_CASE_STUDIES, list):
        raise TypeError('COMPARISON_CASE_STUDIES must be a list.')
    if not isinstance(DISTRIBUTION_CASE_STUDIES, list):
        raise TypeError('DISTRIBUTION_CASE_STUDIES must be a list.')
    if not isinstance(SUPPLY_CURVE_COMPARISON_CASE_STUDIES, list):
        raise TypeError('SUPPLY_CURVE_COMPARISON_CASE_STUDIES must be a list.')
    if not isinstance(SUPPLY_CURVE_COMPARISON_COUNTRIES, list):
        raise TypeError('SUPPLY_CURVE_COMPARISON_COUNTRIES must be a list.')
    if (
        not CASE_STUDIES
        and not COMPARISON_CASE_STUDIES
        and not DISTRIBUTION_CASE_STUDIES
        and not SUPPLY_CURVE_COMPARISON_CASE_STUDIES
    ):
        raise ValueError(
            'Set CASE_STUDIES, COMPARISON_CASE_STUDIES and/or '
            'DISTRIBUTION_CASE_STUDIES and/or '
            'SUPPLY_CURVE_COMPARISON_CASE_STUDIES '
            'in create_publication_plots.py.'
        )
    case_studies = []
    for case_study in CASE_STUDIES:
        if not isinstance(case_study, str) or not case_study.strip():
            raise ValueError('Every entry in CASE_STUDIES must be a non-empty string.')
        case_studies.append(case_study.strip())

    plots_directory = os.path.join(PROJECT_FOLDER, 'results', 'plots')
    for case_study in case_studies:
        output_path = create_publication_plot_1(
            case_study,
            plots_directory,
        )
        print(output_path)

    if SUPPLY_CURVE_COMPARISON_CASE_STUDIES and not SUPPLY_CURVE_COMPARISON_COUNTRIES:
        raise ValueError(
            'Set at least one country in SUPPLY_CURVE_COMPARISON_COUNTRIES.'
        )
    supply_curve_countries = []
    for country in SUPPLY_CURVE_COMPARISON_COUNTRIES:
        if not isinstance(country, str) or not country.strip():
            raise ValueError(
                'Every entry in SUPPLY_CURVE_COMPARISON_COUNTRIES must be a non-empty string.'
            )
        supply_curve_countries.append(country.strip())
    for comparison_case_studies in SUPPLY_CURVE_COMPARISON_CASE_STUDIES:
        if not isinstance(comparison_case_studies, list):
            raise TypeError(
                'Every entry in SUPPLY_CURVE_COMPARISON_CASE_STUDIES must be a list.'
            )
        if len(comparison_case_studies) not in {2, 4}:
            raise ValueError(
                'Every comparison in SUPPLY_CURVE_COMPARISON_CASE_STUDIES '
                'must contain two or four case-study names.'
            )
        for case_study in comparison_case_studies:
            if not isinstance(case_study, str) or not case_study.strip():
                raise ValueError(
                    'Every case study in SUPPLY_CURVE_COMPARISON_CASE_STUDIES '
                    'must be a non-empty string.'
                )
    if SUPPLY_CURVE_COMPARISON_CASE_STUDIES:
        _create_supply_curve_comparison_sources(
            SUPPLY_CURVE_COMPARISON_CASE_STUDIES,
            supply_curve_countries,
            plots_directory,
        )
    for country in supply_curve_countries:
        for comparison_id, comparison_case_studies in enumerate(
            SUPPLY_CURVE_COMPARISON_CASE_STUDIES
        ):
            output_path = create_publication_plot_4(
                comparison_id,
                country,
                plots_directory,
                case_study_count=len(comparison_case_studies),
            )
            print(output_path)

    for case_study in DISTRIBUTION_CASE_STUDIES:
        if not isinstance(case_study, str) or not case_study.strip():
            raise ValueError(
                'Every entry in DISTRIBUTION_CASE_STUDIES must be a non-empty string.'
            )
        output_path = create_publication_plot_3(
            case_study.strip(),
            plots_directory,
        )
        print(output_path)

    for comparison_id, comparison_case_studies in enumerate(COMPARISON_CASE_STUDIES):
        if not isinstance(comparison_case_studies, list):
            raise TypeError('Every entry in COMPARISON_CASE_STUDIES must be a list.')
        if len(comparison_case_studies) not in {2, 4}:
            raise ValueError(
                'Every comparison in COMPARISON_CASE_STUDIES must contain two or '
                'four case-study names.'
            )
        for case_study in comparison_case_studies:
            if not isinstance(case_study, str) or not case_study.strip():
                raise ValueError(
                    'Every case study in COMPARISON_CASE_STUDIES must be a non-empty string.'
                )
        output_path = create_publication_plot_2(
            comparison_id,
            plots_directory,
            case_study_count=len(comparison_case_studies),
        )
        print(output_path)


if __name__ == '__main__':
    main()
