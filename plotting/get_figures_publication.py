import copy
import os
import re
import warnings
import xml.etree.ElementTree as ET


SVG_NAMESPACE = 'http://www.w3.org/2000/svg'
XLINK_NAMESPACE = 'http://www.w3.org/1999/xlink'
MAXIMUM_HEIGHT_MM = 185.0
PANEL_GAP_MM = 1.5
PANEL_CROP_TOP_MM = 3.0
PANEL_CROP_BOTTOM_MM = 3.0
FIGURE_1_FIRST_TOP_CROP_MM = 2.0
FIGURE_1_SECOND_TOP_CROP_MM = 2.0
FIGURE_1_THIRD_TOP_CROP_MM = 2.0
FIGURE_1_LAST_BOTTOM_CROP_MM = 5.0
FIGURE_1_PANEL_GAPS_MM = (-2.5, -5.25)
POINTS_PER_MM = 72.0 / 25.4

ET.register_namespace('', SVG_NAMESPACE)
ET.register_namespace('xlink', XLINK_NAMESPACE)


def _write_publication_outputs(output_root, output_path):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ET.ElementTree(output_root).write(
        output_path,
        encoding='utf-8',
        xml_declaration=True,
    )
    try:
        import cairosvg
    except ImportError as error:
        raise RuntimeError(
            'Creating publication PDFs and PNGs requires CairoSVG. Install the project '
            'requirements before running create_publication_plots.py.'
        ) from error
    output_stem = os.path.splitext(output_path)[0]
    pdf_path = output_stem + '.pdf'
    png_path = output_stem + '.png'
    cairosvg.svg2pdf(url=output_path, write_to=pdf_path)
    cairosvg.svg2png(url=output_path, write_to=png_path, dpi=600)
    return output_path


def _length_in_points(value):
    match = re.fullmatch(
        r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)\s*',
        value or '',
    )
    if not match:
        raise ValueError('Unsupported SVG length: ' + repr(value))
    number = float(match.group(1))
    unit = match.group(2).lower()
    factors = {
        '': 1.0,
        'pt': 1.0,
        'px': 72.0 / 96.0,
        'in': 72.0,
        'mm': POINTS_PER_MM,
        'cm': POINTS_PER_MM * 10.0,
        'pc': 12.0,
    }
    if unit not in factors:
        raise ValueError('Unsupported SVG length unit: ' + repr(unit))
    return number * factors[unit]


def _read_svg(path):
    root = ET.parse(path).getroot()
    if root.findall('.//{' + SVG_NAMESPACE + '}image'):
        raise ValueError('SVG contains raster image elements: ' + path)
    view_box_value = root.get('viewBox')
    if view_box_value:
        view_box = tuple(float(value) for value in view_box_value.replace(',', ' ').split())
        if len(view_box) != 4:
            raise ValueError('Invalid SVG viewBox in ' + path)
    else:
        width = _length_in_points(root.get('width'))
        height = _length_in_points(root.get('height'))
        view_box = (0.0, 0.0, width, height)

    width_points = _length_in_points(root.get('width'))
    height_points = _length_in_points(root.get('height'))
    if width_points <= 0 or height_points <= 0 or view_box[2] <= 0 or view_box[3] <= 0:
        raise ValueError('SVG dimensions must be greater than zero: ' + path)
    return root, view_box, width_points, height_points


def _prefix_svg_ids(root, prefix):
    replacements = {}
    for element in root.iter():
        identifier = element.get('id')
        if identifier:
            replacements[identifier] = prefix + identifier
            element.set('id', replacements[identifier])

    if not replacements:
        return

    href_keys = {'href', '{' + XLINK_NAMESPACE + '}href'}
    for element in root.iter():
        for attribute, value in list(element.attrib.items()):
            updated = value
            for old_identifier, new_identifier in replacements.items():
                updated = updated.replace(
                    'url(#' + old_identifier + ')',
                    'url(#' + new_identifier + ')',
                )
                if attribute in href_keys and updated == '#' + old_identifier:
                    updated = '#' + new_identifier
            if updated != value:
                element.set(attribute, updated)
        if element.text:
            updated_text = element.text
            for old_identifier, new_identifier in replacements.items():
                updated_text = updated_text.replace(
                    'url(#' + old_identifier + ')',
                    'url(#' + new_identifier + ')',
                )
                updated_text = updated_text.replace(
                    '#' + old_identifier,
                    '#' + new_identifier,
                )
            element.text = updated_text


def _primary_axes_bottom(root, view_box):
    axes_group = None
    for element in root.iter():
        if element.get('id') == 'axes_1':
            axes_group = element
            break
    if axes_group is None:
        return view_box[1] + view_box[3]

    for element in axes_group.iter():
        if element.get('id') != 'patch_2':
            continue
        for path in element.iter('{' + SVG_NAMESPACE + '}path'):
            values = [float(value) for value in re.findall(
                r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',
                path.get('d', ''),
            )]
            if len(values) >= 4:
                return max(values[1::2])
    return view_box[1] + view_box[3]


def _path_bounds(path):
    values = [float(value) for value in re.findall(
        r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',
        path.get('d', ''),
    )]
    if len(values) < 4 or len(values) % 2:
        return None
    x_values = values[0::2]
    y_values = values[1::2]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _map_axes_bounds(root, count=4):
    axes_groups = []
    for element in root.iter():
        identifier = element.get('id', '')
        match = re.fullmatch(r'axes_(\d+)', identifier)
        if match:
            axes_groups.append((int(match.group(1)), element))
    axes_groups.sort(key=lambda item: item[0])

    bounds = []
    for _, axes_group in axes_groups[:count]:
        axes_bounds = None
        for child in list(axes_group):
            if not child.get('id', '').startswith('patch_'):
                continue
            for path in child.iter('{' + SVG_NAMESPACE + '}path'):
                axes_bounds = _path_bounds(path)
                if axes_bounds is not None:
                    break
            if axes_bounds is not None:
                break
        if axes_bounds is not None:
            bounds.append(axes_bounds)

    if len(bounds) != count:
        raise ValueError(
            'Expected ' + str(count) + ' map axes in comparison SVG, found '
            + str(len(bounds)) + '.'
        )
    return bounds


def _matplotlib_path_to_svg_d(path, x, baseline_y):
    """Convert a Matplotlib path to SVG coordinates with a fixed baseline."""
    try:
        from matplotlib.path import Path as MplPath
    except ImportError as error:
        raise RuntimeError(
            'Creating vector panel labels requires Matplotlib. Install the project '
            'requirements before running create_publication_plots.py.'
        ) from error

    commands = []
    for vertices, code in path.iter_segments(curves=True, simplify=False):
        if code == MplPath.MOVETO:
            px, py = vertices
            commands.append('M {:.6f} {:.6f}'.format(x + px, baseline_y - py))
        elif code == MplPath.LINETO:
            px, py = vertices
            commands.append('L {:.6f} {:.6f}'.format(x + px, baseline_y - py))
        elif code == MplPath.CURVE3:
            cx, cy, px, py = vertices
            commands.append(
                'Q {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                    x + cx, baseline_y - cy, x + px, baseline_y - py
                )
            )
        elif code == MplPath.CURVE4:
            c1x, c1y, c2x, c2y, px, py = vertices
            commands.append(
                'C {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                    x + c1x, baseline_y - c1y,
                    x + c2x, baseline_y - c2y,
                    x + px, baseline_y - py,
                )
            )
        elif code == MplPath.CLOSEPOLY:
            commands.append('Z')
    return ' '.join(commands)


def _add_vector_label(root, label, x, y, anchor='baseline'):
    """Add a panel label as outlines so SVG and PDF placement is identical."""
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath
    except ImportError as error:
        raise RuntimeError(
            'Creating vector panel labels requires Matplotlib. Install the project '
            'requirements before running create_publication_plots.py.'
        ) from error

    text_path = TextPath(
        (0.0, 0.0),
        '(' + label + ')',
        size=7.0,
        prop=FontProperties(family='DejaVu Sans', weight='bold'),
        usetex=False,
    )
    bounds = text_path.get_extents()

    # TextPath uses Cartesian coordinates (positive y is up); SVG uses positive y down.
    # For the former dominant-baseline="hanging" case, keep y as the visual top.
    baseline_y = y if anchor == 'baseline' else y + bounds.ymax
    path = ET.SubElement(root, '{' + SVG_NAMESPACE + '}path')
    path.set('d', _matplotlib_path_to_svg_d(text_path, x, baseline_y))
    path.set('fill', '#000000')


def _add_panel_label(root, label, panel_y, panel_height, map_bottom_fraction):
    _add_vector_label(
        root,
        label,
        3 * POINTS_PER_MM,
        panel_y + panel_height * map_bottom_fraction - 2.5 * POINTS_PER_MM,
    )


def _add_axes_label(root, label, x, y):
    _add_vector_label(
        root,
        label,
        x + 1.5 * POINTS_PER_MM,
        y - 2.5 * POINTS_PER_MM,
    )


def create_publication_plot_1(case_study, plots_directory, output_path=None):
    source_paths = [
        os.path.join(plots_directory, case_study + '_total_costs_compact.svg'),
        os.path.join(plots_directory, case_study + '_routes_compact.svg'),
        os.path.join(plots_directory, case_study + '_transport_costs_compact.svg'),
    ]
    missing_paths = [path for path in source_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            'Missing compact SVG source plots:\n' + '\n'.join(missing_paths)
        )

    panels = []
    last_panel_index = len(source_paths) - 1
    for panel_index, source_path in enumerate(source_paths):
        root, view_box, width_points, height_points = _read_svg(source_path)
        axes_bottom = _primary_axes_bottom(root, view_box)
        crop_top_mm = (
            FIGURE_1_FIRST_TOP_CROP_MM,
            FIGURE_1_SECOND_TOP_CROP_MM,
            FIGURE_1_THIRD_TOP_CROP_MM,
        )[panel_index]
        crop_bottom_mm = (
            FIGURE_1_LAST_BOTTOM_CROP_MM
            if panel_index == last_panel_index
            else PANEL_CROP_BOTTOM_MM
        )
        crop_top_points = crop_top_mm * POINTS_PER_MM
        crop_bottom_points = crop_bottom_mm * POINTS_PER_MM
        cropped_height_points = height_points - crop_top_points - crop_bottom_points
        if cropped_height_points <= 0:
            raise ValueError('SVG is too short for publication cropping: ' + source_path)
        units_per_point = view_box[3] / height_points
        cropped_view_box = (
            view_box[0],
            view_box[1] + crop_top_points * units_per_point,
            view_box[2],
            view_box[3] - (crop_top_points + crop_bottom_points) * units_per_point,
        )
        map_bottom_fraction = (
            (axes_bottom - cropped_view_box[1]) / cropped_view_box[3]
        )
        panels.append({
            'root': root,
            'view_box': cropped_view_box,
            'width': width_points,
            'height': cropped_height_points,
            'map_bottom_fraction': min(1.0, max(0.0, map_bottom_fraction)),
        })

    common_width = max(panel['width'] for panel in panels)
    natural_heights = [
        panel['height'] * common_width / panel['width']
        for panel in panels
    ]
    panel_gaps = [gap * POINTS_PER_MM for gap in FIGURE_1_PANEL_GAPS_MM]
    maximum_height = MAXIMUM_HEIGHT_MM * POINTS_PER_MM
    output_width = common_width
    panel_heights = natural_heights
    output_height = sum(panel_heights) + sum(panel_gaps)
    if output_height > maximum_height + 1e-9:
        warnings.warn(
            'Publication plot 1 is '
            + format(output_height / POINTS_PER_MM, '.2f')
            + ' mm high after removing the white panel margins; maximum is '
            + format(MAXIMUM_HEIGHT_MM, '.1f')
            + ' mm. The SVG will still be exported.',
            RuntimeWarning,
            stacklevel=2,
        )

    output_root = ET.Element('{' + SVG_NAMESPACE + '}svg', {
        'version': '1.1',
        'width': str(output_width) + 'pt',
        'height': str(output_height) + 'pt',
        'viewBox': '0 0 ' + str(output_width) + ' ' + str(output_height),
    })

    panel_y = 0.0
    pending_labels = []
    for index, (panel, panel_height) in enumerate(zip(panels, panel_heights)):
        source_root = copy.deepcopy(panel['root'])
        _prefix_svg_ids(source_root, 'publication_panel_' + str(index + 1) + '_')
        nested_svg = ET.SubElement(output_root, '{' + SVG_NAMESPACE + '}svg', {
            'x': '0',
            'y': str(panel_y),
            'width': str(output_width),
            'height': str(panel_height),
            'viewBox': ' '.join(str(value) for value in panel['view_box']),
            'preserveAspectRatio': 'xMinYMin meet',
            'overflow': 'hidden',
        })
        for child in list(source_root):
            nested_svg.append(child)
        pending_labels.append(
            (
                chr(ord('a') + index),
                panel_y,
                panel_height,
                panel['map_bottom_fraction'],
            )
        )
        if index < len(panel_gaps):
            panel_y += panel_height + panel_gaps[index]

    for label, label_panel_y, panel_height, map_bottom_fraction in pending_labels:
        _add_panel_label(
            output_root,
            label,
            label_panel_y,
            panel_height,
            map_bottom_fraction,
        )

    if output_path is None:
        output_path = os.path.join(
            plots_directory,
            case_study + '_overview_costs_publication.svg',
        )
    return _write_publication_outputs(output_root, output_path)


def create_publication_plot_2(
    comparison_id,
    plots_directory,
    case_study_count=4,
    output_path=None,
):
    comparison_id = str(comparison_id).strip()
    if not comparison_id:
        raise ValueError('comparison_id must not be empty.')
    if case_study_count not in {2, 4}:
        raise ValueError('case_study_count must be 2 or 4.')
    source_paths = [
        os.path.join(
            plots_directory,
            comparison_id + '_total_supply_costs_comparison.svg',
        ),
        os.path.join(
            plots_directory,
            comparison_id + '_energy_carrier_comparison.svg',
        ),
    ]
    missing_paths = [path for path in source_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            'Missing comparison SVG source plots:\n' + '\n'.join(missing_paths)
        )

    panels = []
    for source_path in source_paths:
        root, view_box, width_points, height_points = _read_svg(source_path)
        panels.append({
            'root': root,
            'view_box': view_box,
            'width': width_points,
            'height': height_points,
            'axes_bounds': _map_axes_bounds(root, count=case_study_count),
        })

    common_width = max(panel['width'] for panel in panels)
    panel_heights = [
        panel['height'] * common_width / panel['width']
        for panel in panels
    ]
    panel_gap = PANEL_GAP_MM * POINTS_PER_MM
    output_width = common_width
    output_height = sum(panel_heights) + panel_gap
    maximum_height = MAXIMUM_HEIGHT_MM * POINTS_PER_MM
    if output_height > maximum_height + 1e-9:
        warnings.warn(
            'Publication plot 2 is '
            + format(output_height / POINTS_PER_MM, '.2f')
            + ' mm high; maximum is '
            + format(MAXIMUM_HEIGHT_MM, '.1f')
            + ' mm. The SVG will still be exported.',
            RuntimeWarning,
            stacklevel=2,
        )

    output_root = ET.Element('{' + SVG_NAMESPACE + '}svg', {
        'version': '1.1',
        'width': str(output_width) + 'pt',
        'height': str(output_height) + 'pt',
        'viewBox': '0 0 ' + str(output_width) + ' ' + str(output_height),
    })

    panel_y = 0.0
    label_index = 0
    for index, (panel, panel_height) in enumerate(zip(panels, panel_heights)):
        source_root = copy.deepcopy(panel['root'])
        _prefix_svg_ids(source_root, 'publication_figure_2_panel_' + str(index + 1) + '_')
        nested_svg = ET.SubElement(output_root, '{' + SVG_NAMESPACE + '}svg', {
            'x': '0',
            'y': str(panel_y),
            'width': str(output_width),
            'height': str(panel_height),
            'viewBox': ' '.join(str(value) for value in panel['view_box']),
            'preserveAspectRatio': 'xMinYMin meet',
            'overflow': 'visible',
        })
        for child in list(source_root):
            nested_svg.append(child)

        view_box_x, view_box_y, view_box_width, view_box_height = panel['view_box']
        for axes_bounds in panel['axes_bounds']:
            axes_left, _, _, axes_bottom = axes_bounds
            label_x = (axes_left - view_box_x) / view_box_width * output_width
            label_y = (
                panel_y
                + (axes_bottom - view_box_y) / view_box_height * panel_height
            )
            _add_axes_label(
                output_root,
                chr(ord('a') + label_index),
                label_x,
                label_y,
            )
            label_index += 1
        panel_y += panel_height + panel_gap

    if output_path is None:
        output_path = os.path.join(
            plots_directory,
            'comparisons_publication_' + comparison_id + '.svg',
        )
    return _write_publication_outputs(output_root, output_path)


def create_publication_plot_3(case_study, plots_directory, output_path=None):
    case_study = str(case_study).strip()
    if not case_study:
        raise ValueError('case_study must not be empty.')
    source_path = os.path.join(
        plots_directory,
        case_study + '_distribution_cost_and_quantities.svg',
    )
    if not os.path.isfile(source_path):
        raise FileNotFoundError('Missing distribution SVG source plot:\n' + source_path)

    root, view_box, width_points, height_points = _read_svg(source_path)
    axes_bounds = _map_axes_bounds(root, count=6)
    output_root = copy.deepcopy(root)
    _, _, view_box_width, view_box_height = view_box
    x_units_per_point = view_box_width / width_points
    y_units_per_point = view_box_height / height_points

    for index, axes_rectangle in enumerate(axes_bounds):
        axes_left, _, _, axes_bottom = axes_rectangle
        _add_vector_label(
            output_root,
            chr(ord('a') + index),
            axes_left + 0.5 * POINTS_PER_MM * x_units_per_point,
            axes_bottom - 1.5 * POINTS_PER_MM * y_units_per_point,
        )

    if output_path is None:
        output_path = os.path.join(
            plots_directory,
            case_study + '_cost_distribution_publication.svg',
        )
    return _write_publication_outputs(output_root, output_path)


def create_publication_plot_4(
    comparison_id,
    country,
    plots_directory,
    case_study_count=4,
    output_path=None,
):
    comparison_id = str(comparison_id).strip()
    country = str(country).strip()
    if not comparison_id:
        raise ValueError('comparison_id must not be empty.')
    if not country:
        raise ValueError('country must not be empty.')
    if case_study_count not in {2, 4}:
        raise ValueError('case_study_count must be 2 or 4.')
    source_path = os.path.join(
        plots_directory,
        comparison_id + '_' + country + '_supply_curves_comparison.svg',
    )
    if not os.path.isfile(source_path):
        raise FileNotFoundError('Missing supply-curve SVG source plot:\n' + source_path)

    root, view_box, width_points, height_points = _read_svg(source_path)
    axes_bounds = _map_axes_bounds(root, count=case_study_count)
    output_root = copy.deepcopy(root)
    _, _, view_box_width, view_box_height = view_box
    x_units_per_point = view_box_width / width_points
    y_units_per_point = view_box_height / height_points

    for index, axes_rectangle in enumerate(axes_bounds):
        axes_left, axes_top, _, _ = axes_rectangle
        _add_vector_label(
            output_root,
            chr(ord('a') + index),
            axes_left + 0.5 * POINTS_PER_MM * x_units_per_point,
            axes_top + 0.5 * POINTS_PER_MM * y_units_per_point,
            anchor='top',
        )

    if output_path is None:
        output_path = os.path.join(
            plots_directory,
            'supply_curves_comparison_publication_' + country + '_' + comparison_id + '.svg',
        )
    return _write_publication_outputs(output_root, output_path)
