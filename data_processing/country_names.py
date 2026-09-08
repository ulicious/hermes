import re
import unicodedata

import pandas as pd


COUNTRY_ALIAS_COLUMNS = (
    'NAME_EN', 'NAME', 'NAME_LONG', 'FORMAL_EN', 'BRK_NAME', 'ABBREV',
    'POSTAL', 'ADM0_A3', 'ISO_A2', 'ISO_A3', 'WB_A2', 'WB_A3',
)


def normalize_country_key(value):
    """Normalize spelling and punctuation for country-name lookup only."""
    if value is None or pd.isna(value):
        return None
    normalized = unicodedata.normalize('NFKD', str(value))
    normalized = ''.join(character for character in normalized
                         if not unicodedata.combining(character))
    return re.sub(r'[^a-z0-9]+', ' ', normalized.casefold()).strip() or None


def build_country_aliases(world):
    """Map Natural Earth names and country codes to its canonical NAME_EN."""
    if world is None or world.empty or 'NAME_EN' not in world.columns:
        raise ValueError('Natural Earth country data with NAME_EN is required.')
    aliases = {}
    ambiguous = set()
    for _, country in world.iterrows():
        canonical = country['NAME_EN']
        if pd.isna(canonical):
            continue
        canonical = str(canonical).strip()
        for column in COUNTRY_ALIAS_COLUMNS:
            if column not in world.columns:
                continue
            alias = country[column]
            key = normalize_country_key(alias)
            if key is None or key == '99':
                continue
            existing = aliases.get(key)
            if existing is not None and existing != canonical:
                aliases.pop(key, None)
                ambiguous.add(key)
                continue
            if key not in ambiguous:
                aliases[key] = canonical

    # Common source-data variants which are not consistent across Natural
    # Earth releases and port datasets.
    us_canonical = next((aliases[key] for key in (
        normalize_country_key('United States of America'),
        normalize_country_key('United States'),
        normalize_country_key('USA'),
        normalize_country_key('US'),
    ) if key in aliases), 'United States of America')
    for alias in ('US', 'U.S.', 'USA', 'U.S.A.', 'United States',
                  'The United States', 'The United States of America'):
        aliases[normalize_country_key(alias)] = us_canonical
    return aliases


def canonicalize_country(value, world=None, aliases=None):
    """Return the Natural Earth NAME_EN spelling for a country value."""
    if value is None or pd.isna(value):
        return None
    if aliases is None:
        aliases = build_country_aliases(world)
    return aliases.get(normalize_country_key(value), str(value).strip())


def canonicalize_country_column(data, column, world):
    """Canonicalize a country column immediately after importing it."""
    result = data.copy()
    if column not in result.columns:
        return result
    aliases = build_country_aliases(world)
    result[column] = result[column].map(
        lambda value: canonicalize_country(value, aliases=aliases))
    return result


def validate_canonical_country_column(data, column, world, source_name):
    """Reject generated files whose country labels are not Natural Earth NAME_EN."""
    if column not in data.columns:
        return
    canonical_names = set(world['NAME_EN'].dropna().astype(str))
    invalid = sorted({str(value) for value in data[column].dropna()
                      if str(value) not in canonical_names})
    if invalid:
        preview = ', '.join(invalid[:10])
        raise ValueError(
            source_name + ' contains non-canonical country names: ' + preview
            + '. Regenerate the file so countries use Natural Earth NAME_EN.')


def canonicalize_country_index(data, world):
    """Canonicalize a country-indexed input table and reject collisions."""
    result = data.copy()
    aliases = build_country_aliases(world)
    result.index = [canonicalize_country(value, aliases=aliases) for value in result.index]
    duplicates = result.index[result.index.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError('Country-name canonicalization created duplicate rows: '
                         + ', '.join(map(str, duplicates)))
    return result


def canonicalize_country_mapping(country_states, world):
    """Canonicalize country keys while retaining configured state selections."""
    aliases = build_country_aliases(world)
    result = {}
    for country, states in country_states.items():
        canonical = canonicalize_country(country, aliases=aliases)
        if canonical in result and result[canonical] != states:
            raise ValueError('Conflicting state selections for country aliases: '
                             + str(canonical))
        result[canonical] = states
    return result
