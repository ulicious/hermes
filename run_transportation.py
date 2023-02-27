import pandas as pd
from joblib import Parallel, delayed
import time, math
from main import transportation

# ---------------------------------------------------------- Modul für multiprocessing des Tools  ----------------------------------------------------------------------------------------------
# Umstellung der transportation Funktion auf zwei Inputwerte notwendig. Startland wird dann aus Inputtabelle 'grid_search_points' eingelesen, da beim multi-processing immer der geolocator versagt hat


# input:
grid_search_points = pd.read_excel('input_data/grid_search_points3.xlsx', sheet_name='Tabelle1')

# output:
location_list = pd.DataFrame({'region':[], 'country':[], 'coordinates': [], 'costs [€/MWh]': []}, dtype=object)


region = 'Africa'
coordinates = []
countries = []

for c, coordinate in enumerate(grid_search_points['country']):
    if grid_search_points.iloc[c, 4] == region:
        coordinates.append((grid_search_points.iloc[c, 2], grid_search_points.iloc[c, 1]))
        countries.append(grid_search_points.iloc[c, 3])

print(coordinates)
start = time.time()

# n_jobs is the number of parallel jobs
results = Parallel(n_jobs=4)(delayed(transportation)(start_coordinates, country) for start_coordinates, country in zip(coordinates, countries))

end = time.time()
minutes = (end-start)/60
print()
print('Runtime: {:.4f} min'.format(minutes))
print(results)

for co in range(len(results)):
    location_list.loc[co] = region, countries[co], coordinates[co], results[co]

name = region + '_location_list.xlsx'
location_list.to_excel(name)