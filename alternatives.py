import pandas as pd
from shipping import shipping

# ------------------------------------- Modul zur Berechnung von Umwandlungsalternativen zum aktuellen Energieträger -----------------------------------------------------------------------

data_costs = pd.read_excel('input_data/Data Aggregation.xlsx', sheet_name='working data')
input_data = pd.read_excel("input_data/Python_Input_Data.xlsx")

requested_medium = input_data.at[0, "Energieträger"]

media_list = pd.DataFrame(input_data['Kürzel'])
media_list['Aggregatzustand'] = input_data['Aggregatzustand']
media_list = media_list.drop(media_list[media_list.Kürzel == 'not defined'].index)

for m, medium in enumerate(media_list['Kürzel']):
    # Datenabfrage
    if medium == 'LH2':
        index_conditioning = data_costs[data_costs.process == 'GH2 liquefication'].index[0]
        index_shipping = data_costs[data_costs.process == 'LH2 shipping'].index[0]
    elif medium == 'CGH2':
        index_conditioning = data_costs[data_costs.process == 'GH2 compression'].index[0]
        index_shipping = False
    elif medium == 'CH4':
        index_conditioning = data_costs[data_costs.process == 'methan synthesis'].index[0]
        index_shipping = False
    elif medium == 'LNG':
        index_conditioning = data_costs[data_costs.process == 'methan liquefication (LNG)'].index[0]
        index_shipping = data_costs[data_costs.process == 'LH2 shipping'].index[0]
    elif medium == 'NH3':
        index_conditioning = data_costs[data_costs.process == 'ammonia synthesis'].index[0]
        index_shipping = data_costs[data_costs.process == 'Ammonia shipping'].index[0]
    elif medium == 'MeOH':
        index_conditioning = data_costs[data_costs.process == 'methanol synthesis'].index[0]
        index_shipping = data_costs[data_costs.process == 'Methanol shipping'].index[0]
    elif medium == 'LOHC DBT':
        index_conditioning = data_costs[data_costs.process == 'Hydrogenation DBT'].index[0]
        index_reconversion = data_costs[data_costs.process == 'Dehydrogenation DBT'].index[0]
        index_shipping = data_costs[data_costs.process == 'LOHC shipping'].index[0]
    elif medium == 'LOHC TOL':
        index_conditioning = data_costs[data_costs.process == 'Hydrogenation TOL'].index[0]
        index_reconversion = data_costs[data_costs.process == 'Dehydrogenation TOL'].index[0]
        index_shipping = data_costs[data_costs.process == 'LOHC shipping'].index[0]
    elif medium == 'FT-fuel':
        index_conditioning = data_costs[data_costs.process == 'FT fuels production'].index[0]
        index_shipping = data_costs[data_costs.process == 'FT fuels shipping'].index[0]


    # Umwandlungkosten zu jeweiligem Energieträger
    conditioning_data = data_costs.iloc[index_conditioning]
    conditioning_cost = conditioning_data['Total process costs  [€/MWh]']
    conversion_capacity = conditioning_data['Capacity']

    if m < 1:
        conditioning_costs = pd.DataFrame({'Energieträger': [medium], 'conditioning cost': conditioning_cost})
        shipping_indices = pd.DataFrame({'Energieträger': [medium], 'index_shipping': index_shipping})
    else:
        conditioning_costs.loc[m] = medium, conditioning_cost
        shipping_indices.loc[m] = medium, index_shipping

conditioning_costs_str = conditioning_costs.to_string(index=False)
shipping_indices_str = shipping_indices.to_string(index=False)
data_update = {'conditioning costs': conditioning_costs_str, 'shipping indices': shipping_indices_str}


def get_alternatives(medium, sea_distance):
    if medium == 'CGH2':
        print('Energieträger CGH2: Kein Schifftransport möglich, Umwandlung erforderlich', '\n', 'Alternativen:')
        if requested_medium == 'LH2' or requested_medium == 'CGH2':               # wenn Wasserstoff gewünscht, keine weitere Umwandlung
            alternatives = ['LH2', 'LOHC DBT', 'LOHC TOL', 'NH3']
        else:
            alternatives = ['LH2', 'MeOH', 'NH3', 'LOHC DBT', 'LOHC TOL', 'FT-fuel']
        for a, alternative in enumerate(alternatives):                              # Überprüfung der Alternativen
            conversion_cost = conditioning_costs[conditioning_costs.Energieträger == alternative]['conditioning cost']  # weitere Umwandlung
            print('Umwandlungskosten CGH2 zu', alternative, ':', conversion_cost.to_string(index=False), '€/MWh')
            index_shipping = shipping_indices[shipping_indices.Energieträger == alternative]['index_shipping']
            shipping_cost = shipping(sea_distance, int(index_shipping))             # Kosten Verschiffung der Alternative
            alternative_cost = round(float(conversion_cost) + shipping_cost, 4)     # Umwandlung + Verschiffung
            print('Umwandlungskosten + Verschiffung:', alternative_cost, '€/MWh')
            if a < 1:
                alternative_costs = pd.DataFrame({'alternative': [alternative], 'conversion_cost': float(conversion_cost),
                                                  'shipping_cost': shipping_cost, 'alternative_cost': alternative_cost})
            else:
                alternative_costs.loc[a] = alternative, float(conversion_cost), shipping_cost, alternative_cost

    elif medium == 'LH2':
        print('Energieträger LH2: Schifftransport möglich, Alternativen möglich:')
        index_shipping = shipping_indices[shipping_indices.Energieträger == medium]['index_shipping']
        index_reconversion = data_costs[data_costs.process == 'LH2 regasification'].index[0]
        reconversion_cost = data_costs.iloc[index_reconversion]['Total process costs  [€/MWh]']  # Verflüssigung vor weiterer Umwandlung
        if requested_medium == 'LH2' or requested_medium == 'CGH2':               # wenn Wasserstoff gewünscht, geringere Auswahl Alternativen
            alternatives = ['LOHC DBT', 'LOHC TOL', 'NH3']
        else:
            alternatives = ['MeOH', 'NH3', 'LOHC DBT', 'LOHC TOL', 'FT-fuel']
        print('Regasifizierungskosten:', round(reconversion_cost, 4), '€/MWh', '\n')
        for a, alternative in enumerate(alternatives):
            conversion_cost = conditioning_costs[conditioning_costs.Energieträger == alternative]['conditioning cost']
            cumulated_conversion_cost = reconversion_cost + float(conversion_cost)
            print('Umwandlungskosten LH2 zu', alternative, ':', round(cumulated_conversion_cost, 4), '€/MWh')
            index_shipping = shipping_indices[shipping_indices.Energieträger == alternative]['index_shipping']
            shipping_cost = shipping(sea_distance, int(index_shipping))
            alternative_cost = round(cumulated_conversion_cost + shipping_cost, 4)
            print('Umwandlungskosten + Verschiffung:', alternative_cost, '€/MWh')
            if a < 1:
                alternative_costs = pd.DataFrame({'alternative': [alternative], 'conversion_cost': cumulated_conversion_cost, 'shipping_cost': shipping_cost, 'alternative_cost': alternative_cost})
            else:
                alternative_costs.loc[a] = alternative, cumulated_conversion_cost, shipping_cost, alternative_cost

    elif medium == 'CH4':
        print('\n', 'Energieträger CH4: Kein Schifftransport möglich, Umwandlung erforderlich', '\n', 'Alternativen:')
        # print('Methan zu Methanol oder Methan zu Wasserstoff und weitere Umwandlung', '\n', 'keine Daten vorhanden', '\n')
        print('Verflüssigung zu LNG')
        alternative = 'LNG'
        index_conversion = data_costs[data_costs.process == 'methan liquefication (LNG)'].index[0]
        conversion_cost = data_costs.iloc[index_conversion]['Total process costs  [€/MWh]']
        index_shipping = data_costs[data_costs.process == 'LH2 shipping'].index[0]
        print('Umwandlungskosten CH4 zu', alternative, ':', round(conversion_cost, 4), '€/MWh')
        shipping_cost = shipping(sea_distance, int(index_shipping))
        alternative_cost = round(conversion_cost + shipping_cost, 4)
        print('Umwandlungskosten + Verschiffung:', alternative_cost, '€/MWh')
        alternative_costs = pd.DataFrame({'alternative': [alternative], 'conversion_cost': conversion_cost, 'shipping_cost': shipping_cost, 'alternative_cost': alternative_cost})

    elif medium == 'NH3':
        print('Energieträger NH3: Schifftransport möglich, Alternativen möglich:')
        index_reconversion = data_costs[data_costs.process == 'ammonia cracking'].index[0]
        reconversion_cost = data_costs.iloc[index_reconversion]['Total process costs  [€/MWh]']  # Rückumwandlung zu H2 vor weiterer Umwandlung
        if requested_medium == 'LH2' or requested_medium == 'CGH2':  # wenn Wasserstoff gewünscht, geringer Auswahl Alternativen
            alternatives = ['LH2', 'LOHC DBT', 'LOHC TOL']
        else:
            alternatives = ['LH2', 'MeOH', 'LOHC DBT', 'LOHC TOL', 'FT-fuel']
        print('Rückumwandlungskosten:', round(reconversion_cost, 4), '€/MWh', '\n')
        for a, alternative in enumerate(alternatives):
            conversion_cost = conditioning_costs[conditioning_costs.Energieträger == alternative]['conditioning cost']
            cumulated_conversion_cost = reconversion_cost + float(conversion_cost)
            print('Umwandlungskosten NH3 zu', alternative, ':', round(cumulated_conversion_cost, 4), '€/MWh')
            index_shipping = shipping_indices[shipping_indices.Energieträger == alternative]['index_shipping']
            shipping_cost = shipping(sea_distance, int(index_shipping))
            alternative_cost = round(cumulated_conversion_cost + shipping_cost, 4)
            print('Umwandlungskosten + Verschiffung:', alternative_cost, '€/MWh')
            if a < 1:
                alternative_costs = pd.DataFrame({'alternative': [alternative], 'conversion_cost': cumulated_conversion_cost,
                     'shipping_cost': shipping_cost, 'alternative_cost': alternative_cost})
            else:
                alternative_costs.loc[a] = alternative, cumulated_conversion_cost, shipping_cost, alternative_cost

    elif medium == 'LOHC DBT':
        index_reconversion = data_costs[data_costs.process == 'Dehydrogenation DBT'].index[0]
        reconversion_cost = data_costs.iloc[index_reconversion]['Total process costs  [€/MWh]']  # Rückumwandlung zu H2 vor weiterer Umwandlung
        print('Energieträger', medium, ': Schifftransport möglich, Alternativen möglich', '\n', 'Alternativen:')
        if requested_medium == 'LH2' or requested_medium == 'CGH2':               # wenn Wasserstoff gewünscht, keine weitere Umwandlung
            alternatives = ['LH2', 'LOHC TOL', 'NH3']
        else:
            alternatives = ['LH2', 'MeOH', 'NH3', 'LOHC TOL', 'FT-fuel']
        print('Rückumwandlungskosten:', round(reconversion_cost, 4), '€/MWh', '\n')
        for a, alternative in enumerate(alternatives):                              # Überprüfung der Alternativen
            conversion_cost = conditioning_costs[conditioning_costs.Energieträger == alternative]['conditioning cost'] # weitere Umwandlung
            cumulated_conversion_cost = reconversion_cost + float(conversion_cost)
            print('Umwandlungskosten', medium, 'zu', alternative, ':', conversion_cost.to_string(index=False), '€/MWh')
            index_shipping = shipping_indices[shipping_indices.Energieträger == alternative]['index_shipping']
            shipping_cost = shipping(sea_distance, int(index_shipping))             # Kosten Verschiffung der Alternative
            alternative_cost = round(float(conversion_cost) + shipping_cost, 4)     # Umwandlung + Verschiffung
            print('Umwandlungskosten + Verschiffung:', alternative_cost, '€/MWh')
            if a < 1:
                alternative_costs = pd.DataFrame({'alternative': [alternative], 'conversion_cost': cumulated_conversion_cost,
                                                  'shipping_cost': shipping_cost, 'alternative_cost': alternative_cost})
            else:
                alternative_costs.loc[a] = alternative, cumulated_conversion_cost, shipping_cost, alternative_cost

    elif medium == 'LOHC TOL':
        index_reconversion = data_costs[data_costs.process == 'Dehydrogenation TOL'].index[0]
        reconversion_cost = data_costs.iloc[index_reconversion]['Total process costs  [€/MWh]']  # Rückumwandlung zu H2 vor weiterer Umwandlung
        print('Energieträger', medium, ': Schifftransport möglich, Alternativen möglich', '\n', 'Alternativen:')
        if requested_medium == 'LH2' or requested_medium == 'CGH2':               # wenn Wasserstoff gewünscht, keine weitere Umwandlung
            alternatives = ['LH2', 'LOHC DBT', 'NH3']
        else:
            alternatives = ['LH2', 'MeOH', 'NH3', 'LOHC DBT', 'FT-fuel']
        print('Rückumwandlungskosten:', round(reconversion_cost, 4), '€/MWh', '\n')
        for a, alternative in enumerate(alternatives):                              # Überprüfung der Alternativen
            conversion_cost = conditioning_costs[conditioning_costs.Energieträger == alternative]['conditioning cost'] # weitere Umwandlung
            cumulated_conversion_cost = reconversion_cost + float(conversion_cost)
            print('Umwandlungskosten', medium, 'zu', alternative, ':', conversion_cost.to_string(index=False), '€/MWh')
            index_shipping = shipping_indices[shipping_indices.Energieträger == alternative]['index_shipping']
            shipping_cost = shipping(sea_distance, int(index_shipping))             # Kosten Verschiffung der Alternative
            alternative_cost = round(float(conversion_cost) + shipping_cost, 4)     # Umwandlung + Verschiffung
            print('Umwandlungskosten + Verschiffung:', alternative_cost, '€/MWh')
            if a < 1:
                alternative_costs = pd.DataFrame({'alternative': [alternative], 'conversion_cost': cumulated_conversion_cost,
                                                  'shipping_cost': shipping_cost, 'alternative_cost': alternative_cost})
            else:
                alternative_costs.loc[a] = alternative, cumulated_conversion_cost, shipping_cost, alternative_cost

    else:
        print('Energieträger:', medium, ': keine Alternative')
        index_shipping = shipping_indices[shipping_indices.Energieträger == medium]['index_shipping']
        alternative_costs = pd.DataFrame()

    return alternative_costs


def calculate_conversion_cost(medium, product):
    energy_carrier = [medium, product]
    for e in range(len(energy_carrier)):
        if energy_carrier[e] == 'LH2' or energy_carrier[e] == 'LNG':
            index_conditioning = data_costs[data_costs.process == 'GH2 liquefication'].index[0]
            index_reconversion = data_costs[data_costs.process == 'LH2 regasification'].index[0]
            reconversion_possible = True
        elif energy_carrier[e] == 'CGH2':
            index_conditioning = data_costs[data_costs.process == 'GH2 compression'].index[0]
            index_reconversion = '-'
            reconversion_possible = True
        elif energy_carrier[e] == 'CH4':
            index_conditioning = data_costs[data_costs.process == 'methan synthesis'].index[0]
            index_reconversion = '-'
            reconversion_possible = False
        elif energy_carrier[e] == 'NH3':
            index_conditioning = data_costs[data_costs.process == 'ammonia synthesis'].index[0]
            index_reconversion = data_costs[data_costs.process == 'ammonia cracking'].index[0]
            reconversion_possible = True
        elif energy_carrier[e] == 'MeOH':
            index_conditioning = data_costs[data_costs.process == 'methanol synthesis'].index[0]
            index_reconversion = '-'
            reconversion_possible = False
        elif energy_carrier[e] == 'LOHC DBT':
            index_conditioning = data_costs[data_costs.process == 'Hydrogenation DBT'].index[0]
            index_reconversion = data_costs[data_costs.process == 'Dehydrogenation DBT'].index[0]
            reconversion_possible = True
        elif energy_carrier[e] == 'LOHC TOL':
            index_conditioning = data_costs[data_costs.process == 'Hydrogenation TOL'].index[0]
            index_reconversion = data_costs[data_costs.process == 'Dehydrogenation TOL'].index[0]
            reconversion_possible = True
        elif energy_carrier[e] == 'FT-fuel':
            index_conditioning = data_costs[data_costs.process == 'FT fuels production'].index[0]
            index_reconversion = '-'
            reconversion_possible = False
        if e == 0:  # medium reconversion to H2 if possible
            if reconversion_possible == False:
                print('keine Umwandlung in anderen Energieträger möglich')
                reconversion_cost = 1000       # sorgt für Überschreitung von Kostenobergrenze, falls gewünschter Energieträger nicht bereitgestellt werden kann
            else:
                if energy_carrier[e] == 'CGH2':
                    reconversion_cost = 0
                else:
                    reconversion_cost = data_costs.iloc[index_reconversion]['Total process costs  [€/MWh]']
            continue

        else:  # conversion to requested medium
            conversion_cost = data_costs.iloc[index_conditioning]['Total process costs  [€/MWh]']

        total_conversion_cost = reconversion_cost + conversion_cost

        return total_conversion_cost




