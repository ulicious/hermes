import pandas as pd
import matplotlib.pyplot as plt
from road_route import truck_transport
from pipelines import pipeline_new, pipeline_repurposed
from shipping import shipping


# Plots für Darstellung Transportkosten in Abhängigkeit von Entfernung und Kapazität


pipelines = pd.read_excel('input_data/network_pipelines_gas.xlsx')
data_costs = pd.read_excel('input_data/Data Aggregation.xlsx', sheet_name='working data')

index_conditioning = 9
index_truck = 51
index_pipeline_new = 32
index_pipeline_repurposed = 38
index_shipping = 44

conditioning_data = data_costs.iloc[index_conditioning]
print(data_costs.iloc[index_conditioning, 0])
print(data_costs.iloc[index_truck, 0])
print(data_costs.iloc[index_pipeline_new, 0])
print(data_costs.iloc[index_pipeline_repurposed, 0])
print(data_costs.iloc[index_shipping, 0])
conditioning_cost = conditioning_data['Total process costs  [€/MWh]']
# conversion_capacity = conditioning_data['Capacity']
# conversion_capacity = 1300

medium = 'NH3'

conversion_capacity = 1300
print('conversion_capacity:', conversion_capacity)
energy_amount = round(conversion_capacity * 8000 / 1000)
distances = [10, 50, 100, 500, 1000, 2000, 5000, 8000]
costs_truck = []
costs_pipeline_new = []
costs_pipeline_repurposed = []
cost_shipping = []

for d in distances:
    duration = d / 45
    costs_truck.append(round(truck_transport(d, duration, index_truck), 2))
    costs_pipeline_new.append(round(pipeline_new(d, medium, index_pipeline_new, conversion_capacity, conditioning_data), 2))
    costs_pipeline_repurposed.append(round(pipeline_repurposed(d, 36, medium, index_pipeline_repurposed, conversion_capacity, conditioning_data, 19753720), 2))
    cost_shipping.append(round(shipping(d, index_shipping), 2))



plt.plot(distances, costs_truck, label="LKW")
plt.plot(distances, costs_pipeline_new, label="neue Pipeline")
plt.plot(distances, costs_pipeline_repurposed, label="bestehende Pipelines")
plt.plot(distances, cost_shipping, label='Schiff')


plt.xlabel('Entfernung [km]')
plt.ylabel('Kosten [€/MW]')
plt.title('Kostenvergleich Transportarten für Ammoniak\n Anlagenkapazität ' + str(conversion_capacity) + 'MW /  Energiemenge ' + str(energy_amount) + ' GWh/Jahr', fontsize=10)
plt.legend()

plt.savefig('C:/Users/gleic/PycharmProjects/transportation_tool' + '/hello', bbox_inches='tight', dpi=100)
plt.show()