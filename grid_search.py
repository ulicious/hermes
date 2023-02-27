import plotly.express as px
import pandas as pd

# --------------------- Erstellung Übersichtsweltkarte ------------------------------------

df = pd.read_excel('location_list.xlsx')

fig = px.scatter_geo(df, lat=df.lat, lon=df.lon,
                     color=df['[€/MWh]'], color_continuous_scale='RdYlGn_r', # size of markers
                     hover_name="country", # column added to hover information
                     size=df.point,
                     projection="natural earth")
fig.update_traces(marker=dict(line=dict(width=0)),
                  selector=dict(mode='markers'))
fig.update_traces(marker={'size': 20})
fig.show()