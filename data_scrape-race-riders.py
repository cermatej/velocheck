import pandas as pd
import pcs_functions as pcsf

# scrape all rider info for stages file
races = pd.read_csv("data/races_df.csv")
race_riders_df = pd.DataFrame()
for year in races['race_year'].unique():

  races_year = races[races['race_year'] == year]
  riders_data = []
  for rider in races_year['rider_id'].unique():
    riders_data.append(pcsf.scrape_rider(rider, year))
  race_riders = pd.DataFrame(riders_data)
  race_riders['rider_stats_year'] = year
  race_riders_df = race_riders_df.append(race_riders)

race_riders_df.to_csv("data/race_riders.csv", index_label='id')

