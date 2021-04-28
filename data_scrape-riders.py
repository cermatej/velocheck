import requests
from bs4 import BeautifulSoup
import pandas as pd
import pcs_functions as pcsf

teams_url = f'{pcsf.PCS_URL}/teams.php'

teams = BeautifulSoup(requests.get(teams_url).content, 'html.parser')

uci_list = teams.find("h3", text="UCI WorldTeams").next_sibling.find_all('li')
uci_list.extend(teams.find("h3", text="UCI ProTeams").next_sibling.find_all('li'))

riders_data = []

for team in uci_list:
  href = team.find("a").attrs.get('href')
  team_url = f'{pcsf.PCS_URL}/{href}'

  team = BeautifulSoup(requests.get(team_url).content, 'html.parser')
  riders = team.find("div", class_='right').find_all("ul", class_='list')[1].find_all('li')

  for rider in riders:
    rider_href = rider.find("a").attrs.get('href')

    riders_data.append(pcsf.scrape_rider(rider_href))

riders_df = pd.DataFrame(riders_data)

# specialty points
# riders_df = pcsf.rider_extract_features(riders_df)

# save to csv
riders_df.to_csv("data/uci_riders.csv", index_label='id')

