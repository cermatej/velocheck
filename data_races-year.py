# %%
import pandas as pd
import pcs_functions as pcsf

# scrape whole PRO calendar for defined years with game picks
races = pd.DataFrame()

game_picks = pd.DataFrame()
for year in range(2016, 2020 + 1): # seasons 16-20s
  year_races = pd.DataFrame()
  races_page = pcsf.get_pcs_page(f'races.php?year={year}&circuit=1&class=&filter=Filter')
  race_links = [a.attrs.get('href') for a in races_page.find('tbody').find_all('a') if a.attrs.get('href').startswith('race')]
  # races
  for race_link in race_links:
    page = pcsf.get_pcs_page(f'{race_link}/gc/overview')
    stages_h3 = page.find("h3", text="Stages")

    if stages_h3:  # race has stages
      st_links = [link.attrs.get('href') for link in stages_h3.next_sibling.find_all('a')]
      st_links = [l for l in st_links if 'stage' in l]  # filter out non-stage links
      for stage_link in st_links:
        year_races = year_races.append(pcsf.get_race_info(stage_link, debug=True))
        game_picks = game_picks.append(pcsf.scrape_game_picks(stage_link))
    else:  # is single race
      year_races = year_races.append(pcsf.get_race_info(race_link, debug=True))
      game_picks = game_picks.append(pcsf.scrape_game_picks(race_link))

  year_races['race_year'] = year
  races = races.append(year_races)

races.to_csv("data/races_df.csv", index_label='id')
game_picks.to_csv("data/gamepicks_df.csv", index_label='id')