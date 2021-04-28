import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

# substrings in race type (only "special" - not stage races)
RACE_TYPE_DEFINITIONS = {
  'One day race': 'oneday',
  'ITT': 'itt',
  'TTT': 'ttt'
}
RACE_TYPE_DEFAULT = 'stage'

PROFILE_DEFINITIONS = {
  'p1': 'flat',
  'p2': 'hills-flatf',
  'p3': 'hills-uphillf',
  'p4': 'mountains-flatf',
  'p5': 'mountains-uphillf'
}
RESULT_TABLE_TYPES = {
  'RACE': 0,
  'GC': 1
}
PCS_URL = 'https://www.procyclingstats.com'
CURRENT_YEAR = datetime.now().year


def get_pcs_page(url_suffix):
  full_url = f'{PCS_URL}/{url_suffix}'
  return BeautifulSoup(requests.get(full_url).content, 'html.parser')

def scrape_game_picks(stage_url):
  picked_page = get_pcs_page(f'{stage_url}/game/most-picked')
  picks_table = picked_page.find("table", class_="basic")
  if not picks_table: # no results found (canceled races etc.) > skip scraping
    return pd.DataFrame()
  pick_list = []
  picks = picks_table.find_all('tr')
  for pick in picks[1:]:
    pick_cols = pick.find_all("td")
    pick_list.append({
      'pick_rider_id': pick_cols[1].find('a').get('href'),
      'pick_npicks': pick_cols[2].get_text()
    })

  picks_df = pd.DataFrame(pick_list)
  picks_df['race_id'] = stage_url
  return picks_df


def scrape_rankings_table(stage_url, results_type):
  # get page
  stage = get_pcs_page(stage_url)
  result_tables = stage.find_all("table", class_="results")
  if not result_tables: # no results found (canceled races etc.) > skip scraping
    return pd.DataFrame()

  res_table = result_tables[RESULT_TABLE_TYPES[results_type]]

  results_list = []
  rankings = res_table.find_all('tr')
  headers = [th.get_text() for th in rankings[0].find_all('th')]
  if "Pnt" not in headers or not len(rankings) > 1:
    return pd.DataFrame()
  pnt_index = headers.index("Pnt")
  for rank in rankings[1:]:
    r = {
      'rider_id': rank.find_all('a')[0].get('href'),
      'rank': rank.find_all('td')[0].get_text()
    }
    if results_type == "RACE":
      r['rider_pcs_points'] = rank.find_all('td')[pnt_index].get_text()
    results_list.append(r)

  results_df = pd.DataFrame(results_list)
  # filter out not numeric 'rank' field (do not start, do not finish etc.)
  results_df = results_df[results_df['rank'].apply(lambda x: x.isnumeric())]
  results_df['rank'] = results_df['rank'].astype(int)
  results_df.rename(columns={'rank': f'race_{results_type.lower()}_rank'}, inplace=True)
  return results_df


def get_stage_profile(stage):
  profile_class = \
    [classname for classname in stage.find('span', class_='profile').get('class') if re.match('p\d', classname)][0]
  return PROFILE_DEFINITIONS[profile_class] if profile_class in PROFILE_DEFINITIONS.keys() else np.nan


def get_race_info(stage_link, debug=False):
  if (debug):
    print(f"Scraping race info: {stage_link}")

  stage_ranking = scrape_rankings_table(stage_link, 'RACE')

  if stage_ranking.empty:
    return stage_ranking

  stage = get_pcs_page(stage_link)  # pull stage information

  race_type_text = stage.find('div', class_='page-title').find('span', class_='blue').get_text()
  stage_num_scrape = [int(s) for s in race_type_text.split() if s.isdigit()]

  race_type = [RACE_TYPE_DEFINITIONS[key] for key in RACE_TYPE_DEFINITIONS.keys() if key in race_type_text]


  st_df = stage_ranking
  st_df['race_id'] = stage_link
  st_df['race_stage-num'] = stage_num_scrape[0] if stage_num_scrape else np.nan # stage number or NaN
  st_df['race_profile'] = get_stage_profile(stage)
  st_df['race_type'] = str(race_type[0]) if race_type else RACE_TYPE_DEFAULT
  st_df['race_profile-score'] = int(stage.find('div', text='ProfileScore: ').parent.find_all('div')[1].get_text())
  st_df['race_distance'] = stage.find('div', text='Distance: ').parent.find_all('div')[1].get_text().split(' ')[0]
  st_df['race_ranking'] = stage.find('div', text='Race ranking:').next_sibling.next_sibling.get_text()

  # if (stage_num > 1):
  #   gc_stand_df = scrape_rankings_table(f'{stage_url_pref}{stage_num-1}', 'GC')
  #   st_df = st_df.merge(gc_stand_df, on='rider_id', how='left')
  # else:
  # st_df['race_gc-rank'] = np.nan

  return st_df


def scrape_rider(rider_href, year=CURRENT_YEAR):
  rider_url = f'{PCS_URL}/{rider_href}'

  GTS_LIST = ['tour-de-france', "giro-d-italia", 'vuelta-a-espana']
  soup = BeautifulSoup(requests.get(rider_url).content, 'html.parser')

  info = soup.find("div", class_="rdr-info-cont")

  name = soup.find('h1').get_text()
  print(f"Scraping rider: {name}")

  # try to get team of specific year, if not - get the last one
  teams_lst = soup.find('ul', class_='rdr-teams')
  year_elm = teams_lst.find('div', text=year)
  if year_elm:
    team = year_elm.parent.find('a').attrs.get('href')
  else:
    all_teams = teams_lst.find_all('a')
    if all_teams:
      team = all_teams[0].attrs.get('href') # get first team
    else:
      team = None

  height_string = info.find("b", text="Height:")
  height = float(re.search('([\d.]+) m', height_string.next_sibling).group(1)) * 100 if height_string else None

  weight_string = info.find("b", text="Weight:")
  weight = float(re.search('([\d.]+) kg', weight_string.next_sibling).group(1)) if weight_string else None

  nationality = info.find("span", class_="flag").next_sibling.get_text()

  # birth
  date_string = info.find("b", text="Date of birth:")
  birth_day = date_string.next_sibling
  birth_month_year = " ".join(birth_day.next_sibling.next_sibling.split()[:2])
  birth_dt = datetime.strptime(f"{birth_day.strip()} {birth_month_year.strip()}", "%d %B %Y")

  # age
  age = (datetime.now() - birth_dt) // timedelta(days=365.2425)

  # grand tour starts
  gt_starts_page = BeautifulSoup(requests.get(f'{rider_url}/statistics/grand-tour-starts').content, 'html.parser')
  gt_table = gt_starts_page.find('table', class_='basic')
  gt_starts = []
  for gt_item in gt_table.find_all('a'):
    gt_starts.append(gt_item.attrs.get('href').split('/')[1])
  gt_starts = Counter(gt_starts)

  # rankings - UCI
  uci_rank_title = soup.find('ul', class_='rdr-rankings').find("a", text="UCI World Ranking")
  uci_rank = int(uci_rank_title.parent.next_sibling.get_text()) if uci_rank_title else None

  # rankings - PCS
  pcs_rank_title = soup.find('ul', class_='rdr-rankings').find("a", text="PCS Ranking")
  pcs_rank = int(pcs_rank_title.parent.next_sibling.get_text()) if pcs_rank_title else None

  # seasons
  season_rank = {}
  seasons_tbl = soup.find('table', class_='rdr-season-stats')
  years_ago = 4
  for i, year in enumerate(range(year - years_ago, year + 1)): # # years ago up until current season
    season_title = seasons_tbl.find("td", text=str(year))
    col_suffix = f'-{years_ago-i}'
    season_rank[f'season_pts_{col_suffix}'] = season_title.parent.find("div",
                                                                   class_="barCont").get_text() if season_title else 0
    season_rank[f'season_rank_{col_suffix}'] = season_title.parent.find("td",
                                                                    class_="ac").get_text() if season_title else None

  # key statistics
  wins = int(soup.find("h3", text='Key statistics').parent.find("a", text="Wins").parent.previous_sibling.get_text())

  # speciality points
  sp_points = {}
  spec_points = soup.find("h3", text='Points per specialty').parent.find('ul').find_all('li')
  for points in spec_points:
    title = points.find("div", class_="title").get_text()
    pts = int(points.find("div", class_="pnt").get_text())
    sp_points[f'sp_{title}'] = pts if pts else 0

  rider_data = {
    'rider_id': rider_href,
    'name': name,
    'recent_team': team,
    'birth_dt': birth_dt,
    'age': age,
    'height': height,
    'weight': weight,
    'nationality': nationality,
    'n_part_tdf': gt_starts[GTS_LIST[0]],
    'n_part_giro': gt_starts[GTS_LIST[1]],
    'n_part_vuelta': gt_starts[GTS_LIST[2]],
    'uci_rank': uci_rank,
    'pcs_rank': pcs_rank,
    'wins': wins
  }

  return {**rider_data, **sp_points, **season_rank}

# deprecated
def   rider_extract_features(r_df):
  # specialty points
  sp_cols = [col for col in r_df.columns if col.startswith('sp_') and not col.endswith("One day races")]
  r_df['sp_sum'] = r_df['sp_Time trial'] + r_df['sp_GC'] + r_df['sp_Climber'] + r_df['sp_Sprint']
  for col in sp_cols:
    r_df[f'{col}_perc'] = r_df[col] / r_df['sp_sum']
    r_df[f'{col}_perc'].fillna(0, inplace=True)
  # riders_df.drop(columns=sp_cols, inplace=True)

  # add prefix to riders
  r_df = r_df.add_prefix("rider_")

  return r_df

