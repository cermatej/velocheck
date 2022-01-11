from pprint import pprint

from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.utils.dates import days_ago

args = {
  'owner': 'airflow',
}

dag = DAG(
  dag_id='riders_pipeline',
  default_args=args,
  schedule_interval=None,
  start_date=days_ago(2),
  tags=['scrape'],
)


def print_context(ds, **kwargs):
  """Print the Airflow context and ds variable from the context."""
  pprint(kwargs)
  print(ds)
  return 'Whatever you return gets printed in the logs'


context = PythonOperator(
  task_id='print_the_context',
  python_callable=print_context,
  dag=dag,
)

def scrape_data():
  from datetime import datetime, timedelta
  import requests
  import re
  from bs4 import BeautifulSoup, NavigableString
  import pandas as pd
  from collections import Counter

  url_base = 'https://www.procyclingstats.com'
  teams_url = f'{url_base}/teams.php'

  teams = BeautifulSoup(requests.get(teams_url).content, 'html.parser')

  wt_title = teams.find("h3", text="UCI WorldTeams")
  wt_list = wt_title.next_sibling.find_all('li')

  riders_data = []

  for team in wt_list:
    href = team.find("a").attrs.get('href')
    team_url = f'{url_base}/{href}'

    team = BeautifulSoup(requests.get(team_url).content, 'html.parser')
    riders = team.find("div", class_='right').find_all("ul", class_='list')[1].find_all('li')

    for rider in riders:
      rider_href = rider.find("a").attrs.get('href')
      rider_url = f'{url_base}/{rider_href}'

      GTS_LIST = ['Tour de France', "Giro d'Italia", 'La Vuelta ciclista a España']
      soup = BeautifulSoup(requests.get(rider_url).content, 'html.parser')

      info = soup.find("div", class_="rdr-info-cont")

      name = soup.find('h1').get_text()
      print(f"Scraping rider: {name}")

      team = soup.find('ul', class_='rdr-teams').find_all('div', class_='name')[0].get_text()

      height_string = info.find("b", text="Height:")
      height = float(re.search('([\d.]+) m', height_string.next_sibling).group(1)) * 100 if height_string else None

      weight_string = info.find("b", text="Weight:")
      weight = float(re.search('([\d.]+) kg', weight_string.next_sibling).group(1)) if weight_string else None

      nationality = info.find("span", class_="flag").next_sibling.get_text()

      # birth
      date_string = info.find("b", text="Date of birth:")
      birth_day = date_string.next_sibling
      birth_month_year = birth_day.next_sibling.next_sibling
      birth_string = re.search("(.*)\s*\(\d+\)", f"{birth_day}{birth_month_year}".strip()).group(1).strip()
      birth_dt = datetime.strptime(birth_string, "%d %B %Y")

      # age
      age = (datetime.now() - birth_dt) // timedelta(days=365.2425)

      # grand tour starts
      gt_starts_page = BeautifulSoup(requests.get(f'{rider_url}/statistics/grand-tour-starts').content, 'html.parser')
      gt_table = gt_starts_page.find('table', class_='basic')
      gt_starts = []
      for gt_item in gt_table.find_all('a'):
        gt_starts.append(gt_item.get_text())
      gt_starts = Counter(gt_starts)

      # rankings
      uci_rank_title = soup.find('ul', class_='rdr-rankings').find("a", text="UCI World Ranking")
      uci_rank = int(uci_rank_title.parent.next_sibling.get_text()) if uci_rank_title else None

      # key statistics
      wins = int(soup.find("h3", text='Key statistics').parent.find("a", text="Wins").parent.previous_sibling.get_text())

      # speciality points
      sp_points = {}
      spec_points = soup.find("h3", text='Points per specialty').parent.find('ul').find_all('li')
      for points in spec_points:
        title = points.find("div", class_="title").get_text()
        pts = int(points.find("div", class_="pnt").get_text())
        sp_points[f'sp_{title}'] = pts

      rider_data = {
        'name': name,
        'recent_team': team,
        'birth_dt': birth_dt,
        'age': age,
        'height': height,
        'weight': weight,
        'nationality': nationality,
        'n_part_tour': gt_starts[GTS_LIST[0]],
        'n_part_giro': gt_starts[GTS_LIST[1]],
        'n_part_vuelta': gt_starts[GTS_LIST[2]],
        'uci_rank': uci_rank,
        'wins': wins
      }

      riders_data.append({**rider_data, **sp_points})

  riders_df = pd.DataFrame(riders_data)
  riders_df.to_csv("riders_df.csv", index_label='id')

virtualenv_scrape = PythonVirtualenvOperator(
  task_id="virtualenv_scrape",
  python_callable=scrape_data,
  requirements=["beautifulsoup4",
                "requests",
                "pandas"],
  system_site_packages=False,
  dag=dag,
)

upload_data_gcs = LocalFilesystemToGCSOperator(
  task_id="upload_file",
  src="riders_df.csv",
  dst="riders_df.csv",
  bucket="velocheck_data",
)

context >> virtualenv_scrape >> upload_data_gcs
