# %%
import pandas as pd
import pcs_functions as pcsf

# %%
all_riders_df = pd.read_csv("data/uae_2021_sprints.csv")
# collect stage information
df = pd.DataFrame()
# for stage_num in range(2,8): # stages 2 (skip missing data - gc rankings) to 7
for stage_num in [1,4,6,7]:
  df = df.append(pcsf.get_race_info(f'race/uae-tour/2021/stage-{stage_num}'))

df = df.merge(all_riders_df, left_on='rider_id', right_on='rider_rider_id', how='left')
df.to_csv("data/uae_2021_sprints.csv", index=False)

df.isnull().sum()

# TODO: get rid of "id" field when merging