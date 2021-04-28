# %%
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn import tree
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

PRED_I = 'pred_target'
RACE_ID = 'race_id'

# %%

# RIDERS DATA PREPARATION

race_riders = pd.read_csv("data/race_riders.csv")

# fill missing data with mean
season_rank_cols = [col for col in race_riders.columns if col.startswith('rider_season_rank_')]
for col in ['rider_height', 'rider_weight', 'rider_uci_rank', 'rider_pcs_rank'] + season_rank_cols:
  race_riders[col].fillna(race_riders[col].mean(), inplace=True)

# fill categorical with modus
for col in ['rider_recent_team']:
  med_val = race_riders['rider_recent_team'].value_counts().sort_values(ascending=False).index[0]
  race_riders[col].fillna(med_val, inplace=True)

for col in [c for c in race_riders.columns if c.startswith('rider_season_pts')]:
  race_riders[col].fillna(0, inplace=True)
race_riders.isnull().sum()

# specialty points per year
sp_cols = [col for col in race_riders.columns if col.startswith('rider_sp_')]
season_cols = [col for col in race_riders.columns if col.startswith('rider_season_pts_')]
sp_sum = race_riders['rider_sp_Time trial'] + race_riders['rider_sp_GC'] + race_riders['rider_sp_Climber'] + \
         race_riders['rider_sp_Sprint'] + race_riders['rider_sp_One day races']
sp_sum = sp_sum.replace(0, 1)
for season_col in season_cols:
  for col in sp_cols:
    race_riders[f'{season_col}_{col}'] = (race_riders[col] / sp_sum) * race_riders[season_col]
race_riders.drop(columns=sp_cols + season_cols, inplace=True)
race_riders.describe()

# %%

# STAGES DATA PREPARATION
races = pd.read_csv("data/races_df.csv")
# todo note: stages without race_profile were flat for 2019
# stages['race_profile'].fillna("flat", inplace=True)

for col in ['rider_pcs_points', 'race_stage-num', 'race_profile']:
  races[col].fillna(0, inplace=True)

races.isnull().sum()

# %%

gamepicks = pd.read_csv("data/gamepicks_df.csv")
gamepicks.drop(columns=['id'], inplace=True)

gamepicks.isnull().sum()
# %%

# INPUT DATA PREPARATION

# stages_f = stages[(stages['race_profile'].isin(['flat'])) & stages['race_type'].isin(['oneday', 'stage'])]
# stages_f = stages[(stages['race_profile'].isin(['hills-flatf', 'flat'])) & stages['race_type'].isin(['oneday', 'stage'])]
# NOTE: filter out only flat/hills-flatf stages & oneday, single stage races
stages_f = races[(races['race_profile-score'] <= 40) & races['race_type'].isin(['oneday', 'stage'])]

# merge: riders
df = stages_f.merge(race_riders, left_on=['rider_id', 'race_year'], right_on=['rider_rider_id', 'rider_stats_year'],
                    how='left')

# merge: gamepicks
df = df.merge(gamepicks, left_on=['rider_id', 'race_id'], right_on=['pick_rider_id', 'race_id'], how='left')
df['pick_npicks'].fillna(0, inplace=True)  # add zeros to nonpicked riders

# longer the race, the skills are more important
season_pts_cols = [col for col in df.columns if col.startswith('rider_season_pts_')]
for col in season_pts_cols:
  df[col] = df[col] * df['race_distance']

# introduce dummy variables
df['rider_recent_team'] = df['rider_recent_team'].str.extract(r'team/(.*)-\d{4}')  # extract name
categorical = ['rider_recent_team', 'rider_nationality']
# for col in categorical:
#   dummies = pd.get_dummies(df[col])
#   df = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
df.drop(columns=categorical, inplace=True)

target = 'rider_pcs_points'  # target = 'pick_npicks'
race_cols = [col for col in df.columns if
             col.startswith("race_")]  # todo note: same for all riders, need to be combined with other features
rider_season_rank_cols = [col for col in df.columns if col.startswith("rider_season_rank")]
excluded_features = rider_season_rank_cols + race_cols + [target, 'rider_birth_dt', 'rider_id', 'rider_rider_id',
                                                          'rider_name', 'race_id',
                                                          'race_type', 'race_profile', 'race_race_rank',
                                                          'rider_pcs_points', 'id_x', 'id_y', 'rider_stats_year',
                                                          'id.1', 'pick_rider_id']

features = list(set(df.columns) - set(excluded_features))

# %%

# ProfileReport(in_df, minimal=True).to_file("report.html")
corr = df[features + [target]].corr().sort_values(target)[target]


# %%

def pred_topn(data, model, top_n=10, test_frac=.2, debug=False):
  u_races = data[RACE_ID].sample(frac=1, random_state=111).unique()  # shuffle dataset
  nu_races = len(u_races)
  # 80/20 train test split cross validation, predicting every race from test iteratively
  test_size = math.ceil(len(u_races) * test_frac)

  accs = []
  # for every CV interval
  for pred_j in range(math.ceil(nu_races / test_size)):

    sel_fr = pred_j * test_size
    sel_to = (pred_j + 1) * test_size if (pred_j + 1) * test_size < nu_races else nu_races
    test_races = list(u_races[sel_fr:sel_to])

    if (debug):
      print(f'Testing on races {sel_fr}-{sel_to}/{nu_races}')

    train = data[~data[RACE_ID].isin(test_races)]
    X_train = train[features]
    y_train = train[target]
    # fit the model for the training data
    pl = Pipeline(steps=[
      ('scaler', StandardScaler()),
      ('model', model)
    ])
    pl.fit(X_train, y_train)

    for i, pred_race in enumerate(test_races):
      test = data[data[RACE_ID] == pred_race]
      X_test = test[features]

      test[PRED_I] = pl.predict(X_test)  # predict values

      top_pred = test.sort_values(PRED_I, ascending=False).head(top_n)
      i_acc = sum(top_pred['race_race_rank'] <= top_n) / top_n
      accs.append(i_acc)
      if (debug):
        print(
          f"[{i + 1}/{len(test_races)}] Race: {pred_race} (ProfileScore: {int(test['race_profile-score'].head(1))})")
        print(top_pred[['rider_rider_id', 'race_race_rank', PRED_I]])
        print(f"Accuracy: {i_acc}")

  print(f"""
    Predicted top competitors: {top_n}
    Unique races: {len(u_races)}
    Overall CV accuracy: {np.mean(accs)}
    Model: {model}
    Test frac.: {test_frac}
  """)

  if (debug and isinstance(pl.named_steps['model'], LinearRegression)):
    coef = pd.DataFrame()
    coef['features'] = features
    coef['coef'] = pl.named_steps['model'].coef_
    print(coef.sort_values('coef'))


# pred_topn(df, top_n=10, model = RandomForestRegressor(random_state=111, n_estimators=100))
for top_n in [10]:
  # for model in [LinearRegression(), RandomForestRegressor(random_state=111, n_estimators=100)]:
  for model in [RandomForestRegressor(random_state=111)]:
    pred_topn(df, model, top_n=top_n, debug=True)

# %%

# GRID SEARCH
data = df
test_frac=0.2
u_races = data[RACE_ID].sample(frac=1, random_state=111).unique()  # shuffle dataset
nu_races = len(u_races)
test_size = math.ceil(len(u_races) * test_frac)
pred_j = 0
sel_fr = pred_j * test_size
sel_to = (pred_j + 1) * test_size if (pred_j + 1) * test_size < nu_races else nu_races
test_races = list(u_races[sel_fr:sel_to])

print(f'Testing on races {sel_fr}-{sel_to}/{nu_races}')

train = data[~data[RACE_ID].isin(test_races)]
X_train = train[features]
y_train = train[target]
# fit the model for the training data
pl = Pipeline(steps=[
  ('scaler', StandardScaler()),
  ('model', model)
])
pl.fit(X_train, y_train)

for i, pred_race in enumerate(test_races):
  test = data[data[RACE_ID] == pred_race]
  X_test = test[features]

  test[PRED_I] = pl.predict(X_test)  # predict values

  top_pred = test.sort_values(PRED_I, ascending=False).head(top_n)
  i_acc = sum(top_pred['race_race_rank'] <= top_n) / top_n

  print(
    f"[{i + 1}/{len(test_races)}] Race: {pred_race} (ProfileScore: {int(test['race_profile-score'].head(1))})")
  print(top_pred[['rider_rider_id', 'race_race_rank', PRED_I]])
  print(f"Accuracy: {i_acc}")


# %%
# import statsmodels.api as sm
#
# X = df[features]
# y = df[target]
#
# X2 = sm.add_constant(X)
# est = sm.OLS(y, X2)
# est2 = est.fit()
# print(est2.summary())

# %%

# mlp = MLPClassifier(random_state=1).fit(X_train, y_train)
# mlp = MLPRegressor(random_state=123, max_iter=500).fit(X_train, y_train)
# y_pred = mlp.predict_proba(X_test)

# rf = RandomForestClassifier().fit(X_train, y_train)
# y_pred = rf.predict_proba(X_test)

# res = test.copy()
# res['prob'] = [x[1] for x in y_pred]
# # res['prob'] = y_pred
# res = res[['name', 'prob', 'stage_rank']].sort_values('prob', ascending=False)
# res.head(10)

# %%
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy_score(y_test, y_pred)
