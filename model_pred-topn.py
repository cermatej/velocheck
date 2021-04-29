# %%
import pickle
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder

pd.options.mode.chained_assignment = None

PRED_I = 'pred_target'
RACE_ID = 'race_id'

# %%
################ RIDERS DATA PREPARATION
race_riders = pd.read_csv("data/race_riders.csv")
# specialty points per year
# todo move to feature extraction section
sp_cols = [col for col in race_riders.columns if col.startswith('rider_sp_')]
season_cols = [col for col in race_riders.columns if col.startswith('rider_season_pts_')]
sp_sum = race_riders['rider_sp_Time trial'] + race_riders['rider_sp_GC'] + race_riders['rider_sp_Climber'] + \
         race_riders['rider_sp_Sprint'] + race_riders['rider_sp_One day races']
sp_sum = sp_sum.replace(0, 1)
for season_col in season_cols:
  for col in sp_cols:
    race_riders[f'{season_col}_{col}'] = (race_riders[col] / sp_sum) * race_riders[season_col]
race_riders.drop(columns=sp_cols + season_cols, inplace=True)

race_riders.isnull().sum()

# %%
################ STAGES DATA PREPARATION
races = pd.read_csv("data/races_df.csv")
# stages['race_profile'].fillna("flat", inplace=True) # note: stages without race_profile were flat for 2019
races.isnull().sum()

# %%
################ USER PICKS IN PCS GAME
gamepicks = pd.read_csv("data/gamepicks_df.csv")
gamepicks.drop(columns=['id'], inplace=True)
gamepicks.isnull().sum()
# %%

################ MERGING DATA
# stages_f = stages[(stages['race_profile'].isin(['flat'])) & stages['race_type'].isin(['oneday', 'stage'])]
# stages_f = stages[(stages['race_profile'].isin(['hills-flatf', 'flat'])) & stages['race_type'].isin(['oneday', 'stage'])]

# NOTE: filter out only flat/hills-flatf stages & oneday, single stage races
stages_f = races[(races['race_profile-score'] <= 5) & races['race_type'].isin(['oneday', 'stage'])]

# MERGE: riders
df = stages_f.merge(race_riders, left_on=['rider_id', 'race_year'], right_on=['rider_rider_id', 'rider_stats_year'],
                    how='left')
# MERGE: gamepicks
df = df.merge(gamepicks, left_on=['rider_id', 'race_id'], right_on=['pick_rider_id', 'race_id'], how='left')

################ FEATURE EXTRACTION

# longer the race, the skills are more important
season_pts_cols = [col for col in df.columns if col.startswith('rider_season_pts_')]
for col in season_pts_cols:
  df[col] = df[col] * df['race_distance']

# extract team name (teams might have different names during the years)
df['rider_recent_team'] = df['rider_recent_team'].str.extract(r'team/(.*)-\d{4}')

################ FILL MISSING VALUES

# fill values with mean
season_rank_cols = [col for col in df.columns if col.startswith('rider_season_rank_')]
for col in ['rider_height', 'rider_weight', 'rider_uci_rank', 'rider_pcs_rank', 'race_stage-num'] + season_rank_cols:
  df[col].fillna(df[col].mean(), inplace=True)

# fill with zeros
zerofill_cols = ['rider_pcs_points', 'pick_npicks']
pts_cols = [c for c in df.columns if c.startswith('rider_season_pts')]
for col in zerofill_cols + pts_cols:
  df[col].fillna(0, inplace=True)

# fill categorical features with modus
for col in ['rider_recent_team', 'race_profile']:
  mod_val = df[col].value_counts().sort_values(ascending=False).index[0]
  print(mod_val)
  df[col].fillna(mod_val, inplace=True)

################ FEATURE SELECTION
target = 'rider_pcs_points'  # target = 'pick_npicks'
race_cols = [col for col in df.columns if
             col.startswith("race_")]  # note: same for all riders for race, need to be combined with other features
# rider_season_rank_cols = [col for col in df.columns if col.startswith("rider_season_rank")]
excluded_features = race_cols + [target, 'rider_birth_dt', 'rider_id', 'rider_rider_id',
                                 'rider_name', 'race_id',
                                 'rider_pcs_points', 'id_x', 'id_y', 'rider_stats_year',
                                 'id.1', 'pick_rider_id']

features = list(set(df.columns) - set(excluded_features))
df[features].isnull().sum()

# %%
################ TRAIN TEST SPLIT BY RACES
test_frac = 0.2  # 80/20 train test split cross validation, predicting every race from test iteratively
u_races = df[RACE_ID].sample(frac=1, random_state=111).unique()  # shuffle dataset
nu_races = len(u_races)
test_size = math.ceil(len(u_races) * test_frac)
test_races = list(u_races[0:test_size])

train = df[~df[RACE_ID].isin(test_races)].copy()
X_train = train[features].copy()
y_train = train[target].copy()

test = df[df[RACE_ID].isin(test_races)].copy()
X_test = test[features].copy()
y_test = test[target].copy()

print(f'Split: train ({len(train[RACE_ID].unique())} unique races), test ({len(test[RACE_ID].unique())} unique races)')
# %%
f_to_scale = list(X_train.columns[X_train.dtypes != 'object'])

################ VARIABLE ENCODING
categorical = ['rider_recent_team', 'rider_nationality']
for c in categorical:
  encoder = TargetEncoder()
  encoder.fit(X_train[c], y_train)  # calculate means of target on training data (=fit the encoder)
  X_train[f'{c}_mean'] = encoder.transform(X_train[c])  # map the training data means to training data
  X_test[f'{c}_mean'] = encoder.transform(X_test[c])  # map the training data means to test data

X_train.drop(columns=categorical, inplace=True)
X_test.drop(columns=categorical, inplace=True)

################ VARIABLE SCALING

for f in f_to_scale:
  norm = MinMaxScaler()
  norm.fit(X_train[[f]])
  X_train[f'{f}_s'] = norm.transform(X_train[[f]])
  X_test[f'{f}_s'] = norm.transform(X_test[[f]])

X_train.drop(columns=f_to_scale, inplace=True)
X_test.drop(columns=f_to_scale, inplace=True)

# %%

# ProfileReport(in_df, minimal=True).to_file("report.html")
corr = df[features + [target]].corr().sort_values(target)[target]

# %%

################ MODELLING

################ RFR GRID SEARCH
hyper_grid = {'n_estimators': [1000, 2000],
              'max_features': ['auto', 10], # default
              'min_samples_split': [4, 6]}
rf = RandomForestRegressor()
gs = GridSearchCV(estimator=rf,
                  param_grid=hyper_grid,
                  cv=7,
                  n_jobs=-1,
                  verbose=3)
gs.fit(X_train, y_train.values.ravel())

# %%
rfr = gs.best_estimator_
y_pred = rfr.predict(X_test)
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')

# %%
################# SAVE MODEL
with open(r"models/rfr_6_400_10f.pickle", "wb") as out_file:
  pickle.dump(rfr, out_file)
# %%
################# LOAD MODEL
with open(r"models/rfr_6_400_10f.pickle", "rb") as out_file:
  rfr = pickle.load(out_file)
# %%

def pred_topn(tst, model, top_n=10, debug=False):
  accs = []
  u_test_races = list(tst[RACE_ID].unique())
  for i, pred_race in enumerate(u_test_races):
    test_race_df = tst[tst[RACE_ID] == pred_race].copy()
    X_test_race = test_race_df.drop(columns=COLS_TO_ADD).copy()

    test_race_df[PRED_I] = model.predict(X_test_race)  # predict values

    top_pred = test_race_df.sort_values(PRED_I, ascending=False).head(top_n)
    i_acc = sum(top_pred['race_race_rank'] <= top_n) / top_n
    accs.append(i_acc)

    if (debug):
      print(
        f"[{i + 1}/{len(u_test_races)}] Race: {pred_race}")
      print(top_pred[['rider_rider_id', 'race_race_rank', PRED_I]])
      print(f"Accuracy: {i_acc}")
  print(f"""
  Predicted top competitors: {top_n}
  Unique races: {len(u_test_races)}
  AVG accuracy: {np.mean(accs)}
  Model: {model}
  """)

tst = X_test.copy()
COLS_TO_ADD = [RACE_ID, 'race_race_rank', 'rider_rider_id']
tst[COLS_TO_ADD] = test[COLS_TO_ADD]
for top_n in [10]:
  pred_topn(tst, model=rfr, top_n=top_n, debug=True)

# %%
def pred_topn_split(data, model, top_n=10, test_frac=.2, debug=False):
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


# pred_topn_split(df, top_n=10, model = RandomForestRegressor(random_state=111, n_estimators=100))
for top_n in [10]:
  # for model in [LinearRegression(), RandomForestRegressor(random_state=111, n_estimators=100)]:
  for model in [RandomForestRegressor(random_state=111)]:
    pred_topn_split(df, model, top_n=top_n, debug=True)

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
