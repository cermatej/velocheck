# %%
import pandas as pd
st_df = pd.read_csv("data/uae_2021_sprints.csv")

for stage_num in [2,3,4,5,6,7]:
  ss = st_df[st_df['race_stage-num'] == stage_num].copy()

  # flat prior - every rider has same probability of winning
  ss['P_flat'] = 1/len(ss)
  # riders that have sprinter abilities (speciality points)
  # ss['sp_Sum'] = (ss['sp_Time trial'] + ss['sp_GC'] + ss['sp_Climber'] + ss['sp_Sprint'])
  ss['Pred_sprinter'] = ss['sp_Sprint_perc'] * int(ss['race_profile'].head(1) == "flat")
  ss['Pred_sprinter'].fillna(0, inplace=True)
  # riders, that are meant to be supported in ride
  ss['Pred_sel_sprinter'] = abs(ss['race_is_sprinter_rider'] - .35)
  # number of stage wins
  ss['Pred_wins'] = ss['wins']
  # age - less is better
  # ss['Pred_age'] = max(ss['age']) - ss['age']
  # uci-rank - less is better
  # ss['Pred_ucirank'] = max(ss['uci_rank']) - ss['uci_rank']


  top_n = 10
  predictors = [col for col in list(ss.columns) if col.startswith("Pred")]

  # for pred in predictors:
  #   range = max(ss[pred]) - min(ss[pred])
  #   ss[f"{pred}_s"] = (ss[pred] - min(ss[pred]) + (.4 * range))/range
  # ss.drop(columns=predictors, inplace=True)

  for i, pred in enumerate(predictors):
    prev_posterior = predictors[i-1] if i != 0 else 'P_flat'
    new_i = f'P_{i}'
    new_ic = f'{new_i}c'

    ss[new_i] = ss[prev_posterior] * ss[pred]
    ss[new_ic] = ss[new_i] / sum(ss[new_i])

    # ss = ss.drop(columns=[new_i])
    top = ss.sort_values(new_ic, ascending=False).head(top_n)
    accuracy = sum(top['stage_rank'] <= top_n) / top_n
    print(f"Iteration: {i}, Predictor: {pred}, Top: {top_n}, Accuracy: {accuracy}")


#
# ss['P_2'] = ss['P_1c'] * ss['Pred_sel_sprinter']
# ss['P_2c'] = ss['P_2'] / sum(ss['P_2'])
# top10_2 = ss.sort_values('P_2c', ascending=False).head(10)
# sprinter_bias = 4
# ss['P_sprinter'] = sprinter_bias if ss['race_is_sprinter_rider'] == 1 else 1


