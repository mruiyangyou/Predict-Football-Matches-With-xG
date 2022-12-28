import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Query specific time team data in database
def select_team_data(data, team, season = None):
    if season:
        return data.loc[(data['Season'] == f'{season}-{season+1}') &((data['opponent'] == team) | (data['team'] == team))]
    else:
        return data.loc[(data['opponent'] == team) | (data['team'] == team)]

# Transform to Particular team xg data
def transform_to_xG(data, team):
    df = select_team_data(data, team)
    df.reset_index(inplace = True)
    for i in range(df.shape[0]):
        if df.loc[i, 'team'] == team:
             pass
        else:
            df.loc[i, ['team_recent_goals', 'team_recent_xG']], df.loc[i, 'oppo_recent_conced'] = df.loc[i, ['oppo_recent_goals', 'oppo_recent_xG']].values,df.loc[i, 'team_recent_conced']
            df.loc[i, ['home', 'team_rank']],df.loc[i, ['oppo_home', 'oppo_rank']] = df.loc[i, ['oppo_home', 'oppo_rank']].values, df.loc[i, ['home', 'team_rank']].values
            df.loc[i, 'team'], df.loc[i, 'opponent'] = df.loc[i, 'opponent'], df.loc[i, 'team']
            df.loc[i, ['team_cumsum_goals', 'team_cumsum_shots','team_cumsum_xg']], df.loc[i, 'oppo_cumsum_conced'] = df.loc[i, ['oppo_cumsum_goals', 'oppo_cumsum_shots','oppo_cumsum_xg']].values, df.loc[i, 'team_cumsum_conced']
            df.loc[i, ['team_cumsum_goal_pergame', 'team_cumsum_shot_pergame','team_cumsum_xg_pergame']], df.loc[i, 'oppo_cumsum_conced_pergame'] = df.loc[i, ['oppo_cumsum_goal_pergame', 'oppo_cumsum_shot_pergame','oppo_cumsum_xg_pergame']].values, df.loc[i, 'team_cumsum_conced_pergame']
            df.loc[i, 'team_history_goals'] = df.loc[i, 'oppo_history_goals']
            df.loc[i, 'team_xG'], df.loc[i, 'oppo_xG'] = df.loc[i, 'oppo_xG'], df.loc[i, 'team_xG']
    df = df[['Season', 'datetime','id', 'team', 'opponent', 'home','team_recent_goals', 'team_recent_xG','oppo_recent_conced', \
            'team_stage', 'team_rank', 'oppo_rank', 'month', 'weekday','game_period', \
            'team_cumsum_goals', 'team_cumsum_shots','team_cumsum_xg', 'team_cumsum_goal_pergame', 'team_cumsum_shot_pergame','team_cumsum_xg_pergame', \
                'oppo_cumsum_conced', 'oppo_cumsum_conced_pergame', 'team_history_goals', 'team_xG']]
    return df

# Utils function
# Label Encode the categorical variable
def encode_coulumns(df, col_list):
    le = LabelEncoder()
    for col in col_list:
        df[col] = le.fit_transform(df[col])
    return df 

# Encode to categorical data
def change_col_category(df, col_list):
    for col in col_list:
        df[col] = df[col].astype('category')
    return df


def xg_predict_pipeline(data, model_choices, model_parameters, 
            encode_list = None, category_list = None, drop_list = None):
    
    if drop_list:
        data.drop(columns = drop_list, inplace = True)
    else:
        pass

    if encode_list:
        data = encode_coulumns(data, encode_list)
    else:
        pass

    if category_list:
        data = change_col_category(data, category_list)

    
