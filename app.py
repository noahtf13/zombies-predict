import pandas as pd
from pandas.util.testing import assert_frame_equal
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import GridSearchCV
import math
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import itertools
import s3fs
import pickle
import datetime
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)
server = app.server

def check_df():
    googleSheetId = '19Cr_YXoGf-mvrEHtPUxdxIOIezs4ozwDMgH0OijhBsI'
    worksheetName = 'Sheet1'
    URL = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
        googleSheetId,
        worksheetName
    )
    df = pd.read_csv(URL).dropna()
    df = df[df['beat_game_attempt'] == 0]
    df.index = range(0, len(df))
    s3 = s3fs.S3FileSystem(anon=False)
    try:
        df_s3 = pd.read_csv('s3://zomb-model-storage/db_extract.csv')
        assert_frame_equal(df, df_s3)

        return df, True
    except:
        with s3.open('s3://zomb-model-storage/db_extract.csv', 'w') as f:
            df.to_csv(f, index=False)
            return df, False


def clean_df(df):
    train = df.drop(['beat_game_attempt', 'win_game', 'rounds_completed', 'date'], axis=1)
    label = np.log(df['rounds_completed'])
    return train, label

def train(use_pickle=0):
    df, answer = check_df()
    train, label = clean_df(df)
    if use_pickle == 0:
        best_model_reg, best_model_raw = gscv(train, label, df)
        fs = s3fs.S3FileSystem(anon=False)
        for model in ['reg','raw']:
            if model == 'reg':
                trained_model = best_model_reg
            else:
                trained_model = best_model_raw
            filename = f"zomb-model-storage/lr_save_{model}.sav"
            pickle.dump(trained_model, fs.open(filename, 'wb'))
    else:
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open('s3://zomb-model-storage/lr_save_reg.sav', 'rb') as handle:
                best_model_reg = pickle.load(handle)
        with fs.open('s3://zomb-model-storage/lr_save_raw.sav', 'rb') as handle:
                best_model_raw = pickle.load(handle)
    for model in ['reg','raw']:
        if model == 'reg':
            model_file = best_model_reg
            reg_coeffs, reg_inter, reg_model = coeffs(train,model_file)
        else:
            model_file = best_model_raw
            raw_coeffs, raw_inter, raw_model = coeffs(train,model_file)
    return reg_coeffs, reg_inter, reg_model, raw_coeffs, raw_inter, raw_model


def coeffs(train, model):
    coeffs = (
        pd.concat(
            [
                pd.DataFrame(train.columns),
                pd.DataFrame(np.transpose(model.coef_))
            ], axis=1
        )
    )
    coeffs.columns = ['variable', 'rounds_added']
    coeffs['rounds_added'] = math.e ** coeffs['rounds_added']
    intercept = [math.e ** model.intercept_]
    inter_df = pd.DataFrame({'intercept': intercept})
    return coeffs, inter_df, model


def gscv(train, label, df):
    multiply = df['rounds_completed'].mean()/math.e
    for model in ['raw','reg']:
        if model == 'reg':
            parameters = {'alpha': [0.1, 1, multiply, 10]}
            model = Ridge()
            gscv_reg = GridSearchCV(model, parameters, scoring='r2', cv=int(round(1 / math.log(len(train)) * 15, 0)))
            gscv_reg.fit(train, label)
        else:
            parameters = {'alpha': [0]}
            model = Ridge()
            gscv_raw = GridSearchCV(model, parameters, scoring='r2', cv=int(round(1 / math.log(len(train)) * 15, 0)))
            gscv_raw.fit(train, label)


    return gscv_reg.best_estimator_, gscv_raw.best_estimator_



app.layout = html.Div([
    dcc.Markdown(open('instructions.markdown', 'r').read()),
    dcc.Dropdown(
        id='regularization-drop',
        options=[
            {'label': 'Least Error', 'value': 'least_error'},
            {'label': 'Raw', 'value': 'raw'}
        ],
        value='least_error'
    ),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1"),
        fullscreen=True
    ),
    dcc.Graph(id="coeff-graph", animate=False),
    html.H2('Select who is playing below:'),
    dcc.Checklist(
        id='playing',
        options=[
            {'label': 'Will', 'value': 'will'},
            {'label': 'Stefan', 'value': 'stefan'},
            {'label': 'Noah', 'value': 'noah'}
        ],
        value=['will', 'stefan', 'noah']
    ),
    dcc.Slider(
        id='slider',
        min=1,
        max=50,
        step=1,
        value=10,
        marks=dict([(i, str(i)) for i in range(0, 50, 5)]),
        tooltip={'always_visible': True}
    ),
    html.Div(
        dcc.Markdown('Desired Rounds Predicted (Put slider at 0 for hardest round possible)'),
        style=dict(display='flex', justifyContent='center')
    ),
    dcc.Markdown(id='hard-rounds'),
    html.H5('Barriers Used:'),
    html.Ul(id='var_list'),
    html.H5('Barriers Not Used:'),
    html.Ul(id='unused-var-list')
])



@app.callback(
    [
        dash.dependencies.Output('coeff-graph', 'figure'),
        dash.dependencies.Output('hard-rounds', 'children'),
        dash.dependencies.Output('var_list', 'children'),
        dash.dependencies.Output('unused-var-list', 'children'),
        dash.dependencies.Output("loading-output-1", "children")
    ],
    [
        dash.dependencies.Input('regularization-drop', 'value'),
        dash.dependencies.Input('playing', 'value'),
        dash.dependencies.Input('slider', 'value'),
    ]
)

def update_graph_scatter(regularization, playing, slider):
    begin_time = datetime.datetime.now()
    df, no_update = check_df()
    if no_update == True:
        print('pandas df was the same')
        try:
            reg_coeffs, reg_inter, reg_model, raw_coeffs, raw_inter, raw_model = train(use_pickle=1)
            print('pickle used')
        except Exception as e:
            print(e)
            reg_coeffs, reg_inter, reg_model, raw_coeffs, raw_inter, raw_model = train(use_pickle=0)
            print('pickle not used')
    else:
        print('pandas df needed to be updated')
        reg_coeffs, reg_inter, reg_model, raw_coeffs, raw_inter, raw_model = train(use_pickle=0)
        print('pickle not used')
    if regularization == 'least_error':
        coeffs, intercept, best_model = reg_coeffs, reg_inter, reg_model
    else:
        coeffs, intercept, best_model = raw_coeffs, raw_inter, raw_model

    fig = px.bar(coeffs, x='rounds_added', y='variable',
                 title=f"The intercept is: {round(intercept['intercept'][0], 1)}")
    fig.update_xaxes(title='Round Multiplier')
    fig.update_yaxes(title='Challenge/Variable')
    fig.update_layout(xaxis=dict(range=[coeffs['rounds_added'].min()*.975, coeffs['rounds_added'].max()*1.025]))
    name_list = ['noah_playing', 'stefan_playing', 'will_playing']
    if playing is None:
        playing = []
    name_cols = [f'{name}_playing' for name in playing]
    if name_cols is None:
        name_cols = []
    name_dict = {}
    for name in name_list:
        if name in name_cols:
            name_dict[name] = 1
        else:
            name_dict[name] = 0
    col_list = coeffs['variable'].tolist()
    lst = [list(i) for i in itertools.product([0, 1], repeat=len(col_list))]
    all_possible = pd.DataFrame(lst, columns=col_list)
    all_possible['prediction'] = math.e**best_model.predict(all_possible)
    relev_games_df = (
        all_possible[
            (all_possible['noah_playing'] == name_dict['noah_playing']) &
            (all_possible['will_playing'] == name_dict['will_playing']) &
            (all_possible['stefan_playing'] == name_dict['stefan_playing'])
        ]
    )
    relev_games_df['round_diff'] = abs(relev_games_df['prediction'] - slider)
    best_game = relev_games_df[relev_games_df['round_diff'] == relev_games_df['round_diff'].min()]
    s = best_game.iloc[0]
    barriers = s.index.values[(s == 1)]
    final_barriers = [barrier for barrier in barriers if barrier not in name_list]
    neg_barr_names = [html.Li(x) for x in final_barriers]
    unused_barr = [html.Li(x) for x in col_list if (x not in final_barriers) and (x not in name_list)]
    num_rounds = "**Chosen Game - Predicted Rounds: **" + str(round(best_game['prediction'].tolist()[0], 1))
    print(datetime.datetime.now() - begin_time)
    loading = ""
    return fig, num_rounds, neg_barr_names, unused_barr, loading



if __name__ == '__main__':
    app.run_server(debug=True,port=1234)