import json
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
from dash.exceptions import PreventUpdate
import os
import time
import scipy.stats as stats

pd.set_option('display.max_columns', None)


app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.MINTY]
)
server = app.server
app.title = "Zombies Predict"
app.css.config.serve_locally = True

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

def rarity_score(df,rarity_dict):
    row_rarity = []
    rarity_col = []
    for index,row in df.iterrows():
        for column in df.columns:
            if column not in ['will_playing', 'stefan_playing', 'noah_playing']:
                row_rarity.append((row[column] * rarity_dict[column] * 10) ** 2)
        rarity_col.append(np.sum(row_rarity)/len(row_rarity))
    df['row_rarity'] = rarity_col
    return df

def train():
    df, answer = check_df()
    train, label = clean_df(df)
    rarity_dict = create_rarity_dict(train)
    best_model_reg, best_model_raw, sd_reg, sd_raw = gscv(train, label, df)
    for model in ['reg','raw']:
        if model == 'reg':
            model_file = best_model_reg
            reg_coeffs, reg_inter, reg_model = coeffs(train,model_file)
        else:
            model_file = best_model_raw
            raw_coeffs, raw_inter, raw_model = coeffs(train,model_file)
    return reg_coeffs, reg_inter, reg_model, raw_coeffs, raw_inter, raw_model, rarity_dict, sd_reg, sd_raw

def create_rarity_dict(df):
    rarity_dict = {}
    for column in df.columns:
        if column not in ['will_playing', 'stefan_playing', 'noah_playing']:
            rarity_dict[column] = 1 - df[column].mean()
    return rarity_dict


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
    for model in ['raw','reg']:
        if model == 'reg':
            parameters = {'alpha': [.1,1,5,10,100,1000]}
            model = Ridge()
            gscv_reg = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error', cv=4)
            gscv_reg.fit(train, label)
            sd_reg = gscv_reg.best_score_
            print(gscv_reg.best_params_)
        else:
            parameters = {'alpha': [0]}
            model = Ridge()
            gscv_raw = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error', cv=4)
            gscv_raw.fit(train, label)
            sd_raw = gscv_raw.best_score_


    return gscv_reg.best_estimator_, gscv_raw.best_estimator_, sd_reg, sd_raw

PLOTLY_LOGO = "https://www.flaticon.com/svg/static/icons/svg/218/218153.svg"

app.layout = html.Div([
    dbc.Navbar(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=PLOTLY_LOGO, height="50px")),
                        dbc.Col(dbc.NavbarBrand("Zombies Predictions", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://plot.ly",
            ),
            dbc.NavbarToggler(id="navbar-toggler")
        ],
        color="#78c2ad",
        dark=True,
        sticky='top'
    ),
    dcc.Markdown(open('instructions.markdown', 'r').read()),
    dcc.Dropdown(
        id='regularization-drop',
        options=[
            {'label': 'Least Error', 'value': 'least_error'},
            {'label': 'Raw', 'value': 'raw'}
        ],
        value='least_error'
    ),
    dcc.Graph(id="coeff-graph", animate=False),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id="loading-output-1"),
        fullscreen=False
    ),
    dcc.Loading(
        id="loading-2",
        type="default",
        children=html.Div(id="loading-output-2"),
        fullscreen=False
    ),
    dcc.Loading(
        id="loading-3",
        type="default",
        children=html.Div(id="loading-output-3"),
        fullscreen=False
    ),
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
    html.Ul(id='unused-var-list'),
    dcc.Graph(id="cdf", animate=False),
    html.Div(id='datasets', style={'display': 'none'}),
    html.Div(id='blah', style={'display': 'none'}),

])

@app.callback(
    [
        dash.dependencies.Output('datasets', 'children'),
        dash.dependencies.Output("loading-output-1", "children")

    ],
        dash.dependencies.Input('blah', 'children')
)



def update_model(value):
    reg_coeffs, reg_inter, reg_model, raw_coeffs, raw_inter, raw_model, rarity_dict, sd_reg, sd_raw = train()
    sd_df = pd.DataFrame(columns=["sd_reg","sd_raw"])
    sd_df.loc[len(sd_df)] = [sd_reg,sd_raw]
    print(sd_df)
    col_list = reg_coeffs['variable'].tolist()
    lst = [list(i) for i in itertools.product([0, 1], repeat=len(col_list))]
    all_possible_raw = pd.DataFrame(lst, columns=col_list).sample(3000)
    all_possible_raw = rarity_score(all_possible_raw,rarity_dict)
    all_possible_le = all_possible_raw.copy()
    for model in ["raw","least_error"]:
        if model == "raw":
            all_possible_raw['prediction'] = (math.e**raw_model.predict(all_possible_raw.loc[:, all_possible_raw.columns != 'row_rarity'])).round(0)
            all_possible_raw = all_possible_raw.sort_values('row_rarity', ascending=False).drop_duplicates(['will_playing','noah_playing','stefan_playing','prediction'])
        else:
            all_possible_le['prediction'] = (math.e**reg_model.predict(all_possible_le.loc[:, all_possible_le.columns != 'row_rarity'])).round(0)
            all_possible_le = all_possible_le.sort_values('row_rarity', ascending=False).drop_duplicates(['will_playing','noah_playing','stefan_playing','prediction'])
    datasets = {
        'reg_coeffs': reg_coeffs.to_json(orient='split'),
        'reg_inter': reg_inter.to_json(orient='split'),
        'raw_coeffs': raw_coeffs.to_json(orient='split'),
        'raw_inter': raw_inter.to_json(orient='split'),
        'all_possible_raw': all_possible_raw.to_json(orient='split'),
        'all_possible_le': all_possible_le.to_json(orient='split'),
        'sd_df': sd_df.to_json(orient='split')
    }
    loading = ""
    return json.dumps(datasets), loading

@app.callback(
    [
        dash.dependencies.Output('coeff-graph', 'figure'),
        dash.dependencies.Output("loading-output-2", "children")
    ],
    [
        dash.dependencies.Input('regularization-drop', 'value'),
        dash.dependencies.Input('datasets', 'children'),
    ]
)


def update_graph(regularization, json_datasets):
    datasets = json.loads(json_datasets)
    if regularization == 'raw':
        coeffs = pd.read_json(datasets['raw_coeffs'], orient='split')
        intercept = pd.read_json(datasets['raw_inter'], orient='split')
    else:
        coeffs = pd.read_json(datasets['reg_coeffs'], orient='split')
        intercept = pd.read_json(datasets['reg_inter'], orient='split')

    fig = px.bar(coeffs, x='rounds_added', y='variable',
                 title=f"Intercept: {round(intercept['intercept'][0], 1)}", color_discrete_sequence =['#78c2ad']*len(coeffs),
)
    fig.update_xaxes(title='Round Multiplier')
    fig.update_yaxes(title='Challenge/Variable')
    fig.update_layout(xaxis=dict(range=[coeffs['rounds_added'].min() * .9, coeffs['rounds_added'].max() * 1.1]))
    loading = ""
    return fig, loading




@app.callback(
    [
        dash.dependencies.Output('hard-rounds', 'children'),
        dash.dependencies.Output('var_list', 'children'),
        dash.dependencies.Output('unused-var-list', 'children'),
        dash.dependencies.Output("loading-output-3", "children"),
        dash.dependencies.Output("cdf", "figure")
    ],
    [
        dash.dependencies.Input('slider', 'value'),
        dash.dependencies.Input('regularization-drop', 'value'),
        dash.dependencies.Input('datasets', 'children'),
        dash.dependencies.Input('playing', 'value'),
    ]
)

def update_chosen_round(slider,regularization, json_datasets, playing):
    datasets = json.loads(json_datasets)
    sd_df = pd.read_json(datasets['sd_df'], orient='split')
    if regularization == 'least_error':
        all_possible = pd.read_json(datasets['all_possible_le'], orient='split')
        reg_coeffs = pd.read_json(datasets['reg_coeffs'], orient='split')
        reg_inter = pd.read_json(datasets['reg_inter'], orient='split')
        coeffs, intercept = reg_coeffs, reg_inter
        sd = sd_df['sd_reg'].tolist()[0]
    else:
        all_possible = pd.read_json(datasets['all_possible_raw'], orient='split')
        raw_coeffs = pd.read_json(datasets['raw_coeffs'], orient='split')
        raw_inter = pd.read_json(datasets['raw_inter'], orient='split')
        coeffs, intercept = raw_coeffs, raw_inter
        sd = sd_df['sd_raw'].tolist()[0]

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
    relev_games_df = (
        all_possible[
            (all_possible['noah_playing'] == name_dict['noah_playing']) &
            (all_possible['will_playing'] == name_dict['will_playing']) &
            (all_possible['stefan_playing'] == name_dict['stefan_playing'])
        ]
    )
    relev_games_df['round_diff'] = abs(relev_games_df['prediction'] - slider)
    best_game = relev_games_df[relev_games_df['round_diff'] == relev_games_df['round_diff'].min()]
    best_game = best_game.drop('round_diff',axis=1)
    s = best_game.iloc[0]
    barriers = s.index.values[(s == 1)]
    final_barriers = [barrier for barrier in barriers if barrier not in name_list]
    neg_barr_names = [html.Li(x) for x in final_barriers]
    col_list = coeffs['variable'].tolist()
    unused_barr = [html.Li(x) for x in col_list if (x not in final_barriers) and (x not in name_list)]
    prediction = math.log(best_game['prediction'].tolist()[0])
    sd = -1 * sd
    num_rounds = "**Chosen Game - Predicted Rounds: **"+ str(int(round(math.e**(prediction))))
    # x = np.linspace(prediction - sd*3, prediction + sd*3, 1000)
    better_num = list(range(math.ceil(math.e**(prediction - 3 * sd)), math.floor(math.e**(prediction + 3 * sd)) + 1))
    log_better = np.log(better_num)
    cdf_df = pd.DataFrame()
    cdf_df['percentile'] = stats.norm.cdf(log_better, prediction, sd)
    cdf_df['possible_game'] = better_num

    fig = px.histogram(
        cdf_df,
        x='possible_game',
        y='percentile',
        title='CDF of Predicted Rounds',
        color_discrete_sequence=['#78c2ad'] * len(cdf_df),
        nbins=len(cdf_df),
    )
    fig.update_xaxes(title="Round Started")
    fig.update_yaxes(title="Percentile")
    loading = ""
    return num_rounds, neg_barr_names, unused_barr, loading, fig



if __name__ == '__main__':
    app.run_server(debug=True,port=1234)