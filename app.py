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


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.MINTY],
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

def train():
    df, answer = check_df()
    train, label = clean_df(df)
    best_model_reg, best_model_raw = gscv(train, label, df)
    fs = s3fs.S3FileSystem(anon=False)
    for model in ['reg','raw']:
        if model == 'reg':
            trained_model = best_model_reg
        else:
            trained_model = best_model_raw
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
        color="primary",
        dark=True,
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
    reg_coeffs, reg_inter, reg_model, raw_coeffs, raw_inter, raw_model = train()
    col_list = reg_coeffs['variable'].tolist()
    lst = [list(i) for i in itertools.product([0, 1], repeat=len(col_list))]
    for model in ["raw","least_error"]:
        if model == "raw":
            all_possible_raw = pd.DataFrame(lst, columns=col_list)
            all_possible_raw['prediction'] = (math.e**raw_model.predict(all_possible_raw)).round(0)
            all_possible_raw = all_possible_raw.drop_duplicates(subset=['will_playing','noah_playing','stefan_playing','prediction'])
        else:
            all_possible_le = pd.DataFrame(lst, columns=col_list)
            all_possible_le['prediction'] = (math.e**raw_model.predict(all_possible_le)).round(0)
            all_possible_le = all_possible_le.drop_duplicates(subset=['will_playing','noah_playing','stefan_playing','prediction'])
    datasets = {
        'reg_coeffs': reg_coeffs.to_json(orient='split'),
        'reg_inter': reg_inter.to_json(orient='split'),
        'raw_coeffs': raw_coeffs.to_json(orient='split'),
        'raw_inter': raw_inter.to_json(orient='split'),
        'all_possible_raw': all_possible_raw.to_json(orient='split'),
        'all_possible_le': all_possible_le.to_json(orient='split')
    }
    print("I was used")
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
                 title=f"The intercept is: {round(intercept['intercept'][0], 1)}")
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
        dash.dependencies.Output("loading-output-3", "children")
    ],
    [
        dash.dependencies.Input('slider', 'value'),
        dash.dependencies.Input('regularization-drop', 'value'),
        dash.dependencies.Input('datasets', 'children'),
    ],
    [
        dash.dependencies.State('playing', 'value'),
    ]
)

def update_chosen_round(slider,regularization, json_datasets, playing):
    datasets = json.loads(json_datasets)
    if regularization == 'least_error':
        all_possible = pd.read_json(datasets['all_possible_le'], orient='split')
        reg_coeffs = pd.read_json(datasets['reg_coeffs'], orient='split')
        reg_inter = pd.read_json(datasets['reg_inter'], orient='split')
        coeffs, intercept = reg_coeffs, reg_inter
    else:
        all_possible = pd.read_json(datasets['all_possible_raw'], orient='split')
        raw_coeffs = pd.read_json(datasets['raw_coeffs'], orient='split')
        raw_inter = pd.read_json(datasets['raw_inter'], orient='split')
        coeffs, intercept = raw_coeffs, raw_inter

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
    s = best_game.iloc[0]
    barriers = s.index.values[(s == 1)]
    final_barriers = [barrier for barrier in barriers if barrier not in name_list]
    neg_barr_names = [html.Li(x) for x in final_barriers]
    col_list = coeffs['variable'].tolist()
    unused_barr = [html.Li(x) for x in col_list if (x not in final_barriers) and (x not in name_list)]
    num_rounds = "**Chosen Game - Predicted Rounds: **" + str(round(best_game['prediction'].tolist()[0], 1))
    loading = ""
    return num_rounds, neg_barr_names, unused_barr, loading



if __name__ == '__main__':
    app.run_server(debug=True,port=1234)