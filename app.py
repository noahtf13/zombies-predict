import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import GridSearchCV
import math
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import itertools

app = dash.Dash(__name__,meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ])
server = app.server

def rerun_model(least_error = True):
    googleSheetId = '19Cr_YXoGf-mvrEHtPUxdxIOIezs4ozwDMgH0OijhBsI'
    worksheetName = 'Sheet1'
    URL = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
        googleSheetId,
        worksheetName
    )
    df = pd.read_csv(URL).dropna()
    df = df[df['beat_game_attempt'] == 0]
    train = df.drop(['beat_game_attempt', 'win_game', 'rounds_completed', 'date'], axis=1)

    label = np.log(df['rounds_completed'])

    full_score, full_estimator = gscv(train, label, df, least_error)

    best_model = full_estimator
    coeffs = (
        pd.concat(
            [
                pd.DataFrame(train.columns),
                pd.DataFrame(np.transpose(best_model.coef_))
            ], axis=1
        )
    )

    best_model = best_model.fit(train, label)
    coeffs.columns = ['variable', 'rounds_added']
    coeffs['rounds_added'] = math.e ** coeffs['rounds_added']
    intercept = [math.e ** best_model.intercept_]
    inter_df = pd.DataFrame({'intercept': intercept})
    return coeffs, inter_df, best_model

def gscv(train, label, df, least_error=True):
    multiply = df['rounds_completed'].mean()/math.e
    if least_error == True:
        parameters = {'alpha':[0.1,1,multiply,10]}
    else:
        parameters = {'alpha': [0]}
    model = Ridge()
    gscv = GridSearchCV(model, parameters, scoring='r2',cv=int(round(1/math.log(len(train))*15,0)))
    gscv.fit(train,label)
    return gscv.best_score_, gscv.best_estimator_

coeffs, intercept, best_model = rerun_model()

app.layout = html.Div([
    dcc.Dropdown(
        id='regularization-drop',
        options=[
            {'label': 'Least Error', 'value': 'least_error'},
            {'label': 'Raw', 'value': 'raw'}
        ],
        value='least_error'
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
        value=['will','stefan','noah']
    ),
    dcc.Slider(
        id='slider',
        min=1,
        max=50,
        step=1,
        value=10,
        marks=dict([(i,str(i)) for i in range(0,50,5)])
    ),
    dcc.Markdown(id='hard-rounds'),
    html.Ul(id='var_list')
])

@app.callback(
    [dash.dependencies.Output('coeff-graph', 'figure'),
    dash.dependencies.Output('hard-rounds', 'children'),
     dash.dependencies.Output('var_list', 'children')],
    [dash.dependencies.Input('regularization-drop','value'),
    dash.dependencies.Input('playing', 'value'),
     dash.dependencies.Input('slider','value')
     ]
)

def update_graph_scatter(regularization, playing, slider):
    if regularization == "least_error":
        coeffs, intercept, best_model = rerun_model(least_error=True)
    else:
        coeffs, intercept, best_model = rerun_model(least_error=False)
    fig = px.bar(coeffs, x='rounds_added', y='variable',
                 title=f"The intercept is: {round(intercept['intercept'][0], 1)}")
    fig.update_xaxes(title='Round Multiplier')
    fig.update_yaxes(title='Challenge/Variable')
    name_list = ['noah_playing','stefan_playing','will_playing']
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
    print(best_game)
    num_rounds = "**Chosen Game - Predicted Rounds: **" + str(round(best_game['prediction'].tolist()[0],1))

    # name_values = coeffs[coeffs['variable'].isin(name_cols)]['rounds_added'].tolist()
    # neg_barriers = coeffs[(~coeffs['variable'].isin(name_list))&(coeffs['rounds_added'] < 1)]['rounds_added'].tolist()
    # neg_barr_names = coeffs[coeffs['rounds_added'] < 1]['variable'].tolist()
    # neg_barr_names = [value for value in neg_barr_names if value not in name_list]
    # if name_values is None:
    #     name_values = []
    # final_list = name_values + neg_barriers
    # multipliers = []
    # for num in final_list:
    #     if len(coeffs[coeffs['rounds_added'] == num]) > 0:
    #         if 'num_perks_removed' in neg_barriers:
    #             multipliers.append(coeffs[coeffs['rounds_added'] == num]['rounds_added'].tolist()[0]**6)
    #         else:
    #             multipliers.append(coeffs[coeffs['rounds_added'] == num]['rounds_added'].tolist()[0])
    # num_rounds = "**Hardest Game - Predicted Rounds: **" + str(round(np.prod(multipliers) * (intercept['intercept'].tolist()[0]),1))
    return fig, num_rounds, neg_barr_names

if __name__ == '__main__':
    app.run_server(debug=True,port=1234)