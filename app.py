import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import GridSearchCV
import math
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

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
    df['num_perks_removed'] = df.iloc[:, 4:10].sum(axis=1)
    train = df.drop(['beat_game_attempt', 'win_game', 'rounds_completed', 'date'], axis=1)
    train_small = train.drop(df.iloc[:, 4:10].columns, axis=1)
    train_full = train.drop(['num_perks_removed'], axis=1)

    label = np.log(df['rounds_completed'])

    full_score, full_estimator = gscv(train_full, label, df, least_error)
    small_score, small_estimator = gscv(train_small, label, df, least_error)

    if full_score > small_score:
        print(full_score)
        best_model = full_estimator
        coeffs = (
            pd.concat(
                [
                    pd.DataFrame(train_full.columns),
                    pd.DataFrame(np.transpose(best_model.coef_))
                ], axis=1
            )
        )
    else:
        print(small_score)
        best_model = small_estimator
        coeffs = (
            pd.concat(
                [
                    pd.DataFrame(train_small.columns),
                    pd.DataFrame(np.transpose(best_model.coef_))
                ], axis=1
            )
        )

    best_model = best_model.fit(train, label)
    coeffs.columns = ['variable', 'rounds_added']
    coeffs['rounds_added'] = math.e ** coeffs['rounds_added']
    intercept = [math.e ** best_model.intercept_]
    inter_df = pd.DataFrame({'intercept': intercept})
    return coeffs, inter_df

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

coeffs, intercept = rerun_model()

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
        ]
    ),
    dcc.Markdown(id='hard-rounds'),
    html.Ul(id='var_list')
])

@app.callback(
    [dash.dependencies.Output('coeff-graph', 'figure'),
    dash.dependencies.Output('hard-rounds', 'children'),
     dash.dependencies.Output('var_list', 'children')],
    [dash.dependencies.Input('regularization-drop','value'),
    dash.dependencies.Input('playing', 'value')]
)

def update_graph_scatter(regularization, playing):
    if regularization == "least_error":
        coeffs, intercept = rerun_model(least_error=True)
    else:
        coeffs, intercept = rerun_model(least_error=False)
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
    name_values = coeffs[coeffs['variable'].isin(name_cols)]['rounds_added'].tolist()
    neg_barriers = coeffs[(~coeffs['variable'].isin(name_list))&(coeffs['rounds_added'] < 1)]['rounds_added'].tolist()
    neg_barr_names = coeffs[coeffs['rounds_added'] < 1]['variable'].tolist()
    neg_barr_names = [value for value in neg_barr_names if value not in name_list]
    if name_values is None:
        name_values = []
    final_list = name_values + neg_barriers
    multipliers = []
    for num in final_list:
        if len(coeffs[coeffs['rounds_added'] == num]) > 0:
            if 'num_perks_removed' in neg_barriers:
                multipliers.append(coeffs[coeffs['rounds_added'] == num]['rounds_added'].tolist()[0]**6)
            else:
                multipliers.append(coeffs[coeffs['rounds_added'] == num]['rounds_added'].tolist()[0])
    num_rounds = "**Hardest Game - Predicted Rounds: **" + str(round(np.prod(multipliers) * (intercept['intercept'].tolist()[0]),1))
    print(name_values)
    print(multipliers)
    print(final_list)
    neg_barr_names = [html.Li(x) for x in neg_barr_names]
    return fig, num_rounds, neg_barr_names

if __name__ == '__main__':
    app.run_server(debug=True,port=1234)