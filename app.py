import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import GridSearchCV
import math
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

app = dash.Dash(__name__)
server = app.server

def rerun_model():
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

    full_score, full_estimator = gscv(train_full, label, df)
    small_score, small_estimator = gscv(train_small, label, df)

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

def gscv(train, label, df):
    multiply = df['rounds_completed'].mean()/math.e
    parameters = {'alpha':[0.00000001,.01,.1,1,multiply,10,100,1000,1*10**10]}
    model = Ridge()
    gscv = GridSearchCV(model, parameters, scoring='r2',cv=int(round(1/math.log(len(train))*15,0)))
    gscv.fit(train,label)
    return gscv.best_score_, gscv.best_estimator_

coeffs, intercept = rerun_model()

fig = px.bar(coeffs, x='rounds_added', y='variable',title = f"The intercept is: {round(intercept['intercept'][0],1)}")
fig.update_xaxes(title='Round Multiplier')
fig.update_yaxes(title='Challenge/Variable')

app.layout = html.Div([
    dcc.Graph(id="bar-chart", figure=fig),
])

if __name__ == '__main__':
    app.run_server(debug=True)