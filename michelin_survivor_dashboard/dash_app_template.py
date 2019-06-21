# standard library
import os

# dash libs
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.figure_factory as ff
import plotly.graph_objs as go

# pydata stack
import pandas as pd
from sqlalchemy import create_engine

# set params
conn = create_engine(os.environ['DB_URI'])


#######################
# Data Analysis / Model
#######################

def fetch_data(q):
    df = pd.read_csv(''
        sql=q,
        con=conn
    )
    return df


#########################
# Dashboard Layout / View
#########################

# Set up Dashboard and create layout
app = dash.Dash()
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

app.layout = html.Div([

    # Page Header
    html.Div([
        html.H1('Project Header')
    ]),

])


#############################################
# Interaction Between Components / Controller
#############################################

# Temlate
@app.callback(
    Output(component_id='selector-id', component_property='figure'),
    [
        Input(component_id='input-selector-id', component_property='value')
    ]
)
def ctrl_func(input_selection):
    return None


# start Flask server
if __name__ == '__main__':
    app.run_server(
        debug=True,
        host='0.0.0.0',
        port=8050
    )
