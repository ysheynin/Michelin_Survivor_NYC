# standard library
import os

# dash libs
import dash
import dash_table
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.graph_objs as go
from IPython.display import IFrame
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *



# pydata stack
import pandas as pd


#######################
# Data Analysis / Model
#######################
global df
df = pd.read_csv('michelin_2019_model_data.csv')
df = df.round(decimals = 2)

###########################
# Data Manipulation / Model
###########################


def get_restos():

    restos = df.name.values

    return restos


def get_resto_data(resto):
    # '''Returns the seasons of the datbase store'''

    main_points = ['name', 'div2_food_quality', 'div2_wait time', 'div4_slope', 'div1_value', 'div3_menu', 'total_service', 'total_mean', 'security']

    # main_points = ['name', 'total_mean', 'total_slope','div3_food_quality', 'div3_menu',  'div3_service', 'div3_value', 'probs']
    resto_series = df.loc[df.name == resto][main_points]

    col_map = {'name': 'restaurant', 'total_mean' : 'average rating', 'div4_slope': 'average trend in rating','div2_food_quality': 'food quality', 'div3_menu': 'offerings', 'total_service' : 'service', 'div1_value': 'value', 'security' : 'Michelin 2020 Security', 'div2_wait time' : 'wait time' }

    resto_series = resto_series.rename(columns = col_map)


    return resto_series

def get_feature_data(resto_series):
        # '''Returns the seasons of the datbase store'''
    return resto_series.columns.values



# def get_match_results(division, season, team):
#     '''Returns match results for the selected prompts'''
#
#     results_query = (
#         f'''
#         SELECT date, team, opponent, goals, goals_opp, result, points
#         FROM results
#         WHERE division='{division}'
#         AND season='{season}'
#         AND team='{team}'
#         ORDER BY date ASC
#         '''
#     )
#     match_results = fetch_data(results_query)
#     return match_results


# def calculate_season_summary(results):
#     record = results.groupby(by=['result'])['team'].count()
#     summary = pd.DataFrame(
#         data={
#             'W': record['W'],
#             'L': record['L'],
#             'D': record['D'],
#             'Points': results['points'].sum()
#         },
#         columns=['W', 'D', 'L', 'Points'],
#         index=results['team'].unique(),
#     )
#     return summary


def plot_resto_scatter_by_feat(feat1, feat2, resto):

    my_map = {'name': 'restaurant', 'total_mean' : 'average rating', 'div4_slope': 'average trend in rating','div2_food_quality': 'food quality', 'div3_menu': 'offerings', 'total_service' : 'service', 'div1_value': 'value', 'security' : 'Michelin 2020 Security', 'div2_wait time' : 'wait time' }

    df_new = df.copy()
    df_new = df_new.rename(columns = my_map)


    x1 = list(df_new[feat1].loc[df_new['at_risk'] == 1])
    x0 = list(df_new[feat1].loc[df_new['at_risk'] == 0])
    y0 = list(df_new[feat2].loc[df_new['at_risk'] == 0])
    y1 = list(df_new[feat2].loc[df_new['at_risk'] == 1])

    color = dict()
    color['ar']  = '#fd80b3'
    color['nar'] = '#CEF982'

    trace0 = go.Scatter(
            x= x0,
            y= y0,
            mode = 'markers',
            hovertext = df_new.loc[df_new['at_risk'] == 0].restaurant.values,
            marker=dict(color=color['nar'] ,size =15, opacity = .7,  line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2))
        )

    trace1 = go.Scatter(
            x= x1,
            y= y1,
            mode='markers',
            hovertext = (df_new.loc[df_new['at_risk'] == 1]).restaurant.values,
            marker=dict(color= color['ar'] , size = 15, opacity = .7,  line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2))
        )

    if resto is not None:

        x2 = df_new[df_new['restaurant'] == resto][feat1]
        y2 = df_new[df_new['restaurant'] == resto][feat2]

        at_risk = int(df_new[df_new['restaurant'] == resto].at_risk)

        if at_risk == 1:
            indiv_color=color['ar']
        else:
            indiv_color=color['nar']

        trace2 = go.Scatter(
                x= x2,
                y= y2,
                mode='markers',
                hovertext = resto,
                marker= dict(color=indiv_color, size = 30, opacity = .7,
                           line = dict(
                color = 'rgb(0, 0, 255)',
                width = 2) )


            )


        data = [trace0, trace1, trace2]


    else:
         data = [trace0, trace1]

    cluster0 = [dict(type='circle',
                         xref='x', yref='y',
                         x0=min(x0), y0=min(y0),
                         x1=max(x0), y1=max(y0),
                         opacity=.25,
                         line=dict(color='#CEF982'),
                         fillcolor='#CEF982')]
    cluster1 = [dict(type='circle',
                         xref='x', yref='y',
                         x0=min(x1), y0=min(y1),
                         x1=max(x1), y1=max(y1),
                         opacity=.25,
                         line=dict(color='#fd80b3'),
                         fillcolor='#fd80b3')]

    updatemenus = list([
            dict(buttons=list([
                    dict(label = 'Not At Risk',
                         method = 'relayout',
                         args = ['shapes', cluster0]),
                    dict(label = 'At Risk',
                         method = 'relayout',
                         args = ['shapes', cluster1]),
                    dict(label = 'All',
                         method = 'relayout',
                         args = ['shapes', cluster0+cluster1])
                ]),
            )
        ])



    layout = dict(title='Why am I voted off Michelin island?', showlegend=False,
                      updatemenus=updatemenus,
                 xaxis=dict(
                            title=feat1,
                            titlefont=dict(
                            family='Avenir',
                            size=18,
                            color='lightgrey'
                            )),
                  yaxis=dict(
                            title=feat2,
                            titlefont=dict(
                            family='Avenir',
                            size=18,
                            color='lightgrey'
                            )))


    figure = go.Figure(data, layout)

    return figure

#########################
# Dashboard Layout / View
#########################


def generate_table(resto_df, max_rows= 1):
    # '''Given dataframe, return template generated using Dash components
    # '''
    return dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in resto_df.columns],
        data=resto_df.to_dict('records'),
        style_table={'overflowX': 'scroll',  'text-align' : 'center'},
    style_cell={
        # all three widths are needed
        'minWidth': '100px', 'width': '110px', 'maxWidth': '180px',
        'whiteSpace': 'normal'
    },
    css=[{
        'selector': '.dash-cell div.dash-cell-value',
        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
    }],
    style_data_conditional=[
        {
            'if': {
                'column_id': 'Michelin 2020 Security',
                'filter_query': '{Michelin 2020 Security} < .5'
            },
            'backgroundColor': 'rgb(255, 0, 0)',
            'color': 'white',
        },

        {
            'if': {
                'column_id': 'Michelin 2020 Security',
                'filter_query': '{Michelin 2020 Security} > .5'
            },
            'backgroundColor': 'rgb(0, 255, 0)',
            'color': 'white',
        }

        ],

    )#html.Table(
    #     # Header
    #     [html.Tr([html.Th(col) for col in resto_df.columns])] +
    #
    #     # Body
    #     [html.Tr([
    #         html.Td(resto_df[col]) for col in resto_df.columns
    #     ]) for i in range(0, max_rows)]
    # )


def onLoad_restaurant_options():
    #'''Actions to perform upon initial page load'''

    resto_options = (
        [{'label': resto, 'value': resto}
         for resto in get_restos()]
    )
    return resto_options

def make_suggestion(resto):

    my_map = {'name': 'restaurant', 'total_mean' : 'average rating', 'div4_slope': 'average trend in rating','div4_food_quality': 'food quality', 'div3_menu': 'offerings', 'total_service' : 'service', 'div1_value': 'value', 'security' : 'Michelin 2019 Security', 'div2_wait time' : 'wait time' }


    df_new     = df.copy()
    df_new     = df_new.rename(columns = my_map)
    resto_df   = df_new[df_new['restaurant'] == resto]
    first_rev      = int(resto_df.first_review.values[0] * 100)
    name           = resto_df.restaurant.str.replace('-', ' ').values[0]
    name           = name[0].upper() + name[1:]
    rating         = int(resto_df['average rating'].values[0] * 100)
    michelin_risk  = int(resto_df[ 'Michelin 2019 Security'].values[0] * 100)
    first_rev      = int(resto_df.first_review.values[0] * 100)
    food_quality   = int(resto_df['food quality'].values[0] * 100)
    value          = int(resto_df['value'].values[0] * 100)
    service        = int(resto_df['service'].values[0] * 100)
    offerings      = int(resto_df['offerings'].values[0] * 100)

    age_str        = f"* **{name}** has been around longer than **{first_rev}%** of NY's Michelin-ranked restaurants in 2018.\n\n"
    rating_str     = f'* Its rating is in the **{rating}th percentile** of ranked restaurants.\n\n'

    if michelin_risk < 50:
        michelin_str = f"### According to our best estimates, there is a **{100-abs(michelin_risk)}%** chance {name} will lose a Michelin star in 2019.\n\n"
    else:
        michelin_str = f"### According to our best estimates, there is a **{abs(michelin_risk)}%** likelihood **{name}** will keep (or gain) a Michelin star in 2019.\n\n"
    if food_quality < 50:
        food_str = (f'* Compared to other Michelin-rated restaurants, **{name}** scored a **{food_quality}%** on food quality according to yelp reviews. Improving the sourcing of ingredients and establishing higher standards with kitchen staff should improve this.\n\n')
    else:
        food_str = ''
    if value < 50:
        value_str = (f'* Compared to other Michelin-rated restaurants, **{name}** scored a **{value}%** on perceived value according to yelp reviews. Reducing your prices will improve your chances of staying listed \n\n')
    else:
        value_str =''
    if service < 50:
        service_str = (f'* Compared to other Michelin-rated restaurants, **{name}** scored a **{service}%** on the quality of service according to yelp reviews. Developing ways to streamline service and hiring top-quality servers will improve your performance \n\n')
    else:
        service_str = ''

    if offerings < 50:
        offerings_str = (f'* Compared to other Michelin-rated restaurants, **{name}** scored a **{offerings}%** on the quality and diversity of its offerings, according to yelp reviews. Focusing on diversifying and updating your menu will improve your current situation\n\n')
    else:
        offerings_str = ''

    total_str =  michelin_str + age_str + rating_str + food_str + value_str + service_str + offerings_str
    return total_str


# Set up Dashboard and create layout
external_stylesheets = ['https://github.com/plotly/dash-app-stylesheets/blob/master/dash-docs-custom.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app.scripts.config.serve_locally = False


d = 'appple'
my_str = ''' f'{d}s are yummy '''

app.layout = html.Div([


# html.H1("Your title")
#                      ],
#                  style = {'padding' : '50px' ,
#                           'backgroundColor' : '#',
#                           'font-family' : 'HelveticaNeue',
#                           'color': 'white',
#                           'text-align' : 'center'})
    # Page Header
    html.Div([
        html.H1('Michelin Survivor:  will stay on the Michelin guide in 2019?')
    ],      style = {'padding' : '50px' ,
                               'backgroundColor' : '#800020',
                               'font-family' : 'Avenir',
                               'color': 'white',
                               'text-align' : 'center'}),

    # Dropdown Grid
    html.Div([
        html.Div([
            # Select Division Dropdown
            html.Div([
                html.Div('Select Michelin Restaurant', className='three columns'),
                html.Div(dcc.Dropdown(id='resto-selector',
                                      options=onLoad_restaurant_options()),
                         className='three columns')
            ]),

            # Select Season Dropdown
            html.Div([
                html.Div('KPI 1', className='three columns'),
                html.Div(dcc.Dropdown(id='perf1-selector'),
                         className='three columns')
            ]),

            # Select Team Dropdown
            html.Div([
                html.Div('KPI 2', className='three columns'),
                html.Div(dcc.Dropdown(id='perf2-selector'),
                         className='three columns')
            ]),
        ], className='six columns'),

        # Empty
        html.Div(className='six columns'),
    ], className='twleve columns'),

    # Match Results Grid
    html.Div([

        # Match Results Table
        html.Div(
            html.Table(id='resto-results'),
            className='six columns', style={'marginBottom': '50px', 'marginTop': 25}
        ),
        html.Div(
            dcc.Markdown(id = 'suggestions'),
             className = 'six columns'),


        # Season Summary Table and Graph
        html.Div([

            # graph
            dcc.Graph(id='scatter-graph'
            )
            # style={},

        ], className='six columns')
    ]),
])


#############################################
# Interaction Between Components / Controller
#############################################

# Load Seasons in Dropdown
@app.callback(
    Output(component_id='perf1-selector', component_property='options'),
    [
        Input(component_id='resto-selector', component_property='value')
    ]
)

def populate_perf1_selector(resto):
    data     =  get_resto_data(resto)
    features =  data.columns.values
    return [
        {'label': feature, 'value': feature}
        for feature in features
    ]





@app.callback(
    Output(component_id='suggestions', component_property='children'),
    [
        Input(component_id='resto-selector', component_property='value')
    ]
)

def populate_suggestions(resto):
    if resto is not None:
        return  [make_suggestion(resto)]
    else:
        return ['']


# Load Teams into dropdown
@app.callback(
    Output(component_id='perf2-selector', component_property='options'),
    [
        Input(component_id='perf1-selector', component_property='value'),
        Input(component_id='resto-selector', component_property='value')
    ]
)


def populate_perf2_selector(resto, feat1):
    data     =  get_resto_data(resto)
    features =  data.columns.values
    return [
        {'label': feature, 'value': feature}
        for feature in features
    ]

# def populate_team_selector(division, season):
#     teams = get_teams(division, season)
#     return [
#         {'label': team, 'value': team}
#         for team in teams
#     ]


# Load Match results
@app.callback(
    Output(component_id='resto-results', component_property='children'),
    [
        Input(component_id='resto-selector', component_property='value'),
        Input(component_id='perf1-selector', component_property='value'),
        Input(component_id='perf2-selector', component_property='value')
    ]
)

def load_resto_results(resto, feat1, feat2):
    results     =  get_resto_data(resto)
    return generate_table(results)


# Update Season Summary Table
# @app.callback(
#     Output(component_id='resto-summary', component_property='figure'),
#     [
#         Input(component_id='division-selector', component_property='value'),
#         Input(component_id='season-selector', component_property='value'),
#         Input(component_id='team-selector', component_property='value')
#     ]
# )
# def load_season_summary(division, season, team):
#     results = get_match_results(division, season, team)
#
#     table = []
#     if len(results) > 0:
#         summary = calculate_season_summary(results)
#         table = ff.create_table(summary)
#
#     return table
#

# Update Season Point Graph
@app.callback(
    Output(component_id='scatter-graph', component_property='figure'),
    [
        Input(component_id='resto-selector', component_property='value'),
        Input(component_id='perf1-selector', component_property='value'),
        Input(component_id='perf2-selector', component_property='value')
    ]
)
def load_season_points_graph(resto, perf1, perf2):

    figure = go.Figure()
    if (perf2 is not None) &  (perf1 is not None) & (resto is not None):

        figure = plot_resto_scatter_by_feat(perf1, perf2, resto)

    return figure


# start Flask server
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug = True, port = 8050)
