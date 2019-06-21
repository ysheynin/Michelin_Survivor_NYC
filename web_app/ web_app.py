import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

app = dash.Dash()
df = pd.read_csv('probabilities.csv')


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Michelin Survivor 2020',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children="Probability of losing a star in next year", style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Graph(
        id='Graph1',
        figure={
            'data': [
                {'x': df['restaurant'], 'y': df['probability'], 'type': 'bar', 'name': 'SF'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    )
])


if __name__ == '__main__':
    app.run_server(debug=True)
