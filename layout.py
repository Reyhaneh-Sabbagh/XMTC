from dash import *

def create_layout():
    return html.Div([
        html.Div(
            id="div-left",
            children=[
                # html.H2("Settings Panel", style={"textAlign": "center"}),
                dcc.Dropdown(
                    id="dropDown-task",
                    options=['Task1', 'Task2', 'Task3'],  # Options set dynamically by callbacks if needed
                    placeholder="Select Task...",
                    clearable=False,
                ),
                html.Button('Apply', id='button-apply', n_clicks=0),
                html.P('Local Exploration:'),
                dcc.RadioItems(id="radioItems-timeSeries"),
                html.P('PDP Exploration:'),
                dcc.Dropdown(               # Select Window size(Model)
                    id="dropDown-model",
                )
            ],
        ),
        html.Div(
            id="div-right",
            children=[
                dcc.Graph(
                    id="figure-scatterPlot-accuracy",
                    config={"displayModeBar": True},
                    clear_on_unhover=True,
                ),
                dcc.Tooltip(id="tooltip-scatterPlot-accuracy", direction='left'),

                # dcc.Graph(
                #     id="figure-scatterPlot-weights",
                #     config={"displayModeBar": True},
                # ),
                dcc.Graph(
                    id='figure-heatmap-probs',
                    config={'displayModeBar': True},
                    ),
                dcc.Graph(
                    id='figure-PDP',
                )
            ],
        )
    ])
