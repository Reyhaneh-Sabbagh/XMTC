from dash import Input, Output, State, html
from numpy._core.strings import title
import dash
from app import app
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from plotly.subplots import make_subplots
import plotly.express as px
import base64
from PIL import Image
import io


num_test_data = {'Task1': 274, 'Task2':278, 'Task3':367}          #update these values for task2,3
MAX_LEN_DICT ={'Task1': 1436, 'Task2': 2842, 'Task3': 1378}
CLASSES = {'Task1': ['l_bottle', 'r_bottle', 'l_cup', 'r_cup', 'l_knife', 'r_knife', 'l_pen', 'r_pen'],
           'Task2': ['l_bar', 'r_bar', 'l_box', 'r_box', 'l_dice', 'r_dice', 'l_plank', 'r_plank'],
           'Task3': ['l_bar', 'r_bar', 'l_box', 'r_box', 'l_dice', 'r_dice', 'l_plank', 'r_plank']}
FEATURES = ['tiax', 'tiay', 'tiaz', 'tmax', 'tmay', 'tmaz', 'trax', 'tray', 'traz', 'tlax', 'tlay', 'tlaz']
# image = 'confusion_matrix_test_Task1_1000.png'


# # Convert an image to base64
# def image_to_base64(img_path):
#     with open(img_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

@app.callback(
    [Output('figure-scatterPlot-accuracy', 'figure'),
     Output('figure-scatterPlot-accuracy', 'style'),
     # Output('figure-scatterPlot-weights', 'figure'),
     # Output('figure-scatterPlot-weights', 'style'),
     Output('radioItems-timeSeries', 'options'),
     Output('dropDown-model', 'options'),
     Output('radioItems-timeSeries', 'value'),
     Output('dropDown-model', 'value')
     ],
    State('dropDown-task', 'value'),
    Input('button-apply', 'n_clicks'),
    prevent_initial_call=True,
)

def apply(Task, n_clicks):
    if n_clicks > 0:
        print(f'n_clicks:{n_clicks}')
        print(f'Task:{Task}')
        # if dropDownValue == 'Task1':
        PATH_DIR = f'data_and_preprocessing/{Task}/drcif'
        run_numbers_array = np.load(os.path.join(PATH_DIR, 'run_numbers_array.npy'))
        test_acc_array = np.load(os.path.join(PATH_DIR, 'test_acc_array.npy'))
        PATH_histogram = os.path.join(PATH_DIR, 'Histogram', 'sequence_lengths.npy')
        sequence_lengths = np.load(PATH_histogram)
        PATH_histogram_test = os.path.join(PATH_DIR, 'Histogram', 'sequence_lengths_test_data.npy')
        sequence_lengths_test = np.load(PATH_histogram_test)
        # print(run_numbers_array)
        # Calculate cumulative histogram counts
        bins = np.arange(0, max(sequence_lengths) + 10, 10)  # Define bins for histogram
        hist_counts, _ = np.histogram(sequence_lengths, bins=bins)
        cumulative_counts = np.cumsum(hist_counts)

        # Calculate cumulative histogram counts for test_data:
        bins_test = np.arange(0, max(sequence_lengths_test) + 10, 10)  # Define bins for histogram
        hist_counts_test, _ = np.histogram(sequence_lengths_test, bins=bins)
        cumulative_counts_test = np.cumsum(hist_counts_test)


        # === plot accuracy:
        # fig_scatterPlot_accuracy = go.Figure()
        # https://community.plotly.com/t/2-plots-same-x-axis-different-y-axis-merge-them-to-same-file/30967/4
        fig_scatterPlot_accuracy = make_subplots(specs=[[{"secondary_y": True}]])
        fig_scatterPlot_accuracy.add_trace(
            go.Scatter(x=run_numbers_array,
                       y=test_acc_array*100,
                       mode='lines+markers',
                       opacity=1,
                       name='accuracy',
                       hovertemplate=
                       # f'<img src="data:image/png;base64,{image_to_base64(image_paths)}" width="20" height="100">'
                       '<b>Window Size:</b> %{x}' +
                       '<br><b>Accuracy:</b> %%{y:.2f}'+
                       '<br><b>Cumulative Count:</b> %{customdata[0]}'+
                       '<br><b>Cumulative Count Test_Data:</b> %{customdata[1]}',
                       customdata=np.column_stack((cumulative_counts, cumulative_counts_test)),
                       ), secondary_y=False,
        )

        fig_scatterPlot_accuracy.add_trace(
            go.Histogram(           # Histogram for whole dataset
                x=sequence_lengths,
                xbins=dict(start=0, end=max(sequence_lengths), size=10),
                marker_color='gray',
                opacity=0.5,
                name='Histogram<br>(All data)',
                cumulative_enabled=False,
                hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>',
            ), secondary_y=True,
        )

        fig_scatterPlot_accuracy.add_trace(
            go.Histogram(       # Histogram for test dataset
                x=sequence_lengths_test,
                xbins=dict(start=0, end=max(sequence_lengths_test), size=10),
                marker_color='yellow',
                opacity=0.5,
                name='Histogram<br>(test data)',
                cumulative_enabled=False,
                hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>',
            ), secondary_y=True,
        )

        # Customize layout (e.g. crosshair lines)
        fig_scatterPlot_accuracy.update_layout(
            title='Test Accuracy over Time',
            showlegend=True,
            barmode='overlay',
            hovermode="closest",  # Show data closest to the mouse pointer
            xaxis=dict(
                title="Run (Number)",
                showspikes=True,  # Show spike line for x-axis
                spikemode="across",  # Extend spike across the plot
                spikesnap="cursor",  # Snap spike to cursor
                spikedash="dash",  # Solid line for spike
                spikethickness=0.5,  # Thickness of the spike line
                spikecolor="gray",  # Color of the spike line
            ),
            yaxis=dict(
                title="Accuracy (%)",
                showspikes=True,  # Show spike line for y-axis
                spikemode="across",
                spikesnap="cursor",
                spikedash="dash",
                spikethickness=0.5,
                spikecolor="gray",
                range=[0, 100],
                tickvals=np.linspace(0, 100, 11),  # Custom ticks
                showgrid=True,
                # gridcolor='gray'
            ),
            yaxis2=dict(
                title='Count',
                overlaying='y',
                side='right',
                range=[0, max(hist_counts)+5],
                tickvals=np.linspace(0, max(hist_counts)+5, 11),  # Custom ticks proportional to left y-axis
                showgrid=False,
            ),
        )
        # ==== plot weights of classifiers:
        weights_df = pd.read_csv(f'data_and_preprocessing/{Task}/hivecote/classifier_weights.csv')
        weights_df["Run"] = weights_df["Run"].astype(int)
        weights_df = weights_df.sort_values(by="Run").reset_index(drop=True)
        weights_df.reset_index(inplace=True)

        fig_scatterPlot_weights = go.Figure()

        for classifier in weights_df.columns[-4:]:  # Skip the 'Run' column
            print(classifier)
            fig_scatterPlot_weights.add_trace(go.Scatter(
                x=weights_df["Run"],
                y=weights_df[classifier],
                mode="markers+lines",
                name=classifier
            ))


        fig_scatterPlot_weights.update_layout(
            title="Classifier Weights Across Runs",
            xaxis=dict(
                title="Run (Number)",
                tickfont=dict(size=20),  # Set x-axis tick font size
            ),
            yaxis=dict(
                title="Weight",
                tickfont=dict(size=20),  # Set y-axis tick font size
            ),
            legend_title="Classifier",
            legend=dict(
                font=dict(size=20),  # Set legend font size
            ),
            title_font=dict(size=16),  # Set title font size
            font=dict(size=20),  # Set general font size
            # height=600,
            # width=900
        )

        fig_scatterPlot_weights.write_html('scatterPlot_weights.html')
        options_radioItems = [{"label": f"TS {i}", "value": f"{i}"} for i in range(num_test_data[Task])]
        options_dropDown = [{"label": f"Model [0-{min(i,MAX_LEN_DICT[Task])}]", "value": f"{min(i, MAX_LEN_DICT[Task])}"} for i in range(10, MAX_LEN_DICT[Task]+10, 10)]
        # return fig_scatterPlot_accuracy, {"display": "block"}, fig_scatterPlot_weights, {"display": "block"}, options
        # return fig_scatterPlot_accuracy, {"display": "block"},fig_scatterPlot_weights, {"display": "block"}, options_radioItems, options_dropDown, None, None
        return fig_scatterPlot_accuracy, {"display": "block"}, options_radioItems, options_dropDown, None, None

    # else:
    #     # return None, None, None, None, None, None
    #     return dash.no_update


@app.callback(
    [Output('figure-heatmap-probs', 'figure'),
     Output('figure-heatmap-probs', 'style'),],
    [Input('radioItems-timeSeries', 'value'),
     Input('figure-heatmap-probs', 'hoverData')],
    State('dropDown-task', 'value'),
    prevent_initial_call=True,
)
def load_probs(radioItemsValue, hoverData, Task):
    if radioItemsValue is not None:
        print(f'radioItemsValue:{radioItemsValue}')
        PATH_DIR = os.path.join('data_and_preprocessing', Task, 'drcif')
        probs_over_time =np.load(os.path.join(PATH_DIR,'probs', 'probs_over_time.npy'))
        y_test = np.load(os.path.join(PATH_DIR, 'probs', 'y_test_10.npy'))
        run_numbers_array = np.load(os.path.join(PATH_DIR, 'run_numbers_array.npy'))
        colorscale = [
            [0.0, "lightgray"],  # 0.0 -> 0.1
            [0.1, "lightgray"],
            [0.1, "lightyellow"],  # 0.1 -> 0.4
            [0.4, "lightyellow"],
            [0.4, "gold"],  # 0.4 -> 0.6
            [0.5, "gold"],
            [0.5, "tomato"],  # 0.6 -> 0.8       # orange
            [0.6, "tomato"],                     # orange
            [0.6, "red"],  # 0.8 -> 0.95
            [0.95, "red"],
            [0.95, "darkred"],  # 0.95 -> 1.0
            [1.0, "darkred"]
        ]
        row_labels = CLASSES[Task]
        reordered_indices = [0,2,4,6,1,3,5,7]
        # Reorder row_labels and probabilities accordingly
        reordered_row_labels = [CLASSES[Task][i] for i in reordered_indices]
        reordered_probs_over_time = probs_over_time[int(radioItemsValue)][reordered_indices]

        fig = make_subplots(
            rows=8, cols=1,  # 8 rows, each with its own heatmap
            shared_xaxes=True,  # Share x-axis across rows
            vertical_spacing=0.05  # Adjust spacing between rows
        )

        # Add heatmaps for each row
        for i, row in enumerate(reordered_probs_over_time):
            fig.add_trace(
                go.Heatmap(x=run_numbers_array,
                    z=[row],  # Each row is a single heatmap
                    colorscale=colorscale,
                    zmin=0,  # Set min range for colorscale
                    zmax=1,  # Set max range for colorscale
                    hovertemplate='<b>Window Size:</b> [0-%{x}]<br><b>Probabilty:</b> %{z}<extra></extra>',
                    colorbar=dict(
                        tickvals=[0.05, 0.25, 0.45, 0.55, 0.8, 0.975],  # Center of ranges
                        ticktext=["0-0.1", "0.1-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.95", "0.95-1"]
                    ),
                ),
                row=i + 1,
                col=1
            )

            # Add annotation for y-axis label (to the left of the subplot)
            fig.add_annotation(
                text=reordered_row_labels[i],  # Use the row label for annotation text
                x=-1,  # Position it to the left of the subplot (negative value moves left)
                y=0,  # Center vertically in the subplot
                showarrow=False,  # No arrow
                font=dict(size=12),  # Optional: Adjust font size
                align="right",  # Align text to the right
                row=i + 1,
                col=1
            )

            # Add vertical highlight box if hoverData is available
        if hoverData:
            hover_x = hoverData['points'][0]['x']  # Get the hovered x value
            fig.add_shape(
                type="line",
                x0=hover_x,  # Adjust width for visibility
                x1=hover_x,
                y0=0,  # Start at the bottom of the plot
                y1=1,  # Extend across all 8 rows
                xref="x",  # Reference x-axis
                yref="paper",  # Reference paper coordinates for vertical span
                line=dict(color="gray", width=0.5, dash="dash"),  # Box line style
                fillcolor="rgba(0, 0, 0, 0.1)"  # Optional: Semi-transparent fill
            )

        fig.update_layout(
            title=f'Probabilities over time, actual label:{row_labels[y_test[int(radioItemsValue)]]}',
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
        )
        # fig.update_layout(
        #     {
        #         "paper_bgcolor": "rgba(0, 0, 0, 0)",
        #         "plot_bgcolor": "rgba(0, 0, 0, 0)",
        #     }
        # )
        fig.update_yaxes(showticklabels=False)

        return fig, {"display": "block"}
    else:
        return {}, {'display': 'none'}

#
@app.callback(
    [Output('figure-PDP', 'figure'),
     Output('figure-PDP', 'style'),],
    [Input('dropDown-model', 'value')],
    State('dropDown-task', 'value'),
    prevent_initial_call=True,
)
def partialDpendencePlot(model_window_size, Task):
    if model_window_size is not None:
        if Task=='Task1' or Task=='Task3':
            print(f'model with window size:{model_window_size}')
            PATH_DIR = f'data_and_preprocessing/{Task}/drcif/runs_PDP_{Task}_drcif/PDP_currentrange_{model_window_size}/PDP_drcif'
            colors = px.colors.qualitative.Set1  # Or any other color palette
            reordered_indices = [0, 2, 4, 6, 1, 3, 5, 7]
            # Reorder row_labels and probabilities accordingly
            reordered_row_labels = [CLASSES[Task][i] for i in reordered_indices]

            fig_scatterPlot_pdp = make_subplots(rows=4, cols=3,
                                                subplot_titles=[f'{FEATURES[i]}' for i in range(12)],
                                                shared_xaxes=True,  # Share x-axis across rows
                                                vertical_spacing=0.04  # Adjust spacing between rows
                                                )
            for f in range(12):             # 12 features: from 0 to 11
                row = int(f / 3)+1
                col = (f % 3)+1
                # print(f'row={row}')
                # print(f'col={col}')
                PDP_feature_values = np.load(os.path.join(PATH_DIR, f'PDP_feature_values_{Task}_{model_window_size}_feature{f}.npy'))
                PDP_values = np.load(os.path.join(PATH_DIR, f'PDP_values_{Task}_{model_window_size}_feature{f}.npy'))
                for c in range(8):      # 8 classes: from 0 to 7
                    class_idx = reordered_indices[c]
                    fig_scatterPlot_pdp.add_trace(
                        go.Scatter(x=PDP_feature_values,
                                   y=PDP_values[:, class_idx],
                                   mode='lines',
                                   opacity=1,
                                   name=f'{CLASSES[Task][class_idx]}' if f == 0 else None,
                                   legendgroup=f'class{class_idx}',  # Group traces by class
                                   line=dict(color=colors[class_idx]),  # Assign consistent colors
                                   showlegend=True if f==0 else False,
                                   ), row=row, col=col
                    )
            fig_scatterPlot_pdp.update_xaxes(range=[0, 1])

            fig_scatterPlot_pdp.update_layout(
                title=f'Partial Dependence Plots for {Task}, selected Model [0-{model_window_size}]',
            )

            for anno in fig_scatterPlot_pdp['layout']['annotations']:
                anno.font = dict(
                    size=14,  # Font size
                    color='blue',  # Font color
                    family='Arial'  # Font family
                )
            # print('fig_scatterPlot_pdp is calculated')
            # fig_scatterPlot_pdp.show()
            return fig_scatterPlot_pdp, {"display": "block", "height":"800px"}

        else:
            return {}, {'display': 'none'}
    elif model_window_size is None:
        return {}, {'display': 'none'}

@app.callback(
    [Output("tooltip-scatterPlot-accuracy", "show"),
    Output("tooltip-scatterPlot-accuracy", "bbox"),
    Output("tooltip-scatterPlot-accuracy", "children"),],
    Input('figure-scatterPlot-accuracy', 'hoverData'),
    State('dropDown-task', 'value'),

)
def display_hover(hoverData, Task):
    if hoverData is not None:
        # print(f'hoverData:{hoverData}')
        hover_data = hoverData['points'][0]
        # print(f'hover_data:{hover_data}')
        bbox = hover_data["bbox"]
        # print(f'bbox:{bbox}')
        # num = hover_data["pointNumber"]
        # print(f'num:{num}')
        x = hover_data['x']
        if x in range(0, MAX_LEN_DICT[Task], 10) or x==MAX_LEN_DICT[Task]:
            print(f'x is in xticks. x:{x}')
            PATH_DIR = f'data_and_preprocessing/{Task}/drcif/confusion_matrix'
            image_path = os.path.join(PATH_DIR, f'confusion_matrix_test_{Task}_{x}.png')
            im_url = image_to_base64(image_path)
            children = [
                html.Div([
                    html.Img(
                        src=im_url,
                        style={"width": "300px", 'display': 'block', 'margin': '0 auto'},
                    ),
                ])
            ]
            return True, bbox, children
        else:
            print(f'x is notttt in xticks. x:{x}')
            return False, dash.no_update, dash.no_update
    else:
        return False, dash.no_update, dash.no_update