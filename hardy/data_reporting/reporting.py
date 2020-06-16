# add funcitons to summarize and visualize the summary reports from the
# the hardy run
# import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import yaml
# def report_plot():
# def reports_summary:
# def performance_comparison():


def report_plot(report_path):
    '''The function that plots the parallel coordinates between report name,
    layers, optimizer, activation function, kernel size, pooling and accuracy.

    Parameters:
    -----------
    report_path: str
                 string representing the location of parent report directory

    Returns:
    --------
    plotly.graph
    '''

    categories = [f for f in os.listdir(report_path) if not f.startswith('.')]

    import_dict = {}

    for i in range(len(categories)):
        yaml_path = report_path+categories[i]+'/report/'
        yaml_file_name = os.listdir(yaml_path)
        with open(yaml_path+yaml_file_name[0], 'r') as file:
            import_dict[categories[i]] = yaml.load(file,
                                                   Loader=yaml.FullLoader)

    column_names = ['report_name', 'layers', 'kernel_size',
                    'activation_function', 'optimizer', 'pooling',
                    'test_accuracy']

    data_dict = {}
    index = 0

    for keys in import_dict.items():
        n = 0
        for keys_1, values_1 in keys[1].items():
            if 'activation_' in keys_1:
                n += 1
                a_function = values_1
            if 'kernel_size_' in keys_1:
                k_size = values_1
            if 'optimizer' in keys_1:
                optimize = values_1
            if 'pooling' in keys_1:
                pool = values_1
            if 'test_accuracy' in keys_1:
                accuracy = values_1
        data_dict[index] = [keys[0], n, k_size, a_function, optimize,
                            pool, accuracy]
        index += 1

    df_test = pd.DataFrame.from_dict(data_dict, orient='index',
                                     columns=column_names)

    fig = go.Figure(data=go.Parcats(
        line=dict(color=px.colors.qualitative.Pastel, colorscale='Electric'),
        dimensions=list([
                    dict(label='Report Name', values=df_test['report_name']),
                    dict(label='Num layers', values=df_test['layers'],
                         categoryorder='category descending'),
                    dict(label='Kernel_Size', values=df_test['kernel_size'],
                         categoryorder='category descending'),
                    dict(label='Activation',
                         values=df_test['activation_function']),
                    dict(label='Optimizer', values=df_test['optimizer']),
                    dict(label='Pooling', values=df_test['pooling'].values),
                    dict(label='Accuracy', values=df_test['test_accuracy'],
                         categoryorder='category descending'),
                ])))

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.show()
