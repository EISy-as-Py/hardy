# add funcitons to summarize and visualize the summary reports from the
# the hardy run
import os
import yaml

import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def report_dataframes(report_path):

    categories = [f for f in os.listdir(report_path) if not f.startswith('.')]

    import_dict = {}

    for i in range(len(categories)):
        yaml_path = report_path+categories[i]+'/report/'
        yaml_file_name = [file for file in os.listdir(yaml_path)
                          if file != 'run_tform_config.yaml']
        with open(yaml_path+yaml_file_name[0], 'r') as file:
            import_dict[categories[i]] = yaml.load(
                file, Loader=yaml.FullLoader)

    column_names = ['report_name', 'layers', 'kernel_size',
                    'activation_function', 'optimizer', 'pooling',
                    'test_accuracy']
    history_names = ['report_name', 'epochs', 'train_loss', 'val_loss',
                     'test_loss', 'train_accuracy', 'val_accuracy',
                     'test_accuracy']

    data_dict = {}
    index = 0
    history_dict = {}

    for keys in import_dict.items():
        n = 0
        for keys_1, values_1 in keys[1].items():
            if 'activation' in keys_1:
                n += 1
                a_function = values_1
            if 'kernel_size' in keys_1:
                k_size = values_1
            if 'optimizer' in keys_1:
                optimize = values_1
            if 'pooling' in keys_1:
                pool = values_1
            if 'test_accuracy' in keys_1:
                accuracy = values_1
        data_dict[index] = [keys[0], n, k_size, a_function,
                            optimize, pool, accuracy]
        history_dict[index] = [keys[0], list(range(1, len(keys[1]['loss'])+1)),
                               keys[1]['loss'], keys[1]['val_loss'],
                               keys[1]['test_loss'],
                               keys[1]['accuracy'], keys[1]['val_accuracy'],
                               keys[1]['test_accuracy']]
        index += 1

    hyperparam_df = pd.DataFrame.from_dict(data_dict, orient='index',
                                           columns=column_names)
    history_df = pd.DataFrame.from_dict(history_dict, orient='index',
                                        columns=history_names)
    return hyperparam_df, history_df


def report_plots(hyperparam_df, history_df):

    # Generate plot for comparing the CNN histories
    fig1 = make_subplots(subplot_titles=('CNN Loss', 'CNN Accuracy'),
                         horizontal_spacing=0.15, rows=1, cols=2)

    # Assigned title to the overall figure, the subplots and their axes
    fig1.update_layout(dict(font=dict(size=12)))
    fig1.update_layout(title_text="CNN Performance History Comparison",
                       height=600)
    fig1.update_xaxes(title_text='Epochs', row=1, col=1)
    fig1.update_xaxes(title_text='Epochs', row=1, col=2)
    fig1.update_yaxes(title_text='Loss', row=1, col=1)
    fig1.update_yaxes(title_text='Accuracy', row=1, col=2)

    # Define the color palette to use
    color = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf',
             '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']

    # Let's plot the trainig and validation loss and accuracy

    for i in range(len(history_df)):

        fig1.add_scatter(
            x=history_df['epochs'][i], y=history_df['train_loss'][i],
            mode='lines', legendgroup=history_df['report_name'][i],
            name=history_df['report_name'][i],
            marker=dict(size=8, color=color[i], colorscale='Electric'),
            row=1, col=1)
        fig1.add_scatter(
            x=history_df['epochs'][i], y=history_df['val_loss'][i],
            mode='markers', legendgroup=history_df['report_name'][i],
            name=history_df['report_name'][i],
            marker=dict(size=8, color=color[i], colorscale='Electric'),
            row=1, col=1, showlegend=False)
        fig1.add_scatter(
            x=history_df['epochs'][i], y=history_df['train_accuracy'][i],
            mode='lines', legendgroup=history_df['report_name'][i],
            name=history_df['report_name'][i],
            marker=dict(size=8, color=color[i], colorscale='Electric'),
            row=1, col=2, showlegend=False)
        fig1.add_scatter(
            x=history_df['epochs'][i], y=history_df['val_accuracy'][i],
            mode='markers', legendgroup=history_df['report_name'][i],
            name=history_df['report_name'][i],
            marker=dict(size=8, color=color[i], colorscale='Electric'),
            row=1, col=2, showlegend=False)

    fig1.update_layout(plot_bgcolor='lightslategray')
    # fig1.show()

    # Generate Parallel Coordinates plot
    fig2 = go.Figure(data=go.Parcats(
        line=dict(color=['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090',
                         '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
                         '#313695'], colorscale='Electric'),
        dimensions=list(
            [dict(label='Report Name', values=hyperparam_df['report_name']),
             dict(label='Num layers', values=hyperparam_df['layers'],
             categoryorder='category descending'),
             dict(label='Kernel_Size', values=hyperparam_df['kernel_size'],
                  categoryorder='category descending'),
             dict(label='Activation',
                  values=hyperparam_df['activation_function']),
             dict(label='Optimizer', values=hyperparam_df['optimizer']),
             dict(label='Pooling', values=hyperparam_df['pooling'].values),
             dict(label='Accuracy', values=hyperparam_df['test_accuracy'],
                  categoryorder='category descending'), ])))

    fig2.update_layout(dict(font=dict(size=12)),
                       title='Parallel Coordinate Plot Comparison',
                       plot_bgcolor='lightblue',
                       paper_bgcolor='white')
    # fig2.show()
    return fig1, fig2


def summary_report(report_path):
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
    hyperparam_df, history_df = report_dataframes(report_path)
    fig1, fig2 = report_plots(hyperparam_df, history_df)
    return fig1, fig2
