import plotly
import shutil
import hardy.data_reporting.reporting as reporting

from hardy.handling.to_catalogue import learning_set
from hardy.recognition import tuner
# from hardy.handling import pre_processing as preprocessing

path = './hardy/test/test_image/'
split = 0.25
classes = ['class_1', 'class_2']
batch_size = 1


def test_report_plot():
    config_path = './hardy/recognition/'
    log_dir = './hardy/test/temp_report/report/'

    train, val = learning_set(path, split=split, classes=classes,
                              iterator_mode=None)

    tuner.build_param(config_path)

    tuned_model = tuner.run_tuner(train, val,
                                  project_name='test_run')
    model, history, metrics = tuner.best_model(tuned_model, train,
                                               val, val)

    tuner.report_generation(model, history, metrics, log_dir,
                            tuner=tuned_model, save_model=False)

    fig = reporting.report_plot('./hardy/test/temp_report/')

    assert isinstance(fig, plotly.graph_objs._figure.Figure),\
        'The returned figure is not a plotly object'

    shutil.rmtree('./hardy/test/temp_report')
    shutil.rmtree('./test_run')
