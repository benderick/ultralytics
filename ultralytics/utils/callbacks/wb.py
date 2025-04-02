# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import SETTINGS, TESTS_RUNNING, oc_to_dict
from ultralytics.utils.torch_utils import model_info_for_loggers
from ultralytics.utils.database import Run

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["wandb"] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, "__version__")  # verify package is not directory
    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


def _custom_table(x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """
    Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of the default wandb precision-recall
    curve while allowing for enhanced customization. The visual metric is useful for monitoring model performance across
    different classes.

    Args:
        x (List): Values for the x-axis; expected to have length N.
        y (List): Corresponding values for the y-axis; also expected to have length N.
        classes (List): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot; defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis; defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis; defaults to 'Precision'.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    import pandas  # scope for faster 'import ultralytics'

    df = pandas.DataFrame({"class": classes, "y": y, "x": x}).round(3)
    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wb.plot_table(
        "wandb/area-under-curve/v0", wb.Table(dataframe=df), fields=fields, string_fields=string_fields
    )


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """
    Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb.
    The curve can represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.

    Args:
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape CxN, where C is the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C. Defaults to [].
        id (str, optional): Unique identifier for the logged data in wandb. Defaults to 'precision-recall'.
        title (str, optional): Title for the visualization plot. Defaults to 'Precision Recall Curve'.
        x_title (str, optional): Label for the x-axis. Defaults to 'Recall'.
        y_title (str, optional): Label for the y-axis. Defaults to 'Precision'.
        num_x (int, optional): Number of interpolated data points for visualization. Defaults to 100.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted. Defaults to True.

    Note:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    import numpy as np

    # Create new x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # Create arrays for logging
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # add new x
            y_log.extend(np.interp(x_new, x, yi))  # interpolate y to new x
            classes.extend([names[i]] * len(x_new))  # add class names
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(plots, step):
    """Logs plots from the input dictionary if they haven't been logged already at the specified step."""
    for name, params in plots.copy().items():  # shallow copy to prevent plots dict changing during iteration
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    """Initiate and start project if module is present."""
    import datetime

    # 获取当前时间
    now = datetime.datetime.now()
    # 将时间格式化为指定格式
    formatted_time = now.strftime("%y%m%d_%H%M")
    
    project = str(trainer.args.project)
    if "/" in project:
        project = project.split("/")[-1]
        
    name = str(trainer.args.name).replace("/", "-")
    name = f"{name}-{formatted_time}"
    
    if not wb.run:
        wb.init(
            project=project if trainer.args.project else "Ultralytics",
            name=name,
            config=vars(trainer.args),
            dir=trainer.save_dir.parent
        )
        logger_info = trainer.logger
        wb.run.notes = str(logger_info.notes)
        wb.run.tags = logger_info.tags
        run = Run.create(id=wb.run.id, location=str(trainer.save_dir.parent),**oc_to_dict(logger_info))
        # run.tags = str(run.tags)
        run.save()


def on_fit_epoch_end(trainer):
    """Logs training metrics and model information at the end of an epoch."""
    wb.run.log(trainer.metrics, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)
    
    # 记录模型的mAP和mAP50到数据库
    epoche_list = [0, 1, 50, 100, 150]
    if trainer.epoch in epoche_list:
        run = Run.get(Run.id == wb.run.id)
        run.map = eval(run.map) + [trainer.metrics["metrics/mAP50-95(B)"]]
        run.map50 = eval(run.map50) + [trainer.metrics["metrics/mAP50(B)"]]
        run.save()
    
        # 对于非基本模型，在低于基本模型的map和map50时停止训练
        if not run.is_basic:
            idx = epoche_list.index(trainer.epoch)
            basics = Run.select().where((Run.project == run.project) & Run.is_basic)
            true_basics_map = []
            true_basics_map50 = []
            try:
                for basic in basics:
                    if set(eval(basic.tags)) & set(eval(run.tags)):
                        true_basics_map.append(eval(basic.map)[idx])    
                        true_basics_map50.append(eval(basic.map50)[idx])  
            except:
                print("在获取基本模型的map和map50时发生错误")
            if len(true_basics_map) > 0 and len(true_basics_map50) > 0:
                run_map = trainer.metrics["metrics/mAP50-95(B)"]
                run_map50 = trainer.metrics["metrics/mAP50(B)"]
                for map in true_basics_map:
                    if run_map < map * 0.95:
                        trainer.need_to_finish = True
                        break
                for map50 in true_basics_map50:
                    if run_map50 < map50 * 0.9:
                        trainer.need_to_finish = True
                        break
                if trainer.need_to_finish:
                    print("触发低指标中断")
                    wb.run.alert(title="低指标中断",
                                text=f"Run {run.id} was killed because of low metrics in epoch {trainer.epoch}. \n In project {run.project}, name {run.name}, tags {run.tags}.",
                                level=wb.AlertLevel.WARN,
                                wait_duration=300,)

def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    """Save the best model as an artifact at end of training."""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=["best"])
    # Check if we actually have plots to save
    if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
        for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(trainer.validator.metrics.names.values()),
                id=f"curves/{curve_name}",
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )
    wb.run.finish()  # required or run continues on dashboard


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
