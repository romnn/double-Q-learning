import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import utils

SMALL_FONTSIZE = 14
MEDIUM_FONTSIZE = 16
BIGGER_FONTSIZE = 16

def plot_values(
    graphs: List[Tuple[Dict[str, Any], np.array]],
    smoothing_radius: int = 50,
    confidence_band=True,
    confidence_band_scale: float = 1.0,
    title: Optional[str] = None,
    legend: Optional[bool] = None,
    xlabel: str = "",
    ylabel: str = "",
    savefig: Optional[str] = None,
    tick_fontsize=SMALL_FONTSIZE,
    title_fontsize=BIGGER_FONTSIZE,
    legend_fontsize=MEDIUM_FONTSIZE,
    axis_fontsize=MEDIUM_FONTSIZE,
    figsize: Tuple[int, int] = (8,4),
    show: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize)
    
    # tick font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)

    # ax.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    for options, data in graphs:
        if len(data.shape) != 2:
            raise ValueError("shape of measurement must be (runs, episodes)")
        
        mean_data = utils.smoothing_window(
            data.mean(axis=0),
            radius=smoothing_radius
        )
        std_data = utils.smoothing_window(
            data.std(axis=0),
            radius=smoothing_radius
        )
        ax.plot(
            mean_data,
            label=options.get("label"),
            color=options.get("color"),
        )
        if confidence_band:
            ax.fill_between(
                np.arange(len(mean_data)),
                mean_data - confidence_band_scale * std_data,
                mean_data + confidence_band_scale * std_data,
                alpha=0.3,
                color=options.get("color"),
            )
    if title is not None:
        plt.title(title, fontsize=title_fontsize)
    if legend == True or (
        legend == None
        and len(graphs) > 1
        and any(["label" in options for options, _ in graphs])
    ):
        plt.legend(fontsize=legend_fontsize)
    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.ylabel(ylabel, fontsize=axis_fontsize)
    plt.tight_layout()
    if savefig is not None:
        Path(savefig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savefig)
    if show:
        plt.show()
    return fig, ax