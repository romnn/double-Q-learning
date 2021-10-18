import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import utils

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
):
    plt.figure(figsize=(15,8))
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
        plt.plot(
            mean_data,
            label=options.get("label"),
            color=options.get("color"),
        )
        if confidence_band:
            plt.fill_between(
                np.arange(len(mean_data)),
                mean_data - confidence_band_scale * std_data,
                mean_data + confidence_band_scale * std_data,
                alpha=0.3,
                color=options.get("color"),
            )
    if title is not None:
        plt.title(title)
    if legend == True or (
        legend == None
        and len(graphs) > 1
        and any(["label" in options for options, _ in graphs])
    ):
        plt.legend()
    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()