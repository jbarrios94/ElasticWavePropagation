import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rc('font', size=16)
mpl.rc('figure', figsize=(8, 6))

def plot_shotrecord(rec, origin, domain_size, t0, tn, factor= 10., cmap = "gray", colorbar=True):

    scale = np.max(rec) / factor
    extent = [1e-3*origin[0], 1e-3*origin[0] + 1e-3*domain_size[0],
              1e-3*tn, t0]

    plt.figure()
    plot = plt.imshow(rec, vmin=-scale, vmax=scale, cmap=cmap, extent=extent, aspect = 'auto')
    plt.xlabel('X position (km)')
    plt.ylabel('Time (s)')
    
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)
    
    plt.tight_layout()

def plot_image(data, extent, vmin=None, vmax=None, colorbar=True, cmap="gray"):

    plt.figure()
    plot = plt.imshow(data,
                      vmin=vmin or 0.9 * np.min(data),
                      vmax=vmax or 1.1 * np.max(data),
                      cmap=cmap, extent= extent)

    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)

    plt.tight_layout()


def plot_seismic_traces(traces, t0, tn):

    plt.figure()
    for trace in traces:
        x = np.linspace(t0, tn, trace.shape[0])
        plt.plot(trace, x)
        plt.ylim(tn, t0)

    plt.tight_layout()

