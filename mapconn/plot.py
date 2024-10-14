import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_mapconn_curve(mfc_obs=None, mfc_null=None, ax=None,
                       errorbar=("pi", 90), label="Sample mean (90% PI)", color="tab:red", alpha=1, lw=2,
                       errorbar_kws=None,
                       plot_individual=True, ind_color="k", ind_alpha=0.1, ind_lw=0.5,
                       xlabel="Percentile", ylabel="FC", title=None, legend=False):
    
    #  validate input
    if mfc_obs is None and mfc_null is None:
        raise ValueError("Provide either mfc_obs or mfc_null, or both.")
    if mfc_obs is not None:
        if not isinstance(mfc_obs, (pd.DataFrame, pd.Series)):
            raise ValueError("Provide mfc_obs as 1D or 2D pd array.")
    if mfc_null is not None:
        if not isinstance(mfc_null, list):
            raise ValueError("Provide mfc_null as list of pd arrays.")

    # ax
    if ax is None:
        fig, ax = plt.subplots()
    
    # title
    if title is None:
        try:
            title = mfc_obs.columns.get_level_values(0).unique()[0]
        except:
            title = None
    elif title in [False, ""]:
        title = None
        
    # observed
    if mfc_obs is not None:
        
        # plot mean curve
        sn.lineplot(
            data=mfc_obs.melt(ignore_index=False).dropna(),
            x="pct",
            y="value",
            ax=ax,
            color=color,
            errorbar=errorbar,
            label=label,
            alpha=alpha,
            lw=lw,
            err_kws=errorbar_kws
        )
        
        # plot individual curves
        if plot_individual:
            for i in range(len(mfc_obs.index)):
                ax.plot(
                    mfc_obs.columns.get_level_values(-1),
                    mfc_obs.values[i,:],
                    c=ind_color,
                    lw=ind_lw,
                    alpha=ind_alpha,
                    zorder=-100
                )
                
    # null
    if mfc_null is not None:
        mfc_null_means = pd.concat([null.mean(axis=0) for null in mfc_null], axis=1).T
        ax.plot(
            mfc_null_means.columns.get_level_values(-1),
            np.quantile(mfc_null_means, 0.5, axis=0),
            color="0.3",
            alpha=1,
            lw=lw,
            label=f"Median of null means",
            zorder=-1000
        )
        for q, c in [(0.25, "0.7"), (0.10, "0.8"), (0.01, "0.9")]:
            ax.fill_between(
                mfc_null_means.columns.get_level_values(-1),
                np.quantile(mfc_null_means, q, axis=0),
                np.quantile(mfc_null_means, 1-q, axis=0),
                color=c,
                alpha=1,
                label=f"{(1-q) * 100:.0f}% PI of null means",
                zorder=-1001 + q
            )
            
    # labels
    ax.set_title(title, weight="semibold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    # legend
    ax.legend()
    
    return ax



def plot_mapconn_curves(mfc_obs=None, mfc_null=None, maps=None,
                        fig=None, axes=None, inset_axes=None, n_cols=6, figsize=None, sharex=True, sharey=False, y_lims=None,
                        titles=True,
                        colors=None, legend="row",
                        plot_kws={}):
    
    #  validate input
    if mfc_obs is None and mfc_null is None:
        raise ValueError("Provide either mfc_obs or mfc_null, or both.")
    if not isinstance(mfc_obs, (pd.DataFrame, pd.Series)):
        raise ValueError("Provide mfc_obs as 1D or 2D pd array.")
    if inset_axes is not None:
        if axes is None:
            raise ValueError("Provide axes if inset_axes is used.")
        if len(np.ravel(inset_axes)) != len(np.ravel(axes)):
            raise ValueError("Provide as many inset_axes as axes.")

    # get maps
    if maps is None:
        maps = mfc_obs.columns.get_level_values(0).unique().tolist()
    elif isinstance(maps, str):
        maps = [maps]
    
    # make dict
    if not isinstance(maps, dict):
        n_rows = int(np.ceil(len(maps) / n_cols))
        maps = {i: maps[i * n_cols : ((i+1) * n_cols) if i < n_rows else len(maps)] for i in range(n_rows)}
    
    # colors
    if colors is None:
        colors = "tab:red"
    if isinstance(colors, (str, tuple)):
        colors = {k: [colors] * len(v) for k, v in maps.items()}
    elif isinstance(colors, list):
        if len(colors) == len(maps):
            colors = {k: [c] * len(v) for c, (k, v) in zip(colors, maps.items())}
        elif len(colors) == len(sum(maps.values(), [])):
            k_lens = np.cumsum([len(v) for v in maps.values()])
            colors = np.split(colors, k_lens)
        else:
            raise ValueError("Provide one single colors as string or a list of colors with one color per map or one color per map category.")
            
    # plot dimensions
    n_rows = len(maps)
    n_cols = int(max([len(v) for v in maps.values()])) 
    if legend == "row":
        n_cols += 1
    if figsize is None:
        figsize = (n_cols * 2.5, n_rows * 2.5)
        
    # initiate
    if axes is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.atleast_2d(axes)
    
    # plot
    axes_ravel = np.ravel(axes)
    if inset_axes is not None:
        inset_axes = np.ravel(inset_axes)
    for r, cat in enumerate(maps.keys()):
        
        for c, m in enumerate(maps[cat]):
            
            ax = axes_ravel[r * n_cols + c]
            subplotspec = ax.get_subplotspec()
            if inset_axes is not None:
                ax = inset_axes[r * n_cols + c]
            ax.set_box_aspect(1)
            
            plot_mapconn_curve(
                mfc_obs=mfc_obs.loc[:, (m, slice(None))],
                mfc_null=([null.loc[:, (m, slice(None))] for null in mfc_null]) if mfc_null is not None else None,
                ax=ax,
                **dict(
                    legend=legend,
                    title=m if titles else False,
                    color=colors[cat][c],
                    xlabel="Percentile" if subplotspec.is_last_row() else "",
                    ylabel="FC" if subplotspec.is_first_col() else ""
                ) | plot_kws
            )
            
            # ylims
            if y_lims is not None:
                y_lims_ax = ax.get_ylim()
                ax.set_ylim(y_lims[0] if y_lims[0] is not None and y_lims[0] < y_lims_ax[0] else y_lims_ax[0],
                            y_lims[1] if y_lims[1] is not None and y_lims[1] > y_lims_ax[1] else y_lims_ax[1])
                
            # legend
            ax.legend().set_visible(False)
            if legend == "all":
                ax.legend()
            elif legend == "row":
                if c == len(maps[cat]) - 1:
                    axes[r, c + 1].set_box_aspect(1)
                    axes[r, c + 1].legend(
                        *ax.get_legend_handles_labels(),
                        loc="center left",
                        bbox_to_anchor=(0, 0.5)
                    )
                else:
                    ax.legend().set_visible(False)
                
        for c in range(c + 1, n_cols):
            axes[r, c].set_axis_off()
            
    return fig, axes
    
    #fig.tight_layout()
    