import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
import os



def merge_2axes(fig1, fig2, file_name1="f1_.png", file_name2="f2_.png", horizontal=True, dpi=600):
  fig1.savefig(file_name1, dpi=dpi, bbox_inches='tight', pad_inches=0.5)
  fig2.savefig(file_name2, dpi=dpi, bbox_inches='tight', pad_inches=0.5)
  h1, h2 = [int(np.ceil(fig.get_figheight())) for fig in (fig1, fig2)]
  w1, w2 = [int(np.ceil(fig.get_figwidth())) for fig in (fig1, fig2)]
  if not horizontal:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(max(w1, w2), h1 + h2))
  else:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(w1 + w2, max(h1, h2)))
  gs = axes[0].get_gridspec()
  for ax in axes.flat:
        ax.remove()
  ax1 = fig.add_subplot(gs[:1])
  ax2 = fig.add_subplot(gs[1:])
  ax1.imshow(plt.imread(file_name1))
  ax2.imshow(plt.imread(file_name2))
  for ax in (ax1, ax2):
        for side in ('top', 'left', 'bottom', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(left=False, right=False, labelleft=False,
                      labelbottom=False, bottom=False)
  fig.tight_layout()
  os.remove(file_name1)
  os.remove(file_name2)
  return fig


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.
    This function creates a RadarAxes projection and registers it.
    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta


def ols_tree_graph(r, title, use_rlm=False, forecolor='mediumorchid', backcolor='thistle'):
    def addlabels(x, y, sig):
      for i, j in enumerate(x):
          if j < 0:
            j -= 0.1
          else:
            j += 0.05
          plt.text(j, y[i], sig[i])
    rb = '_robust' if use_rlm else ''
    d = r[pd.notna(r['sig'+rb])]# & (r['sig'+rb] != '')]
    x1 = d[d['coef'+rb] < 0]['coef'+rb].to_list()
    sig1 = d[d['coef'+rb] < 0]['sig'+rb].to_list()
    x2 = d[d['coef'+rb] > 0]['coef'+rb].to_list()
    sig2 = d[d['coef'+rb] > 0]['sig'+rb].to_list()
    y1 = range(len(x1))
    y2 = range(len(x1), (len(x1) + len(x2))) 
    y = range(len(x1) + len(x2))
    yl1 = d[d['coef'+rb] < 0]['index'].to_list()
    yl2 = d[d['coef'+rb] > 0]['index'].to_list()
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(5, 3.5))
    bar1 = axes[0].barh(y1, x1, align='center', color='red')
    bar2 = axes[1].barh(y2, x2, align='center', color='blue')
    axes[0].set(yticks=y, yticklabels=yl1 + yl2)
    axes[0].set_xlim([-1, -0.001])
    axes[1].set_xlim([0.001, 1])
    for ax in axes.flat:
        ax.margins(0.03)
        ax.grid(True)
    for i, b in enumerate(bar1):
        if len(sig1[i]):
          b.set_color(forecolor)
        else:
          b.set_color(backcolor)
    for i, b in enumerate(bar2):
        if len(sig2[i]):
          b.set_color(forecolor)
        else:
          b.set_color(backcolor)
    addlabels(x1, y1, sig1)
    addlabels(x2, y2, sig2)
    axes[0].tick_params(axis='y', which='both', labelleft=True, labelright=False)
    axes[1].tick_params(axis='y', which='both', labelleft=False, labelright=False)
    fig.suptitle(title, y=1.05)
    fig.text(0.5, 0, 'β', ha='center')
    #fig.tight_layout()
    fig.subplots_adjust(wspace=0, top=0.93)
    #plt.show()
    return fig


def lm_tree_graph(results, title=None, xlabel='$β$', exclude=[], reindex=[], rename_dict={}, file_name='fig1.png', 
                  yaxis=True, xaxis=True, report='stars', dpi=600, figsize=(5, 3.5), xlim=[-1, 1],
                  pos_forecolor='mediumorchid', neg_forecolor='mediumorchid', backcolor='thistle',
                  coef='coef', pvalue='p-value', CIL='CIL', CIR='CIR', sig='sig',
                  title_fontsize=None, xlabel_fontsize=None, ylabel_fontsize=None):
                  # fontsize {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    def addlabels(x, y, rep):
      d = {'fontsize': ylabel_fontsize} if ylabel_fontsize else {}
      for i, j in enumerate(x):
          if j < 0.0:
              plt.text(j-0.04, y[i]-0.04, rep[i], ha='right', **d)
          else:
              plt.text(j+0.04, y[i]-0.04, rep[i], ha='left', **d)

    r = results.copy()
    if len(reindex):
      r = r.reindex(reindex)  
    for e in exclude:
      if e in r.index:
        r.drop(index=e, axis=0, inplace=True)
    r.index = pd.Series(r.index).apply(lambda x: rename_dict[x] 
                                                if x in rename_dict else x)
    neg, pos = r[r[coef] < 0.0], r[r[coef] >= 0.0]
    x1, sig1, x2, sig2 = (neg[coef].to_list(), neg[sig].to_list(), 
                                    pos[coef].to_list(), pos[sig].to_list())
    yl1, yl2 = neg.index.to_list(), pos.index.to_list()
    y1, y2 = range(len(x1)), range(len(x1), len(x1) + len(x2))
    y = range(len(x1) + len(x2))
    if report == 'p':
        xl1 = neg[pvalue].to_list()
        xl1 = ['p='+i if i[0]=='.' else 'p'+i for i in xl1]
        xl2 = pos[pvalue].to_list()
        xl2 = ['p='+i if i[0]=='.' else 'p'+i for i in xl2]
    elif report == 'ci': 
        xl1 = [f'[{i}, {j}]' for i, j in zip(neg[CIL].to_list(), 
                                              neg[CIR].to_list())]
        xl2 = [f'[{i}, {j}]' for i, j in zip(pos[CIL].to_list(), 
                                              pos[CIR].to_list())]
    else:
      xl1, xl2 = sig1, sig2

    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=figsize)
    bar1 = axes[0].barh(y1, x1, align='center', color='red')
    bar2 = axes[1].barh(y2, x2, align='center', color='blue')
    if yaxis:
        axes[0].set(yticks=y, yticklabels=yl1 + yl2)
    else:
        axes[0].set(yticks=y, yticklabels=[])
    if not xaxis:
        axes[0].set(xticklabels=[])
        axes[1].set(xticklabels=[])
    axes[0].set_xlim([xlim[0], -0.001])
    axes[1].set_xlim([0.001, xlim[1]])
    for ax in axes.flat:
          ax.margins(0.03)
          ax.grid(True)
    for i, b in enumerate(bar1):
        if len(sig1[i]):
            b.set_color(neg_forecolor)
        else:
            b.set_color(backcolor)
    for i, b in enumerate(bar2):
        if len(sig2[i]):
            b.set_color(pos_forecolor)
        else:
            b.set_color(backcolor)
    x_left, x_right = plt.xlim()
    addlabels(x1, y1, xl1)
    addlabels(x2, y2, xl2)
    axes[0].tick_params(axis='y', which='both', labelleft=True, labelright=False)
    axes[1].tick_params(axis='y', which='both', labelleft=False, labelright=False)
    if title != None:
        d = {'fontsize': title_fontsize} if title_fontsize else {}
        fig.suptitle(title, y=len(title.split('\n'))*.03+1.015, transform=axes[0].transAxes,
                            x=x_right+.009, **d)
    if xlabel:
        d = {'fontsize': xlabel_fontsize} if xlabel_fontsize else {}
        fig.text(x_right+.009, -0.035, xlabel, ha='center', 
                 transform=axes[0].transAxes, **d)
    fig.subplots_adjust(wspace=0, top=0.93)
    plt.savefig(file_name, dpi=dpi, bbox_inches='tight')
    plt.close()
    return fig


def concat_figures(figures, figsize=(8, 5), axis=1, dpi=600, file_name=None, show=False):
    figures = [np.asarray(f.canvas.buffer_rgba()) for f in figures]
    r = np.sum([i.shape[axis] for i in figures])
    if axis:
          fig, axs = plt.subplots(1, len(figures), figsize=figsize, dpi=dpi, 
                                width_ratios=[i.shape[axis] / r for i in figures])
    else:
          fig, axs = plt.subplots(len(figures), 1, figsize=figsize, dpi=dpi, 
                                height_ratios=[i.shape[axis] / r for i in figures])
    fig.subplots_adjust(wspace=0, hspace=0)
    for ax, a in zip(axs, figures):
        ax.set_axis_off()
        ax.matshow(a, aspect='equal')
    if show:
      plt.show()
    if file_name != None:
      plt.savefig(file_name, dpi=dpi, bbox_inches='tight')
    plt.close()
    return fig
