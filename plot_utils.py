import string
#import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('GTK')
from pylab import *
from matplotlib import collections
from mpl_toolkits.axes_grid.inset_locator import inset_axes

## choose figure or poster defaults
#fig_defaults = 'poster'                                 # poster defaults
#fig_defaults = 'presentation'                           # presentation defaults
fig_defaults = 'paper'                                  # paper figure defaults

if fig_defaults == 'paper':
    ####### paper figure defaults
    label_fontsize = 8 # pt
    plot_linewidth = 0.5 # pt
    linewidth = 1.0#0.5
    axes_linewidth = 0.5
    marker_size = 3.0 # markersize=<...>
    cap_size = 2.0 # for errorbar caps, capsize=<...>
    columnwidth = 85/25.4 # inches
    twocolumnwidth = 174/25.4 # inches
    linfig_height = columnwidth*2.0/3.0
    fig_dpi = 300
elif fig_defaults == 'poster':
    ####### poster defaults
    label_fontsize = 12 # pt
    plot_linewidth = 1.0 # pt
    linewidth = 1.0
    axes_linewidth = 1.0
    marker_size = 3.0
    cap_size = 2.0 # for errorbar caps
    columnwidth = 4 # inches
    linfig_height = columnwidth*2.0/3.0
else:
    ####### presentation defaults for screenshot
    label_fontsize = 20 # pt
    plot_linewidth = 0.5 # pt
    linewidth = 1.0#0.5
    axes_linewidth = 0.5
    marker_size = 3.0 # markersize=<...>
    cap_size = 2.0 # for errorbar caps, capsize=<...>
    columnwidth = 85/25.4 # inches
    twocolumnwidth = 174/25.4 # inches
    linfig_height = columnwidth*2.0/3.0
    fig_dpi = 300


def axes_off(ax,x=True,y=True,xlines=False,ylines=False):
    ''' True is to turn things off, False to keep them on!
        x,y are for ticklabels, xlines,ylines are for ticks'''
    if x:
        for xlabel_i in ax.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
    if y:
        for xlabel_i in ax.get_yticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)
    if xlines:
        for tick in ax.get_xticklines():
            tick.set_visible(False)
    if ylines:
        for tick in ax.get_yticklines():
            tick.set_visible(False)

def set_tick_widths(ax,tick_width):
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markeredgewidth(tick_width)
        tick.tick2line.set_markeredgewidth(tick_width)

def axes_labels(ax,xtext,ytext,adjustpos=False,fontsize=label_fontsize,xpad=None,ypad=None):
    ax.set_xlabel(xtext,fontsize=fontsize,labelpad=xpad)
    # increase xticks text sizes
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    ax.set_ylabel(ytext,fontsize=fontsize,labelpad=ypad)
    # increase yticks text sizes
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)
    if adjustpos:
        ## [left,bottom,width,height]
        ax.set_position([0.135,0.125,0.84,0.75])
    set_tick_widths(ax,axes_linewidth)

def biglegend(legendlocation='upper right',ax=None,fontsize=label_fontsize, **kwargs):
    if ax is not None:
        leg=ax.legend(loc=legendlocation, **kwargs)
    else:
        leg=legend(loc=legendlocation, **kwargs)
    # increase legend text sizes
    for t in leg.get_texts():
        t.set_fontsize(fontsize)

def beautify_plot(ax,x0min=True,y0min=True,
        xticksposn='bottom',yticksposn='left',xticks=None,yticks=None,
        drawxaxis=True,drawyaxis=True):
    """
    x0min,y0min control whether to set min of axis at 0.
    xticksposn,yticksposn governs whether ticks are at
    'both', 'top', 'bottom', 'left', 'right', or 'none'.
    xtickx/yticks is a list of ticks, else [min,0,max] for y and [min,max] for x is set.
    Due to rendering issues,
    axes do not overlap exactly with the ticks, dunno why.
    """
    ax.get_yaxis().set_ticks_position(yticksposn)
    ax.get_xaxis().set_ticks_position(xticksposn)
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    if x0min: xmin=0
    if y0min: ymin=0
    if xticks is None: ax.set_xticks([xmin,xmax])
    else: ax.set_xticks(xticks)
    if yticks is None: ax.set_yticks([ymin,0,ymax])
    else: ax.set_yticks(yticks)
    ### do not set width and color of axes by below method
    ### axhline and axvline are not influenced by spine below.
    #ax.axhline(linewidth=axes_linewidth, color="k")
    #ax.axvline(linewidth=axes_linewidth, color="k")
    ## spine method of hiding axes is cleaner,
    ## but alignment problem with ticks in TkAgg backend remains.
    for loc, spine in ax.spines.items(): # items() returns [(key,value),...]
        spine.set_linewidth(axes_linewidth)
        if loc == 'left' and not drawyaxis:
            spine.set_color('none') # don't draw spine
        elif loc == 'bottom' and not drawxaxis:
            spine.set_color('none') # don't draw spine
        elif loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ### alternate method of drawing axes, but for it,
    ### need to set frameon=False in add_subplot(), etc.
    #if drawxaxis:
    #    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin),\
    #        color='black', linewidth=axes_linewidth))
    #if drawyaxis:
    #    ax.add_artist(Line2D((xmin, xmin), (ymin, ymax),\
    #        color='black', linewidth=axes_linewidth))
    ## axes_labels() sets sizes of tick labels too.
    axes_labels(ax,'','',adjustpos=False)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    return xmin,xmax,ymin,ymax

def beautify_plot3d(ax,x0min=True,y0min=True,
        xticksposn='bottom',yticksposn='left',xticks=None,yticks=None,zticks=None,
        drawxaxis=True,drawyaxis=True):
    """
    x0min,y0min control whether to set min of axis at 0.
    xticksposn,yticksposn governs whether ticks are at
    'both', 'top', 'bottom', 'left', 'right', or 'none'.
    xtickx/yticks/zticks is a list of ticks, else [min,max] is set.
    Due to rendering issues,
    axes do not overlap exactly with the ticks, dunno why.
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    zmin, zmax = ax.get_zlim()
    if x0min: xmin=0
    if y0min: ymin=0
    if xticks is None: ax.set_xticks([xmin,xmax])
    else: ax.set_xticks(xticks)
    if yticks is None: ax.set_yticks([ymin,ymax])
    else: ax.set_yticks(yticks)
    if zticks is None: ax.set_zticks([zmin,zmax])
    else: ax.set_zticks(zticks)
    ### do not set width and color of axes by below method
    ### axhline and axvline are not influenced by spine below.
    #ax.axhline(linewidth=axes_linewidth, color="k")
    #ax.axvline(linewidth=axes_linewidth, color="k")
    ## spine method of hiding axes is cleaner,
    ## but alignment problem with ticks in TkAgg backend remains.
    for loc, spine in ax.spines.items(): # items() returns [(key,value),...]
        spine.set_linewidth(axes_linewidth)
        if loc == 'left' and not drawyaxis:
            spine.set_color('none') # don't draw spine
        elif loc == 'bottom' and not drawxaxis:
            spine.set_color('none') # don't draw spine
        elif loc in ['right','top']:
            spine.set_color('none') # don't draw spine
    ### alternate method of drawing axes, but for it,
    ### need to set frameon=False in add_subplot(), etc.
    #if drawxaxis:
    #    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin),\
    #        color='black', linewidth=axes_linewidth))
    #if drawyaxis:
    #    ax.add_artist(Line2D((xmin, xmin), (ymin, ymax),\
    #        color='black', linewidth=axes_linewidth))
    ## axes_labels() sets sizes of tick labels too.
    axes_labels(ax,'','',adjustpos=False)
    # set zticks text sizes
    for label in ax.get_zticklabels():
        label.set_fontsize(label_fontsize)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_zlim(zmin,zmax)
    return xmin,xmax,ymin,ymax,zmin,zmax

def fig_clip_off(fig):
    ## clipping off for all objects in this fig
    for o in fig.findobj():
        o.set_clip_on(False)

## ------
## from https://gist.github.com/dmeliza/3251476#file-scalebars-py

# Adapted from mpl_toolkits.axes_grid2
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)

from matplotlib.offsetbox import AnchoredOffsetbox
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, label_fontsize=label_fontsize, color='k', **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, fc="none", linewidth=axes_linewidth, color=color))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, fc="none", linewidth=axes_linewidth, color=color))

        if sizex and labelx:
            textareax = TextArea(labelx,minimumdescent=False,textprops=dict(size=label_fontsize,color=color))
            bars = VPacker(children=[bars, textareax], align="center", pad=0, sep=sep)
        if sizey and labely:
            ## VPack a padstr below the rotated labely, else label y goes below the scale bar
            ## Just adding spaces before labely doesn't work!
            padstr = '\n '*len(labely)
            textareafiller = TextArea(padstr,textprops=dict(size=label_fontsize/3.0))
            textareay = TextArea(labely,textprops=dict(size=label_fontsize,rotation='vertical',color=color))
            ## filler / pad string VPack-ed below labely
            textareayoffset = VPacker(children=[textareay, textareafiller], align="center", pad=0, sep=sep)
            ## now HPack this padded labely to the bars
            bars = HPacker(children=[textareayoffset, bars], align="top", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)

def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, \
    label_fontsize=label_fontsize, color='k', **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, label_fontsize=label_fontsize, color=color, **kwargs)
    ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)

    return sb

## from https://gist.github.com/dmeliza/3251476#file-scalebars-py -- ends
## ------

## Broken y-axis
## adapted from https://matplotlib.org/examples/pylab_examples/broken_axis.html
def add_y_break_lines(ax1,ax2):
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')               # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # add the break lines on the y-axis between ax1 and ax2
    # disable clipping; axes coordinates are in (0,1)
    d = .025  # length of diagonal break lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # remember: plot takes x-list, y-list
    ax1.plot((-d, +d), (-d, +d), **kwargs)      # top-left diagonal

    kwargs.update(transform=ax2.transAxes)      # switch to the bottom axes
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)    # bottom-left diagonal

## Broken x-axis
## adapted from https://matplotlib.org/examples/pylab_examples/broken_axis.html
def add_x_break_lines(ax1,ax2,jutOut=0.1,breakSize=.025):
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.set_visible(False)
    # don't draw the x-axis, redrawn later
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # add the break lines on the y-axis between ax1 and ax2
    # disable clipping; axes coordinates are in (0,1)
    d = breakSize                                           # length of diagonal break lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=plot_linewidth)
    # remember: plot takes x-list, y-list
    ax1.plot((0,1+jutOut), (0,0), **kwargs)                 # draw x-axis with a jutOut
    ax1.plot((1+jutOut-d, 1+jutOut+d), (-d, +d), **kwargs)  # bottom-right diagonal

    kwargs.update(transform=ax2.transAxes)                  # switch to the right axes
    ax2.plot((-jutOut,1), (0,0), **kwargs)                  # draw x-axis with a jutOut
    ax2.plot((-jutOut-d, -jutOut+d), (-d, +d), **kwargs)    # bottom-left diagonal
