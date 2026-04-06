import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def add_axis_to_xtick(fig, ax, dx_axs=1):
    plt.draw()

    inv = fig.transSubfigure.inverted()
    if isinstance(fig, mpl.figure.SubFigure):
        inv = fig.transSubfigure.inverted()
    else:
        inv = fig.transFigure.inverted()

    xticks = ax.get_xticks()
    ymin, _ = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    pix_coord = ax.transData.transform([(xtick, ymin) for xtick in xticks])    
    fig_coord = inv.transform(pix_coord)
    
    dx_fig = np.abs(fig_coord[0,0] - fig_coord[1,0])  * dx_axs

    xtick_out = []
    ax_out = []
    for i_tick in range(len(xticks)):        

        if (xticks[i_tick]<xmin) or (xticks[i_tick]>xmax):
            continue
        new_ax_pos = [
            fig_coord[i_tick,0]-dx_fig/2,
            fig_coord[i_tick,1]-dx_fig*2, 
            dx_fig,
            dx_fig
            ]
                
        nax = fig.add_axes(new_ax_pos)
        nax.set_xticks([])
        nax.set_yticks([])
        nax.set_aspect('equal')
        # nax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        ax_out.append(nax)
        xtick_out.append(xticks[i_tick])
    return xtick_out, ax_out


def add_dm_to_x(dm, xtick_out, ax_out, TR=1.5):
    max_t = dm.shape[-1]
    for i,xtick in enumerate(xtick_out):
        dm_idx = int(xtick/TR)
        if dm_idx>=max_t:
            ax_out[i].axis('off')
        else:
            ax_out[i].imshow(dm[:,:,dm_idx], vmin=0, vmax=1,cmap='Greys', extent=[-5,5,-5,5], alpha=1)

def add_dm_to_ts(fig, ax, dm, TR=1.5, dx_axs=1):
    xtick_out, ax_out = add_axis_to_xtick(fig, ax, dx_axs)
    add_dm_to_x(dm, xtick_out, ax_out, TR)
