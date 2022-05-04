import numpy as np 
import matplotlib.pyplot as plt
from uncertainties.unumpy import uarray
import uncertainties
import matplotlib


def heatmap(data, ax=None, vmin=None, vmax=None,
            cbar_kw={}, cbarlabel="",**kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    return im, cbar



pixel_values = np.genfromtxt('firstfocus.ascii_0001.ascii.csv', delimiter=',', unpack=True) #unpack the txt file with pixel values
pixel_values = pixel_values[:160,:160] #somehow there is one line in the file that has just nan in it so we ignore it


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

im, _ = heatmap(pixel_values, ax=ax1, vmin=0,
                cmap="inferno", cbarlabel="counts/pixel")
ax1.set_title('Counts on Camera per Pixel')

im, _ = heatmap(pixel_values[70:90, 55:75], ax=ax2, vmin=0, vmax=np.max(pixel_values),
                cmap="inferno", cbarlabel="counts/pixel")
ax2.set_title('zoom in on spot')

fig.tight_layout()
plt.savefig('camera.pdf')

####################################################################################################


print('Summe aller Pixel normiert mit maximalem Count von einem Pixel?:',np.sum(pixel_values)/np.max(pixel_values))

#How to find the spot?
#lets try masks

detektorsize = np.shape(pixel_values)
fig, ax = plt.subplots()
mask = pixel_values <= 100
pixel_values[mask] = np.nan
cmap = matplotlib.cm.get_cmap("inferno").copy()
cmap.set_bad('white')
ax.imshow(pixel_values)
print('Spotsize with pixel above 100 counts:',detektorsize[0]*detektorsize[1]-np.sum(mask))
plt.savefig('spotsize.pdf')