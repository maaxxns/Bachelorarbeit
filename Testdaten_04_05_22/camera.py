import numpy as np 
import matplotlib.pyplot as plt
from uncertainties.unumpy import uarray
import uncertainties
import matplotlib
import pandas as pd

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
index_max = np.unravel_index(pixel_values.argmax(), pixel_values.shape)

print('Summe aller Pixel normiert mit maximalem Count von einem Pixel?:',np.sum(pixel_values)/np.max(pixel_values))

#How to find the spot?
#lets try masks
detektorsize = np.shape(pixel_values)
fig, ax = plt.subplots()
pixel_values_data = pixel_values
mask = pixel_values <= 100
pixel_values[mask] = np.nan
cmap = matplotlib.cm.get_cmap("inferno").copy()
cmap.set_bad('white')


####################################
#   Draw a circel around the spot
####################################
print('indexmax: ', index_max[0])
R=10
maske_kreis = np.full((160,160), False)
for x in range(160):
    for y in range(160):
        if (abs((x-index_max[0])**2+(y-index_max[1])**2) <= R**2) :
            maske_kreis[x,y] = True

size_circle= np.sum(maske_kreis)


im, _ = heatmap(pixel_values, ax=ax,
                cmap=cmap, cbarlabel="counts/pixel")
ax.set_title('Isolated Spot')
print('Spotsize with pixel above 100 counts:',detektorsize[0]*detektorsize[1]-np.sum(mask))
plt.savefig('spotsize.pdf')

pixel_values_data_1dim = pixel_values_data.reshape(160*160)
sorted_index = np.argsort(pixel_values_data_1dim)
sorted_pixel_values = pixel_values_data_1dim[sorted_index]
percentage = np.array(detektorsize[0]*detektorsize[1] * 0.87)
sorted_pixel_values = sorted_pixel_values.reshape(160*160)
sorted_pixel_values = sorted_pixel_values[range(percentage.astype(int))]
print((sorted_pixel_values))

######## Das funktioniert noch nicht