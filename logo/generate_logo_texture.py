import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from numpy.linalg import matrix_power, norm

def arrow3d(ax, length=1, width=0.05, head=0.2, headwidth=2,
                theta_x=0, theta_z=0, offset=(0,0,0), **kw):
    w = width
    h = head
    hw = headwidth
    theta_x = np.deg2rad(theta_x)
    theta_z = np.deg2rad(theta_z)

    a = [[0,0],[w,0],[w,(1-h)*length],[hw*w,(1-h)*length],[0,length]]
    a = np.array(a)

    r, theta = np.meshgrid(a[:,0], np.linspace(0,2*np.pi,30))
    z = np.tile(a[:,1],r.shape[0]).reshape(r.shape)
    x = r*np.sin(theta)
    y = r*np.cos(theta)

    rot_x = np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x) ],
                      [0,np.sin(theta_x) ,np.cos(theta_x) ]])
    rot_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0 ],
                      [np.sin(theta_z) ,np.cos(theta_z),0 ],[0,0,1]])

    b1 = np.dot(rot_x, np.c_[x.flatten(),y.flatten(),z.flatten()].T)
    b2 = np.dot(rot_z, b1)
    b2 = b2.T+np.array(offset)
    x = b2[:,0].reshape(r.shape);
    y = b2[:,1].reshape(r.shape);
    z = b2[:,2].reshape(r.shape);
    ax.plot_surface(x,y,z, **kw)

#%%

# Parameters of the plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# numbers of spins on the outmost ring on the disk
num_phi_max = 24
# number of rings on the disk
num_r = 10
# colors for the color gradient on the disk
colors = ["pink", "hotpink"]
# set the length of the spins
arrow_length = 0.13
# factor by which the background disk is larger than the region on which the spins sit
disk_factor = 1.3
# cmap for the sprinkles
# sprinkle_map = cm.get_cmap('Pastel2')
sprinkle_map = ListedColormap(['red', 'darkviolet', 'aqua', 'lime', 'deeppink', 'white', 'orange', 'yellow'])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# instantiate 3D figure
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.set_size_inches((15, 15))
ax.set_axis_off()

# set radius to 1
R = 1

# Generate a new colormap for the background
cmap = LinearSegmentedColormap.from_list("mycmap", colors)
newcolors = cmap(np.linspace(0, 1, 256))
num_whites = int((np.sqrt(2) - 1) * 256)
whites = np.array([[255/256, 255/256, 255/256, 1]] * num_whites)
newcolors = np.concatenate((newcolors, whites), axis=0)
cmap = ListedColormap(newcolors)

#scale the radius of the disk
Rp = R * disk_factor
# plot a disk with a colorgradient using the new colormap
xx, yy = np.mgrid[-Rp:Rp:Rp/100, -Rp:Rp:Rp/100]
ax.contourf(xx, yy, xx**2 + yy**2, 500, offset=0.0001, alpha=0.3,
            zdir='z', cmap=cmap, vmin=0, vmax=np.sqrt(2) * Rp)

# parameterize the radii of the rings
rs = np.linspace(0, 1, num_r + 1)[1:]
# get the elevation angles
thetas = np.rad2deg(np.linspace(0, np.pi, num_r, endpoint=False))

# iterate over all the rings on the disk
for (r, theta) in zip(rs, thetas):
    # get number of spins on the ring based on its circumference (relative to the outmost ring).
    num_phi = int(r * num_phi_max / R)
    phis = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)
    # get the cartesian coordinates of the spins
    xs, ys = r * np.cos(phis), r * np.sin(phis)
    # get random colors for the spins
    cs = sprinkle_map(np.random.rand(len(xs)))
    # plot the spins
    for (x, y, phi, c) in zip(xs, ys, phis, cs):
        arrow3d(ax, theta_x=-theta, theta_z=phi, length=arrow_length, width=0.015,
                offset=[x -0.075, y + 0.1, -arrow_length / 2], color=c)


plt.show()
#plt.savefig('topwave_logo_texture.pdf')

