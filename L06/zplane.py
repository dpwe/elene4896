"""Plot complex poles and zeros on a z-plane diagram."""

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib import patches

# from http://gregsurges.com/post/90076189091/python-zplane
def zplane(z, p):

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    unit_circle = patches.Circle((0,0), radius=1, fill=False,
                                 color='black', ls='solid', alpha=0.9)
    ax.add_patch(unit_circle)
    plt.axvline(0, color='0.7')
    plt.axhline(0, color='0.7')
    plt.xlim((-2, 2))
    plt.ylim((-1.25, 1.25))
    plt.grid()

    plt.plot(z.real, z.imag, 'ko', fillstyle='none', ms = 10)
    plt.plot(p.real, p.imag, 'kx', fillstyle='none', ms = 10)
    return fig
    
