import aspsim.room.generatepoints as gp

import numpy as np
import matplotlib.pyplot as plt







def test_show_equidistance_rectangle():

    num_points = 64
    points = gp.equidistant_rectangle(num_points, (2,1), extra_side_lengths=(0.5, 0), offset=0.5, z = 0)

    fig, ax  = plt.subplots()
    for m in range(num_points):
        ax.text(points[m,0], points[m,1], str(m))
    plt.plot(points[:,0], points[:,1], "x")
    plt.show()
