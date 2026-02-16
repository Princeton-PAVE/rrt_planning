import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

app = QtWidgets.QApplication([])

M = np.random.rand(100, 150)

win = pg.ImageView()
win.setImage(M)
win.show()

app.exec()
