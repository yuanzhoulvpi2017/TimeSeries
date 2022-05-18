
import numpy as np

arparams = [.75, -.25]
maparams = [.65, .35]


arparams = np.array(arparams)
maparams = np.array(maparams)


arpoly = np.polynomial.Polynomial(arparams)
mapoly = np.polynomial.Polynomial(maparams)

arcoefs = -arparams[1:]
macoefs = maparams[1:]


