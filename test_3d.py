

import numpy as np
import tth_core
import time

print("--- INITIALISATION TTH 3D (QUATERNIONS) ---")
system = tth_core.TTHSystem3D()

# Ajout de 3 centres avec des axes de rotation différents (X, Y, et Z)
# Centre 1 : Rotation autour de l'axe Z (0, 0, 1)
system.add_center(ox=1.0, oy=1.0, oz=0.0, ax=0.0, ay=0.0, az=1.0, weight=1/3, phase=0.0)

# Centre 2 : Rotation autour de l'axe X (1, 0, 0)
system.add_center(ox=-1.0, oy=2.0, oz=1.0, ax=1.0, ay=0.0, az=0.0, weight=1/3, phase=np.pi/3)

# Centre 3 : Rotation autour de l'axe Y (0, 1, 0)
system.add_center(ox=0.0, oy=-1.0, oz=2.0, ax=0.0, ay=1.0, az=0.0, weight=1/3, phase=np.pi/2)

print("Système 3D configuré.")

# Test sur 1 Million de points 3D
N_POINTS = 1_000_000
points_3d = np.random.rand(N_POINTS, 3)
theta = np.pi / 4

print(f"Lancement du calcul 3D pour {N_POINTS} points...")
start_time = time.time()

# Exécution de la TTH 3D
transformed_3d = system.apply(points_3d, theta)

print(f"Temps d'exécution : {time.time() - start_time:.5f} secondes")
print(f"Aperçu des coordonnées spatiales (X, Y, Z) du premier point :\n{transformed_3d[0]}")
