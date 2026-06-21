
import numpy as np
import tth_core
import time

print("--- INITIALISATION TTH GPU (CUDA) ---")
system = tth_core.TTHSystemGPU()

system.add_center(ox=1.0, oy=1.0, oz=0.0, ax=0.0, ay=0.0, az=1.0, weight=1/3, phase=0.0)
system.add_center(ox=-1.0, oy=2.0, oz=1.0, ax=1.0, ay=0.0, az=0.0, weight=1/3, phase=np.pi/3)
system.add_center(ox=0.0, oy=-1.0, oz=2.0, ax=0.0, ay=1.0, az=0.0, weight=1/3, phase=np.pi/2)

print("Système GPU configuré.")

N_POINTS = 5_000_000 # 5 Millions de points (5 fois plus que le test CPU)
print(f"Génération du nuage de {N_POINTS} points 3D...")
points_3d = np.random.rand(N_POINTS, 3).astype(np.float64)
theta = np.pi / 4

print("Lancement du Kernel CUDA...")
start_time = time.time()

# Exécution fulgurante sur la GTX 1650 Ti
transformed_3d = system.apply(points_3d, theta)

gpu_time = time.time() - start_time
print(f"\n--- RÉSULTATS GPU ---")
print(f"Temps d'exécution (Transfert + Calcul CUDA) : {gpu_time:.5f} secondes")
print(f"Capacité de rendu théorique : {1.0 / gpu_time:.1f} FPS pour {N_POINTS} points")
