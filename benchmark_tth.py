import numpy as np
import time
import tth_core  # L'importation magique de ton module C++ !

print("--- INITIALISATION DU MOTEUR TTH (C++) ---")
# 1. Création du système via le constructeur C++
system = tth_core.TTHSystem2D()

# 2. Ajout des centres (les mêmes que dans la théorie)
system.add_center(x=1.0, y=1.0, weight=1/3, phase=0.0)
system.add_center(x=-1.0, y=2.0, weight=1/3, phase=np.pi/3)
system.add_center(x=2.0, y=-1.0, weight=1/3, phase=np.pi/2)

print("Système TTH configuré avec succès.")

# 3. Création d'un dataset massif : 1 Million de points aléatoires
N_POINTS = 1_000_000
print(f"\nGénération de {N_POINTS} points 2D...")
points = np.random.rand(N_POINTS, 2)
theta = np.pi / 4

# 4. Le Test de Performance
print("Lancement du calcul vectorisé via Eigen...")
start_time = time.time()

# C'est ici que la magie opère : numpy passe ses données au C++ instantanément
transformed_points = system.apply(points, theta)

end_time = time.time()
execution_time = end_time - start_time

print("\n--- RÉSULTATS DU BENCHMARK ---")
print(f"Temps d'exécution pour {N_POINTS} points : {execution_time:.5f} secondes")
print(f"Aperçu des 3 premiers points transformés :\n{transformed_points[:3]}")


