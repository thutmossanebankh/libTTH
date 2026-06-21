import numpy as np
import matplotlib.pyplot as plt

class TTHSystem2D:
    def __init__(self):
        # Initialisation des listes pour stocker les paramètres des centres
        self.centers = []  # O_i
        self.weights = []  # w_i
        self.phases = []   # phi_i

    def add_center(self, x, y, weight, phase=0.0):
        """Ajoute un centre d'influence (O_i) au système TTH."""
        self.centers.append(np.array([x, y]))
        self.weights.append(weight)
        self.phases.append(phase)

    def _rotation_matrix(self, angle):
        """Génère une matrice de rotation 2x2 pour un angle donné."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], 
                         [s,  c]])

    def apply(self, points, theta):
        """
        Applique la Transformation de Thutmos à un ensemble de points.
        points: numpy array de dimension (N, 2)
        theta: paramètre de rotation globale
        """
        # Vérification de la normalisation des poids (somme = 1)
        total_weight = sum(self.weights)
        if not np.isclose(total_weight, 1.0):
            print(f"Attention: La somme des poids est {total_weight}, elle devrait être 1.0")

        # Initialisation du tableau des points transformés (rempli de zéros)
        transformed_points = np.zeros_like(points, dtype=float)

        # Calcul vectorisé de l'équation TTH
        for O_i, w_i, phi_i in zip(self.centers, self.weights, self.phases):
            # Matrice de rotation pour ce centre
            R = self._rotation_matrix(theta + phi_i)
            
            # Application de la transformation relative à O_i pour tous les points
            for j in range(len(points)):
                P = points[j]
                # w_i * (O_i + R * (P - O_i))
                contribution = w_i * (O_i + R @ (P - O_i))
                transformed_points[j] += contribution

        return transformed_points

# ==========================================
# TEST DU PROTOCOLE : Le Triangle sur 3 Cercles
# ==========================================
if __name__ == "__main__":
    # 1. Définition de la figure d'origine (un triangle rectangle ABC)
    triangle = np.array([
        [0.0, 0.0],  # Point A
        [2.0, 0.0],  # Point B
        [0.0, 1.5],  # Point C
        [0.0, 0.0]   # Retour au point A pour fermer le tracé
    ])

    # 2. Initialisation du système TTH
    system = TTHSystem2D()
    
    # Ajout de 3 centres d'influence (O1, O2, O3) avec poids égaux (barycentre parfait)
    system.add_center(x=1.0, y=1.0, weight=1/3, phase=0.0)
    system.add_center(x=-1.0, y=2.0, weight=1/3, phase=np.pi/3) # Déphasage de 60 degrés
    system.add_center(x=2.0, y=-1.0, weight=1/3, phase=np.pi/2) # Déphasage de 90 degrés

    # 3. Visualisation
    plt.figure(figsize=(10, 8))
    
    # Tracé du triangle d'origine
    plt.plot(triangle[:, 0], triangle[:, 1], 'k--', label="Triangle d'origine (ABC)")
    
    # Affichage des centres O_i
    for idx, center in enumerate(system.centers):
        plt.scatter(center[0], center[1], marker='X', s=100, label=f"Centre O{idx+1}")

    # Application de la TTH pour différentes valeurs de l'angle theta
    colors = ['r', 'g', 'b', 'm', 'c']
    thetas = [np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi] # Divers angles de rotation
    
    for theta, color in zip(thetas, colors):
        transformed = system.apply(triangle, theta)
        plt.plot(transformed[:, 0], transformed[:, 1], color=color, alpha=0.7, 
                 label=f"Image TTH (theta={theta:.2f})")

    plt.title("Transformation de Thutmos (TTH) appliquée à un Triangle")
    plt.xlabel("Axe X")
    plt.ylabel("Axe Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

