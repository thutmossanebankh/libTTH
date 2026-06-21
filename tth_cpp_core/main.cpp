#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

// Structure pour stocker un centre d'influence TTH
struct TTHCenter {
    Eigen::Vector2d O;
    double w;
    double phi;
};

// Classe principale du système TTH 2D
class TTHSystem2D {
private:
    std::vector<TTHCenter> centers;

    // Générateur de matrice de rotation 2x2
    Eigen::Matrix2d rotation_matrix(double angle) const {
        double c = std::cos(angle);
        double s = std::sin(angle);
        Eigen::Matrix2d R;
        R << c, -s,
             s,  c;
        return R;
    }

public:
    // Ajouter un centre au système
    void add_center(double x, double y, double weight, double phase = 0.0) {
        centers.push_back({Eigen::Vector2d(x, y), weight, phase});
    }

    // Application de la TTH massivement vectorisée
    Eigen::MatrixXd apply(const Eigen::MatrixXd& points, double theta) const {
        // Initialiser la matrice de sortie (remplie de zéros)
        Eigen::MatrixXd transformed_points = Eigen::MatrixXd::Zero(points.rows(), points.cols());

        for (const auto& center : centers) {
            Eigen::Matrix2d R = rotation_matrix(theta + center.phi);
            
            // 1. P_shifted = P - O_i (Soustraction de O_i à chaque ligne)
            Eigen::MatrixXd P_shifted = points.rowwise() - center.O.transpose();
            
            // 2. Rotation : (P_shifted * R^T)
            // Note géométrique : (R * X^T)^T = X * R^T
            Eigen::MatrixXd rotated = P_shifted * R.transpose();
            
            // 3. Rajouter O_i : rotated + O_i
            rotated = rotated.rowwise() + center.O.transpose();
            
            // 4. Pondérer et cumuler : w_i * (...)
            transformed_points += center.w * rotated;
        }

        return transformed_points;
    }
};

// ==========================================
// TEST DU PROTOCOLE (Équivalent au script Python)
// ==========================================
int main() {
    TTHSystem2D system;
    
    // Ajout des 3 centres (les mêmes que dans la Figure 1)
    system.add_center(1.0, 1.0, 1.0/3.0, 0.0);
    system.add_center(-1.0, 2.0, 1.0/3.0, M_PI/3.0);
    system.add_center(2.0, -1.0, 1.0/3.0, M_PI/2.0);

    // Définition de notre triangle (4 points pour fermer la figure)
    // Format : N lignes, 2 colonnes
    Eigen::MatrixXd triangle(4, 2);
    triangle << 0.0, 0.0,
                2.0, 0.0,
                0.0, 1.5,
                0.0, 0.0;

    double theta = M_PI / 4.0; // 45 degrés
    
    // Mesure de performance basique
    std::cout << "--- TRANSFORMATION DE THUTMOS (C++ EIGEN) ---" << std::endl;
    Eigen::MatrixXd result = system.apply(triangle, theta);
    
    std::cout << "\nPoints transformés (theta = 0.785 rad) :\n" << result << std::endl;

    return 0;
}
