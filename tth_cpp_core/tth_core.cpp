
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// Redéfinition de la structure pour le C++
struct TTHCenterCUDA {
    double ox, oy, oz;
    double ax, ay, az;
    double w;
    double phi;
};

// Signature de la fonction externe compilée par nvcc
extern void run_tth_cuda(const double* h_points_in, double* h_points_out, int N,
                         const TTHCenterCUDA* h_centers, int num_centers, double theta);

class TTHSystemGPU {
private:
    std::vector<TTHCenterCUDA> centers;

public:
    void add_center(double ox, double oy, double oz, 
                    double ax, double ay, double az, 
                    double weight, double phase = 0.0) {
        
        // Normalisation de l'axe de rotation
        double norm = std::sqrt(ax*ax + ay*ay + az*az);
        if (norm > 0) {
            ax /= norm; ay /= norm; az /= norm;
        } else {
            az = 1.0; 
        }
        
        centers.push_back({ox, oy, oz, ax, ay, az, weight, phase});
    }

    // Fonction qui prend un tableau NumPy et renvoie un tableau NumPy
    py::array_t<double> apply(py::array_t<double> input_points, double theta) {
        py::buffer_info buf = input_points.request();

        if (buf.ndim != 2 || buf.shape[1] != 3) {
            throw std::runtime_error("L'entrée doit être une matrice NumPy de dimension (N, 3)");
        }

        int N = buf.shape[0];

        // Création du tableau de sortie NumPy (rempli de zéros)
        auto result = py::array_t<double>({N, 3});
        py::buffer_info res_buf = result.request();

        double* ptr_in = static_cast<double*>(buf.ptr);
        double* ptr_out = static_cast<double*>(res_buf.ptr);

        // Appel à la fonction CUDA
        run_tth_cuda(ptr_in, ptr_out, N, centers.data(), centers.size(), theta);

        return result;
    }
};





// Module Pybind11
PYBIND11_MODULE(tth_core, m) {
    m.doc() = "Noyau TTH accéléré par GPU (CUDA)";

    py::class_<TTHSystemGPU>(m, "TTHSystemGPU")
        .def(py::init<>())
        // AJOUT DES py::arg() POUR ACCEPTER LES KWARGS PYTHON
        .def("add_center", &TTHSystemGPU::add_center,
             py::arg("ox"), py::arg("oy"), py::arg("oz"),
             py::arg("ax"), py::arg("ay"), py::arg("az"),
             py::arg("weight"), py::arg("phase") = 0.0,
             "Ajoute un centre 3D pour le GPU")
        .def("apply", &TTHSystemGPU::apply, 
             py::arg("input_points"), py::arg("theta"),
             "Applique la TTH 3D massivement via CUDA");
}


