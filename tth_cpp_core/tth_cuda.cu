#include <cuda_runtime.h>
#include <math.h>

// Structure légère pour la carte graphique
struct TTHCenterCUDA {
    double ox, oy, oz;
    double ax, ay, az;
    double w;
    double phi;
};

// Fonction __device__ : Exécutée par le GPU pour CHAQUE point
__device__ void rotate_point_quaternion(double& px, double& py, double& pz,
                                        double ax, double ay, double az, double angle) {
    // Calcul du quaternion Q = [cos(a/2), sin(a/2)*V]
    double half_angle = angle * 0.5;
    double s = sin(half_angle);
    double qw = cos(half_angle);
    double qx = ax * s;
    double qy = ay * s;
    double qz = az * s;

    // Produit Hamiltonien optimisé pour la rotation 3D : P' = P + 2w(Q_v x P) + 2(Q_v x (Q_v x P))
    double tx = 2.0 * (qy * pz - qz * py);
    double ty = 2.0 * (qz * px - qx * pz);
    double tz = 2.0 * (qx * py - qy * px);

    px = px + qw * tx + (qy * tz - qz * ty);
    py = py + qw * ty + (qz * tx - qx * tz);
    pz = pz + qw * tz + (qx * ty - qy * tx);
}

// Fonction __global__ : Le Kernel principal qui distribue le travail aux milliers de threads
__global__ void tth_kernel(const double* d_points_in, double* d_points_out,
                           int N, const TTHCenterCUDA* d_centers, int num_centers, double theta) {
    // Identification unique du thread CUDA
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Si le thread est dans les limites de notre nuage de points
    if (idx < N) {
        // Lecture des coordonnées initiales depuis la VRAM
        double px = d_points_in[idx * 3 + 0];
        double py = d_points_in[idx * 3 + 1];
        double pz = d_points_in[idx * 3 + 2];

        double out_x = 0.0, out_y = 0.0, out_z = 0.0;

        // Boucle sur les centres (très rapide car num_centers est petit, ex: 3)
        for (int i = 0; i < num_centers; ++i) {
            TTHCenterCUDA c = d_centers[i];

            // 1. Décalage (P - O_i)
            double shifted_x = px - c.ox;
            double shifted_y = py - c.oy;
            double shifted_z = pz - c.oz;

            // 2. Rotation Quaternionique
            double current_angle = theta + c.phi;
            rotate_point_quaternion(shifted_x, shifted_y, shifted_z, c.ax, c.ay, c.az, current_angle);

            // 3. Retour au centre et pondération barycentrique
            out_x += c.w * (shifted_x + c.ox);
            out_y += c.w * (shifted_y + c.oy);
            out_z += c.w * (shifted_z + c.oz);
        }

        // Écriture du résultat final dans la VRAM
        d_points_out[idx * 3 + 0] = out_x;
        d_points_out[idx * 3 + 1] = out_y;
        d_points_out[idx * 3 + 2] = out_z;
    }
}

// Wrapper C++ classique pour appeler le Kernel depuis Pybind11
void run_tth_cuda(const double* h_points_in, double* h_points_out, int N,
                  const TTHCenterCUDA* h_centers, int num_centers, double theta) {
    double *d_points_in, *d_points_out;
    TTHCenterCUDA *d_centers;

    size_t points_size = N * 3 * sizeof(double);
    size_t centers_size = num_centers * sizeof(TTHCenterCUDA);

    // 1. ALLOCATION SUR LE GPU (VRAM)
    cudaMalloc(&d_points_in, points_size);
    cudaMalloc(&d_points_out, points_size);
    cudaMalloc(&d_centers, centers_size);

    // 2. TRANSFERT RAM -> VRAM
    cudaMemcpy(d_points_in, h_points_in, points_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, h_centers, centers_size, cudaMemcpyHostToDevice);

    // 3. CONFIGURATION DE LA GRILLE CUDA (Blocs de 256 threads)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 4. LANCEMENT DU KERNEL
    tth_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_points_in, d_points_out, N, d_centers, num_centers, theta);
    
    // Synchronisation pour s'assurer que le GPU a fini
    cudaDeviceSynchronize();

    // 5. TRANSFERT VRAM -> RAM (Récupération des résultats)
    cudaMemcpy(h_points_out, d_points_out, points_size, cudaMemcpyDeviceToHost);

    // 6. NETTOYAGE MÉMOIRE GPU
    cudaFree(d_points_in);
    cudaFree(d_points_out);
    cudaFree(d_centers);
}


