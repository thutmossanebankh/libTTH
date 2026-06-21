# 📚 libTTH: The Thutmos Transformation Library

**libTTH** is a high-performance geometric computation engine designed to apply the Thutmos Transformation (TTH) to point clouds and 2D/3D meshes.

Based on a hybrid architecture (C++ / CUDA / Python), this library solves the barycentric multi-centric equation in real time, delivering production-grade performance for procedural generation, robotic kinematics, and non-linear geometric modeling.

## 🧮 Mathematical Foundations

The Thutmos Transformation relies on the interaction of multiple weighted centers of rotation. In 3D space, to prevent gimbal lock, the library utilizes quaternion algebra. The fundamental equation computed by the core kernel is:

$$\mathcal{T}_\theta(P) = \sum_{i=1}^n w_i \Big(O_i + Q_i(\theta+\phi_i)(P - O_i)Q_i^{-1}(\theta+\phi_i)\Big)$$

Where:

* $P$: The original point or vertex.
* $O_i$: The geometric center of influence.
* $w_i$: The barycentric weight (normalized such that $\sum w_i = 1$).
* $Q_i$: The rotation quaternion defined by a spatial axis and a phase-shifted angle of $\phi_i$.

---

## ⚙️ Architecture and Prerequisites

The library is engineered for demanding R&D environments and leverages the following standards:

* **CPU Kernel:** C++17 with matrix vectorization via **Eigen3**.
* **GPU Kernel:** Hardware acceleration via **NVIDIA CUDA 12.8**.
* **Interface:** Native Python bindings via **Pybind11**.

### Benchmark Performances (GTX 1650 Ti / i7-10750H)

* **CPU (Eigen3):** ~870,000 3D points / second.
* **GPU (CUDA):** ~17,330,000 3D points / second (5 million points in 0.288s).


---

## 📖 Python API Reference (`tth_core`)


The compiled module exposes three main classes, tailored for different computational needs.

### 1. `TTHSystem2D` (Vectorized CPU Computation for 2D)

Ideal for planar geometric shapes and polygons.

* `__init__()`: Initializes an empty 2D system.
* `add_center(x: float, y: float, weight: float, phase: float = 0.0) -> None`
* Adds a geometric center. `phase` is in radians.


* `apply(points: numpy.ndarray, theta: float) -> numpy.ndarray`
* **points**: NumPy array of shape `(N, 2)`.
* **theta**: Global angle of the system in radians.
* **Returns**: A new NumPy array of shape `(N, 2)` containing the transformed points.



### 2. `TTHSystem3D` (Vectorized CPU Computation via Quaternions)

Ideal for small 3D meshes (up to 100,000 vertices) tested without hardware acceleration.

* `__init__()`: Initializes a 3D system.
* `add_center(ox: float, oy: float, oz: float, ax: float, ay: float, az: float, weight: float, phase: float = 0.0) -> None`
* `ox, oy, oz`: Coordinates of the center $O_i$.
* `ax, ay, az`: Directional vector of the rotation axis.


* `apply(points: numpy.ndarray, theta: float) -> numpy.ndarray`
* **points**: NumPy array of shape `(N, 3)`.
* **Returns**: NumPy array of shape `(N, 3)`.



### 3. `TTHSystemGPU` (Massively Parallel Computation via CUDA)

The industrial core of the library. Offloads geometric computations to CUDA cores for dense meshes (millions of vertices).

* `__init__()`: Initializes the CUDA context.
* `add_center(ox: float, oy: float, oz: float, ax: float, ay: float, az: float, weight: float, phase: float = 0.0) -> None`
* Identical to the 3D version.


* `apply(input_points: numpy.ndarray, theta: float) -> numpy.ndarray`
* Automatically handles RAM $\to$ VRAM transfer, Kernel execution, and VRAM $\to$ RAM return.
* **input_points**: NumPy array of type `float64` and shape `(N, 3)`.



---

## 🚀 Quick Start Example (GPU)

Here is a quick-start script to deform a spatial point cloud using hardware acceleration:

```python
import numpy as np
import tth_core

# 1. CUDA engine initialization
system = tth_core.TTHSystemGPU()

# 2. Geometric configuration (3 barycentric centers of influence)
system.add_center(ox=1.0, oy=1.0, oz=0.0, ax=0.0, ay=0.0, az=1.0, weight=0.33, phase=0.0)
system.add_center(ox=-1.0, oy=2.0, oz=1.0, ax=1.0, ay=0.0, az=0.0, weight=0.33, phase=1.047)
system.add_center(ox=0.0, oy=-1.0, oz=2.0, ax=0.0, ay=1.0, az=0.0, weight=0.34, phase=1.570)

# 3. Data generation
points = np.random.rand(1_000_000, 3).astype(np.float64)
theta = 0.785 # 45 degrees

# 4. Execution of the Thutmos Transformation
transformed_points = system.apply(points, theta)

print(transformed_points[:3])

```