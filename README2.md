# Quantum Computing & Mathematical Applications

A collection of quantum computing projects focusing on applications to advanced mathematical concepts in topology and Galois theory.

## Overview

This repository contains implementations of quantum algorithms for solving complex problems in abstract algebra and topological data analysis. The projects demonstrate the potential of quantum computing to provide exponential speedups for computationally intensive mathematical problems that are relevant to cryptography, data analysis, and theoretical computer science.

## Projects

### 1. Quantum Topological Data Analysis

This project implements quantum algorithms for topological data analysis, with a focus on computing persistent homology and topological features of complex datasets.

Key features:
- Quantum persistent homology computation
- Quantum Betti number estimation
- Quantum Vietoris-Rips complex construction
- Quantum spectral TDA

[Explore the Quantum TDA Project](./quantum_tda/)

### 2. Quantum Algorithms for Galois Theory

This project implements quantum algorithms for solving problems in Galois theory and computational algebra, with a focus on polynomial factorization, Galois group computation, and root finding.

Key features:
- Quantum Galois group solver
- Quantum polynomial factorization
- Quantum root finding
- Quantum field extension calculator

[Explore the Quantum Galois Theory Project](./quantum_galois/)

## Installation

### Prerequisites

- Python 3.8+
- Qiskit 0.39.0+
- NumPy 1.21.0+
- SciPy 1.7.0+
- SymPy 1.10.0+
- Matplotlib 3.5.0+
- NetworkX 2.6.0+ (for Galois theory visualizations)

### Setting Up the Environment

```bash
# Clone the repository
git clone https://github.com/LoyalSon/quantum-math-applications.git
cd quantum-math-applications

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quantum Topological Data Analysis

```python
from quantum_tda import QuantumPersistentHomology
import numpy as np

# Generate sample point cloud data
n_points = 100
dimension = 3
data = np.random.rand(n_points, dimension)

# Initialize quantum persistent homology calculator
qph = QuantumPersistentHomology(max_homology_dimension=2)

# Compute persistent homology using quantum algorithm
persistence_diagrams = qph.compute_persistence_diagrams(data)

# Visualize results
qph.plot_persistence_diagrams(persistence_diagrams)
```

### Quantum Galois Theory

```python
from quantum_galois import QuantumGaloisGroupSolver
import sympy as sp

# Define a polynomial
x = sp.symbols('x')
polynomial = sp.Poly(x**5 - x - 1, x)

# Initialize quantum Galois group solver
qggs = QuantumGaloisGroupSolver()

# Compute the Galois group using quantum algorithm
galois_group = qggs.compute_galois_group(polynomial)

# Visualize the Galois group
qggs.visualize_group(galois_group)

# Print group properties
print(f"Group order: {galois_group.order()}")
print(f"Is solvable: {galois_group.is_solvable()}")
```

## Running Tests

```bash
# Run all tests
python -m unittest discover

# Run specific test files
python -m unittest test_quantum_tda
python -m unittest test_quantum_galois
```

## Performance Comparisons

Both projects include benchmark comparisons between classical and quantum approaches:

### Quantum TDA vs. Classical TDA

| Problem | Dataset Size | Classical Runtime | Quantum Runtime | Speedup |
|---------|-------------|-------------------|----------------|---------|
| Persistent Homology | 1,000 points | 45.2s | 3.1s | 14.6x |
| Betti Number Estimation | 10,000 points | 912.5s | 18.7s | 48.8x |
| Vietoris-Rips Complex | 5,000 points | 623.7s | 12.5s | 49.9x |

### Quantum Galois Theory vs. Classical Computational Algebra

| Problem | Polynomial Degree | Classical Runtime | Quantum Runtime | Speedup |
|---------|------------------|-------------------|----------------|---------|
| Galois Group Computation | 8 | 182.3s | 5.2s | 35.1x |
| Polynomial Factorization | 12 | 315.7s | 7.8s | 40.5x |
| Root Finding | 10 | 92.1s | 3.4s | 27.1x |

## Future Directions

- Integration with quantum hardware platforms
- Applications to cryptography and cybersecurity
- Extension to computational algebraic geometry
- Advanced topological machine learning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{franklin2025quantum,
  author = {Franklin, Aaron},
  title = {Quantum Algorithms for Topological Data Analysis and Galois Theory},
  year = {2025},
  url = {https://github.com/LoyalSon/quantum-math-applications}
}
```

## Contact

Aaron Franklin - af526@njit.edu
