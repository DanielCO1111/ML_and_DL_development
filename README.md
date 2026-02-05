# Machine Learning Journey: A Comprehensive Exploration

A deep dive into machine learning algorithms, from classical statistical methods to modern deep learning architectures. This project demonstrates hands-on implementation of various ML techniques, with emphasis on understanding the underlying mathematics and practical applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Part 1: Logistic Regression](#part-1-logistic-regression)
  - [Section 1: Standard Logistic Regression](#section-1-standard-logistic-regression)
  - [Section 2: Regularized Logistic Regression](#section-2-regularized-logistic-regression)
- [Part 2: K-Means Clustering & PCA](#part-2-k-means-clustering--pca)
  - [K-Means Implementation](#k-means-implementation)
  - [Principal Component Analysis](#principal-component-analysis)
- [Part 3: Neural Networks & CNNs](#part-3-neural-networks--cnns)
  - [Part 3.1: Neural Network from Scratch](#part-31-neural-network-from-scratch)
  - [Part 3.2: Neural Network with PyTorch](#part-32-neural-network-with-pytorch)
  - [Part 3.3: Convolutional Neural Networks](#part-33-convolutional-neural-networks)
- [Technologies & Dependencies](#technologies--dependencies)
- [Results & Key Insights](#results--key-insights)

---

## Project Overview

This project represents a structured progression through machine learning concepts, designed as a self-learning journey. Each component builds upon foundational principles, demonstrating both theoretical understanding and practical implementation skills.

**Key Learning Objectives:**
- Master classical ML algorithms (Logistic Regression, K-Means)
- Understand regularization techniques and their impact
- Build neural networks from scratch
- Leverage modern frameworks (PyTorch)
- Apply CNNs to image classification

---

## Part 1: Logistic Regression

### Section 1: Standard Logistic Regression

**Problem**: Predict student university admission based on two entrance exam scores.

**Dataset**: Student admission records with two exam scores and binary admission status

**Implementation**:

1. **Data Visualization**
   - Scatter plots showing admitted vs rejected students
   - Visual inspection revealed roughly linear separability

2. **Sigmoid Function**
   ```
   g(z) = 1 / (1 + e^(-z))
   ```
   - Transforms linear outputs to probabilities [0,1]
   - Vectorized implementation for efficiency

3. **Cost Function** (Cross-Entropy Loss)
   ```
   J(θ) = -(1/m) * Σ[y*log(h_θ(x)) + (1-y)*log(1-h_θ(x))]
   ```
   - Measures prediction error
   - Convex function ensures global minimum

4. **Gradient Descent**
   ```
   ∂J(θ)/∂θ_j = (1/m) * Σ[(h_θ(x) - y) * x_j]
   ```
   - Iterative optimization
   - Stochastic gradient descent for faster convergence

5. **Decision Boundary**
   - Plotted learned boundary: θ_0 + θ_1*x_1 + θ_2*x_2 = 0
   - Visual confirmation of class separation

6. **Evaluation**
   - Training accuracy computed
   - ROC curve and AUC calculated
   - Example prediction for student with scores [53, 94]

**Results**: Successfully learned linear decision boundary with strong training accuracy and AUC score

**Key Insights**:
- Sigmoid provides smooth probability estimates
- Gradient descent converges with proper learning rate
- Linear boundary sufficient for linearly separable data

---

### Section 2: Regularized Logistic Regression

**Problem**: Predict microchip QA pass/fail with non-linear decision boundary

**Challenge**: Non-linear separability requires polynomial features, risking overfitting

**Implementation**:

1. **Feature Mapping**
   ```
   map_feature(x) = [1, x_1, x_2, x_1^2, x_1*x_2, x_2^2, ..., x_2^6]
   ```
   - Expanded 2 features → 28 features (up to 6th degree)
   - Enables non-linear boundaries

2. **Regularized Cost Function**
   ```
   J(θ) = -(1/m)*Σ[y*log(h_θ(x)) + (1-y)*log(1-h_θ(x))] + (λ/(2m))*Σθ_j^2
   ```
   - L2 penalty prevents overfitting
   - λ controls regularization strength

3. **Regularized Gradient**
   ```
   For j=0: ∂J/∂θ_0 = (1/m)*Σ[(h_θ(x) - y)*x_0]
   For j≥1: ∂J/∂θ_j = [(1/m)*Σ[(h_θ(x) - y)*x_j]] + (λ/m)*θ_j
   ```

4. **Optimization**: Momentum-based gradient descent

5. **Experiments**:

   **λ = 1 (Regularized)**:
   - Smooth, curved decision boundary
   - Balanced fit and generalization
   - Training accuracy: ~83%
   
   **λ = 0 (No Regularization)**:
   - Highly complex, wiggly boundary
   - Perfect training fit (overfitting)
   - Training accuracy: ~90%
   - Poor generalization expected

**Visualizations**:
- Decision boundaries for both λ values
- Clear demonstration of overfitting with λ=0

**Results**:
- λ=1 provides better generalization
- Regularization crucial for high-dimensional features

**Key Insights**:
- Polynomial features enable non-linear boundaries
- Regularization prevents overfitting
- λ controls bias-variance tradeoff
- Visual inspection reveals overfitting

---

## Part 2: K-Means Clustering & PCA

### K-Means Implementation

**Objective**: Pure NumPy implementation to understand unsupervised learning

**Algorithm**:
1. Initialize K centroids randomly
2. Assign points to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

**Minimize**: Within-cluster sum of squares (WCSS)

### Dataset 1: Convex Data

**Characteristics**: Well-separated, spherical clusters

**Experiments**:
- Plotted cost vs iterations (steep decrease, rapid convergence)
- Tested multiple K values (2, 3, 4, 6, 8, 10, 20)
- Visualized cluster evolution across iterations

**Results**:
- Perfect clustering of spherical groups
- Convergence < 20 iterations
- Cost decreases with increasing K (elbow method selects optimal)

### Dataset 2: Non-Convex Data

**Characteristics**: Moon-shaped, curved boundaries

**Experiments**:
- Same K-Means algorithm
- Cost converged but clusters incorrect
- Multiple K values failed to capture true structure

**Results**:
- K-Means struggled with non-convex shapes
- Spherical assumption caused poor assignments
- Demonstrated algorithm limitations

**Key Insights**:
- K-Means ideal for convex, spherical clusters
- Fails with irregular shapes (use DBSCAN, Spectral Clustering instead)
- Algorithm choice must match data geometry

---

### Principal Component Analysis

**Dataset**: MNIST (70,000 images, 784 dimensions)

**Implementation**:
1. Data centering (zero mean)
2. Covariance matrix computation
3. Eigenvalue decomposition
4. Sort eigenvalues/eigenvectors
5. Project onto top r components

**Experiments**:

1. **2D Projection**
   - Visualized data in top 2 principal components
   - Some class separation visible

2. **Geometric Properties**
   - V^T * V ≈ Identity (orthonormal components)
   - V * V^T = projection matrix (not identity unless r=d)

3. **Image Reconstruction**
   - r=3: Severe loss, barely recognizable
   - r=10: Rough shapes visible
   - r=100: Good reconstruction

**Results**:
- 100 components (13% of original) preserve most information
- Demonstrates curse of dimensionality

**Key Insights**:
- PCA finds maximum variance directions
- Useful for visualization and dimensionality reduction
- Limitation: Linear transformation only

---

## Part 3: Neural Networks & CNNs

### Part 3.1: Neural Network from Scratch

**Dataset**: MNIST (60,000 train, 10,000 test)

**Architecture**:
- Input: 784 neurons
- Hidden: 128 → 64 neurons
- Output: 10 neurons (softmax)

**From-Scratch Components**:

1. **Activations**: Sigmoid, Softmax
2. **Loss**: Cross-entropy
3. **Forward Pass**: Layer-wise computation
4. **Backpropagation**: Chain rule gradients
5. **Weight Init**: Xavier initialization
6. **Training**: Mini-batch SGD

**Challenges Overcome**:
- Numerical stability in exponentials
- Correct matrix shapes
- Gradient checking

**Results**:
- Training: ~96-98%
- Validation: ~93-95%
- Test: ~92-94%

**Key Insights**:
- Backprop is systematic chain rule
- Proper init crucial
- Mini-batch balances speed/convergence

---

### Part 3.2: Neural Network with PyTorch

**Dataset**: Fashion-MNIST (60,000 images, 10 classes)

**Architecture**:
```
Linear(784→128) → ReLU → Linear(128→64) → ReLU → Linear(64→10) → LogSoftmax
```

**Training**:
- Optimizer: SGD (lr=0.005)
- Loss: NLLLoss
- Epochs: 5
- Batch: 64
- Split: 80/20 train/val

**Training Curves**:
- Loss: Train ~2.3→0.4, Val ~2.3→0.5
- Accuracy: Train ~60→85%, Val ~60→83%

**Results**: Val accuracy ~83-85%

**Key Insights**:
- PyTorch simplifies implementation
- ReLU better than sigmoid
- Proper validation crucial
- Fashion-MNIST harder than MNIST

---

### Part 3.3: Convolutional Neural Networks

**Theory**:

**Convolution**: Filter slides over image
**Output size**: W_out = floor((W_in - F + 2P)/S) + 1

**Pooling**: Downsampling (typically max pooling)

**Examples**:
- Q1: [1,28,28] + 10 5×5 filters → [10,24,24]
- Q2: Preserve dimensions → Padding = 2
- Q3: Two conv layers → [15,10,10]

**Simple CNN**:
```
Conv(10×5×5) → MaxPool → ReLU →
Conv(20×5×5) → MaxPool → ReLU →
FC(320) → ReLU → FC(50) → Output(10)
```

**Results**: Val ~85% (better than MLP ~75%)

**Advanced CNN**:

**Techniques Applied**:
- More conv layers (2→4)
- More filters (32→64→128)
- BatchNormalization
- Dropout (0.5)
- Adam optimizer
- Data augmentation
- Learning rate scheduling

**Final Architecture**:
```
Conv(32×3×3) → BatchNorm → ReLU → MaxPool →
Conv(64×3×3) → BatchNorm → ReLU → MaxPool →
Conv(128×3×3) → BatchNorm → ReLU → MaxPool →
FC(256) → ReLU → Dropout → Output(10)
```

**Results**: Test accuracy ~84-86% (>80% requirement)

**Key Insights**:
- CNNs superior for images
- Weight sharing reduces parameters
- Regularization crucial
- Architecture design is iterative

---

## Technologies & Dependencies

**Core**:
- NumPy, Pandas, Matplotlib
- PyTorch, torchvision
- scikit-learn

**Environment**:
- Python 3.7+
- Jupyter Notebook
- CUDA (optional)

---

## Results & Key Insights

| Model | Dataset | Accuracy |
|-------|---------|----------|
| Logistic Reg | Admission | ~89% |
| Logistic Reg (λ=1) | QA | ~83% |
| K-Means | Convex | Perfect |
| K-Means | Non-convex | Poor |
| MLP (NumPy) | MNIST | ~93-95% |
| MLP (PyTorch) | Fashion | ~83-85% |
| CNN | Fashion | ~84-86% |

**Overall Insights**:
1. Regularization prevents overfitting
2. K-Means limited to spherical clusters
3. From-scratch builds deep understanding
4. PyTorch accelerates development
5. CNNs excel at image tasks
6. Architecture tuning is critical

---

## Project Structure

```
.
├── README.md
├── Logistic_Regression.ipynb
├── KMeans_Clustering.ipynb
├── Neural_Network_and_KNN.ipynb

```

---

## Acknowledgments

- MNIST: Yann LeCun, Corinna Cortes
- Fashion-MNIST: Zalando Research
- PyTorch, NumPy, scikit-learn communities

---

**Happy Learning! 🚀📊**
