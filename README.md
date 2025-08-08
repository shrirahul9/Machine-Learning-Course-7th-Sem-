# 🔢 Digit Recognition with Machine Learning

A comprehensive implementation of handwritten digit recognition using multiple machine learning approaches, from simple pixel counting to neural networks. This project demonstrates the progression from basic computer vision techniques to advanced ML algorithms.


## 📸 Project Overview

This project implements **4 different approaches** to recognize handwritten digits (0-9), showcasing the evolution from simple rule-based methods to sophisticated machine learning algorithms:

1. **Simple Pixel Intensity Classifier** - Rule-based approach using pixel counting
2. **Manual K-Nearest Neighbors** - Distance-based similarity matching
3. **Template Matching** - Statistical pattern matching with digit averages
4. **Simple Neural Network** - Basic deep learning implementation

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib scikit-learn
```

### Running the Code

```bash
git clone https://github.com/yourusername/digit-recognition-ml.git
cd digit-recognition-ml
python digit_recognition.py
```

## 📊 Results Comparison

| Method | Accuracy | Complexity | Code Lines | Best For |
|--------|----------|------------|------------|----------|
| Simple Pixel | ~15-25% | Very Low | ~20 lines | Learning basics |
| Manual KNN | ~95-97% | Medium | ~50 lines | Understanding ML |
| Template Match | ~85-90% | Medium | ~40 lines | Pattern recognition |
| Neural Network* | ~10% | High | ~60 lines | Deep learning intro |

*Neural network shows low accuracy as it's randomly initialized (not trained)

## 🛠️ Implementation Details

### 1. Simple Pixel Intensity Classifier

The most basic approach that classifies digits based on total pixel intensity:

```python
def simple_pixel_classifier(X):
    for image in X:
        total_intensity = np.sum(image)
        if total_intensity < 200: return 1    # Thin digit
        elif total_intensity < 600: return 8  # Thick digit
        else: return 0                        # Very thick outline
```

**How it works:** Different digits have different amounts of "ink" (black pixels). Digit 1 is thin, digit 8 is thick.

### 2. Manual K-Nearest Neighbors

Finds the 3 most similar training examples and votes on the prediction:

```python
def manual_knn_classifier(X_train, y_train, X_test, k=3):
    for test_sample in X_test:
        # Calculate distance to all training samples
        distances = [euclidean_distance(test_sample, train_sample) 
                    for train_sample in X_train]
        # Get k nearest neighbors and vote
        nearest_k = get_k_smallest(distances, k)
        prediction = most_common_digit(nearest_k)
```

**How it works:** "Show me your friends, and I'll tell you who you are" - finds similar digit images and predicts based on their labels.

### 3. Template Matching

Creates average templates for each digit and matches new images to the closest template:

```python
def template_matching_classifier(X_train, y_train, X_test):
    # Create average digit templates
    templates = {}
    for digit in range(10):
        templates[digit] = np.mean(X_train[y_train == digit], axis=0)
    
    # Match test images to the closest template
    for test_sample in X_test:
        distances = [euclidean_distance(test_sample, template) 
                    for template in templates.values()]
        prediction = digit_with_minimum_distance
```

**How it works:** Creates a "perfect average" of each digit and compares new images to these averages.

### 4. Simple Neural Network

A basic neural network with random weights (for demonstration):

```python
class SimpleNeuralNetwork:
    def __init__(self, input_size=64, hidden_size=50, output_size=10):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.softmax(self.z2)
```

**Note:** This neural network is not trained, so it performs poorly. It's included to show the structure of neural networks.

## 📈 Dataset Information

- **Source:** scikit-learn's built-in digits dataset
- **Images:** 1,797 handwritten digits (0-9)
- **Format:** 8×8 grayscale images (64 pixels per image)
- **Split:** 80% training (1,437 images), 20% testing (360 images)

```python
# Dataset loading
digits = load_digits()
X = digits.data      # Shape: (1797, 64) - flattened 8x8 images
y = digits.target    # Shape: (1797,) - digit labels 0-9
```

## 🔧 Key Features

- **Multiple Algorithms:** Compare 4 different approaches side-by-side
- **Manual Implementations:** See exactly how each algorithm works
- **Windows Compatible:** Fixes common OpenBLAS threading issues
- **Progress Tracking:** Monitor processing of large datasets
- **Visualization:** See sample predictions with accuracy colors
- **Educational:** Detailed comments explaining every step

## 🐛 Troubleshooting

### Common Issues

1. **OpenBLAS Threading Error on Windows:**
   ```python
   import os
   os.environ['OMP_NUM_THREADS'] = '1'
   ```

2. **Slow KNN Performance:**
   - Normal behavior - KNN compares every test sample to all training samples
   - Reduce test set size for faster execution

3. **Low Neural Network Accuracy:**
   - Expected! The network uses random weights (not trained)
   - Real training would require gradient descent implementation

## 📚 Learning Objectives

This project teaches:

- **Computer Vision Basics:** How images are represented as numbers
- **Machine Learning Fundamentals:** Classification, training/testing, accuracy
- **Algorithm Progression:** From simple rules to complex neural networks
- **Implementation Details:** What happens "under the hood" of ML libraries
- **Performance Trade-offs:** Accuracy vs. complexity vs. interpretability

## 📄 Code Structure

```
digit-recognition-ml/
├── digit_recognition.py       # Main implementation file
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── examples/
    ├── sample_predictions.png # Example outputs
    └── confusion_matrix.png   # Model performance analysis
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **scikit-learn** for the digits dataset and ML utilities
- **NumPy** for efficient numerical computations
- **Matplotlib** for visualization capabilities
- The machine learning community for educational resources

## 📞 Contact

E-mail - shrirahul06@gmail.com

Project Link: https://github.com/shrirahul9/Machine-Learning-Course-7th-Sem-
---

⭐ **Star this repo if you found it helpful!**

## 📋 Requirements

Create a `requirements.txt` file:

```txt
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
```

---

