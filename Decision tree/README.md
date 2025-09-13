# 🌸 Iris Dataset Decision Tree Analysis

A comprehensive Python notebook for building and analyzing decision trees on the famous Iris dataset. This project demonstrates fundamental machine learning concepts, including classification, model evaluation, hyperparameter tuning, and data visualization.

## 📊 Dataset Overview

The Iris dataset contains 150 samples of iris flowers from three species:
- **Iris Setosa** (50 samples)
- **Iris Versicolor** (50 samples)
- **Iris Virginica** (50 samples)

Each sample has 4 features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

## 🎯 Project Objectives

- Build and evaluate decision tree classifiers
- Demonstrate key machine learning concepts (overfitting, cross-validation, feature importance)
- Create comprehensive visualizations for data understanding
- Compare model performance across different hyperparameters
- Provide interpretable results through decision tree visualization

## 🚀 Features

### 1. **Data Exploration & Visualization**
- Statistical summaries and data distribution analysis
- Pairwise scatter plots showing species separation
- Feature distribution histograms
- Correlation matrix heatmap

### 2. **Decision Tree Modeling**
- Model training with optimal hyperparameters
- Training and test accuracy evaluation
- Cross-validation for robust performance assessment

### 3. **Model Evaluation**
- Detailed classification report
- Confusion matrix visualization
- Performance metrics across different tree depths

### 4. **Decision Tree Visualization**
- Text-based tree rules
- Graphical tree structure
- Feature importance analysis

### 5. **Decision Boundaries**
- 2D visualization of decision boundaries
- Separate plots for petal and sepal features
- Interactive understanding of classification regions

### 6. **Hyperparameter Tuning**
- Comparison across different `max_depth` values
- Bias-variance tradeoff demonstration
- Optimal model selection

### 7. **Predictions on New Data**
- Example predictions with probability estimates
- Practical application demonstration

## 📋 Requirements

```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/iris-decision-tree-analysis.git
cd iris-decision-tree-analysis
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. **Run the notebook:**
```bash
jupyter notebook iris_decision_tree_analysis.ipynb
```

## 📁 Project Structure

```
iris-decision-tree-analysis/
│
├── iris_decision_tree_analysis.ipynb    # Main analysis notebook
├── iris_decision_tree_analysis.py       # Python script version
├── requirements.txt                     # Package dependencies
├── README.md                           # Project documentation
└── images/                             # Generated plots and visualizations
    ├── exploratory_data_analysis.png
    ├── decision_tree_structure.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── decision_boundaries.png
    └── model_comparison.png
```

## 🎨 Key Visualizations

### 1. Exploratory Data Analysis
- **Scatter plots**: Show clear species separation, especially in petal measurements
- **Histograms**: Reveal feature distributions for each species
- **Correlation heatmap**: Identifies relationships between features

### 2. Decision Tree Structure
- **Tree diagram**: Visual representation of decision rules
- **Feature importance**: Bar chart showing which features matter most
- **Decision boundaries**: 2D plots showing classification regions

### 3. Model Performance
- **Confusion matrix**: Detailed classification results
- **Performance comparison**: Training vs test accuracy across different tree depths
- **Cross-validation**: Robust performance assessment

## 📊 Results Summary

### Model Performance
- **Test Accuracy**: ~97.8% (typical result)
- **Training Accuracy**: ~98.1% (typical result)
- **Cross-validation Score**: ~96.0% ± 4.0%

### Key Insights
- **Petal measurements** are more discriminative than sepal measurements
- **Iris Setosa** is perfectly separable from other species
- **Optimal tree depth**: 3-4 levels for best generalization
- **Most important feature**: Petal length (typically ~0.9 importance)

### Decision Rules
The tree typically creates rules like:
1. If petal length ≤ 2.45 cm → **Setosa**
2. If petal length > 2.45 cm and petal width ≤ 1.65 cm → **Versicolor**
3. If petal length > 2.45 cm and petal width > 1.65 cm → **Virginica**

## 🔧 Customization

### Modify Hyperparameters
```python
dt_classifier = DecisionTreeClassifier(
    max_depth=3,           # Tree depth limit
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples in leaf
    criterion='gini'       # Split criterion
)
```

### Add New Visualizations
```python
# Example: Add a new scatter plot
plt.figure(figsize=(10, 6))
for species in iris.target_names:
    mask = df['species'] == species
    plt.scatter(df[mask]['feature_x'], df[mask]['feature_y'], label=species)
plt.legend()
plt.show()
```

## 🎓 Learning Outcomes

After running this analysis, you'll understand:

- **Decision Tree Mechanics**: How trees make decisions and split data
- **Overfitting vs Underfitting**: The importance of model complexity
- **Feature Engineering**: Which features matter most for classification
- **Model Evaluation**: Proper techniques for assessing performance
- **Data Visualization**: Effective ways to explore and present data
- **Hyperparameter Tuning**: Finding optimal model settings


### Development Setup
```bash
# Fork the repository
git fork https://github.com/yourusername/iris-decision-tree-analysis.git

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes and commit
git commit -m "Add your feature description"

# Push and create a pull request
git push origin feature/your-feature-name
```

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Iris Dataset Details](https://archive.ics.uci.edu/ml/datasets/iris)
- [Decision Trees Explained](https://www.analyticsvidhya.com/blog/2020/10/decision-trees-tutorial/)
- [Machine Learning Course](https://www.coursera.org/learn/machine-learning)


## 🙏 Acknowledgments

- **Ronald Fisher** for creating the original Iris dataset (1936)
- **UCI Machine Learning Repository** for hosting the dataset
- **Scikit-learn** team for the excellent machine learning library
- **Matplotlib/Seaborn** teams for visualization tools


---

⭐ **Found this helpful?** Give it a star and share with others learning machine learning!

🐛 **Found a bug?** Please open an issue with details and steps to reproduce.

💡 **Have suggestions?** Open an issue or submit a pull request!
