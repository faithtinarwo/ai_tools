AI Tools for Software
# AI Tools & Frameworks Mastery: Group Assignment

## Project Overview

This repository houses the collaborative work for our AI Tools & Frameworks assignment, designed to evaluate our understanding and practical application of key AI libraries and platforms. We will demonstrate proficiency in selecting, implementing, and critically analyzing AI tools to solve real-world problems across classical machine learning, deep learning, and natural language processing.

The assignment is structured into three main parts:
* **Part 1: Theoretical Understanding:** Demonstrating knowledge of core AI tools.
* **Part 2: Practical Implementation:** Applying tools to solve specific AI tasks.
* **Part 3: Ethics & Optimization:** Addressing ethical considerations and troubleshooting.

## Team Members

* **Faith Tinarwo** 
* **Mpho Hlalele** 
* **Steven Asumba]** 


---

## Table of Contents

1.  [Tools & Technologies](#tools-and-technologies)
2.  [Repository Structure](#repository-structure)
3.  [Getting Started (Local Setup)](#getting-started-local-setup)
4.  [Working with Google Colab (Recommended)](#working-with-google-colab-recommended)
    * [Opening Notebooks from GitHub](#opening-notebooks-from-github)
    * [Saving Your Changes Back to GitHub](#saving-your-changes-back-to-github)
    * [Handling Non-Notebook Files](#handling-non-notebook-files)
5.  [Project Tasks & Deliverables](#project-tasks-and-deliverables)
    * [Part 1: Theoretical Understanding](#part-1-theoretical-understanding)
    * [Part 2: Practical Implementation](#part-2-practical-implementation)
    * [Part 3: Ethics & Optimization](#part-3-ethics-and-optimization)
    * [Bonus Task: Model Deployment](#bonus-task-model-deployment)
6.  [Collaboration Guidelines](#collaboration-guidelines)
    * [Branching Strategy](#branching-strategy)
    * [Commit Messages](#commit-messages)
    * [Pull Requests (PRs)](#pull-requests-prs)
    * [Conflict Resolution](#conflict-resolution)
    * [Communication](#communication)
7.  [Submission Guidelines](#submission-guidelines)
8.  [Resources](#resources)
9.  [License](#license)

---

## 1. Tools & Technologies

This project leverages a range of industry-standard AI tools and platforms:

* **Machine Learning Frameworks:** TensorFlow, PyTorch, Scikit-learn, spaCy
* **Development Environments:** Google Colab (recommended), Jupyter Notebook
* **Programming Language:** Python 3.x
* **Version Control:** Git, GitHub
* **Deployment (Bonus):** Streamlit / Flask

---

## 2. Repository Structure

This repository is organized to keep our project clean and manageable: 

Conversation with Gemini
Part 1: Theoretical Understanding (40%)

1. Short Answer Questions

Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

Q2: Describe two use cases for Jupyter Notebooks in AI development.

Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

2. Comparative Analysis

Compare Scikit-learn and TensorFlow in terms of:

Target applications (e.g., classical ML vs. deep learning).

Ease of use for beginners.

Community support.

Part 2: Practical Implementation (50%)

Task 1: Classical ML with Scikit-learn

Dataset: Iris Species Dataset

Goal:

Preprocess the data (handle missing values, encode labels).

Train a decision tree classifier to predict iris species.

Evaluate using accuracy, precision, and recall.

Deliverable: Python script/Jupyter notebook with comments explaining each step.

Task 2: Deep Learning with TensorFlow/PyTorch

Dataset: MNIST Handwritten Digits

Goal:

Build a CNN model to classify handwritten digits.

Achieve >95% test accuracy.

Visualize the model’s predictions on 5 sample images.

Deliverable: Code with model architecture, training loop, and evaluation.

Task 3: NLP with spaCy

Text Data: User reviews from Amazon Product Reviews.

Goal:

Perform named entity recognition (NER) to extract product names and brands.

Analyze sentiment (positive/negative) using a rule-based approach.

Deliverable: Code snippet and output showing extracted entities and sentiment.

Part 3: Ethics & Optimization (10%)

1. Ethical Considerations

Identify potential biases in your MNIST or Amazon Reviews model. How could tools like TensorFlow Fairness Indicators or spaCy’s rule-based systems mitigate these biases?

2. Troubleshooting Challenge

Buggy Code: A provided TensorFlow script has errors (e.g., dimension mismatches, incorrect loss functions). Debug and fix the code.

Bonus Task (Extra 10%)

Deploy Your Model: Use Streamlit or Flask to create a web interface for your MNIST classifier. Submit a screenshot and a live demo link.

 This assignment evaluates your understanding of AI tools/frameworks and their real-world applications through a mix of theoretical and practical tasks. You’ll demonstrate proficiency in selecting, implementing, and critically analyzing AI tools to solve problems.



The assignment has three parts: theory, which tests the theoretical understanding of AI Tools; Practical, which tests the implementation skills of AI tools; and Ethics and optimization, which tests ethical AI development and AI software optimization skills. 



The assignment should be handled as group work formed by 3-5 people. This is to enhance teamwork and AI engineering collaboration.



Submission Guidelines



Code: Well-commented scripts/Jupyter notebooks. (To be submitted on GitHub)



Report: A PDF with:



Answers to theoretical questions. 



Screenshots of model outputs (e.g., accuracy graphs, NER results).



Ethical reflection.



(Share this PDF as an article in the Community, Peer Group Review and include it in the GitHub Repo)



Presentation: Create 3-minute video explaining your approach. (All members in the group should participate in the video. Share the video on the Community platform)



Grading Rubric

Criteria Weight

Theoretical Accuracy 30%

Code Functionality & Quality 40%

Ethical Analysis 15%

Creativity & Presentation 15%

Tools & Resources



Frameworks: TensorFlow, PyTorch, Scikit-learn, spaCy.



Platforms: Google Colab (free GPU), Jupyter Notebook.



Datasets: Kaggle, TensorFlow Datasets.



Why This Matters



Real-World Impact: These tools power industries from healthcare to finance.



Skill Validation: Employers seek proficiency in TensorFlow, PyTorch, and Scikit-learn.



Deadline: 7 days. Showcase your AI toolkit mastery! 🚀



Need Help?



Use official documentation: TensorFlow, PyTorch, spaCy.



Post questions on the LMS Community with #AIToolsAssignment.



Pro Tip: Test code incrementally—small wins lead to big successes! 💡



Please note: As an AI, I cannot directly execute code, create files, or interact with external platforms like GitHub, Google Colab, or the LMS Community. Therefore, for the "Practical Implementation" and "Bonus Task" sections, I will provide the Python code snippets and explanations that you would use to achieve the stated goals. You would then execute this code in your environment (e.g., Jupyter Notebook, Google Colab) and generate the required deliverables (screenshots, reports, etc.).

AI Tools and Frameworks Assignment
Part 1: Theoretical Understanding (40%)
1. Short Answer Questions
Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

Primary Differences:

Dynamic vs. Static Computation Graphs:
PyTorch: Uses dynamic computation graphs (define-by-run). This means the graph is built on the fly as operations are executed, making debugging easier and allowing for more flexible model architectures, especially with variable-length inputs or recurrent neural networks.
TensorFlow (prior to TF 2.0): Primarily used static computation graphs (define-and-run). The entire graph had to be defined before execution, which could be more challenging for debugging and less flexible for certain dynamic models. However, TensorFlow 2.0 largely adopted eager execution, making its behavior much closer to PyTorch's dynamic graphs.
Ease of Debugging:
PyTorch: Due to its dynamic graphs, debugging PyTorch models is generally more straightforward, as standard Python debugging tools can be used directly.
TensorFlow (pre-2.0): Debugging static graphs was more complex, often requiring specialized TensorFlow debugging tools. TensorFlow 2.0's eager execution significantly improved debugging capabilities.
Production Deployment:
TensorFlow: Historically, TensorFlow has had a stronger ecosystem for production deployment (e.g., TensorFlow Serving, TensorFlow Lite), making it a preferred choice for deploying models at scale.
PyTorch: While PyTorch has made significant strides in this area (e.g., TorchScript, ONNX export), TensorFlow still often has an edge in very large-scale, enterprise-level deployments.
Community and Industry Adoption:
TensorFlow: Backed by Google, TensorFlow has a vast community and widespread adoption in industry, particularly in large enterprises.
PyTorch: Backed by Facebook (Meta), PyTorch has gained immense popularity in the research community and is rapidly growing in industry, especially for its ease of use and flexibility.
API Design:
PyTorch: Often perceived as more "Pythonic" and intuitive for Python developers, with a more direct and less abstract API.
TensorFlow: While TensorFlow 2.0 has greatly simplified its API with Keras integration, earlier versions could be more verbose and abstract.
When to Choose One Over the Other:

Choose PyTorch if:
You are primarily focused on research and rapid prototyping, especially with complex or dynamic model architectures.
You prefer a more "Pythonic" and imperative programming style.
You value easier debugging and direct use of standard Python debugging tools.
You are working with recurrent neural networks or models with variable input sizes.
Choose TensorFlow if:
You are building large-scale production applications and require robust deployment tools (e.g., mobile, edge devices, distributed training).
You are already familiar with the Keras API, as it's deeply integrated into TensorFlow 2.0.
You need extensive pre-trained models or highly optimized solutions for specific Google Cloud services.
You are part of a team or organization that has standardized on TensorFlow.
Q2: Describe two use cases for Jupyter Notebooks in AI development.

Exploratory Data Analysis (EDA) and Data Preprocessing: Jupyter Notebooks are excellent for the initial stages of AI projects. Data scientists can use them to load datasets, inspect their structure, visualize distributions, identify missing values, and perform various data cleaning and preprocessing steps. The cell-based execution allows for an iterative and interactive approach, making it easy to experiment with different transformations and see immediate results. For example, a data scientist could load a CSV, plot histograms of features, and then write code in separate cells to handle outliers, all while documenting their thought process with Markdown.

 Model Prototyping, Training, and Evaluation: Jupyter Notebooks provide an ideal environment for rapidly prototyping and experimenting with different AI models. Developers can define model architectures, train them on small subsets of data, visualize training progress (e.g., loss curves, accuracy plots), and evaluate model performance. The ability to run code in small blocks and view outputs immediately facilitates iterative model refinement. For instance, a developer could define a neural network architecture in one cell, train it in another, and then use a third cell to calculate metrics like accuracy, precision, and recall, all within the same document, making it easy to share and reproduce results.

Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

spaCy significantly enhances NLP tasks compared to basic Python string operations by providing a robust, efficient, and pre-trained framework for various linguistic annotations and analyses. Here's how:

 Tokenization: While basic string operations might split text by spaces, spaCy's tokenizer intelligently handles punctuation, contractions (e.g., "don't" as "do" and "n't"), and special cases, providing a more linguistically accurate breakdown of text into tokens.

Linguistic Annotation: spaCy goes far beyond simple string manipulation by providing out-of-the-box support for:

Part-of-Speech (POS) Tagging: Assigns grammatical categories (e.g., noun, verb, adjective) to each token, which is impossible with basic string operations.
Dependency Parsing: Identifies grammatical relationships between words in a sentence, allowing for a deeper understanding of sentence structure.
Named Entity Recognition (NER): Automatically identifies and classifies named entities (e.g., persons, organizations, locations, products) in text, which would require complex regex patterns or manual lookups with basic string operations and would be far less accurate.
Lemmatization: Reduces words to their base form (e.g., "running" to "run," "better" to "good"), crucial for consistent analysis, whereas string operations would treat these as distinct.
 Efficiency and Performance: spaCy is designed for speed and efficiency, especially for large volumes of text. Its underlying implementation is optimized in Cython, providing significantly faster processing than what could be achieved with custom Python string functions for similar tasks.

 Pre-trained Models: spaCy comes with pre-trained statistical models for various languages, allowing users to perform complex NLP tasks (like NER or dependency parsing) without needing to train models from scratch. Basic string operations offer no such built-in intelligence.

 Rule-based Matching: While string operations might rely on rigid regex, spaCy provides a more powerful and flexible rule-based Matcher that can combine token attributes (POS tags, lemmas, text) and regular expressions for more sophisticated pattern matching, enhancing tasks like sentiment analysis or information extraction.

 Vector Representations/Word Embeddings: spaCy models often include word vectors, enabling semantic similarity calculations and other advanced NLP applications that are completely beyond the scope of basic string manipulations.

In essence, spaCy provides a high-level, linguistically aware, and performant toolkit, abstracting away the complexities of low-level text processing and allowing developers to focus on higher-level NLP applications.

2. Comparative Analysis
Compare Scikit-learn and TensorFlow in terms of:

Feature	Scikit-learn	TensorFlow
Target Applications	Classical Machine Learning algorithms:	Deep Learning:
- Classification (e.g., Logistic Regression,	- Building and training deep neural networks (CNNs, RNNs, Transformers).
SVMs, Decision Trees, Random Forests,	- Computer Vision (image classification, object detection).
Gradient Boosting)	- Natural Language Processing (machine translation, text generation).
- Regression (e.g., Linear Regression, Ridge)	- Reinforcement Learning.
- Clustering (e.g., K-Means, DBSCAN)	- Large-scale data processing and distributed training.
- Dimensionality Reduction (e.g., PCA, t-SNE)	
- Model Selection and Preprocessing.	
Ease of Use for Beginners	Generally easier for beginners.	Can be more challenging for absolute beginners (though TF 2.0 with Keras greatly simplified it).
- Consistent API across various algorithms.	- Requires understanding of neural network architectures, layers, and optimizers.
- Clear documentation and examples.	- Steeper learning curve for core concepts like tensors, gradients, and computation graphs (even with eager execution).
- Focuses on high-level API for common ML tasks.	- Keras API within TensorFlow makes it much more user-friendly for building standard neural networks.
Community Support	Excellent and mature community.	Massive and active community.
- Extensive documentation.	- Backed by Google, widely adopted in industry and research.
- Large user base, many online tutorials, forums (e.g., Stack Overflow).	- Extensive official documentation, tutorials, and examples.
- Widely taught in academic courses.	- Large number of pre-trained models available.
- Active development with regular updates.	- Strong presence in conferences and research papers.

Export to Sheets
In summary, Scikit-learn is your go-to for traditional ML tasks with its user-friendly API, while TensorFlow is the powerhouse for deep learning, offering unparalleled flexibility and scalability for complex neural network architectures.

Part 2: Practical Implementation (50%)
(As an AI, I cannot directly execute code or generate files. I will provide the Python code snippets and explanations. You would run these in your preferred environment like Jupyter Notebook or Google Colab.)

Task 1: Classical ML with Scikit-learn
Dataset: Iris Species Dataset
Goal: Preprocess data, train a decision tree classifier, evaluate using accuracy, precision, and recall.
Deliverable: Python script/Jupyter notebook with comments.

Python

# Task 1: Classical ML with Scikit-learn

# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the Iris dataset
# The Iris dataset is a classic and clean dataset, typically without missing values,
# but we'll include a check for demonstration purposes.
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target) # Target is already numerical (0, 1, 2)

print("--- Task 1: Classical ML with Scikit-learn ---")
print("\nOriginal Iris Dataset Info:")
print(X.info())
print("\nFirst 5 rows of features:")
print(X.head())
print("\nFirst 5 rows of target labels (numerical):")
print(y.head())

# --- Data Preprocessing ---

# 1. Handle Missing Values (Check for missing values, although Iris is typically clean)
print("\nChecking for missing values...")
if X.isnull().sum().sum() == 0:
    print("No missing values found in features.")
else:
    print("Missing values found. Handling strategy (e.g., imputation) would go here.")
    # For demonstration, if there were missing values, a common approach for numerical data:
    # from sklearn.impute import SimpleImputer
    # imputer = SimpleImputer(strategy='mean')
    # X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 2. Encode Labels (Iris target is already numerical, but if it were strings, we'd encode)
# For demonstration, let's pretend y was ['setosa', 'versicolor', 'virginica']
# and show how to encode it if needed.
# Since iris.target is already numerical (0, 1, 2), we don't strictly need LabelEncoder here
# for the model to work, but it's good practice to demonstrate for categorical targets.
print("\nEncoding target labels (if necessary)...")
# Let's create a dummy string target for demonstration
y_species_names = iris.target_names[y]
# print(f"Example of target names: {y_species_names[:5]}")

# If y_species_names was our original target:
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_species_names)
print(f"Original numerical target first 5: {y.head().tolist()}")
print(f"Encoded target from string names first 5: {y_encoded[:5].tolist()}")
print(f"Classes found by LabelEncoder: {label_encoder.classes_}")

# We will use the original numerical 'y' for consistency with load_iris output
y_final = y

# 3. Split the data into training and testing sets
# Stratify ensures that the proportion of target classes is the same in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.3, random_state=42, stratify=y_final)

print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Distribution of classes in training set:\n{pd.Series(y_train).value_counts(normalize=True)}")
print(f"Distribution of classes in test set:\n{pd.Series(y_test).value_counts(normalize=True)}")


# --- Train a Decision Tree Classifier ---

print("\nTraining Decision Tree Classifier...")
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
print("Decision Tree Model trained successfully.")

# --- Evaluate the Model ---

# Make predictions on the test set
y_pred = decision_tree_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
# precision_score and recall_score require 'average' parameter for multiclass classification
# 'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
# 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# 'micro': Calculate metrics globally by counting the total true positives, false negatives, and false positives.
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("\n--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")

# Optional: Display classification report for more detailed metrics per class
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# You can also visualize the decision tree (requires graphviz)
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plt.figure(figsize=(15, 10))
# plot_tree(decision_tree_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
# plt.title("Decision Tree Visualization")
# plt.show()
Explanation of Steps:

Import Libraries: Import pandas for data manipulation, load_iris for the dataset, train_test_split for splitting data, DecisionTreeClassifier for the model, and evaluation metrics from sklearn.metrics. LabelEncoder is included for general label encoding demonstration.
Load Data: The load_iris() function from sklearn.datasets is used to load the Iris dataset. Features are put into a DataFrame X, and target labels into a Series y.
Handle Missing Values: A check for missing values is included. The Iris dataset is clean, so this section primarily serves as a placeholder to demonstrate where one would implement imputation or removal strategies if missing values were present.
Encode Labels: The iris.target is already numerical (0, 1, 2). However, LabelEncoder is demonstrated to show how one would convert categorical string labels (e.g., 'setosa', 'versicolor') into numerical format if they were not already. We use the original numerical y for the model training.
Split Data: The dataset is split into training (70%) and testing (30%) sets using train_test_split. random_state ensures reproducibility, and stratify=y_final ensures that the proportion of each iris species is maintained in both training and test sets, which is crucial for balanced datasets.
Train Model: A DecisionTreeClassifier is initialized and trained using the fit() method on the training data (X_train, y_train).
Evaluate Model:
Predictions (y_pred) are made on the X_test.
accuracy_score, precision_score (with average='weighted' for multiclass), and recall_score (with average='weighted') are calculated to evaluate the model's performance. The weighted average accounts for class imbalance.
A classification_report is also printed to provide detailed per-class metrics.
Task 2: Deep Learning with TensorFlow/PyTorch
Dataset: MNIST Handwritten Digits
Goal: Build a CNN model to classify handwritten digits, achieve >95% test accuracy, visualize predictions.
Deliverable: Code with model architecture, training loop, and evaluation.

(I will provide a TensorFlow 2.x example, as it's widely adopted and integrates Keras well for ease of use.)

Python

# Task 2: Deep Learning with TensorFlow (Keras API)

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("--- Task 2: Deep Learning with TensorFlow ---")

# --- Load and Preprocess the MNIST Dataset ---

print("\nLoading and preprocessing MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Reshape images to add a channel dimension (for CNNs: height, width, channels)
# MNIST images are grayscale, so channel is 1.
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test labels shape: {test_labels.shape}")

# --- Build a CNN Model ---

print("\nBuilding CNN model...")
model = models.Sequential([
    # Convolutional Layer 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    # Convolutional Layer 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Convolutional Layer 3
    layers.Conv2D(64, (3, 3), activation='relu'), # Another conv layer before flattening

    # Flatten the 3D output to 1D for the Dense layers
    layers.Flatten(),
    # Dense (Fully Connected) Layers
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # Output layer with 10 classes (digits 0-9)
])

# Display model summary
model.summary()

# --- Compile the Model ---

print("\nCompiling model...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Use this for integer labels (0-9)
              metrics=['accuracy'])
print("Model compiled.")

# --- Train the Model ---

print("\nTraining model...")
# Using a callback to stop training if accuracy doesn't improve after a few epochs
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(train_images, train_labels, epochs=10, # Number of training iterations
                    validation_split=0.1, # Use 10% of training data for validation during training
                    batch_size=64, # Number of samples per gradient update
                    verbose=1) # Display progress bar during training
                    # callbacks=[early_stopping]) # Uncomment to use early stopping
print("Model training complete.")

# --- Evaluate the Model ---

print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Check if target accuracy is met
if test_accuracy > 0.95:
    print("Achieved >95% test accuracy! 🎉")
else:
    print("Test accuracy is below 95%. Consider more epochs, different architecture, or hyperparameter tuning.")

# --- Visualize Model's Predictions on 5 Sample Images ---

print("\nVisualizing model predictions on 5 sample images...")
# Get 5 random indices from the test set
sample_indices = np.random.choice(len(test_images), 5, replace=False)
sample_images = test_images[sample_indices]
sample_labels = test_labels[sample_indices]

# Make predictions
predictions = model.predict(sample_images)

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {sample_labels[i]}\nPred: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.suptitle("Model Predictions on Sample Images (True vs. Predicted)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
plt.show()

# Optional: Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
Explanation of Steps:

Import Libraries: Import tensorflow and its keras API components for building neural networks, matplotlib.pyplot for plotting, and numpy for numerical operations.
Load and Preprocess Data:
tf.keras.datasets.mnist.load_data() efficiently loads the MNIST dataset, splitting it into training and testing sets.
Images are reshaped to add a channel dimension (28x28x1 for grayscale images), which is required by Conv2D layers.
Pixel values are normalized to the range [0, 1] by dividing by 255.0, which helps in faster and more stable training of neural networks.
Build CNN Model:
A Sequential model is used, which is a linear stack of layers.
Conv2D Layers: These are the core building blocks of CNNs, learning spatial hierarchies of features. relu is used as the activation function. input_shape is specified for the first layer.
MaxPooling2D Layers: These layers reduce the spatial dimensions of the output volume, reducing the number of parameters and computational cost, and helping to make the model more robust to small shifts in the input.
Flatten Layer: Converts the 3D output of the convolutional layers into a 1D vector to be fed into the dense layers.
Dense Layers: Standard fully connected neural network layers. The last Dense layer has 10 units (for 10 digits) and a softmax activation function, which outputs probabilities for each class.
Compile Model:
optimizer='adam': A popular and effective optimization algorithm.
loss='sparse_categorical_crossentropy': Appropriate for classification tasks where labels are integers (0-9).
metrics=['accuracy']: To monitor accuracy during training and evaluation.
Train Model:
model.fit() trains the model using the training data.
epochs: Number of times the model will iterate over the entire training dataset.
validation_split: A portion of the training data is used for validation during training to monitor for overfitting.
batch_size: The number of samples processed before the model's parameters are updated.
callbacks: EarlyStopping is commented out but can be used to stop training automatically if validation accuracy stops improving, preventing overfitting and saving computational resources.
Evaluate Model:
model.evaluate() calculates the loss and accuracy on the unseen test dataset.
The results are printed, and a check is performed to see if the >95% accuracy goal is met.
Visualize Predictions:
Random sample images from the test set are selected.
model.predict() generates probability distributions for these images.
np.argmax() finds the predicted digit (the one with the highest probability).
matplotlib is used to display the sample images along with their true labels and the model's predictions.
Plots for training/validation accuracy and loss history are also included to show the learning curve.
Task 3: NLP with spaCy
Text Data: User reviews from Amazon Product Reviews.
Goal: Perform named entity recognition (NER) to extract product names and brands. Analyze sentiment (positive/negative) using a rule-based approach.
Deliverable: Code snippet and output showing extracted entities and sentiment.

(Since I don't have direct access to a specific Amazon Product Reviews dataset, I will use a few example review texts to demonstrate the functionality.)

Python

# Task 3: NLP with spaCy

# Import necessary libraries
import spacy

print("--- Task 3: NLP with spaCy ---")

# --- Load a pre-trained spaCy model ---
# You need to download a spaCy model first if you haven't:
# For example, run in your terminal: python -m spacy download en_core_web_sm
# Or for a larger model: python -m spacy download en_core_web_md
try:
    nlp = spacy.load("en_core_web_sm")
    print("\nspaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("\nError: spaCy model 'en_core_web_sm' not found.")
    print("Please run 'python -m spacy download en_core_web_sm' in your terminal and try again.")
    exit() # Exit if the model isn't available

# --- Example Amazon Product Review Text Data ---
# In a real scenario, you'd load this from a CSV, JSON, or database.
amazon_reviews = [
    "I absolutely love my new Echo Dot! The sound quality is surprisingly good for its size, and Alexa understands my commands perfectly. Highly recommend this smart speaker.",
    "The Samsung Galaxy S24 Ultra has an amazing camera, but the battery life could be better. Still, a fantastic phone overall, especially for gaming and productivity.",
    "This 'PowerBoost Pro' blender is powerful, but it's incredibly noisy. It blends well, but I wish it was quieter. Good for smoothies, but not for a peaceful morning.",
    "My Apple AirPods Pro 2 are comfortable and the noise cancellation is superb. Best earbuds I've ever owned. Great product!",
    "I bought a cheap generic USB cable, and it broke after a week. Always go for original brands like Anker for cables.",
    "This 'Gourmet Chef' Air Fryer makes cooking so much easier! It's quick, efficient, and everything tastes great. A must-have kitchen appliance."
]

# --- Perform Named Entity Recognition (NER) ---

print("\n--- Named Entity Recognition (NER) Results ---")
for i, review in enumerate(amazon_reviews):
    doc = nlp(review)
    print(f"\nReview {i+1}: \"{review}\"")
    found_entities = []
    for ent in doc.ents:
        # Common entity types for products/brands might be ORG, PRODUCT, GPE, NORP (for nationalities sometimes used as brand-like)
        # We'll primarily look for ORG (organizations, often brands) and PRODUCT if available in the model.
        # en_core_web_sm might not have a dedicated 'PRODUCT' label, but 'ORG' or 'LOC' might capture some.
        # We can also add custom rules or patterns for specific product names.
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "PERSON", "NORP", "FAC", "GPE"]: # Expanding for more possibilities
             found_entities.append(f"'{ent.text}' (Label: {ent.label_})")
    if found_entities:
        print("  Extracted Entities:")
        for entity_str in found_entities:
            print(f"    - {entity_str}")
    else:
        print("  No significant entities extracted by default model.")

# --- Analyze Sentiment using a Rule-Based Approach ---
# This is a very basic rule-based approach for demonstration.
# For more robust sentiment, consider TextBlob, VADER, or fine-tuning a classifier.

print("\n--- Rule-Based Sentiment Analysis Results ---")

positive_keywords = ["love", "amazing", "fantastic", "superb", "great", "excellent", "highly recommend", "must-have", "good", "efficient", "easier", "perfectly"]
negative_keywords = ["noisy", "broken", "wish it was quieter", "could be better", "cheap", "errors", "bad", "disappointing"]

def analyze_sentiment_rule_based(text):
    text_lower = text.lower()
    positive_score = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_score = sum(1 for keyword in negative_keywords if keyword in text_lower)

    if positive_score > negative_score:
        return "Positive"
    elif negative_score > positive_score:
        return "Negative"
    else:
        return "Neutral"

for i, review in enumerate(amazon_reviews):
    sentiment = analyze_sentiment_rule_based(review)
    print(f"Review {i+1}: \"{review[:70]}...\"") # Print first 70 chars
    print(f"  Sentiment: {sentiment}")

# Example of how spaCy can be used for custom rule-based entity extraction
# For specific product names not covered by standard NER labels,
# you can use spaCy's PhraseMatcher or EntityRuler.
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab)
# Define patterns for specific products/brands that might not be caught by default NER
product_patterns = [
    nlp("Echo Dot"), nlp("Samsung Galaxy S24 Ultra"), nlp("PowerBoost Pro"),
    nlp("Apple AirPods Pro 2"), nlp("Anker"), nlp("Gourmet Chef Air Fryer")
]
matcher.add("PRODUCT_BRAND", product_patterns)

print("\n--- Custom Product/Brand Extraction with PhraseMatcher ---")
for i, review in enumerate(amazon_reviews):
    doc = nlp(review)
    matches = matcher(doc)
    extracted_custom_entities = []
    for match_id, start, end in matches:
        span = doc[start:end]
        extracted_custom_entities.append(span.text)

    print(f"Review {i+1}: \"{review[:70]}...\"")
    if extracted_custom_entities:
        print(f"  Custom Extracted Product/Brand: {', '.join(extracted_custom_entities)}")
    else:
        print("  No specific products/brands found by custom matcher.")
Explanation of Steps:

Load spaCy Model: spacy.load("en_core_web_sm") loads a small English language model. This model includes components for tokenization, POS tagging, dependency parsing, and Named Entity Recognition (NER). You must download this model first using the command python -m spacy download en_core_web_sm in your terminal.
Example Text Data: A list of amazon_reviews is defined to simulate real-world input. In a practical application, this data would be loaded from a file or database.
Named Entity Recognition (NER):
Each review text is processed by the nlp object, which creates a Doc object.
doc.ents provides access to the recognized entities in the text.
The code iterates through doc.ents and prints entities with labels that might correspond to products or brands (ORG, PRODUCT, WORK_OF_ART, etc.). Note that general spaCy models might not have a specific 'PRODUCT' label, and often brands fall under 'ORG' (Organization) or are missed entirely without custom training.
Rule-Based Sentiment Analysis:
A simple analyze_sentiment_rule_based function is created.
It defines lists of positive_keywords and negative_keywords.
It converts the review to lowercase and counts the occurrences of these keywords.
Based on which type of keywords appears more, it classifies the sentiment as "Positive," "Negative," or "Neutral."
Limitation: This approach is very basic and easily fooled by nuance, sarcasm, or context. For real-world applications, more sophisticated methods (e.g., lexicon-based like VADER, machine learning models, or fine-tuned transformers) are preferred.
Custom Entity Extraction (Bonus/Refinement):
The PhraseMatcher from spaCy is demonstrated. This is useful when you have a predefined list of product names or brands that the general NER model might not consistently identify.
Patterns (e.g., "Echo Dot") are added to the matcher.
The matcher is applied to each doc, and any exact matches are extracted, providing a more targeted way to find specific product/brand mentions.
Part 3: Ethics & Optimization (10%)
1. Ethical Considerations
Potential Biases in MNIST or Amazon Reviews Model:

MNIST Model (Handwritten Digits Classification):

Training Data Bias:
Representational Bias: If the dataset (MNIST) is primarily composed of handwriting styles from a specific demographic (e.g., a particular age group, region, or education level), the model might perform poorly on handwriting from other demographics. For instance, if most training samples are from young adults, it might struggle with elderly handwriting or children's handwriting.
Class Imbalance: While MNIST is generally well-balanced for digit classes (0-9), if a specific digit were underrepresented in the training set, the model might exhibit lower accuracy for that digit.
Historical Bias: Although less prevalent in simple digit recognition, if the dataset was collected under historical circumstances where certain groups were excluded or their data was not properly captured, it could lead to biases.
Amazon Reviews Model (Sentiment Analysis & NER):

Training Data Bias (if using a pre-trained sentiment model or training one):
Demographic Bias: Sentiment models trained on general web text might reflect societal biases present in that text. For example, if reviews by certain demographic groups (e.g., based on gender, race, or age) consistently use language interpreted as less positive or more negative by the model, even when intended otherwise, the model will propagate this bias.
Language & Dialect Bias: If the training data is predominantly standard English, the model might perform poorly on reviews written in regional dialects, slang, or non-native English, leading to inaccurate sentiment classification for these users.
Contextual Bias: The rule-based sentiment analysis, as demonstrated, is highly susceptible to contextual bias. A positive word like "sick" (meaning cool) might be classified as negative, or negative connotations of a word might be missed in a positive context.
Selection Bias: If the reviews dataset itself is skewed (e.g., collected only from highly engaged customers, or from a product niche that attracts a particular type of reviewer), the model's performance may not generalize well to the broader customer base.
Labeling Bias (if the sentiment labels were human-annotated): Human annotators might have their own biases, leading to inconsistent or prejudiced labeling, which the model would then learn.
Product/Brand Bias (NER): If the NER model (or custom rules) is only trained or configured to recognize popular brands/products, it might ignore or misclassify smaller brands or niche products, creating an unfair advantage or disadvantage in product visibility or analysis.
How tools could mitigate these biases:

TensorFlow Fairness Indicators:

Grouped Performance Metrics: This tool allows developers to evaluate model performance (e.g., accuracy, precision, recall) across different demographic slices (e.g., age groups, gender, regions). For the MNIST model, one could potentially group test data by source (if such metadata were available) to check if accuracy varies significantly. For Amazon reviews, you could group reviews by inferred user demographics (if permissible and available) and analyze if sentiment accuracy differs across these groups.
Bias Detection: It helps identify if a model is performing worse for specific subgroups. If the MNIST model consistently misclassifies digits written by a certain age group, Fairness Indicators would highlight this. Similarly, for sentiment analysis, it could show if reviews containing gendered language are more likely to be misclassified.
Counterfactual Fairness: While more advanced, some concepts within fairness tools aim to assess if a model's prediction would change if a sensitive attribute (e.g., gender, race) were counterfactually changed while keeping other features similar.
Mitigation Techniques: Although not a mitigation technique itself, Fairness Indicators guides mitigation by pinpointing where bias exists, allowing developers to then apply techniques like re-sampling, re-weighting, or adversarial debiasing.
spaCy's Rule-Based Systems (and general NLP strategies):

Mitigating Sentiment Bias:
Context-Aware Rules: Instead of simple keyword matching, more sophisticated rule-based systems in spaCy can leverage dependency parsing and POS tagging. For example, a rule could look for negation preceding a positive word ("not good") or intensification ("very good"). This makes the sentiment analysis less prone to misinterpreting context.
Lexicon Expansion and Customization: For specific domains (like product reviews), sentiment lexicons can be expanded with domain-specific terms and their polarity. This can mitigate bias stemming from out-of-domain terms.
Handling Slang/Dialect: Rules can be developed to normalize slang or specific dialectal expressions before sentiment analysis, reducing bias against non-standard English.
Mitigating NER Bias:
Custom EntityRuler: As demonstrated with PhraseMatcher, spaCy's EntityRuler allows developers to define custom patterns for product names and brands using linguistic features (e.g., specific POS tags, surrounding words). This ensures that niche or new brands, or products not recognized by the pre-trained model, are correctly identified, preventing bias towards widely known entities.
Active Learning for Annotation: When creating custom training data for NER, active learning strategies can be used to prioritize annotating examples where the model is uncertain or performing poorly on specific entity types or contexts, thereby improving coverage and reducing bias.
Regular Model Updates: Continuously updating and retraining NER models with diverse and representative data from new reviews can help keep pace with evolving product names and branding, preventing historical bias.
In essence, these tools facilitate a more systematic approach to identifying and addressing biases by providing quantitative measures of fairness and flexible mechanisms for injecting domain-specific knowledge and contextual understanding into the models.

2. Troubleshooting Challenge
Buggy Code: A provided TensorFlow script has errors (e.g., dimension mismatches, incorrect loss functions). Debug and fix the code.

(As I cannot be "provided" with a live buggy script, I will simulate a common TensorFlow error and provide the "buggy" code and then the "fixed" code with explanations.)

Scenario: A common error is incorrect input shape for the first layer of a CNN or using the wrong loss function for the label type.

"Buggy" TensorFlow Script (Example):

Python

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Bug 1: Forget to reshape for CNN (missing channel dimension)
# train_images = train_images.astype('float32') / 255.0
# test_images = test_images.astype('float32') / 255.0

# Bug 2: Using categorical_crossentropy with integer labels without one-hot encoding
# train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
# test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=10)

model = models.Sequential([
    # Bug 3: Incorrect input_shape for Conv2D (expecting 3D, got 2D if not reshaped)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28)), # Should be (28, 28, 1)
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Bug 4: Wrong loss function for integer labels
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Should be 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1) # Will likely fail here due to shape mismatch
Debugging Process & Fixes:

Running the Buggy Code: If you run the buggy code, you would typically encounter an error message like:

ValueError: Input 0 of layer conv2d is incompatible with the layer: expected ndim=4, found ndim=3. Full shape received: (None, 28, 28)
This immediately points to the input_shape of the Conv2D layer and the dimensions of train_images. It's expecting 4 dimensions (batch, height, width, channels) but received 3 (batch, height, width).
Addressing Bug 1 & 3 (Reshaping and input_shape):

The error message tells us that Conv2D needs a 4D input for (batch_size, height, width, channels). MNIST images are 28x28 grayscale, meaning they have 1 channel.
Fix: Reshape the images from (num_samples, 28, 28) to (num_samples, 28, 28, 1). Update the input_shape in Conv2D accordingly.
Addressing Bug 2 & 4 (Loss Function):

If you fix the shape issue, the code might run for an epoch but give incorrect or extremely poor results, or you might encounter a ValueError related to the loss function if your labels are integers (0-9) and you use categorical_crossentropy.
categorical_crossentropy expects labels to be one-hot encoded (e.g., [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] for digit 2).
sparse_categorical_crossentropy is designed for integer labels (e.g., 2 for digit 2), which is how MNIST labels are loaded by default.
Fix: Change the loss function to sparse_categorical_crossentropy. Alternatively, one could one-hot encode the labels using tf.keras.utils.to_categorical(train_labels, num_classes=10) and then use categorical_crossentropy, but sparse_categorical_crossentropy is simpler when labels are already integers.
Fixed TensorFlow Script:

Python

# Fixed TensorFlow Script

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt # Added for visualization from original task

print("--- Troubleshooting Challenge: Fixed TensorFlow Script ---")

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Fix 1: Reshape images to include a channel dimension (for CNNs)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

print(f"\nFixed Train images shape: {train_images.shape}")
print(f"Fixed Test images shape: {test_images.shape}")

model = models.Sequential([
    # Fix 2: Correct input_shape for Conv2D (added 1 for channel)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Fix 3: Use 'sparse_categorical_crossentropy' for integer labels
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nTraining fixed model for 1 epoch to demonstrate functionality...")
history = model.fit(train_images, train_labels, epochs=1, validation_split=0.1, batch_size=64, verbose=1)

test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest Accuracy (after fixing): {test_accuracy:.4f}")

# Re-adding prediction visualization from Task 2 for completeness
print("\nVisualizing predictions with fixed model on 5 sample images...")
sample_indices = np.random.choice(len(test_images), 5, replace=False)
sample_images = test_images[sample_indices]
sample_labels = test_labels[sample_indices]

predictions = model.predict(sample_images)

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {sample_labels[i]}\nPred: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.suptitle("Fixed Model Predictions on Sample Images", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
Key Debugging Takeaways:

Read Error Messages Carefully: TensorFlow/Keras error messages are often very informative, pointing directly to the layer or operation causing the issue and what was expected vs. received.
Check Data Shapes: Always verify the shapes of your input data and ensure they match the input_shape expected by the first layer of your model, and the subsequent layers. print(data.shape) is your friend.
Understand Loss Functions: Choose the correct loss function based on your target variable's format (e.g., sparse_categorical_crossentropy for integer labels, categorical_crossentropy for one-hot encoded labels, binary_crossentropy for binary classification).
Incremental Development: Build and test your model in small steps. Compile and run after adding a few layers, check the model.summary(), and ensure data preprocessing is correct before full training.
Bonus Task (Extra 10%)
Deploy Your Model: Use Streamlit or Flask to create a web interface for your MNIST classifier. Submit a screenshot and a live demo link.

(As an AI, I cannot create a live web demo or take screenshots. However, I can provide the Python code for a simple Streamlit application that would deploy the MNIST classifier from Task 2. You would then run this code locally and deploy it if you wish.)

Streamlit Application Code (app.py):

Python

# Bonus Task: Deploy MNIST Classifier with Streamlit

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image # For handling image uploads

# --- Function to build and load the pre-trained MNIST model ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_mnist_model():
    # Re-build the same model architecture as in Task 2
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    # Compile the model (required for evaluation/prediction even if not training)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # In a real deployment, you would save and load the trained weights
    # For this demo, we'll quickly train a model if not already trained,
    # or assume a pre-trained model is available.
    # For a robust solution, save the model after training in Task 2:
    # model.save('mnist_cnn_model.h5')
    # And then load here:
    # model.load_weights('mnist_cnn_model.h5') # If you saved only weights
    # Or:
    # model = tf.keras.models.load_model('mnist_cnn_model.h5') # If you saved the whole model

    # For demonstration, let's load a tiny pre-trained model (or quickly train if needed)
    # This assumes you have run Task 2 and have a trained model.
    # If you have saved your model from Task 2, uncomment the line below and point to your saved model path.
    # model = tf.keras.models.load_model("path/to/your/mnist_cnn_model.h5") # Replace with your saved model path

    # For simplicity, let's load the MNIST dataset and do a quick dummy train
    # This is NOT how you should do it in production, you should load a saved model.
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
    model.fit(x_train[:1000], y_train[:1000], epochs=1, verbose=0) # Train on a small subset for quick demo
    return model

# Load the model
model = load_mnist_model()

# --- Streamlit App Interface ---

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

st.title("MNIST Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) and the model will try to predict it.")
st.write("---")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L") # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized) / 255.0 # Normalize
    img_array = img_array.reshape(1, 28, 28, 1) # Add batch and channel dimensions

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    st.success(f"**Prediction: Digit {predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")

    # Display probabilities for all digits
    st.subheader("Prediction Probabilities:")
    prob_df = tf.DataFrame({
        'Digit': [str(i) for i in range(10)],
        'Probability': [f"{p*100:.2f}%" for p in predictions[0]]
    })
    st.dataframe(prob_df.set_index('Digit'))

    st.write("---")
    st.write("Try another image!")

else:
    st.info("Upload an image to get started!")

st.markdown("---")
st.markdown("Model trained on MNIST dataset using TensorFlow/Keras.")
st.markdown("Created for the AI Tools & Frameworks Assignment.")

# Instructions to run this:
# 1. Save the code above as `app.py`.
# 2. Make sure you have `streamlit` and `tensorflow` installed:
#    `pip install streamlit tensorflow numpy Pillow`
# 3. Make sure you have trained and saved your MNIST model from Task 2 as 'mnist_cnn_model.h5'
#    and update the `load_mnist_model` function to load it properly.
#    (The current `load_mnist_model` does a dummy train if no saved model is found).
# 4. Run from your terminal: `streamlit run app.py`
# 5. This will open a web browser with your app.
To use this Streamlit application:

Save the code: Save the Python code above as app.py in a new directory.
Install dependencies: Open your terminal or command prompt, navigate to that directory, and run:
Bash

pip install streamlit tensorflow numpy Pillow
Train and Save Your Model (from Task 2):
Ensure you have successfully completed Task 2 and trained your MNIST CNN model.
After training in your Task 2 script, add the following line to save your model:
Python

model.save('mnist_cnn_model.h5')
Make sure this mnist_cnn_model.h5 file is in the same directory as your app.py file, or update the load_mnist_model function in app.py to point to the correct path. (The current load_mnist_model has a simplified dummy train for quick testing if a saved model isn't immediately available, but a proper solution requires loading a pre-trained model.)
Run the Streamlit App: In your terminal, from the directory where app.py is saved, run:
Bash

streamlit run app.py
Access the App: Your web browser will automatically open to http://localhost:8501 (or a similar address) where your Streamlit app is running.
Screenshot & Live Demo Link Submission:

Screenshot: Take a screenshot of your running Streamlit application in your web browser after uploading an image and seeing a prediction. Include this in your PDF report.

Live Demo Link: To provide a live demo link, you would typically deploy your Streamlit app to a cloud platform. Some popular options include:

Streamlit Community Cloud: Easiest for Streamlit apps. You connect it to a GitHub repository containing your app.
Heroku: A general-purpose platform that can host Python web apps.
Google Cloud Run, AWS App Runner, Azure App Service: More advanced cloud services for containerized applications.
You will need to follow the specific deployment instructions for your chosen platform to get a live demo link.

This structure ensures you cover all aspects of the assignment, from theoretical understanding to practical implementation and ethical considerations, with clear instructions for each part.


Sources





can i run the bonus task on colab



Show thinking
Yes, you can run the Streamlit bonus task on Google Colab, but it requires a workaround because Colab doesn't directly expose web ports to your browser. You'll need to use a tunneling service like ngrok to create a public URL for your app.

Here’s how you can do it.

Step-by-Step Guide
Install Libraries
In a Colab cell, install Streamlit and pyngrok.

Python

!pip install streamlit pyngrok
Write the Streamlit App File
Use the %%writefile magic command to save your Streamlit app code (from the bonus task) to a file named app.py.

Python

%%writefile app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

@st.cache_resource
def load_mnist_model():
    # This function should load your pre-trained model.
    # For a real use case, load a saved .h5 file.
    # For this example, we'll quickly train a model for demonstration.
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Quick dummy training
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
    model.fit(x_train[:1000], y_train[:1000], epochs=1, verbose=0)
    return model

model = load_mnist_model()

st.title("MNIST Digit Classifier")
uploaded_file = st.file_uploader("Upload a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.success(f"Prediction: {predicted_class}")
Set Up and Run ngrok
You need an authtoken from ngrok.

Go to the ngrok dashboard and sign up for a free account.
Copy your authtoken.
Now, run the following code in a new Colab cell, pasting your token where indicated.

Python

from pyngrok import ngrok

# Terminate any existing tunnels
ngrok.kill()

# Paste your ngrok authtoken here
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTHTOKEN"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Open a tunnel to the default streamlit port (8501)
public_url = ngrok.connect(8501)
print(f"Streamlit App URL: {public_url}")
Run the Streamlit App
Finally, run the app.py file. The & at the end runs the process in the background, so you can continue using your Colab notebook.

Python

!streamlit run app.py &
After running the last two cells, a URL will be printed by ngrok. Click that URL to open and interact with your Streamlit application, all hosted from your Colab session. 🚀


Sources





connect collab to github


Connecting Google Colab to GitHub is a fantastic way to manage your AI projects, collaborate with others, and ensure version control. There are a few primary ways to do this, depending on what you want to achieve:

Saving a Colab Notebook to GitHub: This is the most common use case for versioning your work.
Loading a Colab Notebook from GitHub: To open and run existing notebooks from your repositories.
Cloning a GitHub Repository: If you have a larger project with multiple scripts, datasets, or modules in a repo and want to work on them in Colab.
Committing Changes to a Cloned Repository: To push your changes back to the GitHub repo.
Let's go through each method.

Method 1: Saving a Colab Notebook to GitHub (Easiest for Versioning Notebooks)
This method is ideal when you've created or modified a Colab notebook and want to save its current state directly to a GitHub repository.

Open your Colab Notebook.
Go to File > Save a copy in GitHub....
Authorize Colab to access GitHub:
If it's your first time, a pop-up will ask for authorization. Click "Authorize Google Colaboratory."
You'll be redirected to GitHub to grant permissions. Follow the prompts.
Select Repository:
Choose your GitHub organization/user from the dropdown.
Select the repository where you want to save the notebook.
You can optionally create a new repository directly from this dialog.
Set File Path:
Specify the File path. This is the path within your repository where the .ipynb file will be saved (e.g., my_projects/my_notebook.ipynb).
You can also add a commit message.
Click OK.
Your Colab notebook will now be saved as a new commit in the specified GitHub repository.

Method 2: Loading a Colab Notebook from GitHub
To open an existing .ipynb notebook from your GitHub repository in Colab:

Go to File > Open notebook.
In the "Open notebook" dialog, select the GitHub tab.
Search for a repository:
You can search for a specific GitHub user/organization and repository.
You can also paste the full URL of the GitHub repository or the direct link to the .ipynb file.
Select the .ipynb file you want to open.
Click Select.
The notebook will open in a new Colab tab.

Method 3: Cloning a GitHub Repository (for Full Project Work)
This method is useful when your project involves more than just a single notebook, like helper scripts, data files, or complex directory structures. You'll clone the entire repository into your Colab environment.

Get your GitHub Repository URL:

Go to your GitHub repository.
Click the green "Code" button.
Copy the HTTPS URL (e.g., https://github.com/your-username/your-repo.git).
Open a new Colab Notebook.

Run the git clone command in a cell:

Python

!git clone https://github.com/your-username/your-repo.git
Replace https://github.com/your-username/your-repo.git with your actual repository URL.

If your repository is private, you'll need to provide credentials:

Python

# Option A: Using a Personal Access Token (Recommended for private repos)
# 1. Go to GitHub -> Settings -> Developer settings -> Personal access tokens (classic)
# 2. Generate a new token with 'repo' scope. Copy it.
# 3. Use it like this:
!git clone https://your-username:YOUR_PERSONAL_ACCESS_TOKEN@github.com/your-username/your-repo.git

# Option B: Prompt for username/password (less secure for automated scripts)
# !git clone https://github.com/your-username/your-repo.git
# This will prompt you for your GitHub username and password/PAT in the output.
Verify the clone:
You'll see a new folder appear in the Colab file browser (left sidebar) with the name of your repository. You can verify its contents:

Python

!ls -F
# This will list the top-level directories, including your cloned repo
Navigate into your repository:
You'll typically want to change your current working directory to the cloned repository to access its files easily.

Python

import os
os.chdir('/content/your-repo-name') # Replace 'your-repo-name' with the actual repo folder name
!pwd # Print current working directory to confirm
Method 4: Committing Changes to a Cloned Repository
After you've cloned a repository (Method 3) and made changes to files within that cloned directory in Colab, you'll want to commit those changes back to GitHub.

Navigate to your repository directory (if you're not already there):

Python

import os
os.chdir('/content/your-repo-name')
Configure Git (first time per session/repo):
You need to tell Git who you are.

Python

!git config user.email "your.email@example.com"
!git config user.name "Your Name"
Check Status:
See which files have been modified or are untracked.

Python

!git status
Add Changes:
Stage the files you want to commit.

Python

!git add . # Adds all changed/new files
# Or for specific files:
# !git add my_script.py my_data_folder/
Commit Changes:

Python

!git commit -m "Your descriptive commit message"
Push Changes to GitHub:
This is where you'll need to authenticate with GitHub.

Python

# If using a private repo or first push
# You will be prompted for your GitHub username and Personal Access Token (PAT)
# IMPORTANT: Use a PAT instead of your regular password for security.
# See Method 3, Option A for how to generate a PAT.
!git push origin main # Or 'master' depending on your branch name
When prompted:

Username for 'https://github.com': your-github-username
Password for 'https://your-github-username@github.com': YOUR_PERSONAL_ACCESS_TOKEN (Paste your PAT here, it won't show on screen)
If you're pushing to a public repo and haven't set credentials, it might just work if you've previously authorized Colab for saving. However, using PATs for pushes is the most robust and secure method.

By using these methods, you can effectively integrate your Google Colab workflow with GitHub for robust version control and collaborative development.






does colab automatic update github


No, Google Colab does not automatically update GitHub when you make changes to a notebook or files within a cloned repository. You always need to explicitly initiate a save or a commit/push operation.

Here's a breakdown:

Saving a Colab Notebook to GitHub (File > Save a copy in GitHub...):

This is a manual one-time save operation for the current state of that specific notebook.
It creates a new commit in your chosen GitHub repository.
It does not automatically sync subsequent changes you make in Colab. If you continue working on the notebook, you'll need to go through File > Save a copy in GitHub... again to push those new changes. Each time it will create a new commit.
Working with a Cloned GitHub Repository (using !git clone, !git add, !git commit, !git push):

When you clone a repository into your Colab session, you're essentially getting a copy of that repository on the Colab VM's temporary storage.
Any changes you make to files within that cloned directory in Colab are local to that Colab session.
To get those changes back to GitHub, you must explicitly use Git commands:
!git add . (to stage your changes)
!git commit -m "Your commit message" (to record the changes locally)
!git push origin <branch_name> (to send the commits to your remote GitHub repository)
These are manual operations you perform via code cells. Colab does not monitor your files and automatically push them.
Why no automatic updates?

Version Control Philosophy: Git (and GitHub) is designed for explicit version control. Every save to the repository is a "commit" with a message, allowing you to track changes, revert to previous versions, and collaborate. Automatic, implicit saves would undermine this control and could lead to messy, untracked changes.
Performance/Resource Usage: Constantly monitoring file changes and pushing them to a remote server would consume significant resources and bandwidth, especially for large projects.
User Control: Users need to decide when to save, what to save (which files to stage), and what commit message to associate with the changes. This is fundamental to effective version control.
So, while Colab provides excellent integration with GitHub, it's always an on-demand, user-initiated process to update your GitHub repositories.


Sources





Saving a Colab Notebook to GitHub (File > Save a copy in GitHub...) not working


It's frustrating when a seemingly straightforward feature like "Save a copy in GitHub" doesn't work! This can happen for a few reasons. Let's troubleshoot it.

Here are the most common reasons and solutions:

GitHub Authorization Issues (Most Common):

Expired or Revoked Token: Colab uses an OAuth token to connect to your GitHub account. This token might have expired or you might have accidentally revoked Colab's access from your GitHub settings.
Solution:
Go to your GitHub settings: Settings > Applications > Authorized OAuth Apps.
Find "Google Colaboratory" (or similar) and revoke its access.
Go back to your Colab notebook, close the problematic tab, open it again, and try File > Save a copy in GitHub... again. This should prompt you to re-authorize Colab, allowing you to grant the necessary permissions.
Important for Organizations/Private Repos: When re-authorizing, ensure you grant access to private repositories and organizations if that's where you're trying to save. Sometimes, you need to explicitly check a box for this. For organizations, the organization owner might also need to approve Colab's access in the organization's settings.
Pop-up Blockers:

The GitHub authorization process often involves a pop-up window. If you have a strict pop-up blocker enabled in your browser, it might be preventing this window from appearing, thus blocking the authorization flow.
Solution: Temporarily disable your pop-up blocker for colab.research.google.com or google.com and try again.
Browser Cookies/Cache:

Corrupted browser data can sometimes interfere with authentication.
Solution: Try clearing your browser's cache and cookies for Google Colab and GitHub. Or, try saving using an Incognito/Private Browse window to rule out browser extensions or cached data.
GitHub Repository State/Permissions:

No write access: Ensure that the GitHub account you've authorized Colab with actually has write permissions to the specific repository and branch where you're trying to save. If it's a repository owned by an organization, you might not have direct write access.
Large Notebooks: Very large notebooks can sometimes cause issues.
Network Issues/Temporary Glitch: Sometimes it's just a transient network issue or a temporary glitch on either Colab or GitHub's side.
Solution: Wait a few minutes and try again. Refresh the Colab page (F5 or Cmd+R) before retrying.
Incorrect Branch/File Path:

Double-check that you're selecting the correct repository and a valid file path within that repository in the "Save to GitHub" dialog.
"You do not have permission to save this notebook" Error:

If you opened the notebook directly from GitHub in Colab, Colab often opens it as a read-only "copy" initially. You cannot directly "Save" (Ctrl+S) back to GitHub. You must always use File > Save a copy in GitHub... to create a new commit. The error message is a bit misleading here, as it implies you can't save at all, but it means you can't overwrite the original source directly.
Solution: Always use File > Save a copy in GitHub... even if you opened it from GitHub. This will create a new commit with your changes.
Troubleshooting Steps to Follow:
Reload Colab: Close the current Colab tab and open the notebook again.
Try Incognito Mode: Open Colab in an Incognito/Private window in your browser. This bypasses extensions and cached data.
Re-authorize GitHub:
Go to GitHub: Settings > Applications > Authorized OAuth Apps.
Revoke access for "Google Colaboratory".
Go back to Colab and attempt File > Save a copy in GitHub.... Follow the authorization prompts carefully, ensuring you grant all necessary permissions, especially for private repos/organizations.
Check Pop-up Blocker: Ensure no pop-up blocker is interfering.
Small Test Notebook: Create a brand new, very simple Colab notebook (e.g., just print("Hello")). Try saving that to a new, empty public GitHub repository. This helps isolate if the issue is with the notebook itself, the target repository, or the general connection.
Check GitHub Status: Rarely, GitHub itself might be experiencing issues. Check the GitHub Status page.
If you've tried these steps and are still facing issues, provide any specific error messages you see, and that can help narrow down the problem further.






github still shows they is no code added


Okay, let's figure out why your GitHub repository isn't showing the code you saved from Colab. Here's a breakdown of things to check:

Check the Correct Repository and Branch:

Repository Selection: Double-check you saved (or pushed) to the correct repository. If you have multiple repositories, it's easy to accidentally save to the wrong one.
Branch: GitHub repositories have branches (usually main or master by default). Make sure you're looking at the branch where you saved/pushed your code. The default branch is usually displayed when you first open a repository. You can switch branches using the branch dropdown menu (usually near the top-left, it will say something like "main" or "master").
File Path: If you specified a file path when saving from Colab (e.g., my_notebooks/my_colab_notebook.ipynb), ensure you are navigating to that directory within your repository to find the file.
Verify Successful Save/Push from Colab:

"Save a copy in GitHub...": After using File > Save a copy in GitHub... in Colab, did you see a confirmation message that the save was successful? If there were any errors or prompts you dismissed, the save might not have completed. Try the save again, watch for any error messages, and ensure you go through the complete authorization flow if prompted.
Git Push (if cloning): If you were using !git clone, !git add, !git commit, and !git push, carefully review the output of those commands in the Colab cells. Look for any error messages during the push operation, especially related to authentication or permissions. If the push failed, the changes wouldn't have made it to GitHub.
Check Commit History:

On your GitHub repository page, look for the "commits" link (usually near the top). This shows a history of all changes made to the repository.
See if there's a recent commit corresponding to your save or push from Colab. The commit message should give you a clue. If there's no recent commit, the save/push likely failed.
GitHub Authorization (Again):

Even if you authorized Colab previously, the authorization might have expired or been revoked. Follow the steps in the previous response to re-authorize Colab with GitHub. Ensure you grant access to the correct repositories (especially private ones or those within organizations).
Small Test:

Create a very simple new Colab notebook (just a few lines of code).
Try saving or pushing that to GitHub. This helps isolate if the issue is with a specific large/complex notebook or with the general connection.
Network/GitHub Issues:

Occasionally, there might be temporary network issues or problems on GitHub's side. Check the GitHub Status page to see if there are any reported outages.
Private Repository Permissions (If Applicable):

If you're working with a private repository, double-check that your GitHub account has write access to it. If it's an organization-owned repository, the organization owner might need to grant you access.
If you've checked all of these things and you're still not seeing your code, please provide more specific details:

How are you saving/pushing? (Using File > Save a copy in GitHub... or git commands?)
Are you seeing any error messages in Colab? If so, what are they?
Can you link to the GitHub repository you're trying to save to? (If it's public).
What's the commit history showing?
Did you grant Colab access to private repositories or organizations during authorization?
The more information you give, the better I can assist you!






Save a copy in GitHub...": After using File > Save a copy in GitHub... in Colab, did you see a confirmation message that the save was successful? If there were any errors or prompts you dismissed, the save might not have completed. Try the save again, watch for any error messages, and ensure you go through the complete authorization flow if prompted. no errors just been asked where must my code go then I choose the folder




Show thinking
Okay, this is a common point of confusion. If Colab asks you where to save and doesn't show an error, it usually means the Colab-side process completed successfully. The issue might be in how you're verifying it on GitHub, or a subtle detail in the save process.

Let's re-examine the "Save a copy in GitHub..." flow and then how to check on GitHub very carefully.

1. Reconfirm the "Save a copy in GitHub..." Steps in Colab
When you go to File > Save a copy in GitHub..., you should see a dialog box. Please walk through these specific points and confirm what you are selecting/typing for each:

GitHub organization/user: This dropdown lets you choose your personal GitHub account or any organizations you are part of.
Are you selecting your personal username (e.g., your-username) or an organization (e.g., my-company)?
Repository: This dropdown lists all repositories under the selected user/organization.
Are you sure you are selecting the exact repository where you expect the file to appear? (e.g., my_ai_projects, not my_website).
Is this repository public or private? (If private, ensure Colab has private repo access granted during authorization).
File path: This is crucial. It defines the name of your notebook and its location within the selected GitHub repository.
What exactly are you typing here?
If you type my_notebook.ipynb, it will appear directly in the root of the repository.
If you type my_folder/my_notebook.ipynb, it will create my_folder (if it doesn't exist) and save the notebook inside it.
Is it possible you typed a path that leads to a subfolder you're not looking in?
Are you giving it a .ipynb extension? (Colab usually adds this automatically, but confirm).
Commit message: This is the message associated with the change in Git.
You'll usually see a default like "Initial commit" or "Colaboratory via GitHub".
Branch (optional, if shown): Sometimes, if you have multiple branches, it might ask which branch to save to (e.g., main or master).
Are you saving to the correct branch? (The one you usually view).
Once you click "OK" in this dialog, do you see any small "Saving..." or "Saved to GitHub" notification at the bottom of the Colab window? Even if it's quick, a success message usually appears briefly.

2. How to Verify on GitHub (The Definitive Check)
The best way to confirm if anything was sent to GitHub is to check the commit history of the repository you selected.

Go to your GitHub repository in your web browser. (e.g., https://github.com/your-username/your-repo-name)
Look for the "commits" link. It's usually a small link next to the branch name, or you can click on the "Code" tab and then find "commits" (it will often show the number of commits).
Example: (Image is illustrative, actual GitHub UI might vary slightly)
Examine the commit history:
Do you see a new commit at the top of the list that corresponds to your save attempt (based on the time and the commit message)?
If you see a commit from "Colaboratory" or with your chosen commit message: This means the save did happen on GitHub's end.
Click on that commit message. This will show you the specific files that were changed or added in that commit.
Does your .ipynb file appear in the list of changed files for that commit? If it does, then the file is definitely in the repository. The problem then is just locating it on the main repository page (maybe you saved it in a subfolder).
If you don't see any new commit from Colab/yourself: This means the save operation, despite no explicit error message, did not actually complete the push to GitHub. This would bring us back to troubleshooting Colab's connection or authorization (Step 1 in my previous answer).
Common Scenario if a Commit is Found but File is "Missing":
It's in a Subfolder: You specified a file path like my_notebooks/my_awesome_project.ipynb. When you view the main repository page, you need to click into the my_notebooks folder to see your file.
Wrong Branch: You saved to a branch different from the one you're currently viewing on GitHub. Use the branch selector on GitHub to switch to main, master, or whatever branch you saved to.
Please try these detailed checks and let me know what you find, especially about the commit history on GitHub!









Analysis
Thank you for the screenshot! This is extremely helpful.

Based on the screenshot, here's what I observe and what it implies:

Repository: faithtinarwo/ai_tools - This confirms you're selecting a specific repository.
File path: AI_software_tools.ipynb - This means the notebook will be saved directly into the root of the ai_tools repository, with that exact filename. It's not going into a subfolder.
Branch: "No branches" - This is the most crucial piece of information here!
The Problem: "No branches"
"No branches" in that dialog indicates that GitHub hasn't found any existing branches in your ai_tools repository.

A GitHub repository must have at least one branch (traditionally master or, more recently and commonly, main) for you to commit files to it.

Why would it say "No branches"?

You created an empty repository on GitHub: When you initialize a new repository on GitHub without checking "Add a README file" or "Add .gitignore", it's created as an empty shell without any initial commits or branches. Colab needs some branch to push to.
A temporary glitch: Less likely, but possible.
How to Fix It: Initialize Your GitHub Repository
You need to ensure your ai_tools repository has at least one initial commit and a default branch (main or master).

Option 1: Add a README (Easiest)

Go to your ai_tools repository on GitHub in your web browser: https://github.com/faithtinarwo/ai_tools (assuming faithtinarwo is your GitHub username).
If it's truly empty, you should see a prompt like "Quick setup — if you've done this kind of thing before" or buttons to "create a new file", "upload files", etc.
Look for a button or link that says "Add a README" or "Initialize this repository with a README". Click it.
You can leave the README content as default or add a description.
Scroll down and click the green "Commit new file" button.
This action will:

Create a main (or master) branch.
Add the README.md file to it.
Make the first commit.
Option 2: Manually create a new file via GitHub web UI

Go to your ai_tools repository on GitHub.
Click on "Add file" -> "Create new file".
Name it test.txt and type some text inside.
Commit the new file.
After doing either Option 1 or 2, your repository will have a branch.

Then, Try Saving from Colab Again:
Go back to your Colab notebook tab.
Refresh the Colab page (F5 or Ctrl+R/Cmd+R). This is important for Colab to re-fetch the repository's branch information from GitHub.
Go to File > Save a copy in GitHub... again.
Now, the "Branch" dropdown should show main (or master). Select it.
Click OK.
Your AI_software_tools.ipynb notebook should now successfully appear in the root of your faithtinarwo/ai_tools repository on GitHub!







worked create a read me aand instructions as you will be collaborating with other peers



Show thinking
Excellent! I'm glad we got that sorted. Having a proper branch is essential.

Now that your GitHub repository is correctly initialized, let's create a comprehensive README.md file and clear collaboration instructions. This will be the first thing your peers see and will guide your teamwork effectively.

Here's a template for your README.md, incorporating best practices for collaborative AI projects, especially when using Google Colab.

README.md Template
You can copy this into a new file named README.md directly on GitHub, or create it in your Colab environment using %%writefile README.md in a cell, then commit and push it.

Markdown

# AI Tools & Frameworks Mastery: Group Assignment

## Project Overview

This repository houses the collaborative work for our AI Tools & Frameworks assignment, designed to evaluate our understanding and practical application of key AI libraries and platforms. We will demonstrate proficiency in selecting, implementing, and critically analyzing AI tools to solve real-world problems across classical machine learning, deep learning, and natural language processing.

The assignment is structured into three main parts:
* **Part 1: Theoretical Understanding:** Demonstrating knowledge of core AI tools.
* **Part 2: Practical Implementation:** Applying tools to solve specific AI tasks.
* **Part 3: Ethics & Optimization:** Addressing ethical considerations and troubleshooting.

## Team Members

* **[Your Name 1]** - [Your Role/Contributions]
* **[Peer Name 2]** - [Peer's Role/Contributions]
* **[Peer Name 3]** - [Peer's Role/Contributions]
* *(Add more names as needed)*

---

## Table of Contents

1.  [Tools & Technologies](#tools-and-technologies)
2.  [Repository Structure](#repository-structure)
3.  [Getting Started (Local Setup)](#getting-started-local-setup)
4.  [Working with Google Colab (Recommended)](#working-with-google-colab-recommended)
    * [Opening Notebooks from GitHub](#opening-notebooks-from-github)
    * [Saving Your Changes Back to GitHub](#saving-your-changes-back-to-github)
    * [Handling Non-Notebook Files](#handling-non-notebook-files)
5.  [Project Tasks & Deliverables](#project-tasks-and-deliverables)
    * [Part 1: Theoretical Understanding](#part-1-theoretical-understanding)
    * [Part 2: Practical Implementation](#part-2-practical-implementation)
    * [Part 3: Ethics & Optimization](#part-3-ethics-and-optimization)
    * [Bonus Task: Model Deployment](#bonus-task-model-deployment)
6.  [Collaboration Guidelines](#collaboration-guidelines)
    * [Branching Strategy](#branching-strategy)
    * [Commit Messages](#commit-messages)
    * [Pull Requests (PRs)](#pull-requests-prs)
    * [Conflict Resolution](#conflict-resolution)
    * [Communication](#communication)
7.  [Submission Guidelines](#submission-guidelines)
8.  [Resources](#resources)
9.  [License](#license)

---

## 1. Tools & Technologies

This project leverages a range of industry-standard AI tools and platforms:

* **Machine Learning Frameworks:** TensorFlow, PyTorch, Scikit-learn, spaCy
* **Development Environments:** Google Colab (recommended), Jupyter Notebook
* **Programming Language:** Python 3.x
* **Version Control:** Git, GitHub
* **Deployment (Bonus):** Streamlit / Flask

---

## 2. Repository Structure

This repository is organized to keep our project clean and manageable:

.
├── README.md                           # Project overview and instructions
├── notebooks/                          # Jupyter/Colab notebooks for practical tasks
│   ├── classical_ml_iris.ipynb         # Task 1: Scikit-learn with Iris
│   ├── deep_learning_mnist.ipynb       # Task 2: TensorFlow/PyTorch with MNIST
│   └── nlp_spacy_reviews.ipynb         # Task 3: spaCy with Amazon Reviews
├── reports/                            # PDF report, screenshots, ethical reflections
│   ├── AI_Tools_Assignment_Report.pdf
│   └── screenshots/
├── scripts/                            # Standalone Python scripts (e.g., for Streamlit app)
│   └── streamlit_mnist_app.py          # Bonus Task: Streamlit deployment code
├── data/                               # Placeholder for any datasets not loaded directly
├── .gitignore                          # Specifies intentionally untracked files to ignore
└── LICENCE                             # Project license (e.g., MIT)

## 3. Getting Started (Local Setup)

While Google Colab is recommended, you can set up the environment locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/faithtinarwo/ai_tools.git](https://github.com/faithtinarwo/ai_tools.git)
    cd ai_tools
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # (You will need to create a requirements.txt by running 'pip freeze > requirements.txt' in your Colab or local env once you have all libs installed)
    ```
    *Alternatively, install individually:*
    ```bash
    pip install tensorflow scikit-learn spacy matplotlib pandas numpy Pillow streamlit pyngrok
    python -m spacy download en_core_web_sm # For spaCy's English model
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

---

## 4. Working with Google Colab (Recommended)

Google Colab offers free GPU access and a collaborative environment, making it ideal for this assignment.

### Opening Notebooks from GitHub

1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Click on `File` > `Open notebook`.
3.  Go to the `GitHub` tab.
4.  In the "Enter a GitHub URL or organization/user" field, type `faithtinarwo/ai_tools` and press Enter.
5.  Select the desired `.ipynb` notebook from the list (e.g., `notebooks/deep_learning_mnist.ipynb`).

### Saving Your Changes Back to GitHub

**This is critical for collaboration and version control.**

* **For `.ipynb` notebooks:**
    1.  After making changes in your Colab notebook, go to `File` > `Save a copy in GitHub...`.
    2.  Ensure `faithtinarwo/ai_tools` is selected as the repository.
    3.  **Crucially, ensure the "Branch" dropdown shows `main` (or the branch you are working on, see "Branching Strategy" below).** If it says "No branches", your repo might not be initialized, or you need to refresh Colab.
    4.  Enter a descriptive `Commit message` (e.g., "Feat: Completed MNIST CNN model for Task 2").
    5.  Click `OK`.
    6.  **Important:** This creates a new commit directly on GitHub. It does *not* automatically pull changes made by others. Always be aware of potential merge conflicts if multiple people are editing the same notebook.

* **For non-notebook files (e.g., `.py` scripts, data files):**
    If you're working with Python scripts or other files that are part of the repository (e.g., `scripts/streamlit_mnist_app.py`), you'll need to use standard Git commands within Colab cells to push changes.

    1.  **Clone the repository at the start of your Colab session:**
        ```python
        !git clone [https://github.com/faithtinarwo/ai_tools.git](https://github.com/faithtinarwo/ai_tools.git)
        %cd ai_tools # Change directory into the cloned repo
        ```
    2.  **Configure Git (first time in a new Colab session):**
        ```python
        !git config user.email "your.email@example.com" # Use your GitHub email
        !git config user.name "Your GitHub Username"
        ```
    3.  **Make your changes.** You can edit files directly in Colab's file browser (left sidebar) or use `%%writefile` magic command.
    4.  **Check status, add, commit, and push:**
        ```python
        !git status # See what files have changed
        !git add .  # Add all changed files to staging
        !git commit -m "Your descriptive commit message" # Commit changes
        !git push origin main # Push to the main branch (or your feature branch)
        ```
        *If prompted for username/password, use your GitHub username and a Personal Access Token (PAT).*

---

## 5. Project Tasks & Deliverables

Here's a breakdown of the specific tasks and deliverables for our assignment:

### Part 1: Theoretical Understanding
* **Deliverable:** Answers to short answer questions and comparative analysis within the final PDF report.

### Part 2: Practical Implementation

* **Task 1: Classical ML with Scikit-learn**
    * **Dataset:** Iris Species Dataset
    * **Goal:** Preprocess, train a Decision Tree Classifier, evaluate with accuracy, precision, and recall.
    * **Code Location:** `notebooks/classical_ml_iris.ipynb`
* **Task 2: Deep Learning with TensorFlow/PyTorch**
    * **Dataset:** MNIST Handwritten Digits
    * **Goal:** Build a CNN, achieve >95% test accuracy, visualize predictions.
    * **Code Location:** `notebooks/deep_learning_mnist.ipynb`
* **Task 3: NLP with spaCy**
    * **Text Data:** User reviews from Amazon Product Reviews (example data will be used within the notebook).
    * **Goal:** Perform NER (product names, brands), analyze sentiment (rule-based).
    * **Code Location:** `notebooks/nlp_spacy_reviews.ipynb`

### Part 3: Ethics & Optimization

* **Deliverable:** Answers to ethical considerations and debugging the provided buggy code, included in the final PDF report and potentially within relevant notebooks.

### Bonus Task: Model Deployment

* **Goal:** Deploy the MNIST classifier using Streamlit or Flask.
* **Code Location:** `scripts/streamlit_mnist_app.py`
* **Deliverable:** Screenshot of the web interface in the report, and if possible, a live demo link.

---

## 6. Collaboration Guidelines

To ensure smooth teamwork and effective version control, please adhere to these guidelines:

### Branching Strategy

* **`main` Branch:** This branch (`main`) will always contain the latest stable, working version of our project. **Do not commit directly to `main` unless specifically instructed (e.g., for minor README updates).**
* **Feature Branches:** For every task or significant change you work on, create a new branch from `main`.
    * **Naming Convention:** Use descriptive names like `feature/task1-sklearn`, `bugfix/mnist-accuracy-fix`, `refactor/nlp-sentiment-rules`.
    * **Example:** If you're working on Task 1, you'd create `feature/classical-ml-iris`.
    ```bash
    # In a Colab cell after cloning the repo:
    %cd ai_tools
    !git checkout main
    !git pull origin main # Ensure your main is up-to-date
    !git checkout -b feature/your-task-name
    ```
* **Work on your feature branch.** Once your changes are complete and tested, push your feature branch and open a Pull Request to `main`.

### Commit Messages

* Write clear, concise, and descriptive commit messages.
* Start with a verb indicating the type of change (e.g., `Feat:`, `Fix:`, `Docs:`, `Refactor:`, `Chore:`).
* **Example:** `Feat: Implement Decision Tree Classifier for Iris dataset (Task 1)`

### Pull Requests (PRs)

* Once your feature branch is ready, create a Pull Request on GitHub from your feature branch to the `main` branch.
* **Request Reviews:** Tag at least one other team member to review your code.
* **Describe Changes:** Clearly describe what you did and why in the PR description. Link to the relevant assignment task.
* **Address Feedback:** Respond to comments and make necessary changes.

### Conflict Resolution

* **Pull Frequently:** Before starting work and pulling a new branch, always `!git pull origin main` to get the latest changes.
* **Communicate:** If you anticipate working on the same file as someone else, discuss it beforehand.
* **Resolve Locally:** If a merge conflict occurs when pulling or merging, resolve it locally in your Colab environment using Git's conflict markers, or coordinate with the team member. **Do not force push unless you are absolutely sure what you are doing.**

### Communication

* Beyond GitHub, we will use our **WhatsApp Group** to coordinate tasks, discuss progress.

---

## 7. Submission Guidelines

* **Code:** All code (Jupyter/Colab notebooks and Python scripts) must be well-commented and submitted on this GitHub repository.
* **Report:** A PDF report will be shared as an article in the LMS Community and included in the GitHub Repo (`reports/AI_Tools_Assignment_Report.pdf`). It should contain:
    * Answers to theoretical questions.
    * Screenshots of model outputs (e.g., accuracy graphs, NER results).
    * Ethical reflection.
* **Presentation:** A 3-minute video explaining our approach will be shared on the Community platform, with all members participating.

---

## 8. Resources

* [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
* [spaCy Documentation](https://spacy.io/usage)
* [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [GitHub Guides](https://guides.github.com/) (for Git basics and collaboration workflow)

---

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
