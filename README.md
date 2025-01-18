# Sentiment Analysis Project: Binary Classification of Movie Reviews

# DS Part :

## Introduction

This project involves building a binary sentiment classification model for movie reviews, with labels indicating positive or negative sentiments. The dataset used contains 50,000 movie reviews, split into training and testing sets. The objective is to accurately predict the sentiment of a review using machine learning techniques.

## Tools and Libraries Used

- **Google Colab** (for implementation)
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and NLTK

## Exploratory Data Analysis (EDA)

### Key Insights:

- The dataset is balanced, with an equal number of positive and negative reviews (25,000 each in the training set).
  
### Review Lengths:

- **Word count**: Ranges from 10 to 2,500 words, with an average of ~300 words.
- **Character count**: Average of ~1,500 characters per review.

Sentiments are equally distributed, eliminating the need for data balancing techniques.

### Visualizations:

- **Histograms** for word and character counts revealed right-skewed distributions.
- **Word clouds** highlighted common words for both positive and negative sentiments.

## Text Preprocessing

### Steps Performed:

- **Tokenization**: Split text into individual words.
- **Stop-word Removal**: Removed common words (e.g., "the", "and", "is") to focus on meaningful terms.

### Comparison: Stemming vs Lemmatization:

- **Stemming**: Reduced words to their base forms but introduced readability issues (e.g., "running" → "run").
- **Lemmatization**: Retained meaningful forms of words and preserved context (e.g., "running" → "running").

**Conclusion**: Lemmatization was chosen for better semantic retention.

### Vectorization:

- **TF-IDF Vectorization**: Performed better than Count Vectorization by emphasizing important words while down-weighting frequent but less meaningful ones.

## Modeling

### Models Explored:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Random Forest Classifier**

### Performance Comparison:

| Model               | Precision (Negative) | Recall (Negative) | F1-Score (Negative) | Precision (Positive) | Recall (Positive) | F1-Score (Positive) | Accuracy |
|---------------------|----------------------|-------------------|---------------------|----------------------|-------------------|---------------------|----------|
| Logistic Regression  | 0.90                 | 0.89              | 0.89                | 0.89                 | 0.90              | 0.89                | 0.89     |
| SVM                 | 0.90                 | 0.90              | 0.90                | 0.90                 | 0.90              | 0.90                | 0.90     |
| Naive Bayes         | 0.85                 | 0.89              | 0.87                | 0.89                 | 0.85              | 0.87                | 0.87     |
| Random Forest       | 0.84                 | 0.86              | 0.85                | 0.85                 | 0.83              | 0.84                | 0.84     |

## Final Model Selection:

- **SVM** was chosen as the final model due to its highest overall accuracy (90%) and balanced performance across all metrics.

## Overall Performance Evaluation

The **SVM model** achieved:

- **Accuracy**: 90%
- **F1-Score**: 0.90 (both positive and negative classes)

This demonstrates the model's ability to generalize well on unseen data.

## Business Applications

1. **Customer Feedback Analysis**: Understanding customer sentiment from reviews can help businesses improve their products or services.
2. **Content Moderation**: Automatically flagging negative reviews for quick resolution.
3. **Market Research**: Identifying trends and patterns in customer opinions.

---

# MLE Part :


---