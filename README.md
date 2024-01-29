# Sentiment Exploration in IMDb Reviews

This Jupyter notebook explores sentiment analysis on 50K IMDb reviews using various natural language processing (NLP) techniques and machine learning models.

## Overview

The notebook is structured to guide you through the process of analyzing sentiment in IMDb reviews. It covers the following tasks:

### 1. Data Loading and Exploration

- **Libraries Used:** `pandas`, `matplotlib`, `seaborn`, `plotly`, `nltk`, `wordcloud`, `scikit-learn`
- **Dataset:** IMDb dataset with 50K reviews

### 2. Data Preprocessing

- Cleaning and processing the text data for analysis.
- Removing HTML tags, URLs, mentions, and non-alphanumeric characters.
- Tokenization, stemming, and removing stopwords.

### 3. Exploratory Data Analysis (EDA)

- Descriptive statistics of the dataset.
- Visualization of sentiment distribution using a count plot.

### 4. Visualization of Sentiment Distribution

- Utilizing `plotly` to create an interactive histogram showcasing sentiment distribution.

### 5. Word Count Analysis

- Creating histograms to analyze the word count distribution in reviews for both positive and negative sentiments.

### 6. Word Clouds for Positive and Negative Reviews

- Generating word clouds to visually represent the most frequent words in positive and negative reviews.

### 7. Common Words Analysis for Positive and Negative Reviews

- Analyzing and visualizing the most common words in positive and negative reviews using bar charts.

### 8. Model Building and Evaluation

- **Models:** Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Machine (LinearSVC), Random Forest Classifier, Decision Tree Classifier.
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report.

### 9. Model Comparison and Selection

- Comparing the accuracy of each model and selecting the best-performing model.

## Usage

1. Open the Jupyter notebook in a compatible environment.
2. Execute the cells in order to run the analysis step by step.
3. Explore the visualizations, model performance, and insights.

## Dependencies

Ensure that you have the required Python libraries installed. You can install them using the following:

```bash
pip install pandas matplotlib seaborn plotly nltk wordcloud scikit-learn
