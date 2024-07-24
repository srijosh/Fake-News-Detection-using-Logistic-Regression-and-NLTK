# Fake News Detection using Logistic Regression and NLTK

This repository contains a project for detecting fake news articles using a Logistic Regression model. The project utilizes Natural Language Toolkit (NLTK) for text preprocessing and Term Frequency-Inverse Document Frequency (TF-IDF) vectorization for feature extraction.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Code](#code)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Fake news detection is a crucial task in today's world, where misinformation can spread rapidly through social media and other online platforms. This project aims to build a machine learning model to classify news articles as real or fake using text-based features.

## Dataset

The dataset used in this project is from Kaggle's Fake News competition. It contains news articles labeled as either real or fake. The dataset is split into training and testing sets to evaluate the model's performance.

- [Fake News Dataset on Kaggle](https://www.kaggle.com/competitions/fake-news/data)

## Installation

To run this project, you need to have Python installed on your machine. You can install the required dependencies using `pip`.

```
pip install numpy pandas nltk scikit-learn
```

Requirements
Python 3.x
NumPy
Pandas
NLTK
Scikit-learn

## Usage

1. Clone the repository to your local machine:

```
   git clone https://github.com/srijosh/Fake-News-Detection-using-Logistic-Regression-and-NLTK.git
```

2. Navigate to the project directory:
   cd Fake-News-Detection-using-Logistic-Regression-and-NLTK

3. Download the dataset from Kaggle and place it in the project folder

4. Open and run the Jupyter Notebook:
   jupyter notebook Fake_News_Prediction.ipynb

## Model

The model used in this project is a Logistic Regression classifier. The text data is preprocessed using NLTK, and TF-IDF vectorization is applied to convert the text into numerical features suitable for machine learning.

Text Preprocessing

- Stop words removal: Removing common words that do not contribute much to the meaning of the text (e.g., "and", "the").
- Stemming: Reducing words to their root form using the Porter Stemmer.

Feature Extraction

- TF-IDF vectorization: Converting text into numerical features based on term frequency and inverse document frequency.
