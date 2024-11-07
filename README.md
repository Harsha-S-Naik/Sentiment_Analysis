# Sentiment Analysis of IMDb Movie Reviews

## Objective
The goal of this project is to classify IMDb movie reviews as positive or negative based on the textual content of the reviews. This is achieved by preprocessing the data using NLTK (Natural Language Toolkit), transforming the text data into numerical vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer, and training a Logistic Regression model with cross-validation. The model is then evaluated with an accuracy of 89%.

## Methodology

### Data Collection
The dataset used in this project consists of movie reviews from IMDb, where each review is labeled as either positive or negative. This dataset can be found in public datasets like the IMDb dataset available through sources such as Kaggle.
- **Dataset Link** = https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews

### Data Preprocessing
- **Text Cleaning:** The text data is preprocessed by removing unnecessary characters, punctuation, and stop words using the NLTK library.
- **Tokenization:** Words in the reviews are split into individual tokens (words).
- **TF-IDF Vectorization:** The text data is transformed into numerical vectors using the TF-IDF vectorizer, which captures the importance of words based on their frequency and inverse document frequency.

### Model Training
- **Logistic Regression with Cross-Validation (LogisticRegressionCV):** A Logistic Regression model is trained with cross-validation to select the best regularization parameter and avoid overfitting. This method ensures robust performance across different subsets of the data.
- **Model Evaluation:** The model's performance is evaluated using accuracy, with the final model achieving an impressive accuracy of 89% in classifying reviews correctly.

## Key Results
- Achieved **89% accuracy** in classifying IMDb movie reviews as positive or negative, demonstrating the effectiveness of the model.
- The use of TF-IDF vectorization and Logistic Regression with cross-validation contributed significantly to the model's performance.

## Tools & Libraries
- **Python**
- **NLTK** (Natural Language Toolkit)
- **Scikit-learn** (for TF-IDF Vectorizer and LogisticRegressionCV)
- **Pandas** (for data manipulation)
- **NumPy** (for numerical operations)

## How to Run
1. Clone the repository:  
   `git clone https://github.com/Harsha-S-Naik/Sentiment_Analysis.git`
   
2. Install the required libraries:  
   `pip install -r requirements.txt`
   
3. Run the Jupyter Notebook to see the sentiment analysis in action:  
   `jupyter notebook`

4. To  Run the Application
   `python app.py`

## Preview
![sentiment](https://github.com/user-attachments/assets/81d9db3a-959a-4c28-aa8d-4642316e1c3b)


# Live Link:
https://sentiment-analysis-review.onrender.com/


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
