 Fake News Detection Using NLP 🧠

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to detect whether a news article is **real** or **fake**, based on its title and content.

 Built with `Python`, `Scikit-learn`, `TF-IDF`, and `Logistic Regression`

---

 Project Overview

- ✅ Loaded real & fake news datasets
- ✅ Preprocessed text with custom cleaning function
- ✅ Vectorized using **TF-IDF**
- ✅ Trained **Logistic Regression** model
- ✅ Achieved **98.7% accuracy**
- ✅ Saved model and vectorizer for future predictions

 Files Included

| File Name              | Description                             |
|------------------------|-----------------------------------------|
| `Fake.csv`             | Fake news articles dataset              |
| `True.csv`             | Real news articles dataset              |
| `fake_news_detection.py` | Full Python code (ML pipeline)         |
| `fake_news_model.pkl`  | Saved Logistic Regression model         |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer                 |

---

 Sample Output

```python
Model Accuracy: 0.9873

Confusion Matrix:
[[4670   63]
 [  51 4196]]
added README.md file
