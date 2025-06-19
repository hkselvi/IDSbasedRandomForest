This project presents a simple yet effective **Intrusion Detection System (IDS)** that detects anomalies and cyber attacks in network traffic using the **Random Forest** machine learning algorithm. The system is trained and tested on the **NSL-KDD dataset**, a popular benchmark for intrusion detection research.

## 📌 Features

- Supervised ML-based attack classification
- Uses `RandomForestClassifier` from scikit-learn
- Dataset preprocessing including encoding and splitting
- Accuracy, precision, recall, and F1-score evaluation
- Easily extendable for real-time IDS or other models (SVM, Isolation Forest, etc.)

## 📁 Dataset

- **NSL-KDD Dataset**
- Download from: [Kaggle - NSL-KDD Dataset](https://www.kaggle.com/datasets/harmannahal/nsl-kdd-dataset)
- Files used:
  - `KDDTrain_filtered.csv` (for training)

## ⚙️ Requirements

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies:

- Python 3.x
- pandas
- scikit-learn
- seaborn (optional, for visualization)
- matplotlib (optional)

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/IDSbasedRandomForest.git
cd IDSbasedRandomForest
```

2. Place the NSL-KDD dataset files in the same directory.

3. Run the application:

```bash
python application.py
```

## 📊 Sample Output

```
Accuracy: 0.98
Precision: 0.97
Recall: 0.96
F1-score: 0.965
```

Optional visualizations (confusion matrix, feature importance) can be enabled in the script.

## 🧠 Model Explanation

The Random Forest algorithm is a robust ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction. It handles high-dimensional data and class imbalance effectively, which is essential for intrusion detection.

## 📄 Report

The research report includes:
- System design
- ML methodology
- Evaluation metrics
- Discussion of results
- Citations from relevant academic studies

## 📚 References

1. [Computers Journal, 2025](https://doi.org/10.3390/computers14030087)
2. [Computer & Security, 2025](https://doi.org/10.1016/j.cose.2025.104542)
3. [Journal of Network and Computer Applications, 2024](https://doi.org/10.1016/j.jnca.2024.103868)

## 🙌 Acknowledgements

- Dataset provided via Kaggle.
- Project developed as part of academic coursework.
