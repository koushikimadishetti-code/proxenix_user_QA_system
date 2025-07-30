# proxenix_user_QA_system
# 🤖 Question Answering System using BERT and SQuAD

This project aims to build a machine learning model capable of answering questions based on a given context using the **BERT transformer architecture**. We use the **SQuAD v1.1 (Stanford Question Answering Dataset)** for training and evaluation.

---

## 📚 Dataset

We use the **SQuAD v1.1** dataset from Hugging Face's `datasets` library, which consists of:

- ~87,000 training question-context-answer triples
- ~10,000 validation triples

Each example includes:
- A paragraph (context)
- A question based on the paragraph
- The answer span from the paragraph

---

## 🚀 Features

- Uses `bert-base-uncased` pre-trained model from Hugging Face
- Fine-tunes BERT on SQuAD dataset for span-based QA
- Includes custom question answering interface (CLI)
- Supports saving and loading trained models
- Evaluation using F1 score and Exact Match (optional)

---

## 🛠️ Installation
Install dependencies:
pip install transformers datasets pandas sklearn torch

##🙌 Acknowledgments
- Hugging Face 🤗 Transformers & Datasets

- SQuAD Dataset from Stanford NLP

- Google Colab for free GPU runtime
