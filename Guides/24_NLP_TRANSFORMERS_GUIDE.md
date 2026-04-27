# Natural Language Processing and Transformers Guide

NLP enables machines to understand and generate human language. This guide covers text preprocessing, classical NLP methods, and modern transformer-based approaches as used in Weeks 2-4 of the course.

**Table of Contents**

1. [Text Preprocessing](#1-text-preprocessing)
2. [Bag of Words and TF-IDF](#2-bag-of-words-and-tf-idf)
3. [Word Embeddings Overview](#3-word-embeddings-overview)
4. [Transformer Architecture (Simplified)](#4-transformer-architecture-simplified)
5. [HuggingFace Transformers Library](#5-huggingface-transformers-library)
6. [Common NLP Tasks](#6-common-nlp-tasks)
7. [Fine-Tuning Pre-trained Models](#7-fine-tuning-pre-trained-models)
8. [Working with the Spam Dataset](#8-working-with-the-spam-dataset)
9. [Quick Reference Tables](#9-quick-reference-tables)
10. [Resources](#10-resources)

---

## 1. Text Preprocessing

Raw text must be cleaned and standardized before feeding it to a model.

### 1.1 Basic Text Cleaning

```python
import re

text = "  Hello World!!! Check out https://example.com for ML resources.  "

# Lowercase
text = text.lower()                              # "  hello world!!! check out..."

# Remove URLs
text = re.sub(r'http\S+|www\.\S+', '', text)    # Remove URLs

# Remove special characters (keep letters and spaces)
text = re.sub(r'[^a-zA-Z\s]', '', text)         # "hello world check out for ml resources"

# Remove extra whitespace
text = ' '.join(text.split())                     # Clean up spaces
```

### 1.2 Tokenization

Splitting text into individual words or subwords.

```python
# Simple split (basic)
tokens = text.split()
# ['hello', 'world', 'check', 'out', 'for', 'ml', 'resources']

# NLTK tokenizer (handles punctuation better)
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize

tokens = word_tokenize("It's a beautiful day!")
# ["It", "'s", "a", "beautiful", "day", "!"]
```

### 1.3 Stopword Removal

**Stopwords** are common words (the, is, at, which) that add little meaning.

```python
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

tokens = ['machine', 'learning', 'is', 'a', 'subset', 'of', 'artificial', 'intelligence']
filtered = [word for word in tokens if word not in stop_words]
# ['machine', 'learning', 'subset', 'artificial', 'intelligence']
```

### 1.4 Stemming vs Lemmatization

Both reduce words to their base form, but differently.

| Method | How It Works | Example | Result |
|--------|-------------|---------|--------|
| Stemming | Chops off word endings (crude) | "running", "runs", "ran" | "run", "run", "ran" |
| Lemmatization | Uses dictionary to find root (accurate) | "running", "runs", "ran" | "run", "run", "run" |

```python
# Stemming (faster, less accurate)
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ['running', 'runs', 'easily', 'fairly']
stemmed = [stemmer.stem(word) for word in words]
# ['run', 'run', 'easili', 'fairli']  -- note: not always real words

# Lemmatization (slower, more accurate)
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ['running', 'runs', 'easily', 'fairly']
lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in words]
# ['run', 'run', 'easily', 'fairly']
```

### 1.5 Complete Text Preprocessing Function

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and preprocess a text string."""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply to a DataFrame column
df['clean_text'] = df['text'].apply(preprocess_text)
```

---

## 2. Bag of Words and TF-IDF

### 2.1 Bag of Words (CountVectorizer)

Converts text into a **matrix of word counts**. Each document becomes a vector of how many times each word appears.

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "machine learning is great",
    "deep learning is a subset of machine learning",
    "natural language processing"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print(f"Vocabulary: {vectorizer.get_feature_names_out()}")
print(f"Matrix shape: {X.shape}")  # (3 documents, N unique words)
print(X.toarray())
```

### 2.2 TF-IDF (Term Frequency-Inverse Document Frequency)

Like Bag of Words but **reduces the weight of common words** and increases the weight of rare, informative words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(texts)

print(f"Shape: {X.shape}")
print(f"Feature names (first 10): {tfidf.get_feature_names_out()[:10]}")
```

**Key parameters:**

| Parameter | What It Does | Typical Value |
|-----------|-------------|---------------|
| `max_features` | Limit vocabulary size | 5000-10000 |
| `stop_words` | Remove common words | `'english'` |
| `ngram_range` | Include word pairs/triples | `(1, 2)` for unigrams + bigrams |
| `min_df` | Minimum document frequency | 2 (ignore very rare words) |
| `max_df` | Maximum document frequency | 0.95 (ignore words in >95% of docs) |

### 2.3 TF-IDF + Classifier Pipeline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Create pipeline
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train and evaluate
text_pipeline.fit(X_train, y_train)
y_pred = text_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 3. Word Embeddings Overview

### 3.1 What Are Word Embeddings?

Word embeddings represent words as **dense vectors** where similar words have similar vectors. Unlike Bag of Words (sparse, high-dimensional), embeddings are compact (typically 100-768 dimensions).

```
"king"  -> [0.2, -0.1, 0.8, 0.3, ...]  (300 dimensions)
"queen" -> [0.3, -0.2, 0.7, 0.4, ...]  (similar vector)
"car"   -> [-0.5, 0.9, -0.1, 0.2, ...] (very different)
```

### 3.2 Popular Embedding Methods

| Method | Year | Key Idea |
|--------|------|----------|
| Word2Vec | 2013 | Predict word from context (or vice versa) |
| GloVe | 2014 | Global word co-occurrence statistics |
| FastText | 2016 | Includes subword information |
| BERT | 2018 | Context-dependent (same word, different meanings) |

### 3.3 Sentence Transformers

For getting embeddings of entire sentences (not just individual words).

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    'Machine learning is great',
    'Deep learning uses neural networks',
    'I like pizza'
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  # (3, 384)

# Compare similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(embeddings)
print(f"Similarity between sentence 0 and 1: {similarity[0][1]:.3f}")
print(f"Similarity between sentence 0 and 2: {similarity[0][2]:.3f}")
```

---

## 4. Transformer Architecture (Simplified)

### 4.1 Key Concepts

The **Transformer** (2017) is the architecture behind modern NLP. Its key innovation is **self-attention**: the ability to look at all words in a sentence simultaneously and learn which words are most relevant to each other.

**Why transformers replaced RNNs:**

| Feature | RNN/LSTM | Transformer |
|---------|----------|-------------|
| Processing | Sequential (word by word) | Parallel (all words at once) |
| Long-range dependencies | Struggles with long texts | Handles easily via attention |
| Training speed | Slow (sequential) | Fast (parallelizable) |
| Modern use | Mostly replaced | Standard for NLP |

### 4.2 Encoder vs Decoder

| Component | What It Does | Models That Use It |
|-----------|-------------|-------------------|
| **Encoder** | Understands/encodes input text | BERT, RoBERTa |
| **Decoder** | Generates output text | GPT, Llama |
| **Both** | Translates / transforms text | T5, BART |

### 4.3 Common Transformer Models

| Model | Type | Parameters | Best For |
|-------|------|-----------|----------|
| **BERT** | Encoder | 110M-340M | Classification, NER, Q&A |
| **GPT-2/3/4** | Decoder | 124M-175B | Text generation |
| **T5** | Encoder-Decoder | 60M-11B | Translation, summarization |
| **RoBERTa** | Encoder | 125M-355M | Better BERT (more training) |
| **Llama 2/3** | Decoder | 7B-70B | Open-source text generation |
| **DistilBERT** | Encoder | 66M | Fast BERT (60% faster, 97% performance) |

---

## 5. HuggingFace Transformers Library

### 5.1 Pipeline API (Easiest Way)

The `pipeline` function provides a high-level API for common NLP tasks with just one line.

```python
from transformers import pipeline

# Sentiment analysis
sentiment = pipeline('sentiment-analysis')
result = sentiment('This machine learning course is really helpful!')
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Analyze multiple texts
texts = ['I love this!', 'This is terrible.', 'It was okay.']
results = sentiment(texts)
for text, res in zip(texts, results):
    print(f"{text:30s} -> {res['label']} ({res['score']:.3f})")
```

### 5.2 Tokenizers

Tokenizers convert text to numbers that models can process.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize
text = "Machine learning is powerful"
tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

print(f"Input IDs: {tokens['input_ids']}")
print(f"Tokens:    {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
# ['[CLS]', 'machine', 'learning', 'is', 'powerful', '[SEP]']
```

### 5.3 Loading Models

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# For text generation
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('gpt2')
```

---

## 6. Common NLP Tasks

### 6.1 Text Classification

```python
classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
result = classifier('This product is amazing and works perfectly!')
print(result)  # [{'label': 'POSITIVE', 'score': 0.9999}]
```

### 6.2 Named Entity Recognition (NER)

Identifies people, organizations, locations, and other entities in text.

```python
ner = pipeline('ner', grouped_entities=True)
result = ner('Elon Musk founded SpaceX in Los Angeles')
for entity in result:
    print(f"{entity['word']:20s} -> {entity['entity_group']} ({entity['score']:.3f})")
# Elon Musk            -> PER (0.998)
# SpaceX               -> ORG (0.997)
# Los Angeles          -> LOC (0.999)
```

### 6.3 Text Summarization

```python
summarizer = pipeline('summarization')

long_text = """
Machine learning is a subset of artificial intelligence that focuses on building
systems that learn from data. Instead of being explicitly programmed, these systems
improve their performance through experience. There are three main types: supervised
learning, unsupervised learning, and reinforcement learning.
"""

summary = summarizer(long_text, max_length=50, min_length=20)
print(summary[0]['summary_text'])
```

### 6.4 Question Answering

```python
qa = pipeline('question-answering')
result = qa(
    question='What is machine learning?',
    context='Machine learning is a subset of AI that learns from data.'
)
print(f"Answer: {result['answer']} (confidence: {result['score']:.3f})")
```

### 6.5 Zero-Shot Classification

Classify text into categories **without training on those categories**.

```python
classifier = pipeline('zero-shot-classification')
result = classifier(
    'The stock market crashed today',
    candidate_labels=['politics', 'finance', 'sports', 'technology']
)
for label, score in zip(result['labels'], result['scores']):
    print(f"{label:15s} {score:.3f}")
```

---

## 7. Fine-Tuning Pre-trained Models

### 7.1 When to Fine-Tune

- Your task is **domain-specific** (medical, legal, financial text)
- Pre-trained models don't perform well on your data
- You have enough labeled data (hundreds to thousands of examples)

### 7.2 Fine-Tuning Workflow

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Prepare data
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Convert pandas DataFrame to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 2. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)

# 3. Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    save_strategy='epoch',
    load_best_model_at_end=True
)

# 4. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
```

### 7.3 Fine-Tuning Tips

| Parameter | Recommended Value | Why |
|-----------|------------------|-----|
| Learning rate | 2e-5 to 5e-5 | Pre-trained weights need gentle updates |
| Epochs | 2-5 | More can overfit on small datasets |
| Batch size | 8-32 | Limited by GPU memory |
| Max length | 128-512 | Depends on text length, affects memory |

---

## 8. Working with the Spam Dataset

### 8.1 Loading and Exploring

```python
import pandas as pd

df = pd.read_csv('../Datasets/Spam_Ham_Dataset.csv')
print(f"Shape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"\nSample messages:")
print(df.head())
```

### 8.2 Classical Approach (TF-IDF + Classifier)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Adjust column names to match your dataset
X = df['text']       # or 'message', 'content', etc.
y = df['label']      # or 'category', 'class', etc.

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 8.3 Transformer Approach (Zero-Shot)

Classify without any training using a pre-trained model.

```python
from transformers import pipeline

classifier = pipeline('zero-shot-classification')

# Test on a few examples
sample_texts = X_test.head(5).tolist()
for text in sample_texts:
    result = classifier(text, candidate_labels=['spam', 'ham'])
    print(f"Text: {text[:50]}...")
    print(f"  Prediction: {result['labels'][0]} ({result['scores'][0]:.3f})")
```

---

## 9. Quick Reference Tables

### 9.1 Text Preprocessing Steps

| Step | Tool | Code |
|------|------|------|
| Lowercase | Python | `text.lower()` |
| Remove URLs | regex | `re.sub(r'http\S+', '', text)` |
| Remove special chars | regex | `re.sub(r'[^a-zA-Z\s]', '', text)` |
| Tokenize | split/NLTK | `text.split()` |
| Remove stopwords | NLTK | `[w for w in tokens if w not in stop_words]` |
| Stemming | NLTK | `PorterStemmer().stem(word)` |
| Lemmatization | NLTK | `WordNetLemmatizer().lemmatize(word)` |

### 9.2 sklearn Text Vectorizers

| Vectorizer | What It Produces | When to Use |
|-----------|-----------------|-------------|
| CountVectorizer | Word count matrix | Simple baseline |
| TfidfVectorizer | TF-IDF weighted matrix | Better than counts (standard choice) |
| HashingVectorizer | Fixed-size hash matrix | Very large vocabularies |

### 9.3 HuggingFace Pipeline Tasks

| Task | Pipeline String | Example |
|------|----------------|---------|
| Sentiment | `'sentiment-analysis'` | Positive/Negative |
| Classification | `'text-classification'` | Custom categories |
| NER | `'ner'` | People, places, organizations |
| Summarization | `'summarization'` | Shorten long text |
| Q&A | `'question-answering'` | Answer from context |
| Translation | `'translation_en_to_fr'` | Language translation |
| Text generation | `'text-generation'` | Continue a prompt |
| Zero-shot | `'zero-shot-classification'` | Classify without training |

### 9.4 Common Pre-trained Models

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| distilbert-base-uncased | 66M | Fast | Quick classification tasks |
| bert-base-uncased | 110M | Medium | General NLP tasks |
| roberta-base | 125M | Medium | Better accuracy than BERT |
| gpt2 | 124M | Medium | Text generation |
| all-MiniLM-L6-v2 | 22M | Fast | Sentence embeddings |

---

## 10. Resources

- [HuggingFace Documentation](https://huggingface.co/docs)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Sentence Transformers Documentation](https://www.sbert.net/)

---

**For most NLP tasks, start with the HuggingFace pipeline API -- you can get impressive results in just a few lines of code!**

---

[← Previous: Computer Vision](23_COMPUTER_VISION_GUIDE.md) | [Index](README.md) | [Next: Generative AI →](25_GENERATIVE_AI_GUIDE.md)
