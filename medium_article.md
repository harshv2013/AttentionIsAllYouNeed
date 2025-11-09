# 🧠 Attention Is All You Need — Building a Mini Transformer from Scratch in TensorFlow

![Transformer Architecture](https://via.placeholder.com/1200x400/0066cc/ffffff?text=Attention+Is+All+You+Need)

*Learn how to build a working Transformer model from scratch — no libraries, no shortcuts, just pure understanding.*

---

## 🚀 Introduction

Transformers **revolutionized deep learning**.

They replaced recurrent neural networks (RNNs) and convolutional architectures for NLP, computer vision, and even generative AI. From GPT to BERT to Stable Diffusion — Transformers are everywhere.

But **how do they really work inside?**

To answer that question, I built a **Mini Transformer model** — from scratch, using TensorFlow and Keras — inspired by the original ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper by Vaswani et al. (2017).

This post walks through:
- ✅ How I built it
- ✅ What I learned
- ✅ How you can try it yourself

Even if you're just starting with Transformers, this guide will give you hands-on intuition.

---

## 🧩 What is a Transformer?

At its core, a Transformer processes input sequences (like sentences) using an operation called **Self-Attention** — a way for the model to "look at" different parts of the sequence to decide which words are most relevant to each other.

### Example:

> *"The animal didn't cross the street because **it** was too tired"*

The model learns that "**it**" refers to "**animal**" — not "street" — via self-attention.

**No recurrence. No convolution.**  
**Just Attention.**

This is what makes Transformers so powerful and efficient at capturing long-range dependencies.

---

## 💡 Project Overview

I created a **Mini Transformer Encoder** — small enough to train on a toy dataset, but powerful enough to demonstrate all the key ideas:

- ✅ **Tokenization**
- ✅ **Positional Encoding**
- ✅ **Multi-Head Self-Attention**
- ✅ **Feed-Forward Network**
- ✅ **Residual Connections + Layer Normalization**
- ✅ **Masked Loss & Accuracy**

This isn't just theory — it's a working model that predicts the next word in a sentence.

---

## 🧠 The Dataset

To keep it simple and interpretable, I trained on a **tiny toy corpus** of short sentences:

```python
corpus = [
    "i love deep learning",
    "i love artificial intelligence",
    "deep learning is fun",
    "artificial intelligence is cool",
    "i love models",
    "models learn patterns"
]
```

### The Goal:

**Predict the next word in a sentence.**

**Example:**

```
Input:  "i love deep"
Target: "learning"
```

Think of it as a **"baby GPT"** that learns basic word associations through context.

---

## 🔧 Step 1: Tokenization and Input Preparation

First, we need to convert text into numbers that the model can understand.

I used Keras's `Tokenizer` to convert words into integer tokens:

```python
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters='', 
    lower=True, 
    oov_token='<OOV>'
)
tokenizer.fit_on_texts(corpus)
```

Each sentence becomes a **sequence of integers**.

Then, we create **(input → target)** pairs for next-word prediction and **pad** them to equal length.

### Example Transformation:

```
Sentence: "i love deep learning"

Input sequences:     Target words:
[i]                  → love
[i, love]            → deep
[i, love, deep]      → learning
```

This is how the model learns to predict what comes next!

---

## 🧮 Step 2: Positional Encoding

Here's a critical insight:

**Transformers don't have a built-in sense of order.**

Unlike RNNs that process sequences step-by-step, Transformers see all tokens at once. So we need to **inject position information** into the embeddings.

We do this using **positional encoding** with sine and cosine functions:

```python
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    
    sines = np.sin(angle_rads[:, 0::2])
    coses = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = np.zeros(angle_rads.shape)
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = coses
    
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
```

This gives each token a **unique signature** based on its position in the sentence.

The beauty? Positions that are close together have similar encodings, and the model can learn relative positions naturally.

---

## ⚙️ Step 3: Scaled Dot-Product Attention

Here's where the **magic** happens.

Given **Q** (query), **K** (key), and **V** (value) matrices, attention computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### What does this mean?

1. **QK^T** — Measures similarity between all pairs of tokens
2. **Divide by √d_k** — Prevents values from getting too large
3. **Softmax** — Converts scores into probabilities
4. **Multiply by V** — Aggregates information from relevant tokens

### Intuitive Example:

When processing the word "**love**", the attention mechanism might assign high weights to "**i**" and "**deep**" but low weights to "**artificial**".

This is how the model learns context!

---

## 🧠 Step 4: Multi-Head Self-Attention

Instead of computing attention once, we compute it **multiple times in parallel** — each time with different learned projection matrices.

This is called **Multi-Head Attention**.

```python
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.Wq = tf.keras.layers.Dense(d_model)
        self.Wk = tf.keras.layers.Dense(d_model)
        self.Wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        # Split into multiple heads
        # Compute scaled dot-product attention
        # Concatenate heads
        # Final linear projection
        ...
```

### Why multiple heads?

Each head learns a **different "view"** of relationships:

- Some focus on **local context** (nearby words)
- Others capture **long-term dependencies** (distant words)
- Some learn **syntactic relationships** (grammar)
- Others learn **semantic relationships** (meaning)

---

## 🧱 Step 5: The Transformer Encoder Layer

Each encoder layer is a carefully designed block that contains:

1. **Multi-head self-attention**
2. **Feed-forward neural network**
3. **Residual connections** (skip connections)
4. **Layer normalization**

```python
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
```

### Why residual connections?

They allow gradients to flow directly through the network, making it possible to stack many layers without vanishing gradients.

The formula: `output = LayerNorm(x + Sublayer(x))`

This is the secret sauce that enables training very deep networks!

---

## 🧩 Step 6: Training Setup

I used a custom training loop with `tf.GradientTape` for maximum control:

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions, _ = model(inputs, training=True)
        loss = masked_loss(targets, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss
```

### Key Implementation Details:

- **Masked Loss** — Ignores padded positions when calculating loss
- **Masked Accuracy** — Only evaluates predictions on real tokens
- **1000 Epochs** — Small dataset needs more iterations to learn patterns

---

## 🎯 Step 7: Inference

Once trained, you can feed the model partial sentences and watch it predict:

```python
test_sentence = "i love deep"

# Model processes:
# 1. Tokenize → [1, 2, 3]
# 2. Pad → [1, 2, 3, 0, 0]
# 3. Embed + Positional Encoding
# 4. Pass through Transformer layers
# 5. Output logits → softmax → prediction

Predicted next word: learning ✅
```

### What the Model Learned:

```
"i love deep"                    → "learning"
"artificial intelligence"        → "is"
"models learn"                   → "patterns"
```

Even with a tiny dataset, the model captures **meaningful associations**!

---

## 📊 Step 8: Attention Visualization

One of the coolest things about Transformers? You can **visualize what they're paying attention to**.

```python
# Extract attention weights from a specific layer and head
layer_to_plot = 0
head_to_plot = 0

_, all_attn = model(sample_input, training=False)
attn_matrix = all_attn[layer_to_plot].numpy()[0, head_to_plot]

# Visualize
plt.imshow(attn_matrix, cmap='viridis')
plt.title(f'Layer {layer_to_plot+1} Head {head_to_plot+1} Attention')
plt.xlabel('Key position')
plt.ylabel('Query position')
plt.colorbar()
plt.show()
```

This produces a **heatmap** showing which tokens attend to which other tokens.

### What You'll See:

- Diagonal patterns (tokens attending to themselves)
- Clusters of attention (related words focusing on each other)
- Different patterns in different heads (diverse learning)

It's beautiful to see the model's "thought process" visualized!

---

## 🧩 What I Learned

Building this from scratch gave me **deep intuition** about:

### 1. **Why Attention Scales Better Than Recurrence**
RNNs process sequentially, creating bottlenecks. Attention processes everything in parallel, making it faster and better at capturing long-range dependencies.

### 2. **How Positional Encoding Works**
Without position information, "love deep learning" and "learning deep love" would be identical to the model. Positional encoding solves this elegantly.

### 3. **Multi-Head Attention is Powerful**
Each head learns a unique aspect of context — syntax, semantics, local patterns, global patterns. Together, they create a rich representation.

### 4. **Residual Connections Are Essential**
They stabilize training and allow information to flow freely through many layers without degradation.

### 5. **From Logits to Predictions**
Understanding how raw model outputs (logits) are converted to probabilities (softmax) and finally to predicted tokens is crucial for debugging and improvement.

---

## 💡 Key Takeaway

**It's one thing to read about Transformers.**

**It's another level to build one yourself.**

When you implement each component by hand — tokenization, attention, residuals, normalization — you develop an intuition that no amount of reading can provide.

---

## 🔗 GitHub Repository

Want to explore the full code and experiment yourself?

👉 **[GitHub: AttentionIsAllYouNeed (Mini Transformer)](https://github.com/yourusername/AttentionIsAllYouNeed)**

The repository includes:
- Complete implementation with detailed comments
- Jupyter notebook for interactive learning
- Attention visualization utilities
- Training and inference examples

**Clone it, run it, modify it, break it, fix it — that's how you learn!**

---

## 💬 Final Thoughts

This project is **small**, but it builds the **foundation** for understanding large-scale LLMs like GPT, BERT, and T5.

Once you understand:
- Tokenization
- Attention mechanisms
- Residual learning
- Layer normalization

**You can understand any large model** — because at their core, they all use these same building blocks.

The difference between a "Mini Transformer" and GPT-4? Mostly just:
- More layers
- More attention heads
- Bigger embeddings
- Massive datasets
- Computational resources

But the **fundamental architecture**? It's the same.

👉 **[GitHub: AttentionIsAllYouNeed (Mini Transformer)](https://github.com/harshv2013/AttentionIsAllYouNeed)**

---

## 🧑‍💻 About the Author

**Harsh Vardhan**  
*AI/ML Engineer | Deep Learning Enthusiast | Researching Transformers, RAG, and Generative AI*

I'm passionate about making complex AI concepts accessible through hands-on implementations and clear explanations.

📍 Follow me on [LinkedIn](https://www.linkedin.com/in/harsh-vardhan-60b6aa106/)  
💻 Check out my [GitHub](https://github.com/harshv2013) for more projects  
📧 Contact: harsh2013@gmail.com

---

## 📚 References

1. Vaswani, A., et al. (2017). ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). *NeurIPS 2017*
2. [TensorFlow Official Documentation](https://www.tensorflow.org)
3. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — Harvard NLP
4. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Christopher Olah
5. [Jay Alammar's Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

---

## ❤️ Thanks for Reading!

If you enjoyed this deep dive into Transformers, please:

👏 **Clap** (up to 50 times!)  
💬 **Comment** with your thoughts or questions  
🔄 **Share** so more learners can understand Transformers from scratch  
➕ **Follow** for more deep learning tutorials

---

*Building from scratch is the best way to truly understand. What will you build next?*

---

### Tags
`#MachineLearning` `#DeepLearning` `#Transformers` `#NLP` `#TensorFlow` `#AI` `#ArtificialIntelligence` `#Python` `#DataScience` `#NeuralNetworks`