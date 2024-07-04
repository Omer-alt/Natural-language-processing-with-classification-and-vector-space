# Natural-language-processing-with-classification-and-vector-space


## 1. Lab1
### 1.1 Naive Bayes Implementation

The Naive Bayes classifier uses the following formula to calculate the probability of a label \( C \) given a sentence \( S \):

$$ P(C|S) = P(C) \cdot \prod_{i=1}^{n} P(w_i|C) $$

Where:

- \( P(C|S) \) is the posterior probability of label \( C \) given the sentence \( S \).
- \( P(C) \) is the prior probability of label \( C \).
- \( P(w_i|C) \) is the likelihood of word \( w_i \) given the label \( C \).

To handle unseen words, Laplacian smoothing is applied to the likelihood calculation:

$$ P(w_i|C) = \frac{\text{count}(w_i, C) + \mu}{\text{count}(C) + \mu \cdot |V|} $$

Where:

- $ \text{count}(w_i, C) $ is the number of times word $ w_i  $ appears in sentences with label \( C \).
- $\text{count}(C)$ is the total number of words in sentences with label \( C \).
- $\mu$ is the smoothing parameter.
- $|V|$ is the size of the vocabulary (the number of unique words in the training data).


#### Impact of Mu ($\mu$) on Validation Accuracy

The smoothing parameter $\mu$ impacts the validation accuracy by addressing the issue of zero probabilities for unseen words. A higher $\mu$ value increases the likelihood of rare or unseen words, which can prevent overfitting but may lead to underfitting if set too high. Conversely, a lower $\mu$ value may overfit the model to the training data, as it will not sufficiently account for unseen words. Therefore, selecting an appropriate $\mu$ value is crucial for balancing the bias-variance tradeoff and achieving optimal validation accuracy.

|         $\mu$   |Validation acuracy| 
|-----------------|------------------|
|       0.1       |    80.9         | 
|       1.0       |    80.9          | 
|       10.0      |    73.9          | 

Tab1: Impact of $\mu$ on Validation Accuracy (Training dataset: train2.txt, Validation dataset: valid2.txt )


### 1.2 Logistic Regression Model
Created a logistic regression model for classification tasks, specifically to implement a language identifier that accurately distinguishes between different languages.

#### Softmax Function

The softmax function is utilized to compute probabilities over multiple classes (languages in this case). For a vector of scores \( x \), the softmax function calculates the probability \( \sigma(x)_i \) for each class \( i \) as follows:

$ \sigma(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}} $

where:
- $ e $ denotes the exponential function,
- $ x_i $ is the score for class \( i \),
- $ \max(x) $ is the maximum score in vector \( x \),
- $ \sum_j e^{x_j - \max(x)} $ sums over all classes.

The softmax function ensures that the output probabilities sum to 1, making it suitable for multiclass classification tasks. Subtracting $ \max(x) $ improves numerical stability by preventing large exponentiation results, which could lead to overflow or loss of precision.

#### Loss Function

The loss function used here is the negative log likelihood: `Logistic Loss`

$$ \text{loss} = -\sum_{i=1}^{N}   \log(1 + \exp(w^\top x_i))  $$


where:
- $ x_i $ is the feature vector for example \( i \).
- $ w $ are the model weights.

## 2. Lab2: Word Translation Using Vector Space

This project implements various operations on word vectors, including cosine similarity computation, nearest neighbor search, analogy, association strength evaluation, WEAT test, word vector alignment, and translation between languages using word embeddings.

#### Cosine Similarity

Cosine similarity between two vectors \( u \) and \( v \) is defined as:

$$ \text{cosine}(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|} $$

where:
- $ \|u\| $ and $ \|v\| $ denote the Euclidean norms of vectors \( u \) and \( v \), respectively.

#### nearest neighbor search: Using a priority queue such as heapq 

This approach ensures efficient computation of nearest neighbors by maintaining a min-heap structure, allowing for rapid insertion of potential neighbors and quick access to the neighbor with the highest cosine similarity among those already added. This efficiency is crucial as it optimizes both time complexity—logarithmic time for push (heappush) and pop (heappop) operations—and memory usage, ensuring that only the top $ k $ neighbors are stored at any given time.

#### analogy
To find the analogies, we find the nearest neighbour associated with the wordvector d
$$ d = \frac{c}{\Vert {c} \Vert} + \frac{b}{\Vert {b} \Vert} - \frac{a}{\Vert {a} \Vert}$$

#### translation between languages: English and French
The `align` function computes a linear mapping between English and French word vectors based on a provided lexicon, using least squares regression `(np.linalg.lstsq)`.The translate function applies this mapping to predict the French translation of an English word by finding its nearest neighbor in the French word vector space.


## 3. Lab3: N-gram Model and Stupid Backoff 

The project features the implementation of an n-gram language model, enhanced with a stupid backoff mechanism to handle rare words (Out of Vocabulary words). This model, implemented through functions like build_ngram, calculates the probability of word sequences based on their contexts. The get_prob function handles the probability computation for words given their contexts, while perplexity evaluates how well the n-gram model predicts sequences from provided text data. This approach not only improves prediction accuracy but also robustly manages unseen or infrequent words encountered during language modeling tasks.












### Build With

**Language:** Python

**Package:**  numpy, heapq, tqdm, Pytorch

### Authors

- [@Fotso Omer](https://portfolio-omer-alt.vercel.app/)

### License

[MIT](https://choosealicense.com/licenses/mit/)