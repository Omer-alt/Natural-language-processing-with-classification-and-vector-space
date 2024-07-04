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

## 1. Lab2: Word Translation Using Vector Space


## 1. Lab3: N-gram Model and Stupid Backoff 

Constructed an n-gram language model, incorporating a stupid backoff mechanism to handle rare words (Out of Vocabulary words), thus improving the model's ability to predict sequences of words accurately.





### Build With

**Language:** Python

**Package:**  numpy, tqdm, Pytorch

### Authors

- [@Fotso Omer](https://portfolio-omer-alt.vercel.app/)

### License

[MIT](https://choosealicense.com/licenses/mit/)