
# LDA
Linear Discriminant Analysis (LDA) is a statistical technique used for dimensionality reduction, as well as for classification purposes. At a high level, LDA aims to project a dataset onto a lower-dimensional space with the goal of maximizing the separation between multiple classes. This makes LDA particularly useful in preparing data for classification tasks, enhancing the performance of classifiers by reducing the complexity of the data.

The core idea behind LDA is to find a linear combination of features that characterizes or separates two or more classes of objects or events. The method does this by drawing a decision boundary between the different classes in such a way that the distance between the means of the classes is maximized while minimizing the variance within each class. This criterion is known as the Fisher’s criterion, named after Ronald Fisher who introduced the method.

High-Level Breakdown

1. **Compute the within-class and between-class scatter matrices:** The within-class scatter matrix measures how spread out the classes are internally, while the between-class scatter matrix measures the separation between different classes.

2. **Calculate eigenvectors and eigenvalues:** This step involves solving an eigenvalue problem for the matrix derived from the within-class and between-class scatter matrices. The eigenvectors represent the directions in the new feature space, and the eigenvalues give the magnitude of variance in those directions.

3. **Select the top k eigenvectors:** Based on the eigenvalues, the top k eigenvectors are selected to form a new subspace, with k typically chosen based on the desired number of dimensions in the reduced space or based on the eigenvalues themselves.

4. **Project the data:**: Finally, the original data is projected onto this new subspace, resulting in a lower-dimensional representation of the dataset that maintains the class discriminatory information.

LDA is particularly effective when the assumptions of normality and equal covariance matrices across groups are met, though it can be robust to some deviations from these assumptions. It's widely used in various fields such as pattern recognition, machine learning, and biometrics for tasks like face recognition and voice classification.

# Bayes' Prediction Rule

Bayes' prediction rule in the context of Linear Discriminant Analysis (LDA) is used to assign a new observation to one of several predefined classes based on the probabilities that the observation falls into each class. This rule is derived from Bayes' theorem, which provides a way to update our prior beliefs based on new evidence.

For LDA, the goal is to classify a new observation $x$ into one of $K$ classes $(C_{1}, C_{2}, \ldots, C_{K})$ based on the discriminant functions computed for each class. These discriminant functions are essentially linear combinations of the features in $x$, designed to separate the classes as clearly as possible.

Bayes' prediction rule states that an observation $x$ should be assigned to the class $C_{K}$ for which the posterior probability $\mathbb{P}(C_{k} | x)$ is highest. The posterior probability can be expressed using Bayes' Theorem as follows:

$$\mathbb{P}(C_{k} | x) = \frac{\mathbb{P}(x | C_{k}) \mathbb{P}(C_{k})}{\mathbb{P}(x)}$$

where:

- $\mathbb{P}(C_{k} | x)$ is the posterior probability of class $C_{k}$ given observation $x$.
- $\mathbb{P}(x | C_{k})$ is hte likelihood of observing $x$ given that it belongs to class $C_{k}$.
- $\mathbb{P}(C_{k})$ is the prior probability of class $C_{k}$, representing our initial belief about the distribution of classes before observing $x$.
- $\mathbb{P}(x)$ is the evidence, a normalization factor ensuring that the probabilities across all classes sum to $1$. It's calculated as $$\sum_{j = 1}^{K}\mathbb{P}(x | C_{j}) \mathbb{P}(C_{j})$$, though for classification purposes, this term can often be ignored since it does not affect the ranking of the posterior probabilities.


In LDA, the likelihoods $\mathbb{P}(x | C_{k})$ are typically assumed to be normally distributed with a mean specific to the class and a covariance matrix that is common across all classes. This simplification allows for the discriminant functions to be linear with respect to $x$.

Thus, the Bayes' prediction rule for LDA assigns a new observation $x$ to the class $C_{k}$ if and only if:

$$\delta_{k}(x) > \delta_{j}(x) \text{ for all } j \neq k$$

where $\delta_{k}(x)$ is the discriminant function for class $C_{k}$, calculated based on the log of the posterior probability $\mathbb{P}(C_{k} | x)$. The specific form of $\delta_{k}(x)$ takes into account the means of the classes, the chared covariance matrix, and the prior probabilities of the classes, reflecting the linear nature of LDA.

A couple other equivalent ways to state the discriminant function for LDA are:

- $\mathbb{P}(C_{k} | x) > \mathbb{P}(C_{j} | x) \forall j \neq k$

This expression states that an observation $x$ should be assigned to class $C_{k}$ if the posterior probability of $C_{k}$ given $x$ is greater than the posterior probabilities of all other classes. This is a direct application of Bayes' theorem, where the classification decision is based on which class has the highest posterior probability given the observation.

- $\hat{k}(x) = \text{arg max}_{k} \delta_{k}(x)$

This formulation specifies that the predicted class $\hat{k}(x)$ for an observation $x$ is the one for which the discriminant function $\delta_{k}(x)$ yields the highest value. In the context of LDA, the discriminant function is derived to reflect the log of the posterior probabilities, taking into account the class priors, the likelihood of the data given the class, and the evidence (the data itself). Thus, maximizing $\delta_{k}(x)$ is equivalent to choosing the class with the highest posterior probability.

- (note: already mentioned) Assign a new observation $x$ to the class $C_{k}$ if and only if:

    $\delta_{k}(x) > \delta_{j}(x) \text{ for all } j \neq k$

This statement is a specific case of the second formulation meant to clarify that we are not just finding the maximum of $\delta_{k}(x)$, but ensuring that this maximum is strictly greater than the discriminant values for all other classes.

## Variable Mapping

The following is a variable map that shows how the above information corresponds to the notation used in the [problem](#Problem-Statement) that follows after.

- $Y$ represents the class variable with possible outcomes $1$ (for class $C_{1}$) and $0$ (for class $C_{0}$)
- $\pi_{1} = \mathbb{P}(Y = 1)$ represents the prior probability of class $C_{1}$ and $1 - \pi_{1} = \mathbb{P}(Y = 0)$ represents the prior probability of class $C_{0}$.
- $\mathbf{X}$ represents the feature vector
- $\mathbb{P}(\mathbf{X} | Y = k) \sim \mathcal{N}(\mathbf{\mu}_{k}, \mathbf{\Sigma})$ indicates that the conditional distribution of $\mathbf{X}$ given class $Y = k$ follows a multivariate normal distribution with mean vector $\mathbf{\mu}_{k}$ and a covariance matrix $\mathbf{\Sigma}$, which is common across both classes.
- $f_{k}(x)$ corresponds to $\delta_{k}(x)$, the discriminant function.

The discriminant function for LDA can be expressed as (we'll derive later):

$$\delta_{k}(x) = \mathbf{x}^{\top}\mathbf{\sigma}^{-1}\mathbf{\mu}_{k} - \frac{1}{2}\mathbf{\mu}_{k}^{\top}\mathbf{\Sigma}^{-1}\mathbf{\mu}_{k} + log(\pi_{k})$$

To find discriminant functions for $f_{k}(x)$ for $k = 0, 1$ that are linear in $x$ and to express the decision rule $\hat{k}(x)$ in terms of these functions, we apply the LDA approach, which leverages the assumption of normally distributed classes with equal covariance matrices. 

The discriminant function for LDA after substituting relevant variables be expressed as:

$$f_{k}(x) = \mathbf{x}^{\top}\mathbf{\sigma}^{-1}\mathbf{\mu}_{k} - \frac{1}{2}\mathbf{\mu}_{k}^{\top}\mathbf{\Sigma}^{-1}\mathbf{\mu}_{k} + log(\pi_{k})$$

Since $\pi_{k}$ refers to the prior probability of class $k$, for $k = 1$, it is $\pi_{1}$, and for $k = 0$, it is $1 - \pi$.

The decision rule $\hat{k}(x) = \text{arg max}_{k}f_{k}(x)$ across the two classes, effectively clasifying $x$ into the class with the higher discriminant function value, which directly corresponds to the higher probability $\mathbb{P}(Y = k | \mathbf{X} = x)$ as per Bayes' rule. 

## Problem-Statement (Bayes' Prediction Rule for LDA)

Let the distribution of $(\boldsymbol{X}, Y)$ be $\mathbb{P}(Y = 1) = \pi_1$, $\mathbb{P}(Y = 0) = 1 - \pi_1$, and $\mathbb{P}(\boldsymbol{X} \mid Y = k) \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma})$. Here $\pi_1 \in [0, 1]$, $\boldsymbol{\mu}_1, \boldsymbol{\mu}_0 \in \mathbb{R}^d$ and $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$. Please find functions $f_k(\boldsymbol{x})$ for $k = 0, 1$ which are **linear** in $\boldsymbol{x}$ such that the decision rule $\hat{k}(\boldsymbol{x})$ which is defined as
$$
\hat{k}(\boldsymbol{x}) \equiv \arg\max_{k} \mathbb{P}(Y= k \mid \boldsymbol{X} = \boldsymbol{x})
$$
can be rewritten as 
$$
\hat{k}(\boldsymbol{x}) = \arg\max_k f_k(\boldsymbol{x}). 
$$

## Solution

### Bayes' Theorem:

$$\mathbb{P}(C_{k} | x) = \frac{\mathbb{P}(x | C_{k}) \mathbb{P}(C_{k})}{\mathbb{P}(x)}$$

we have:

$$\mathbb{P}(Y = k | \mathbf{X} = \mathbf{x}) = \frac{\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = k) \mathbb{P}(Y = k)}{\mathbb{P}(\mathbf{X} = \mathbf{x})}$$

Since $Y = k$ refers to the event that the class label is $k$, where $k$ indexes into the class --its equivalent to saying the class belongs to class $C_{k}$, for $k \in [0, 1]$.

Similarly, $\mathbf{X} = \mathbf{x}$ refers to the feature vector of the observation being $\mathbf{x} \in \mathbb{R}^{d}$.

where:

- $\mathbb{P}(Y = k | \mathbf{X} = \mathbf{x})$ is the posterior probability that the class label $Y$ is $k$ given the feature vector $X = \mathbf{x}$

- $\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = k)$ is the likelihood, which is the probability of observing the feature vector $\mathbf{X} = \mathbf{x}$ give that the class label is $k$

- $\mathbb{P}(Y = k)$ is the prior probability of class $k$, which is our belief about the frequency of class $k$ before observing the features.

- $\mathbf{P}(\mathbf{X} = \mathbf{x})$ is the evidence, which is the overall probability of observing the feature vector $\mathbf{X} = \mathbf{x}$. It acts as a normalizing constant to ensure that the posterior probabilities across all classes sum to $1$.

### Log of Posterior Odds:

In order for the classification descision to be made, we need to look at the log of posterior odds. The posterior odds compares the probabilities of two hypotheses (our class labels), given some evidence (the feature vector $\mathbf{x}$).

$$log \left( \frac{\mathbb{P}(Y = 1 | \mathbf{X} = \mathbf{x})}{\mathbb{P}(Y = 0 | \mathbf{X} = \mathbf{x})} \right)$$

Using Baye's theorem on each term we have

$$\mathbb{P}(Y = 1 | \mathbf{X} = \mathbf{x}) = \frac{\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 1)\mathbb{P}(Y = 1)}{\mathbb{P}(\mathbf{X} = \mathbf{x})}$$

and 

$$\mathbb{P}(Y = 0 | \mathbf{X} = \mathbf{x}) = \frac{\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 0)\mathbb{P}(Y = 0)}{\mathbb{P}(\mathbf{X} = \mathbf{x})}$$

thus, 

$$\frac{\mathbb{P}(Y = 1 | \mathbf{X} = \mathbf{x})}{\mathbb{P}(Y = 0 | \mathbf{X} = \mathbf{x})} = \frac{\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 1)\mathbb{P}(Y = 1)}{\mathbb{P}(\mathbf{X} = \mathbf{x})} \frac{\mathbb{P}(\mathbf{X} = \mathbf{x})}{\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 0)\mathbb{P}(Y = 0)}$$

and the log of posterior odds simplifies to 

$$log \left( \frac{\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 1)\mathbb{P}(Y = 1)}{\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 0) \mathbb{P}(Y = 0)} \right)$$

$$= log(\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 1) \mathbb{P}(Y = 1)) - log(\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 0) \mathbb{P}(Y = 0))$$

$$= log(\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 1)) + log(\mathbb{P}(Y = 1)) - [log(\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 0)) + log(\mathbb{P}(Y = 0))]$$

This expression is what we use to make a classification decision and its comprised of our two discriminant functions. In otherwords, it can be viewed as: 

$$f_{1}(\mathbf{x}) - f_{0}(\mathbf{x})$$

see why in [Substitution of Normal Distribution and Priors](#Substitution-of-Normal-Distribution-and-Priors)

A few advantages to using the log of the odds instead of the odds include:

1. Logarithms will transform our product of likelihood and priors into a sum, thus simplifying our calculations.

2. Probabilities in especially large datasets with many features or in cases of very unlikely events can become extremely small and working with this can lead to numerical underflow.

2. The log function is monotonic, meaning we preserve the order of the odds. 

### Interpretability (For Me)

**Positive Log Odds**

If we have:

$$log \left( \frac{\mathbb{P}(Y = 1 | \mathbf{X} = \mathbf{x})}{\mathbb{P}(Y = 0 | \mathbf{X} = \mathbf{x})} \right) > 0 $$

then:

$$\frac{\mathbb{P}(Y = 1 | \mathbf{X} = \mathbf{x})}{\mathbb{P}(Y = 0 | \mathbf{X} = \mathbf{x})} > 1$$

This is due to the exponential function is the inverse of the log function and is also monotonic. This means that the posterior probability of class $1$, given $\mathbf{X} = \mathbf{x}$ is greater than the posterior probability of class $0$, given $\mathbf{X} = \mathbf{x}$

$$\mathbb{P}(Y = 1 | \mathbf{X} = \mathbf{x}) > \mathbb{P}(Y = 0 | \mathbf{X} = \mathbf{x})$$

**Negative Log Odds**

If we have:

$$log \left( \frac{\mathbb{P}(Y = 1 | \mathbf{X} = \mathbf{x})}{\mathbb{P}(Y = 0 | \mathbf{X} = \mathbf{x})} \right) < 0 $$

then:

$$\frac{\mathbb{P}(Y = 1 | \mathbf{X} = \mathbf{x})}{\mathbb{P}(Y = 0 | \mathbf{X} = \mathbf{x})} < 1$$

Similar to before, this means that the posterior probability of class $1$, given $\mathbf{X} = \mathbf{x}$ is less than the posterior probability of class $0$, given $\mathbf{X} = \mathbf{x}$

$$\mathbb{P}(Y = 1 | \mathbf{X} = \mathbf{x}) < \mathbb{P}(Y = 0 | \mathbf{X} = \mathbf{x})$$

The positive log odds indicates that the evidence (feature vector $\mathbf{x}$) makes it more likely that the observation belongs to class $1$ rather than class $0$.

The negative log odds indicates that the evidence makes it more likely that the observation belongs to class $0$ rather than class $1$. 

All of the logic would "flip" if we were to change the numerator denominator terms from the beginning expession.

### Substitution of Normal Distribution and Priors

We're given that $\mathbb{P}(\mathbf{X} | Y = k) \sim \mathcal{N}(\mathbf{\mu}_{k}, \mathbf{\Sigma})$

The Gaussian density function for a multivariate normal distribution, where $\mathbf{X}$ is a $d$-dimensional feature vector is:

$$\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = k) = \frac{1}{(2\pi)^{d /2}|\mathbf{\Sigma}|^{1/2}}exp\left( - \frac{1}{2}(\mathbf{x} - \mathbf{\mu}_{k})^{\top} \mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu}_{k}) \right)$$

where:

- $\mathbf{x}$ is a feature vector for which we're calculating the likelihood.

- $\mathbf{\mu}_{k}$ is the mean vector of the features for class $k$.

- $\mathbf{\Sigma}$ is the common covariance matrix for all classes (by assumption of the LDA model)

- $|\mathbf{\Sigma}|$ is the determinant of the covariance matrix.

- $d$ is the number of dimensions (features) in $\mathbf{x}$

Now, we can substitute the normal distribution and priors back into:

$$f_{1}(\mathbf{x}) - f_{0}(\mathbf{x}) = log(\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 1)) + log(\mathbb{P}(Y = 1)) - [log(\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = 0)) + log(\mathbb{P}(Y = 0))]$$

The log of each the Gaussian terms become

$$= log \left( \frac{1}{(2\pi)^{d /2}|\mathbf{\Sigma}|^{1/2}} \right) - \frac{1}{2}(\mathbf{x} - \mathbf{\mu}_{k})^{\top}\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu}_{k})$$

The log term at the begining of each Gaussian term is constant for all classes and will thus cancel out when we take the difference between classes. This leaves us with:

$$- \frac{1}{2}(\mathbf{x} - \mathbf{\mu}_{1})^{\top}\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu}_{1}) + log(\mathbb{P}(Y = 1)) - [- \frac{1}{2}(\mathbf{x} - \mathbf{\mu}_{0})^{\top}\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu}_{0}) + log(\mathbb{P}(Y = 0))]$$

and after substituting priors,

$$- \frac{1}{2}(\mathbf{x} - \mathbf{\mu}_{1})^{\top}\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu}_{1}) + log(\pi_{1}) - [- \frac{1}{2}(\mathbf{x} - \mathbf{\mu}_{0})^{\top}\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu}_{0}) + log(1 - \pi_{1})]$$

and therefore our two discriminant functions are 

For class $1$ ($k = 1$):

$$f_{1}(\mathbf{x}) = - \frac{1}{2}(\mathbf{x} - \mathbf{\mu}_{1})^{\top}\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu}_{1}) + log(\pi_{1})$$

For class $0$ ($k = 0$):

$$f_{0}(\mathbf{x}) = - \frac{1}{2}(\mathbf{x} - \mathbf{\mu}_{0})^{\top}\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu}_{0}) + log(1 - \pi_{1})$$

where:

- $\mathbf{\Sigma}^{-1}$ is the inverse of the common covariance matrix shared by all classes (by assumption of the LDA model)

- $\mathbf{\mu}_{k}$ is the same as before.

- $\pi_{1}$ is the prior probability of class $1$ ($Y = 1$), and $1 - \pi_{1}$ is the prior probability of class $0$ ($Y = 0$)

- $\mathbf{x}^{\top}\mathbf{\Sigma}^{-1}\mathbf{\mu}_{k}$ contributes a term linear in $\mathbf{x}$

- $-\frac{1}{2}\mathbf{\mu}_{k}^{\top}\mathbf{\Sigma}^{-1}\mathbf{\mu}_{k}$ is a constant with respect to $\mathbf{x}$ and affects the intercept of the linear function

- $log(\pi_{k})$ incorporates the prior probability of class $k$, thus adjusting the functions offset.

As previously mentioned, we can compute each $f_{k}(\mathbf{x})$ and then:

$$\hat{k}(\mathbf{x}) = \arg \max_{k \in \{ 0, 1 \}} f_{k}(\mathbf{x})$$

means we assign $\mathbf{x}$ to class $1$, if $f_{1}(\mathbf{x}) > f_{0}(\mathbf{x})$, else class $0$.
