# Machine Learning Queries

### Why do we use Deep Learning instead of traditional Machine Learning Algorithms?
- One of the best answer is the use of *Feature Engineering* is minimized in Deep Learning. The Deep Learning is nothing but the use of Deep Neural Networks as an generic Algorithm. The whole time of Feature Engineering are reduced and therefore save a lot of work load. 
- The another thing which is more exciting is the exceptional performance of Neural Nets as compared to any other algorithm.

#### Why do we see concepts been explained/visualize in two dimensions commonly? Whereas in Deep Learning we dealt with too many dimensions?

### How *Collaborative Filtering* are implemented with the help of Machine Learning?
- One of the way to implement is by Matrix Factorization method where we have a huge and sparse matrix as our dataset. However this have some trade offs and not the state of the art model for Colloborative Filtering.

### What is the reason behind the use of Maxpool function in Object Recognition Task?
- The Maxpool function prohibits translation variance in the sequential sequences of images (i.e consecutive scene of
frames in the real time video) so that the model is able to determine the object more steadily in Object Recognition.

- The above reason also justify why it is never used in Reinforcement Learning task where the agent learns itself to maximize reward for variant of states. Here we try to imply each consecutive frames to be different as possible, as it helps the agent to determinate each frames as a new state.

### What if my Training loss is higher than Validation loss?
- The Training must not greater than validation loss, it infers that the model need to be trained more or the learning rate is too low.

### Why TanH function is used over to sigmoid function as an activation function ?
-  The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer.

### Why we initialize weights randomly rather than initializing it to zero?
- Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”.

### The shape of a Weights of any layer L neural network can be determined by-
(no of units in the layer L, no of units in layer L-1)

### Why L2 Regularization is termed as decay rate?

### How does Regularization helps from overfitting?
- As we add the extra term lamba times the weights euclidean by no of training examples in the cost function and also for dw in backprop.
Since the values initialized of weight matrix is containing less than 1, then the gradient descent becomes slow because of the small
values updated. 

### What is Mini-batch Gradient Descent?
- Rather than taking all the training examples one at a time to compute loss (Batch Gradient Descent) we actually divide the training
examples into a set of mini-batches. These set of mini-batches are used to compute loss and backprop and update the parameters, which
will be used for the gradient descent of another set of training examples.
- tip: we usually set 2^n training examples in a set consisting the total training examples.

### What is Gradient Descent with Momentum?
- When we do Gradient Descent it turns out to be oscillating up and down with moving forward to the global minimum. This oscillating
nature makes it slow, so to avoid the oscillating nature of this we take averages to make it smoother. What actually happens is that when
- we take average in the horizontal direction then the up and down distances cancel out to give a less oscillating nature. The average
is calcluated by the hyperparameter Beta(usually 0.9). The formaula : $Vt = Beta*(Vt-1)+(1-Beta)*(Theta)$ where theta represents the 
current value whereas the Vt-1 represents the average of the values of the past datapoints.
The averages we took is usually termed to be exponentially weighted averages.

### What is Adam?
- Combination of gradient descent with momentum and RMSProp.

### What is Batch Normalization and what it does?
- The Barch Normalization is a process in which each hidden layer is normalized by subtracting the mean and dividing by square root of
sum of varience + epsilon. This normalization helps when there is a shifting of datasets regardless of the same function. It acts like
a regularizer but have very small impact. To avail the functionality for choice of normalization or not, we introduce two parameters 
beta and gamma, by multiplying and then subtracting. In this method the bias b is not useful because the mean acts his work so we
neglect the b and db.

### What is Convolution or Convolving?
- The convolving is an process in which we usually squash a given matrix(Filter) to the bigger(Image) one, by squashing we actually
multiply those elements projected of filter and then add those elements. The no. of squashing depends upon the total ability to move
the filter around the matrix(Image).

### What the filter contains?
- The filter are of several types, generally vertical and horizontal. These filters contain 1,0 and -1. As the higher value represents
the Brightness of the image whereas the lower value corresponds to the darker part of any Image.

### What is padding and why we implement it?
- Padding is done to an image when applying ConvNet. When we apply ConvNet, the image shrinks and hence it loses a lot of information
of a single pixel because of convolving with the other more value pixels. To prevent this we usually make another set of pixels around
the boundary of our image as a padding size of 1 to enlarge the image.

### How can we calculate the image size after convolution?
- If n, f, p are the size of Image, size of Filter, and Padding size respectively then n+2p-f+1 is the resultant size of the image.

### What is gradient clipping?
- The gradient clipping is a method of thresholding the gradients when it explodes in training the model, it is generally used whrere the gradients explode at an exponential rates.

### Why One-hot encoding is used as a input feature in every machine-learning model?

### Why we use pooling after applying relu function?
### What are minibactches? How they are used?
- Minibatches are the composite blocks which are basically done to create the time period of the updating of gradients. It is usually done to avoid the fluctuations caused in convergence method(like Gradient Descent).

### What are the advantages of choosing large  no of filters? Why we generally use large no of filters?
### How using a skip-connection helps the gradient to backpropagate?

### What is the Identity Function referred in the ResNet50?
### What is interpolation and extrapolation? What is the difference btw them?(ref- NALU)
### How to reset the computation graph in Tensorflow?
- tf.reset_default_graph()

### Differentiate between PCA and SVD?
- 1. PCA -> Square Matrix whereas SVD -> Any general matrix

### Why we compute the transpose while reshaping any tensor(while maintainig order)?
- We can reshape any tensor but for maintaining order we must reshape it corresponding to its axis. Like suppose we want to reshape a tensor of shape (a,b,c) into (c,b*a), so we would first reshape it in (b*a,c) and then will compute transpose of it to get the required shape (c,b*a).
