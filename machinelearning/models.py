from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        "*** YOUR CODE HERE ***"
        wTensor = tensor(data=[[1.0 for _ in range(dimensions)]])  #, dtype=torch.float64
        self.w = Parameter(wTensor, requires_grad=False) #Initialize your weights here

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        "*** YOUR CODE HERE ***"
        # tensordot not the standard Matrix Dot Product in Math
        return tensordot(self.w, x, dims=([1], [1]))  


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if self.run(x).item() >= 0 else -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            "*** YOUR CODE HERE ***"
            converged = False
            while not converged:
                converged = True
                for data in dataloader:
                    feature = data['x']
                    label = data['label']
                    if label != self.get_prediction(feature):
                        converged = False
                        self.w += label * feature


class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()
        n = 100
        # layer1 shape: batch_size x n
        self.layer1 = Linear(1, n, bias=True)
        # layer2 shape: n x 1
        self.layer2 = Linear(n, 1, bias=True)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # neural network forward pass:
        #   x  => layer1 => layer2 =>  y
        #  bx1 =>  1xn   =>  nx1   => bx1
        h1 = relu(self.layer1(x))
        y = self.layer2(h1)
        return y 
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        "*** YOUR CODE HERE ***"
        predict_y = self.forward(x)
        return mse_loss(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        params = self.parameters()
        optimizer = optim.Adam(params, lr=0.003)
        done = False
        target_loss = 0.02
        while not done:
            acc_loss, cnt_loss = 0, 0
            # train over and over again on the data set
            for data in dataloader:
                feature = data['x']
                label = data['label']
                # 1.Forward pass: Compute predictions and loss
                loss = self.get_loss(feature, label)
                # 2.Backward pass: Compute gradients
                loss.backward()
                # 3.Step: Update parameters
                optimizer.step()
                # 4.Zero gradients: Reset gradients before next batch
                optimizer.zero_grad()
                # 5.accumulate loss value
                acc_loss += loss.item()
                cnt_loss += 1
            # check if the target loss has been achieved
            if acc_loss / cnt_loss <= target_loss:
                done = True


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        # define all hyperparameters
        # trial 1, pass rate: 4/5
        # self.learning_rate = 0.003
        # self.batch_size = 16
        # self.hidden1_size = 200

        # trial 2, pass rate: 5/5
        self.learning_rate = 0.003
        self.batch_size = 64  # accelerate trainning
        self.hidden1_size = 200

        # define neural network
        self.layer1 = Linear(input_size, self.hidden1_size, bias=True)
        self.layer2 = Linear(self.hidden1_size, output_size, bias=True)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        h1 = relu(self.layer1(x))
        y = self.layer2(h1)
        return y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        predict_y = self.run(x)
        return cross_entropy(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        epochs = 0
        target_accuracy = 0.975
        while dataset.get_validation_accuracy() < target_accuracy:
            for data in dataloader:
                feature = data['x']
                label = data['label']
                loss = self.get_loss(feature, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epochs += 1
        print(f"trainning {epochs} epochs")


class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        # define all hyperparameters
        # trial 1, pass_rate: 10/10, average_epoch: 112/10=11.2
        self.learning_rate = 0.003
        self.batch_size = 64
        self.hidden1_size = 200 # The hidden size d should be sufficiently large
        self.target_accuracy = 0.85

        # trial 2, pass_rate: 5/5, average_epoch: 106/5=21.2
        # self.learning_rate = 0.003
        # self.batch_size = 128   # Larger batches provide less weight/parameter updates
        # self.hidden1_size = 200
        # self.target_accuracy = 0.85

        # trial 3, pass_rate: 5/5, average_epoch: 99+306+136+105+854
        # self.learning_rate = 0.001
        # self.batch_size = 128
        # self.hidden1_size = 200
        # self.target_accuracy = 0.85

        # Initialize your model parameters here
        self.layer1 = Linear(self.num_chars, self.hidden1_size, bias=True)
        self.layer2 = Linear(self.hidden1_size, self.hidden1_size, bias=True)
        self.layer3 = Linear(self.hidden1_size, len(self.languages), bias=True)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.
        xs[i] represents i-th character of each word

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # RNN: h_i+1 = x_i * W_1 + h_i * W_2,  i >= 1
        h = relu(self.layer1(xs[0]))  # h_1 = relu(x_0 * W_1)
        for x in xs[1:]:
            # x shape: batch_size x self.num_chars
            # layer1 shape: self.num_chars x self.hidden1_size
            # h shape: batch_size x self.hidden1_size
            # layer2 shape: self.hidden1_size x self.hidden1_size
            h = relu(self.layer1(x) + self.layer2(h))
        return self.layer3(h)
    
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict_y = self.run(xs)
        return cross_entropy(predict_y, y)

    def train(self, dataset):
        """
        Trains the model.

        Note that when you iterate through dataloader, each batch will returned as its own vector in the form
        (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
        get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
        that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
        as follows:

        movedim(input_vector, initial_dimension_position, final_dimension_position)

        For more information, look at the pytorch documentation of torch.movedim()
        """
        "*** YOUR CODE HERE ***"
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        epochs = 0
        while dataset.get_validation_accuracy() < self.target_accuracy:
            for data in dataloader:
                feature = data['x']
                label = data['label']
                new_feature = movedim(feature, 0, 1)
                loss = self.get_loss(new_feature, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epochs += 1
            print(f"trainning {epochs} epochs")
        

def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    "*** YOUR CODE HERE ***"
    out_h = input_tensor_dimensions[0] - weight_dimensions[0] + 1
    out_w = input_tensor_dimensions[1] - weight_dimensions[1] + 1
    Output_Tensor = empty((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            Output_Tensor[i, j] = tensordot(
                input[i:i+weight_dimensions[0], j:j+weight_dimensions[1]], 
                weight, 
                dims=([0,1],[0,1])
            )
    "*** End Code ***"
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10
        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        # define all hyperparameters
        self.learning_rate = 0.003
        self.batch_size = 64
        self.hidden1_size = 200
        self.target_accuracy = 0.85
        # define neural network
        # Convolution: You use a 3x3 kernel, stride 1, no padding
        #              Output size per dimension: 28 - 3 + 1 = 26
        self.layer1 = Linear(26*26, self.hidden1_size, bias=True)
        self.layer2 = Linear(self.hidden1_size, output_size, bias=True)

    def run(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        h1 = relu(self.layer1(x))
        y = self.layer2(h1)
        return y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        """ YOUR CODE HERE """
        predict_y = self.run(x)
        return cross_entropy(predict_y, y)
        
    def train(self, dataset):
        """
        Trains the model.
        """
        """ YOUR CODE HERE """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        epochs = 0
        while dataset.get_validation_accuracy() < self.target_accuracy:
            for data in dataloader:
                feature = data['x']
                label = data['label']
                loss = self.get_loss(feature, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epochs += 1
            print(f"trainning {epochs} epochs")