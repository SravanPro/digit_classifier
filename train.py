#old train
# from functions import *

# training dataset
train = np.array(pd.read_csv(r"C:\codes\MLStuff\datasets\mnist\mnist_train.csv"))

# number of neurons in each layer
nn0 = 784   # input layer, number of pixels in each image
nn1 = 24   # hidden layer, number of neurons in hidden layer    
nn2 = 10   # output layer, number of neurons in output layer, which is the number of classes (0-9)

# our datset:   
# [[5 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [4 0 0 ... 0 0 0]
#  [1 0 0 ... 0 0 0]
#  [9 0 0 ... 0 0 0]]
# 1st column: actual number, which is our output
# rest of the columns: pixel values of the image, which is our input

train_input = train[:,1:].T   # all columns except the first one, transpose taken coz each column must be an image.
train_output = train[:,0]   # first column
train_input = train_input/255   # normalizing the pixel values to be between 0 and 1

learning_rate = 0.01
batch_size = 64
epochs = 200

w1, b1, w2, b2 = train_model(train_input, train_output, learning_rate, batch_size, epochs, nn0, nn1, nn2)

np.savez(
    r"C:\codes\MLStuff\outputs\_0_digitClassifier\mnist_model.npz",
    w1=w1,
    b1=b1,
    w2=w2,
    b2=b2
)