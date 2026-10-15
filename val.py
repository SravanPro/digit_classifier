#old val
from functions import *

# validation dataset
val = np.array(pd.read_csv(r"C:\codes\MLStuff\datasets\mnist\mnist_test.csv"))

val_input = val[:, 1:].T
val_output = val[:, 0]

val_input = val_input / 255

# load trained weights and biases
model = np.load(r"C:\codes\MLStuff\outputs\_0_digitClassifier\mnist_model.npz")

w1 = model["w1"]
b1 = model["b1"]
w2 = model["w2"]
b2 = model["b2"]

# inference

num_batches = min(100, val_output.size)
batches = np.array_split(np.arange(val_output.size), num_batches)

total_correct = 0
total_examples = 0

for batch_num, indices in enumerate(batches, start=1):

    # input batch
    a0 = val_input[:, indices]
    y = val_output[indices]

    # forward propagation
    z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, a0)

    # predicted digit for each image
    predictions = np.argmax(a2, axis=0)

    # calculate batch accuracy
    batch_correct = np.sum(predictions == y)
    batch_accuracy = batch_correct / len(y)

    total_correct += batch_correct
    total_examples += len(y)

    print(
        f"Batch {batch_num:3d}: "
        f"{batch_correct}/{len(y)} correct | "
        f"Accuracy: {batch_accuracy:.2%}"
    )

# overall accuracy
total_accuracy = total_correct / total_examples

print("\n========== Final Results ==========")
print(f"Total correct: {total_correct}/{total_examples}")
print(f"Overall accuracy: {total_accuracy:.2%}")
