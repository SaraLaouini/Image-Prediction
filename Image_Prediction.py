import cv2
import os
import h5py
import numpy as np
from PIL import Image



def clean_image(image_path, output_path, target_size=(224, 224), grayscale=False):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Skipping {image_path} - Could not read image.")
        return
    
    image = cv2.resize(image, target_size)
    
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(output_path, image)
    print(f"Cleaned and saved: {output_path}")

input_dir = "cat_other_trainBC"
output_dir = "cat_other_train"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    clean_image(input_path, output_path, target_size=(224, 224), grayscale=True)



def clean_image(image_path, output_path, target_size=(224, 224), grayscale=False):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Skipping {image_path} - Could not read image.")
        return
    
    image = cv2.resize(image, target_size)
    
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(output_path, image)
    print(f"Cleaned and saved: {output_path}")

input_dir = "cat_other_testBC"
output_dir = "cat_other_test"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    clean_image(input_path, output_path, target_size=(224, 224), grayscale=True)


source_folder = 'cat_other_train'

target_folder = 'cat_other_trainN'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

def label_image(image_path):
   
    if "cat" in os.path.basename(image_path):
        label = 1
    else:
        label = 0

    return label

for filename in os.listdir(source_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(source_folder, filename)
        label = label_image(image_path)

        target_path = os.path.join(target_folder, f"{label}_{filename}")
        
        image = Image.open(image_path)
        
        image.save(target_path)

print("Images labeled and saved to the target folder.")


source_folder = 'cat_other_trainN'  

image_paths = [os.path.join(source_folder, filename) for filename in os.listdir(source_folder) if filename.endswith(('.jpg', '.png'))]


labels = [0 if "0_" in filename else 1 for filename in os.listdir(source_folder) if filename.endswith(('.jpg', '.png'))]


with h5py.File('train.h5', 'w') as file:
    image_data_list = []

    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert('RGB')  
        image_array = np.array(image)
        image_data_list.append(image_array)

    max_shape = tuple(max(image.shape[i] for image in image_data_list) for i in range(3))

    image_dataset = file.create_dataset('X_train', shape=(len(image_data_list),) + max_shape, dtype=np.uint8)

    for i, image_data in enumerate(image_data_list):
        image_dataset[i, :image_data.shape[0], :image_data.shape[1], :image_data.shape[2]] = image_data

    label_dataset = file.create_dataset('Y_train', data=labels)

    class_list = [b'non-cat', b'cat']
    class_dataset = file.create_dataset('list_classes', data=class_list)

print("Labeled images (converted to RGB) added to H5 file 'train.h5'.")


with h5py.File('train.h5', 'r') as file:
    for key in file.keys():
        print("Key:", key)
        
        if isinstance(file[key], h5py.Dataset):
            dataset = file[key]
            data = dataset[:]
            print("Contents:", data)
        elif isinstance(file[key], h5py.Group):
            print("This is a group.")

file.close()

source_folder = 'cat_other_test'

target_folder = 'cat_other_testN'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)


def label_image(image_path):
   
    if "cat" in os.path.basename(image_path):
        label = 1
    else:
        label = 0

    return label

for filename in os.listdir(source_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(source_folder, filename)
        label = label_image(image_path)

        target_path = os.path.join(target_folder, f"{label}_{filename}")
        
        image = Image.open(image_path)
        
        image.save(target_path)

print("Images labeled and saved to the target folder.")




source_folder = 'cat_other_testN'  

image_paths = [os.path.join(source_folder, filename) for filename in os.listdir(source_folder) if filename.endswith(('.jpg', '.png'))]

labels = [0 if "0_" in filename else 1 for filename in os.listdir(source_folder) if filename.endswith(('.jpg', '.png'))]

with h5py.File('test.h5', 'w') as file:
    image_data_list = []

    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert('RGB')  
        image_array = np.array(image)
        image_data_list.append(image_array)

    max_shape = tuple(max(image.shape[i] for image in image_data_list) for i in range(3))

    image_dataset = file.create_dataset('X_test', shape=(len(image_data_list),) + max_shape, dtype=np.uint8)

    for i, image_data in enumerate(image_data_list):
        image_dataset[i, :image_data.shape[0], :image_data.shape[1], :image_data.shape[2]] = image_data

    label_dataset = file.create_dataset('Y_test', data=labels)

    class_list = [b'non-cat', b'cat']
    class_dataset = file.create_dataset('list_classes', data=class_list)

print("Labeled images (converted to RGB) added to H5 file 'test.h5'.")


with h5py.File('test.h5', 'r') as file:
    for key in file.keys():
        print("Key:", key)
        
        if isinstance(file[key], h5py.Dataset):
            dataset = file[key]
            data = dataset[:]
            print("Contents:", data)
        elif isinstance(file[key], h5py.Group):
            print("This is a group.")

file.close()


import copy
import matplotlib.pyplot as plt


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
train_dataset = h5py.File('train.h5', "r")

test_dataset = h5py.File('test.h5', "r")

print("File format of train_dataset:",train_dataset)
print("File format of test_dataset:" ,test_dataset)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

X_trainN = np.array(train_dataset["X_train"][:]) 
Y_train = np.array(train_dataset["Y_train"][:]) 

X_testN = np.array(test_dataset["X_test"][:])
Y_test = np.array(test_dataset["Y_test"][:]) 



classes = ["non-cat", "cat"]
index = 180

plt.imshow(X_trainN[index])

print("y = " + str(Y_train[index]) + ", it's a '" + classes[np.squeeze(Y_train[index])] + "' picture.")



m_train = X_trainN.shape[0]

m_test = X_testN.shape[0]

n = X_trainN.shape[1]


print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: n = " + str(n))
print ("Each image is of size: (" + str(n) + ", " + str(n) + ", 3)")
print ("X_train shape: " + str(X_trainN.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_testN.shape))
print ("Y_test shape: " + str(Y_test.shape))


X_train_flatten = X_trainN.reshape(X_trainN.shape[0],-1).T

X_test_flatten = X_testN.reshape(X_testN.shape[0],-1).T

Y_train = Y_train.reshape(1, -1)
Y_test = Y_test.reshape(1, -1)

print ("X_train_flatten shape: " + str(X_train_flatten.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test_flatten shape: " + str(X_test_flatten.shape))
print ("Y_test shape: " + str(Y_test.shape))


X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.


def sigmoid(z):
   
    s = 1/(1+np.exp(-z))
        
    return s



def initialize_with_zeros(dim):
  
    w = np.zeros(shape=(dim, 1))

    b = 0.0

    return w, b



def propagate(w, b, X, Y):
    
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    cost = -(1/m)*np.sum((np.dot(Y,np.log(A).T))+(np.dot((1-Y),np.log(1-A).T)))
    

    dw = (1/m)*np.dot(X,((A-Y).T))
    
    db = (1/m)*np.sum(A-Y)
    
    cost = np.squeeze(np.array(cost))

    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
   
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
     
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
     
        w = w-learning_rate * dw
        
        b = b-learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
        
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):
  
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
     
        if A[0, i] >0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
            
    return Y_prediction



def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
     
        w, b = initialize_with_zeros(dim=X_train.shape[0])

        params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        w = params['w']
        b = params['b']

        Y_prediction_test = predict(w, b, X_test)
        Y_prediction_train = predict(w, b, X_train)
 
        if print_cost:
            print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
            print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : w, 
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}

        return d

logistic_regression_model = model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.001, print_cost=True)

my_image = "cat.jpg"   

image = np.array(Image.open(my_image).resize((n, n)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, n * n * 3)).T
my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

predicted_class = classes[int(np.squeeze(my_predicted_image))]
print(f"y = {int(np.squeeze(my_predicted_image))}, your algorithm predicts a \"{predicted_class}\" picture.")




