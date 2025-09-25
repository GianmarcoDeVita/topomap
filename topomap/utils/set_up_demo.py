import numpy as np
import os
import tensorflow as tf
import torch
from keras.datasets import mnist, cifar10
from keras import backend as K, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

ROOT = "input"


# Routine to delete already existing datasets input files
def clear_demo_input_data(data_name):

    # Define the target filename suffixes
    target_filenames = {
        "train_x.npy",
        "valid_x.npy",
        "test_x.npy",
        "train_y.npy",
        "valid_y.npy",
        "test_y.npy"
    }

    deleted_files = []

    # Walk through all files in the directory
    for root, _, files in os.walk(os.path.join(ROOT, data_name)):
        for file in files:
            if file in target_filenames:
                file_path = os.path.join(root, "_".join((data_name, file)))
                try:
                    os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")


# Save Keras model architecture
def demo_keras_model_architecture(dataset_name):

    if dataset_name == "cifar10":

        # Define model for CIFAR-10
        img_width, img_height, img_num_channels = 32, 32, 3
        input_shape = (img_width, img_height, img_num_channels)

        # Create the model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

    else:

        # Define model for MNIST
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

    return model


# Save downloaded dataset into proper folder
def save_data(dataset_name, x_train, y_train, x_test, y_test, x_valid=None, y_valid=None):

    # Create the directory
    try:
        os.mkdir(os.path.join(ROOT, dataset_name))
        print(f"Directory '{dataset_name}' created successfully!")
    except FileExistsError:
        print(f"Directory '{dataset_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{dataset_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Save training set
    np.save(os.path.join(ROOT, dataset_name, "_".join((dataset_name, "train", "x.npy"))), x_train)
    np.save(os.path.join(ROOT, dataset_name, "_".join((dataset_name, "train", "y.npy"))), y_train)

    # Save test set
    np.save(os.path.join(ROOT, dataset_name, "_".join((dataset_name, "test", "x.npy"))), x_test)
    np.save(os.path.join(ROOT, dataset_name, "_".join((dataset_name, "test", "y.npy"))), y_test)

    # Save validation set
    if x_valid is not None:
        np.save(os.path.join(ROOT, dataset_name, "_".join((dataset_name, "valid", "x.npy"))), x_valid)
        np.save(os.path.join(ROOT, dataset_name, "_".join((dataset_name, "valid", "y.npy"))), y_valid)

    return


# Download MNIST dataset from Pytorch
def download_data_mnist_pytorch():
    # Load the MNIST dataset
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='/tmp/data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='/tmp/data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    data_train = train_images.numpy()
    labels_train = train_labels.numpy()

    data_test = test_images.numpy()
    labels_test = test_labels.numpy()

    print("SUCCESS: Dataset MNIST downloaded successfully!")
    return data_train, data_test, labels_train, labels_test


# Download MNIST dataset from Keras
def download_data_mnist_keras():
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    (img_rows, img_cols) = (28, 28)
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print("SUCCESS: Dataset MNIST downloaded successfully!")

    return x_train, x_test, y_train, y_test


# Download CIFAR-10 dataset from Pytorch
def download_data_cifar10_pytorch():
    # Load the CIFAR dataset
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    train_data = datasets.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    data_train = train_images.numpy()
    labels_train = train_labels.numpy()

    data_test = test_images.numpy()
    labels_test = test_labels.numpy()

    print("SUCCESS: Dataset CIFAR-10 downloaded successfully!")
    return data_train, data_test, labels_train, labels_test


# Download CIFAR-10 dataset from Keras
def download_data_cifar10_keras():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Determine shape of the data
    img_width, img_height, img_num_channels = 32, 32, 3
    input_shape = (img_width, img_height, img_num_channels)

    # Parse numbers as floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize data
    x_train = x_train / 255
    x_test = x_test / 255

    print("SUCCESS: Dataset CIFAR-10 downloaded successfully!")

    return x_train, x_test, y_train, y_test


# Initialize a demo dataset from Keras
def load_demo_data_keras(dataset_name="mnist", validation_split=0):

    # Download the dataset chosen by the user
    if dataset_name == "cifar10":
        # Download the CIFAR-10 dataset
        x_train_, x_test_, y_train_, y_test_ = download_data_cifar10_keras()
        y_train_ = y_train_.ravel()
        y_test_ = y_test_.ravel()
    else:
        # Otherwise, download the default option, MNIST
        x_train_, x_test_, y_train_, y_test_ = download_data_mnist_keras()

    # If validation split is given (> 0), create a validation set out of the training set
    if validation_split > 0:
        x_train_, x_valid_, y_train_, y_valid_ = train_test_split(x_train_, y_train_, test_size=validation_split,
                                                              random_state=0)

        y_train_ = y_train_.ravel()
        y_valid_ = y_valid_.ravel()
        y_test_ = y_test_.ravel()

        save_data(dataset_name, x_train_, y_train_, x_test_, y_test_, x_valid_, y_valid_)

        # Print stats on data
        print("Dataset ", dataset_name)
        print(x_train_.shape[0], 'train samples')
        print(x_valid_.shape[0], 'validation samples')
        print(x_test_.shape[0], 'test samples')

    else:
        save_data(dataset_name, x_train_, y_train_, x_test_, y_test_)

        # Print stats on data
        print("Dataset ", dataset_name)
        print(x_train_.shape[0], 'train samples')
        print(x_test_.shape[0], 'test samples')

    # Save data model
    data_model = demo_keras_model_architecture(dataset_name)
    # Serialize model to JSON
    model_json = data_model.to_json()
    # Save to file
    model_path = os.path.join(ROOT, dataset_name, ".".join(("_".join(("model_architecture", dataset_name, "keras")), "json")))
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    return


# Initialize a demo dataset from PyTorch
def load_demo_data_pytorch(dataset_name="mnist", validation_split=0):

    # Download the dataset chosen by the user
    if dataset_name == "cifar10":
        # Download the CIFAR-10 dataset
        x_train_, x_test_, y_train_, y_test_ = download_data_cifar10_pytorch()
        y_train_ = y_train_.ravel()
        y_test_ = y_test_.ravel()
    else:
        # Otherwise, download the default option, MNIST
        x_train_, x_test_, y_train_, y_test_ = download_data_mnist_pytorch()

    # If validation split is given (> 0), create a validation set out of the training set
    if validation_split > 0:
        x_train_, x_valid_, y_train_, y_valid_ = train_test_split(x_train_, y_train_, test_size=validation_split,
                                                              random_state=0)

        y_train_ = y_train_.ravel()
        y_valid_ = y_valid_.ravel()
        y_test_ = y_test_.ravel()

        save_data(dataset_name, x_train_, y_train_, x_test_, y_test_, x_valid_, y_valid_)

        # Print stats on data
        print("Dataset ", dataset_name)
        print(x_train_.shape[0], 'train samples')
        print(x_valid_.shape[0], 'validation samples')
        print(x_test_.shape[0], 'test samples')

    else:
        save_data(dataset_name, x_train_, y_train_, x_test_, y_test_)

        # Print stats on data
        print("Dataset ", dataset_name)
        print(x_train_.shape[0], 'train samples')
        print(x_test_.shape[0], 'test samples')

    return
