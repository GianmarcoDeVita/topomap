import os
import keras
import ast
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import class_weight
from keras.models import model_from_json
import numpy as np
import importlib.util
import inspect
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from utils.apply_clustering import predict_clusters_valid_set
from utils.load_data import load_pseudolabels

BATCH_SIZE = 128
EPOCHS = 15


# Compute weighted pairwise accuracy
def compute_weighted_pairwise_accuracy(clusters_in_test_set, y_test_labels, predicted_class_labels, embedding,
                                       clustering, num_classes, experiment, log_file):

    weighted_accuracy_by_pair_nn = []

    for picked_cluster in clusters_in_test_set:
        for other_cluster in clusters_in_test_set:
            if picked_cluster != other_cluster:
                elements_in_picked_cluster = np.where(y_test_labels == picked_cluster)[0]
                elements_in_other_cluster = np.where(y_test_labels == other_cluster)[0]

                predictions_in_picked_cluster = predicted_class_labels[elements_in_picked_cluster]
                predictions_in_other_cluster = predicted_class_labels[elements_in_other_cluster]

                # Classifications in picked cluster
                correct_predictions_picked_cluster = len(np.where(predictions_in_picked_cluster == picked_cluster)[0])
                mispredictions_picked_cluster = len(np.where(predictions_in_picked_cluster == other_cluster)[0])

                # Classifications in other cluster
                correct_predictions_other_cluster = len(np.where(predictions_in_other_cluster == other_cluster)[0])
                mispredictions_other_cluster = len(np.where(predictions_in_other_cluster == picked_cluster)[0])

                # Compute weighted accuracy
                size_picked_cluster = correct_predictions_picked_cluster + mispredictions_picked_cluster
                size_other_cluster = correct_predictions_other_cluster + mispredictions_other_cluster

                n_AA = correct_predictions_picked_cluster  # Number of elements correctly predicted as belonging to cluster A
                n_BB = correct_predictions_other_cluster  # Number of elements correctly predicted as belonging to cluster B
                n_BA = mispredictions_picked_cluster  # Number of elements belonging to cluster A but predicted to belong to cluster B
                n_AB = mispredictions_other_cluster  # Number of elements belonging to cluster B but predicted to belong to cluster A

                if size_picked_cluster > size_other_cluster:
                    if size_other_cluster == 0:
                        continue
                    w = size_picked_cluster / size_other_cluster
                    wacc_pairs = (n_AA + w * n_BB) / (n_AA + w * n_BB + n_BA + w * n_AB)
                else:
                    if size_picked_cluster == 0:
                        continue
                    w = size_other_cluster / size_picked_cluster
                    wacc_pairs = (w * n_AA + n_BB) / (w * n_AA + n_BB + w * n_BA + n_AB)

                # Write pairwise accuracy
                log_file.write(embedding + "," + clustering + "," + str(num_classes) + "," + str(experiment) + "," + str(
                    picked_cluster) + "," + str(other_cluster) + "," + str(wacc_pairs) + "\n")
                log_file.flush()

                weighted_accuracy_by_pair_nn.append(wacc_pairs)

    return weighted_accuracy_by_pair_nn


# Replace last layer of the model to the new number of classes
def change_last_model_layer_pytorch(model, num_classes):

    # 1. Check for common attributes
    for name in ['fc', 'classifier', 'head']:
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                setattr(model, name, nn.Linear(in_features, num_classes))
                return model
            elif isinstance(layer, nn.Sequential):
                # Replace the last linear in the sequential block
                for i in reversed(range(len(layer))):
                    if isinstance(layer[i], nn.Linear):
                        in_features = layer[i].in_features
                        layer[i] = nn.Linear(in_features, num_classes)
                        setattr(model, name, layer)
                        return model

    # 2. If it's a Sequential model itself
    if isinstance(model, nn.Sequential):
        for i in reversed(range(len(model))):
            if isinstance(model[i], nn.Linear):
                in_features = model[i].in_features
                model[i] = nn.Linear(in_features, num_classes)
                return model

    # 3. Fallback: scan all modules (last matching nn.Linear will be replaced)
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            parent = model
            components = name.split(".")
            for c in components[:-1]:
                parent = getattr(parent, c)
            last_name = components[-1]
            in_features = module.in_features
            setattr(parent, last_name, nn.Linear(in_features, num_classes))
            return model

    raise ValueError("No nn.Linear layer found to replace.")


# Initialized newly substituted lazy layer
def initialize_lazy_layers(model: nn.Module, input_shape: tuple, device=None):
    model.eval()
    input_shape = (1, *input_shape[1:])  # Ensure batch size is 1
    dummy_input = torch.zeros(*input_shape).to(device or 'cpu')
    with torch.no_grad():
        _ = model(dummy_input)


# Load model architecture stored as a JSON file
def load_model_architecture(data_name, num_classes, model_library, input_shape_):

    if model_library == "pytorch":

        model_path = os.path.join("input", data_name, ".".join(("_".join(("model_architecture", data_name, "pytorch")), "py")))

        # Load the module
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Get source code of the module
        source = inspect.getsource(model_module)

        # Parse the AST
        parsed = ast.parse(source)

        # Find the last class in the file that is a subclass of nn.Module
        model_class = None
        for node in parsed.body:
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                cls = getattr(model_module, class_name, None)
                if isinstance(cls, type) and issubclass(cls, nn.Module):
                    model_class = cls  # Last matching class will be kept

        if model_class is None:
            raise ValueError("No nn.Module subclass found in the model file.")

        # Instantiate the model
        model = model_class()
        initialize_lazy_layers(model, input_shape=input_shape_)  # Adjust to your input shape
        model = change_last_model_layer_pytorch(model, num_classes)

    if model_library == "keras":

        # Define model filename and path
        model_path = os.path.join("input", data_name, ".".join(("_".join(("model_architecture", data_name, "keras")), "json")))
        # Load JSON and create model
        with open(model_path, "r") as json_file:
            loaded_model_json = json_file.read()

        # Replace the output layer of the model with a new layer with softmax activation and num_clusters classes
        model = model_from_json(loaded_model_json)
        x = model.layers[-2].output
        x = Dense(num_classes, activation='softmax', name="NewOutputLayer")(x)
        model = keras.models.Model(inputs=model.input, outputs=x)

    return model


# Evaluate topographical map (with validation set)
def dnn_eval_with_val(data_name, original_train_set, train_cluster_labels, original_valid_set, valid_cluster_labels,
                      original_test_set, test_cluster_labels, num_classes, embedding, clustering, experiment,
                      log_file, model_library, verbose=0):

    if model_library == "pytorch":

        # Define model filename and path
        model_save_filename = "_".join((data_name, model_library, "cluster_discriminator", str(embedding),
                                        str(clustering), str(num_classes), ".".join((str(experiment), "pth"))))

    if model_library == "keras":

        # Define model filename and path
        model_save_filename = "_".join((data_name, model_library, "cluster_discriminator", str(embedding),
                                        str(clustering), str(num_classes), ".".join((str(experiment), "h5"))))

    model_path_name = os.path.join(data_name, "data", "trained_dnn_models", model_save_filename)
    input_shape_ = original_train_set.shape

    if not os.path.exists(model_path_name):
        # Load the architecture of the model that we want to test
        model = load_model_architecture(data_name, num_classes, model_library, input_shape_)

        if verbose == 1:
            print(original_train_set.shape)
            print(train_cluster_labels.shape)
            print(original_test_set.shape)
            print(test_cluster_labels.shape)

        class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                          classes=np.unique(train_cluster_labels),
                                                          y=train_cluster_labels)

        if model_library == "pytorch":
            # Define the device to be used
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            if verbose == 1:
                # Check if CUDA is available
                print("CUDA available:", torch.cuda.is_available())
                print("Using device:", device)
                print(torch.version.cuda)

            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            criterion.weight = criterion.weight.to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=1.000)

            # Load training set as Tensor
            X_train_tensor = torch.tensor(original_train_set)
            y_train_tensor = torch.tensor(train_cluster_labels)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Load test set as Tensor
            X_valid_tensor = torch.tensor(original_valid_set)
            y_valid_tensor = torch.tensor(valid_cluster_labels)

            valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
            valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Load test set as Tensor
            X_test_tensor = torch.tensor(original_test_set)

            for epoch in range(EPOCHS):
                model.train()  # Set model to training mode
                running_loss = 0.0

                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).long()

                    optimizer.zero_grad()  # Clear gradients
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Calculate loss
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Update weights

                    running_loss += loss.item() * inputs.size(0)

                epoch_train_loss = running_loss / len(train_dataset)

                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device).long()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                epoch_val_loss = val_loss / len(valid_dataset)
                val_accuracy = correct / total

                if verbose == 1:
                    print(f"Epoch {epoch + 1}/{EPOCHS} | "
                          f"Train Loss: {epoch_train_loss:.4f} | "
                          f"Val Loss: {epoch_val_loss:.4f} | "
                          f"Val Acc: {val_accuracy:.4f}")

                # Save model's weights
                torch.save(model.state_dict(), model_path_name)

            predicted_outputs = model(torch.tensor(X_test_tensor).to(device))
            _, predicted_class_labels = torch.max(predicted_outputs, 1)
            predicted_class_labels = predicted_class_labels.cpu().numpy()

        if model_library == "keras":
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

            class_weight_dict = dict(enumerate(class_weights))

            y_train_categorical = keras.utils.to_categorical(train_cluster_labels, num_classes)
            y_valid_categorical = keras.utils.to_categorical(valid_cluster_labels, num_classes)

            model.fit(original_train_set, y_train_categorical,
                      class_weight=class_weight_dict,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=verbose,
                      validation_data=(original_valid_set, y_valid_categorical))
            model.save(model_path_name)

            pred_labels = model.predict(original_test_set, verbose=0)
            predicted_class_labels = np.argmax(pred_labels, axis=1)

    else:

        if model_library == "pytorch":
            model = load_model_architecture(data_name, num_classes, model_library, input_shape_)
            model.load_state_dict(torch.load(model_path_name))
            model.eval()
            predicted_outputs = model(torch.tensor(original_test_set))
            _, predicted_class_labels = torch.max(predicted_outputs, 1)
            predicted_class_labels = predicted_class_labels.cpu().numpy()
        if model_library == "keras":
            model = keras.models.load_model(model_path_name)
            pred_labels = model.predict(original_test_set, verbose=0)
            predicted_class_labels = np.argmax(pred_labels, axis=1)
        del model

    clusters_in_test_set = np.unique(test_cluster_labels)

    weighted_accuracy_by_pair_nn = compute_weighted_pairwise_accuracy(clusters_in_test_set, test_cluster_labels,
                                                                      predicted_class_labels, embedding, clustering,
                                                                      num_classes, experiment, log_file)

    min_weighted_accuracy_pair = np.min(weighted_accuracy_by_pair_nn)
    mean_weighted_accuracy_pair = np.mean(weighted_accuracy_by_pair_nn)

    return min_weighted_accuracy_pair, mean_weighted_accuracy_pair


# Evaluate topographical map (without validation set)
def dnn_eval(data_name, original_train_set, train_cluster_labels, original_test_set, test_cluster_labels, num_classes,
             embedding, clustering, experiment, log_file, model_library, verbose=0):

    if model_library == "pytorch":

        # Define model filename and path
        model_save_filename = "_".join((data_name, model_library, "cluster_discriminator", str(embedding),
                                        str(clustering), str(num_classes), ".".join((str(experiment), "pth"))))

    if model_library == "keras":

        # Define model filename and path
        model_save_filename = "_".join((data_name, model_library, "cluster_discriminator", str(embedding),
                                        str(clustering), str(num_classes), ".".join((str(experiment), "h5"))))

    model_path_name = os.path.join(data_name, "data", "trained_dnn_models", model_save_filename)

    input_shape_ = original_train_set.shape

    if not os.path.exists(model_path_name):
        # Load the architecture of the model that we want to test
        model = load_model_architecture(data_name, num_classes, model_library, input_shape_)

        if verbose == 1:
            print(original_train_set.shape)
            print(train_cluster_labels.shape)
            print(original_test_set.shape)
            print(test_cluster_labels.shape)

        class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                          classes=np.unique(train_cluster_labels),
                                                          y=train_cluster_labels)

        if model_library == "pytorch":
            # Define the device to be used
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            if verbose == 1:
                # Check if CUDA is available
                print("CUDA available:", torch.cuda.is_available())
                print("Using device:", device)
                print(torch.version.cuda)

            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            criterion.weight = criterion.weight.to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=1.000)

            # Load training set as Tensor
            X_train_tensor = torch.tensor(original_train_set)
            y_train_tensor = torch.tensor(train_cluster_labels)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Load test set as Tensor
            X_test_tensor = torch.tensor(original_test_set)
            y_test_tensor = torch.tensor(test_cluster_labels)

            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            for epoch in range(EPOCHS):
                model.train()  # Set model to training mode
                running_loss = 0.0

                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).long()

                    optimizer.zero_grad()  # Clear gradients
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Calculate loss
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Update weights

                    running_loss += loss.item() * inputs.size(0)

                epoch_train_loss = running_loss / len(train_dataset)

                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device).long()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                epoch_val_loss = val_loss / len(test_dataset)
                val_accuracy = correct / total

                if verbose == 1:
                    print(f"Epoch {epoch + 1}/{EPOCHS} | "
                          f"Train Loss: {epoch_train_loss:.4f} | "
                          f"Val Loss: {epoch_val_loss:.4f} | "
                          f"Val Acc: {val_accuracy:.4f}")

                # Save model's weights
                torch.save(model.state_dict(), model_path_name)

            predicted_outputs = model(torch.tensor(X_test_tensor).to(device))
            _, predicted_class_labels = torch.max(predicted_outputs, 1)
            predicted_class_labels = predicted_class_labels.cpu().numpy()

        if model_library == "keras":

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

            class_weight_dict = dict(enumerate(class_weights))

            y_train_categorical = keras.utils.to_categorical(train_cluster_labels, num_classes)
            y_test_categorical = keras.utils.to_categorical(test_cluster_labels, num_classes)

            model.fit(original_train_set, y_train_categorical,
                      class_weight=class_weight_dict,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      verbose=verbose,
                      validation_data=(original_test_set, y_test_categorical))
            model.save(model_path_name)

            pred_labels = model.predict(original_test_set, verbose=0)
            predicted_class_labels = np.argmax(pred_labels, axis=1)

    else:

        if model_library == "pytorch":
            model = load_model_architecture(data_name, num_classes, model_library, input_shape_)
            model.load_state_dict(torch.load(model_path_name))
            model.eval()
            predicted_outputs = model(torch.tensor(original_test_set))
            _, predicted_class_labels = torch.max(predicted_outputs, 1)
            predicted_class_labels = predicted_class_labels.cpu().numpy()
        if model_library == "keras":
            model = keras.models.load_model(model_path_name)
            pred_labels = model.predict(original_test_set, verbose=0)
            predicted_class_labels = np.argmax(pred_labels, axis=1)
        del model

    # Evaluate model's performance through pairwise weighted accuracy
    clusters_in_test_set = np.unique(test_cluster_labels)

    weighted_accuracy_by_pair_nn = compute_weighted_pairwise_accuracy(clusters_in_test_set, test_cluster_labels,
                                                                      predicted_class_labels, embedding, clustering,
                                                                      num_classes, experiment, log_file)

    min_weighted_accuracy_pair = np.min(weighted_accuracy_by_pair_nn)
    mean_weighted_accuracy_pair = np.mean(weighted_accuracy_by_pair_nn)

    return min_weighted_accuracy_pair, mean_weighted_accuracy_pair