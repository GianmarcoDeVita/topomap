# TopoMap: Topographical Mapper of the Test Input Space

This repository contains all the code needed to generate an input topographical map with TopoMap.

## 🏗️ Project Structure

The project directories are structured as follows:

```
project_root/
├── dockerfile.cpu/		# Dockerfile for CPU devices
│   ├── Dockerfile
│   └── requirements.txt
├── dockerfile.gpu/		# Dockerfile for GPU devices
│   ├── Dockerfile
│   └── requirements.txt
└── topomap/
    ├── input/
    ├── run_demo.py		# Demo run
    ├── topomap.py		# Main TopoMap file
    └── utils/
```

## 🐳 Docker Installation

### 💻 CPU

If you are using CPU, move files in ``dockerfile.cpu/`` into the ``project_root/`` folder. 
1. First, build the Docker image:

```bash
docker build -t topomap_env .
```

2. Then, launch the container:

```bash
docker run -it --name topomap_container \
  -v ./topomap:/workspace/topomap \
  -w /workspace/topomap \
  topomap_env
```

### 🖥️ GPU

If you are using GPU, move files in ``dockerfile.gpu/`` into the ``project_root/`` folder. 

1. First, build the Docker image:

```bash
docker build -t topomap_env .
```

2. Then, launch the container:

```bash
docker run --gpus all -it --name topomap_container \
  -v ./topomap:/workspace/topomap \
  -w /workspace/topomap \
  topomap_env
...
```

## 🐁 Launch TopoMap

### 📊 Load Input Data

In order to generate the topographical map of the inputs of a dataset ``dataset_name``, the training, validation (if present) and test sets must be provided as ``.npy`` arrays into the ``topomap/input/`` folder, according to the following scheme and naming:

```
...
topomap/
└── input/		
    └── dataset_name/													# Folder named after the dataset
        ├── [dataset_name]_train_x.npy									# Training set data (mandatory)
        ├── [dataset_name]_train_y.npy									# Training set labels (mandatory)
        ├── [dataset_name]_valid_x.npy									# Validation set data (if available)
        ├── [dataset_name]_valid_y.npy									# Validation set labels (if available)
        ├── [dataset_name]_test_x.npy									# Test set data (mandatory)
        ├── [dataset_name]_test_y.npy									# Test set labels (mandatory)
		└── model_architecture_[dataset_name]_[model_library].[ext] 	# DNN model architecture
...
```

Where [dataset_name] is the name of the dataset, and [model_library] can be either ``keras`` or ``pytorch``. As for Keras models, TopoMap expects a ``.json`` file containing the architecture of the model, while for PyTorch, the model code in a ``.py`` class is expected. 

### 🧪 Experimental Configuration

TopoMap supports the following embedding and clustering algorithms:

| **Embedding**         | **Type**        |
|-----------------------|-----------------|
| PCA                   | Linear          |
| LDA                   | Linear          |
| UMAP                  | Non-linear      |
| Isomap                | Non-linear      |

| **Clustering**        | **Type**        |
|-----------------------|-----------------|
| K-means               | Parametric      |
| BIRCH                 | Parametric      |
| Affinity Propagation  | Non-parametric  |
| HDBSCAN               | Non-parametric  |

### 📝 Handling Configurations

In file ``topomap/topomap.py``, in function ``generate_map``, you can find the set of available embedding and clustering algorithms. You can edit the content of the ``embeddings`` and ``clusterings`` lists to your necessities.

```python
    # Experimental configurations
    embeddings = ["pca", "umap", "lda", "isomap"]
    clusterings = ["kmeans", "birch", "affinity_propagation", "hdbscan"]
    experiments = 10
```

As per performance, we suggest *CPU users* to use only ``"kmeans"`` and ``"hdbscan"`` for their experiments, depending on the size of the input dataset.

### 🛠️ Map Computation

In order to launch TopoMap, go into folder ``topomap/`` and run the following command:

```bash
python topomap.py [dataset_name] [--args]
```

TopoMap also accepts some extra elective arguments:

```
'-p', '--package', type=str, default="keras", 'Python model's package: either "keras" or "pytorch".'
'-b', '--backup', type=bool, default=False, 'Create a compressed backup copy of the existing data for this dataset.'
'-c', '--clear', type=bool, default=False, 'Perform a clean run of the code for this dataset. WARNING: You'll lose all your data!'
'-s', '--select', type=str, default="[]", 'Select classes to be shown in the TopoMap plot. Default: all classes.'
'-i', '--imgplot', type=bool, default=False, 'Plot clusters medoids. WARNING: Only works with image datasets.'
'-r', '--regression', type=bool, default=False, 'Indicate whether the input dataset pertains a regression problem.'
```

## 🎬 Launch a Demo Run

As an example, TopoMap is released along with a couple of examples, pertaining to the MNIST/CIFAR-10 datasets and models implemented in Keras and PyTorch.

In order to execute the demo, launch the following command in ``topomap`` folder:

```bash
python run_demo.py [dataset_name] [model_package]
```

Where ``[dataset_name]`` can be either ``mnist`` or ``cifar10`` and ``[model_package]`` can be either ``keras`` or ``pytorch``.

## 🗺️ Get the Map

At the end of the execution, a folder with the name ``[dataset_name]/`` will appear in the ``topomap/`` directory and will be structured as follows:

```
...
topomap/
└── dataset_name/						# Folder named after the dataset
    ├── data/
    │	├── embedded_data/				# Embedded data (.npy)
    │	├── experiments/				# Pairwise accuracy by experiment (.npy)
    │	├── experiments_clusters/		# Cluster labels computed at each experiment (.npy)
    │	├── metrics/					# Data about cluster quality metrics (.npy)
    │	├── trained_cluster_models/		# Fitted clustering models
    │	├── trained_dnn_models/			# Fitted DNN cluster evaluators or weights (PyTorch)
    │	└── trained_embedding_models/	# Fitted embedding models
    ├── reports/
    │	├── topomap_report_clustering_[dataset_name].csv 				# Cluster Accuracy
    │	├── topomap_report_configurations_pairwise_[dataset_name].csv 	# Cluster DNN Quality
    │	├── topomap_report_ncomp_[dataset_name].csv 					# Embedding Components
    │	└── topomap_report_pairwise_accuracies_[dataset_name].csv 		# Pairwise Accuracies
    └── topographical_map/				# Selected Topographical Map Labels
...

```

The labels of the final topographical map generated and selected by TopoMap, can be found in subfolder ``topographical_map/``.

### References:

[1] De Vita, G., Humbatova, N., & Tonella, P. (2025). TopoMap: A Feature-based Semantic Discriminator of the Topographical Regions in the Test Input Space. *arXiv preprint arXiv:2509.03242*. [[arXiv](https://arxiv.org/abs/2509.03242)]

