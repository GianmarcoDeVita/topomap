import argparse
import sys
from topomap import main
from utils.set_up_demo import *
from utils.set_up_directories import *


# Function to run demo
def run_demo(data_name="mnist", model_library="keras"):
    
    print(data_name)

    # Remove all existing input files for this dataset, to avoid conflicts
    clear_demo_input_data(data_name)

    # Download specific files from the relevant package
    if data_name == "mnist":
        if model_library == "pytorch":
            load_demo_data_pytorch("mnist")
        else:
            load_demo_data_keras("mnist")
    else:
        if model_library == "pytorch":
            load_demo_data_pytorch("cifar10", 0.2)
        else:
            load_demo_data_keras("cifar10", 0.2)

    welcome()

    sys.argv = [
        'topomap.py',
        data_name,
        '-p', model_library,
        '-c', 'True',
        '-i', 'True',
    ]

    main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo run for TopoMap: MNIST or CIFAR10")

    # Positional args
    parser.add_argument('input_dataset_name', type=str, help='Input dataset name')
    parser.add_argument('model_package_name', type=str, help='Model package name')
    args = parser.parse_args()

    run_demo(args.input_dataset_name, args.model_package_name)



