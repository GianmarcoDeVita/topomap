import os
import shutil
import datetime
import sys
import time
import threading

data_folder_name = "data"
data_subfolders_names = ["embedded_data", "experiments", "experiments_clusters", "metrics",
                         "trained_cluster_models", "trained_dnn_models", "trained_embedding_models"]
reports_folder_name = "reports"
topographical_map_folder_name = "topographical_map"


# Initialize directories for saving intermediate and final results
def initialize_directories(data_name):

    try:
        # Create root directory
        os.mkdir(data_name)
        # Create reports directory
        os.mkdir(os.path.join(data_name, reports_folder_name))
        # Create topographical map directory
        os.mkdir(os.path.join(data_name, topographical_map_folder_name))
        # Create data directory
        os.mkdir(os.path.join(data_name, data_folder_name))
        # Create subfolders
        for subfolder_name in data_subfolders_names:
            os.mkdir(os.path.join(data_name, data_folder_name, subfolder_name))
        print(f"Directory '{data_folder_name}' and sub-directories created successfully for '{data_name}'!")
    except FileExistsError:
        print(f"Directory '{data_folder_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{data_folder_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None


# Clear data space to make the code suitable for re-runs
def clear_directories(data_name):

    # Clear files in subfolders inside the main data folder
    for subdirectory in data_subfolders_names:
        folder_path = os.path.join(data_name, data_folder_name, subdirectory)
        if os.path.isdir(folder_path):  # Check if folder exists
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    # Clear files in reports directory
    folder_path = os.path.join(data_name, reports_folder_name)
    if os.path.isdir(folder_path):  # Check if folder exists
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Clear files in topographical_map directory
    folder_path = os.path.join(data_name, topographical_map_folder_name)
    if os.path.isdir(folder_path):  # Check if folder exists
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return None


# Add an animation of movimg mouse
def walking_mouse(stop_event):
    mouse = "ᓚᘏᕐᐷ"
    total_dots = 20
    i = 0
    while not stop_event.is_set():
        left_dots = i % (total_dots + 1)
        right_dots = total_dots - left_dots
        line = f"{'.' * left_dots}{mouse}{'.' * right_dots}"
        sys.stdout.write('\r' + line)
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1


# Create a backup with the current results
def backup_directories(data_name):

    # Create a thread-safe stop signal
    stop_event = threading.Event()

    # Start the animation
    t = threading.Thread(target=walking_mouse, args=(stop_event,))
    t.start()

    backup_archive_name = "data_backup_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    shutil.make_archive(backup_archive_name, 'zip', os.path.join(data_name, data_folder_name))

    # Stop the animation
    stop_event.set()
    t.join()

    # Clear the animation line
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()

    return None


# Show intro logo
def welcome():
    print(r"""
  ______                  __  ___          
 /_  __/___  ____  ____  /  |/  /___ _____ 
  / / / __ \/ __ \/ __ \/ /|_/ / __ `/ __ \
 / / / /_/ / /_/ / /_/ / /  / / /_/ / /_/ /
/_/  \____/ .___/\____/_/  /_/\__,_/ .___/ 
         /_/                      /_/      
    """)

