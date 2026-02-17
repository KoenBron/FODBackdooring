# Federated Object Detection Backdooring
This repository implements existing dissapearance and mislabelling backdoor attacks on CNN-based Object Detection models in a Federate Learning session.
These attacks are established weaknesses of OD-models trained by a centralized server entity, but this work shows their implementation in a FL setting through utilizing Distributed Backdoor Attacks.

## Setup
This project is made in Python version **3.10.2**.
To run this project as is locally:
1. Install pytorch with pip (NOTE: We used pytorch version 2.10.0 with CUDA 13.0 in a Linux environment)
    ```
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    ```
2. Install the remaining dependencies
    ```
    pip install -r requirements.txt
    ```
3. Make sure you have your dataset in COCO-annotation stored in a directory called dataset at the project root
    ```
    mkdir dataset
    ```
    1. Though this behviour is alterable in the `utils/dataset.py` file, the program expects the following folders and files to be present (NOTE: though modifiable, the program expects .jpg files for the images)

    ```
    /dataset/training_annotations.json
    /dataset/validation_annotations.json
    /dataset/training_images/
    /dataset/validation_images/
    ```
4. You can start the training of the model (NOTE: training parameters can be altered in `train.sh`)
    ```
    chmod +x run_train.sh
    ./run_train.sh
    ```
5. You can start the training of the model (NOTE: training parameters can be altered in `train.sh`)
    ```
    chmod +x run_evaluation.sh
    ./run_evaluation.sh
    ```
