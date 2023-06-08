#!/bin/bash
set -e

DATASET=$1

function prepare_yoochoose {
    echo "Downloading yoochoose"
    wget -nc https://s3-eu-west-1.amazonaws.com/yc-rdata/yoochoose-data.7z -P datasets/yoochoose/
    7zz x -aos datasets/yoochoose/yoochoose-data.7z -odatasets/yoochoose/
    
    echo "Preprocessing yoochoose"
    pipenv run python -m src.preprocessing --dataset yoochoose
}

function download_digitinica {
    if [ ! -f datasets/diginetica/dataset-train-diginetica.zip ]; then
        mkdir -p datasets/diginetica
        echo "Please download the dataset and save it to datasets/diginetica/dataset-train-diginetica.zip"
        if [ "$(uname)" == "Darwin" ]; then
            open https://drive.google.com/uc?id=0B7XZSACQf0KdenRmMk8yVUU5LWc
        else
            xdg-open https://drive.google.com/uc?id=0B7XZSACQf0KdenRmMk8yVUU5LWc
        fi
        echo "Press enter to continue"
        read
    fi
}

function prepare_diginetica {
    echo "Downloading diginetica"
    download_digitinica
    unzip -n datasets/diginetica/dataset-train-diginetica.zip -d datasets/diginetica/
    
    echo "Preprocessing diginetica"
    pipenv run python -m src.preprocessing --dataset diginetica
}

function prepare_otto {
    echo "Downloading otto"
    pipenv run kaggle datasets download -d otto/recsys-dataset -p datasets/otto/
    unzip -n datasets/otto/recsys-dataset.zip -d datasets/otto/
    
    echo "Preprocessing otto"
    pipenv run python -m src.preprocessing --dataset otto
}

if [ "$DATASET" = "yoochoose" ]; then
    prepare_yoochoose
elif [ "$DATASET" = "diginetica" ]; then
    prepare_diginetica
elif [ "$DATASET" = "otto" ]; then
    prepare_otto
else
    echo "Unknown dataset"
    exit 1
fi
