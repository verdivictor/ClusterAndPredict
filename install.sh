#!/bin/sh

curl http://archive.ics.uci.edu/static/public/352/online+retail.zip --output online_retail.zip

unzip -LL online_retail.zip && mv "online retail.xlsx" "online_retail.xlsx"

rm online_retail.zip

conda env list | grep -q "pytorch" || conda create --name pytorch

conda activate pytorch

for package in pytorch::pytorch numpy pandas anaconda::scikit-learn xgboost imbalanced-learn; do
    conda list | grep -q "$package" || conda install -y "$package"
done