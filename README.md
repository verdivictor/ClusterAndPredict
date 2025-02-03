
# ClusterAndPredict

Short script to analyze the Online Retail dataset (http://archive.ics.uci.edu/dataset/352/online+retail) containing "transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail".

Splits up the data in categories, clusters them using K-Means by Customer-Value, assigns supercategories, and attempts to predict if Customer will move up, down, or remain in the same Supercategory in the next month. 

Greatest accuracy achieved was around 83% with the Deep Learning model, which was pretty good at estimating decreases and static cases, but very bad at predicting increases in Customer-Value.




## How to use this program

#### Initializes conda environment and installs necessary packages

```http
  chmod +x install.sh && ./install.sh
```

#### Run the program

```http
  python3 main.py --model nn
```

| Parâmetro   | Tipo       | Descrição                                   |
| :---------- | :--------- | :------------------------------------------ |
| `model`      | `string` | **Obligatory**. "nn" or "xgboost"|
| `debug`      | `string` | View debug data|
| `view`      | `string` | View scatter plots of Customer clusters|

