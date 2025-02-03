import torch
from torch import nn
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # No CUDA :(
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.utils import class_weight
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def assign_supercategory(cluster, low_threshold, high_threshold):
    if cluster >= high_threshold:
        return 'HIGH'
    elif cluster <= low_threshold:
        return 'LOW'
    else:
        return 'MID'

def get_basic_columns(df):
    df = df.sort_values('InvoiceDate', ascending=True)

    df['Original_Quantity'] = df['Quantity']
    df['Original_UnitPrice'] = df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Hour'] = df['InvoiceDate'].dt.hour
    df['Month'] = df['InvoiceDate'].dt.to_period('M')

    df.loc[df['Quantity'] < 0, 'Quantity'] = 0
    df.loc[df['Quantity'] > 10000, 'Quantity'] = 0
    df.loc[df['UnitPrice'] < 0, 'UnitPrice'] = 0

    df['Sales'] = df['Quantity'] * df['UnitPrice']

    excluded_products = ['DOT', 'POST', 'M']

    df = df[~df['StockCode'].isin(excluded_products)]
    df = df.dropna(subset=['StockCode'])

    return df

def get_customer_stats(df):
    total_spend = df.groupby(['Month', 'CustomerID'])['Sales'].sum().unstack().dropna(axis=1, how='all')
    purchase_frequency = df.groupby(['Month', 'CustomerID'])['InvoiceNo'].nunique().unstack().dropna(axis=1, how='all')
    avg_hour = df.groupby(['Month', 'CustomerID'])['Hour'].mean().unstack().dropna(axis=1, how='all')
    avg_spend_per_transaction = total_spend / purchase_frequency
    return total_spend, purchase_frequency, avg_spend_per_transaction, avg_hour

def view_cluster(total):
    num_clusters = total['Cluster'].nunique()
    colors = plt.cm.get_cmap('tab10', num_clusters)

    plt.figure(figsize=(10, 6))
    for cluster in np.sort(total['Cluster'].unique()):
        cluster_data = total[total['Cluster'] == cluster]
        plt.scatter(cluster_data['TotalSpend'], cluster_data['PurchaseFrequency'], 
                color=colors(cluster), label=f'Cluster {cluster}', alpha=0.6)

    plt.xlabel('Total Spend')
    plt.ylabel('Purchase Frequency')
    plt.title('K-Means Clusters')
    plt.legend()
    plt.show()

def view_categories(total):
    num_categories = total['Supercategory'].nunique()
    colors = plt.cm.get_cmap('tab10', num_categories)

    plt.figure(figsize=(10, 6))
    for category in np.sort(total['Supercategory'].unique()):
        category_data = total[total['Supercategory'] == category]
        plt.scatter(category_data['TotalSpend'], category_data['PurchaseFrequency'], 
                color=colors(np.where(np.sort(total['Supercategory'].unique()) == category)[0][0]), label=f'Supercategory {category}', alpha=0.6)

    plt.xlabel('Total Spend')
    plt.ylabel('Purchase Frequency')
    plt.title('K-Means Category')
    plt.legend()
    plt.show()

def train_and_view_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob',
        num_class=3
    )

    #classes_weights = class_weight.compute_sample_weight(
    #    class_weight='balanced',
    #    y=y_train
    #)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

parser = argparse.ArgumentParser(description="A simple script to analyze the Online Retail dataset")
parser.add_argument("--debug", action="store_true", help="Show debug data")
parser.add_argument("--view", action="store_true", help="Display scatter plots of the clusters and supercategories")
parser.add_argument("--model", help="Which model to utilize (xgboost, nn)")

args = parser.parse_args()

df = pd.read_excel('online_retail.xlsx')  

df = get_basic_columns(df)

total_spend, purchase_frequency, avg_spend_per_transaction, avg_hour = get_customer_stats(df)

combined_df = pd.DataFrame({
    'TotalSpend': total_spend.stack(),
    'PurchaseFrequency': purchase_frequency.stack(),
    'AvgSpendPerTransaction': avg_spend_per_transaction.stack(),
    'AvgHourOfPurchase': avg_hour.stack(),
}).reset_index()

combined_df = combined_df.merge(
    df[['CustomerID', 'Country']].drop_duplicates(),
    on='CustomerID',
    how='left'
)

features = combined_df[['TotalSpend', 'PurchaseFrequency', 'AvgSpendPerTransaction']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

best_state = None
best_inertia = float('inf')

for state in range(1, 101):
    kmeans = KMeans(n_clusters=12, random_state=state, init='k-means++')
    kmeans.fit(scaled_features)
    if kmeans.inertia_ < best_inertia:
        best_inertia = kmeans.inertia_
        best_state = state

if args.debug:
    print(f"Best random_state: {best_state} with inertia: {best_inertia}")

kmeans = KMeans(n_clusters=12, random_state=best_state)
combined_df['Cluster'] = kmeans.fit_predict(scaled_features)

num_clusters = combined_df['Cluster'].nunique()
if args.debug:
    print(f"Number of unique clusters: {num_clusters}")

unique_clusters = combined_df['Cluster'].unique()
if args.debug:
    print(f"Unique cluster labels: {unique_clusters}")

cluster_distribution = combined_df['Cluster'].value_counts().sort_index()

if args.debug:
    for cluster, count in cluster_distribution.items():
        print(f"Cluster {cluster}: {count} Customers")

cluster_summary = combined_df.groupby('Cluster')[['TotalSpend', 'PurchaseFrequency', 'AvgSpendPerTransaction']].mean()

cluster_summary['TotalSpend_Cluster'] = cluster_summary.index.map(lambda x: cluster_distribution.get(x, 0) * cluster_summary.loc[x, 'TotalSpend'])

if args.debug:
    print(cluster_summary)

if args.view:
    view_cluster(combined_df)

high_threshold = cluster_summary['TotalSpend'].quantile(0.7)
low_threshold = cluster_summary['TotalSpend'].quantile(0.25)

cluster_summary['Supercategory'] = cluster_summary['TotalSpend'].apply(
    lambda x: assign_supercategory(x, low_threshold, high_threshold)
)

cluster_to_supercategory = cluster_summary['Supercategory'].to_dict()

combined_df['Supercategory'] = combined_df['Cluster'].map(cluster_to_supercategory)

if args.view:
    view_categories(combined_df)

# ML MODEL

combined_df['Target'] = combined_df.groupby('CustomerID')['Supercategory'].shift(-1)  # Next month's supercategory
combined_df = combined_df.dropna(subset=['Target'])

combined_df['Target'] = combined_df.apply(
    lambda row: 2 if row['Target'] > row['Supercategory'] 
                else 1 if row['Target'] == row['Supercategory'] 
                else 0, 
    axis=1
)

label_encoder = LabelEncoder()
combined_df['Country_Encoded'] = label_encoder.fit_transform(combined_df['Country'])

X = combined_df[['CustomerID', 'TotalSpend', 'PurchaseFrequency', 'AvgSpendPerTransaction', 'Cluster']]
y = combined_df['Target']

# 0 -> WENT DOWN
# 1 -> STAY SAME
# 2 -> WENT UP
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

class CVPredictorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=5, out_features=25)
        self.layer_2 = nn.Linear(in_features=25, out_features=3)
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x))

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

model_1 = CVPredictorModel().to(device)

model_0 = nn.Sequential(
    nn.Linear(in_features=5, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=3),
)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.01)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

if args.model == 'xgboost':
    train_and_view_xgboost(X_train, X_test, y_train, y_test)
elif args.model == 'nn':
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

    with torch.inference_mode():
        untrained_preds = model_0(X_test_tensor.to(device))
    if args.debug:
        print(untrained_preds)
        print(f'Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}')
        print(f'Length of test samples: {len(X_test_tensor)}, Shape: {X_test_tensor.shape}')
        print(f'Length of label samples: {len(y_test_tensor)}, Shape: {y_test_tensor.shape}')
        print(f'First 10 predictions: \n{untrained_preds[:10]}')
        print(f'First 10 targets: \n{y_test_tensor[:10]}')
    epochs = 15000
    for epoch in range(epochs):
        model_0.train()
        y_logits = model_0(X_train_tensor)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits, y_train_tensor)
        acc = accuracy_fn(y_train_tensor, y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {acc:.2f}%")
    model_0.eval() 
    with torch.inference_mode():
        test_logits = model_0(X_test_tensor)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test_tensor)
        test_acc = accuracy_fn(y_test_tensor, test_pred)
        print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")

        # Print predictions and targets
        print(classification_report(y_test_tensor, test_pred))



