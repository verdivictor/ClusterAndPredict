import torch
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # MPL BACKEND
import matplotlib.pyplot as plt

df = pd.read_excel('online_retail.xlsx', index_col=0)  

df = df.sort_values('InvoiceDate', ascending=True)

df['Original_Quantity'] = df['Quantity']
df['Original_UnitPrice'] = df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['Hour'] = df['InvoiceDate'].dt.hour
df['Month'] = df['InvoiceDate'].dt.to_period('M')

df.loc[df['Quantity'] < 0, 'Quantity'] = 0
df.loc[df['Quantity'] > 10000, 'Quantity'] = 0
df.loc[df['UnitPrice'] < 0, 'UnitPrice'] = 0

excluded_products = ['DOT', 'POST', 'M']

#df['StockCode'] = pd.to_numeric(df['StockCode'], errors='coerce')
df = df[~df['StockCode'].isin(excluded_products)]

df = df.dropna(subset=['StockCode'])

df['Sales'] = df['Quantity'] * df['UnitPrice']

hourly_purchases = df.groupby('Hour')['Sales'].sum()

top_products = df.groupby('StockCode')['Sales'].sum().nlargest(10).index

df_top_products = df[df['StockCode'].isin(top_products)]

monthly_hourly_sales = df_top_products.groupby(['Month', 'Hour', 'StockCode'])['Sales'].sum().unstack()

stockcode_to_description = df.drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()

for month in monthly_hourly_sales.index.levels[0]:
    plt.figure(figsize=(14, 8))
    for stock_code in monthly_hourly_sales.columns:
        #plt.scatter(
        #    monthly_hourly_sales.loc[month].index,
        #    monthly_hourly_sales.loc[month][stock_code],
        #    label=f'StockCode: {stock_code}',
        #    alpha=0.7
        #)
        description = stockcode_to_description[stock_code]
        plt.plot(
        monthly_hourly_sales.loc[month].index,
        monthly_hourly_sales.loc[month][stock_code],
        #label=f'StockCode: {stock_code}',
        label=description,  
        marker='o',
        linestyle='-',
        alpha=0.7
        )


        plt.xlabel('Hour of the Day')
        plt.ylabel('Total Sales')
        plt.title(f'Top 10 Products: Sales by Hour of the Day ({month})')
        plt.xticks(range(24))  # Ensure all hours are shown
        plt.legend(title='StockCode', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot to a PNG file
        plt.savefig(f'views/sales_per_hour_of_day_{month}_line.png', bbox_inches='tight', dpi=300) 
#plt.show()
