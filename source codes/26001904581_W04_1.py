#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Tian Xiaoyang
# 26001904581


# In[ ]:


import pandas as pd
import numpy as np


# In[30]:


customer_master = pd.read_csv('customer_master.csv')
print("------------------------------------------------------")
print("customer_master")
print(customer_master.columns)
print(customer_master.head())
print("------------------------------------------------------")

item_master = pd.read_csv('item_master.csv')
print("item_master")
print(item_master.columns)
print(item_master.head())
print("------------------------------------------------------")


transaction_1 = pd.read_csv('transaction_1.csv')
print("transaction_1")
print(transaction_1.columns)
print(transaction_1.head())
print("------------------------------------------------------")

transaction_detail_1 = pd.read_csv('transaction_detail_1.csv')
print("transaction_detail_1")
print(transaction_detail_1.columns)
print(transaction_detail_1.head())
print("------------------------------------------------------")

transaction_2 = pd.read_csv('transaction_2.csv')
print("transaction_2")
print(transaction_2.columns)  
print(transaction_2.head())
print("------------------------------------------------------")

transaction_detail_2 = pd.read_csv('transaction_detail_2.csv')
print("transaction_detail_2")
print(transaction_detail_2.columns)  
print(transaction_detail_2.head())
print("------------------------------------------------------")

transaction = pd.concat([transaction_1, transaction_2])
print("transaction_join")
print(transaction.columns)
print(transaction.head())
print("------------------------------------------------------")

transaction_detail = pd.concat([transaction_detail_1, transaction_detail_2])
print("transaction_detail_join")
print(transaction_detail.columns)
print(transaction_detail.head())
print("------------------------------------------------------")


join_data = pd.merge(transaction, transaction_detail)
print("join_data")
print(join_data.columns)
print(join_data.head())
print("------------------------------------------------------")

join_data_1 = pd.merge(join_data, customer_master)
print("join_data_1")
print(join_data_1.columns)
print(join_data_1.head())
print("------------------------------------------------------")

join_data_2 = pd.merge(join_data_1, item_master)
print("join_data_2")
print(join_data_2.columns)
print(join_data_2.head())
join_data_2.columns
print("------------------------------------------------------")


# In[ ]:




