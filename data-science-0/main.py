#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[4]:


black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[7]:


def q1():
   x1 = black_friday.shape
   return (x1)
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[9]:


def q2():
    x2 = black_friday.query("Gender == 'F' & Age == '26-35'").shape[0]
    return (x2)
q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[10]:


def q3():
    x3 = len(black_friday['User_ID'].unique())
    return (x3)
q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[11]:


def q4():
    x4 = len(black_friday.dtypes.unique())
    return (x4)
q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    x5 = black_friday.isnull().sum().max() / black_friday.shape[0]
    return (x5)
q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[19]:


def q6():
    x6 = black_friday.isna().sum().max()
    return (x6)
q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[21]:


def q7():
    x7 = black_friday['Product_Category_3'].mode().values[0]
    return (x7)
q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[7]:


def q8():
    scaler = MinMaxScaler()
    x8 = scaler.fit_transform(black_friday['Purchase'].values.reshape(-1, 1)).mean()
    return float(x8)
q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[46]:


def q9():
    standard = StandardScaler()
    x9 = standard.fit_transform(black_friday['Purchase'].values.reshape(-1, 1))
    return (len(x9[(x9 > -1) & (x9 < 1)]))
q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[4]:


def q10():
    x10 = black_friday[black_friday['Product_Category_2'].isnull()]
    return bool(x10['Product_Category_2'].sum() == x10['Product_Category_3'].sum())
q10()


# In[ ]:




