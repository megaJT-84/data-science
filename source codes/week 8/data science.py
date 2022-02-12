import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from seaborn.rcmod import set_style
from sklearn import preprocessing
plt.rc("font", size = 14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sb
sb.set(style = "white")
sb.set(style = "whitegrid", color_codes = True)