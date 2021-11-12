import tensorflow as tf
from pandas import DataFrame
import pandas as pd
from sklearn import datasets

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
print("x_data:\n", x_data)
print("y_data:\n", y_data)
# 为表格增加行索引（左侧）和列标签（上方）
x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
# 设置列名对齐
pd.set_option('display.unicode.east_asian_width', True)
print("x_data add index:\n", x_data)
# 新加一列，列标签为‘类别’，数据为y_data
x_data['类别'] = y_data
print("x_data add a column:\n", x_data)
