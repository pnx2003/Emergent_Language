import pickle
import numpy as np
incom_symbol = 'data/dqn_incom_symbol.pkl'
out_symbol = 'data/dqn_out_symbol.pkl'

#read the pkl
with open(incom_symbol, 'rb') as f:
    incom_symbol = pickle.load(f)

print(incom_symbol)
# change the array in incom_symbol to the histogram
# normalize the incom_symbol
incom_symbol = np.array(incom_symbol)
incom_symbol = incom_symbol / np.sum(incom_symbol)

#incom_symbol 是 纵坐标
#0-31为纵坐标，画直方图
import matplotlib.pyplot as plt
plt.bar(range(32), incom_symbol)

# the x-axis is called by "symbol index"
# the y-axis is labled by "probability"
plt.xlabel('symbol index')
plt.ylabel('probability')
plt.show()
# save the image in the 16vocabin.png
plt.savefig('data/16vocabin.png')
