import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
from scipy import ndimage
import csv

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

a = []
b = [3,4]


a.append(b)
a.append([5,6,7])
a.append([5,6,8,9])

#a=np.array(a)
print(np.array(a))

#np.savetxt('hist_vector_list.csv', (a), delimiter=',')

# with open('hist_vector_list.csv','w',newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(a)

hist_vector_list = []
with open('hist_vector_list.csv','r') as f:
    reader = csv.reader(f)
    for r in reader:
        hist_vector_list.append(list(map(int, r[0:1000])))
    #print(hist_vector_list)
    # hist_vector_list = list(reader)
    
digits  = np.array(hist_vector_list)
print(len(digits[30]))
colors =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink","lime"]

print((digits.shape))

tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits)

plt.figure(figsize=(10,10))
plt.xlim(digits_tsne[:,0].min(), digits_tsne[:,0].max()+1)
plt.ylim(digits_tsne[:,1].min(), digits_tsne[:,1].max()+1)

for i in range(len(digits[:])):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str('o'))
plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")
plt.savefig('tsne.png')

