import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
from scipy import ndimage
import csv

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pandas as pd
from sklearn.preprocessing import StandardScaler

def detection_point(img):
    #検出

    pts_list = np.where(img == 255)
    pts_list = np.array(pts_list)

    copylist = pts_list.copy()

    flg = 0
    center_list = []
    center_list = np.array(center_list)
    
    while flg == 0:
        nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(copylist.T)
        distances, indices = nbrs.kneighbors(copylist.T)

        # 近接点の閾値
        th = 5

        # 閾値以下の近接点との重心を計算。
        center_list = []
        for d,i in zip(distances,indices):
            i = i[np.where(d < th)]
            pts = copylist.T[i]
            center = np.mean(pts,axis=0)
            center_list.append(center)
        center_list = np.array(center_list).astype(np.int32)

        #重複を削除
        center_list = np.unique(center_list,axis=0)    
        center_list = center_list.T

        if copylist.shape == center_list.shape:
            flg = 1
        copylist = center_list

    return center_list

def show_point_scatter(center_list, x, y, file_name=None):
    fig = plt.figure(dpi=150, facecolor='white')
    plt.gray()
    plt.imshow(img)
    plt.scatter(center_list[1],center_list[0],s=1,c='red',alpha=0.7)
    plt.xlim(0,x)
    plt.ylim(y,0)
    if not file_name is None:
        plt.savefig(file_name)
    plt.show()
    
def cal_rdf(distances, indices, th=100):
    #thピクセル以内に存在する他粒子との距離を計算し、ヒストグラムを作成
    dist_list = []

    for d,i in zip(distances,indices):
        d = d[np.where(d < th)][1:]
        dist_list.extend(d)
    dist_list = np.array(dist_list)
    dist_hist = np.histogram(dist_list,range=(0,th),bins=th)

    #RDFの計算
    rdf = dist_hist[0]/(2*np.pi*dist_hist[1][1:])

    return rdf

def show_rdf(rdf, th, l, h, file_name=None):
    #rdfの最大値とその時のピクセルの値
#     mm = np.argmax(rdf)
#     mi = rdf[np.argmax(rdf)]

    #rdfの全体表示
    fig = plt.figure(dpi=150,facecolor='white')
#     fig.text(0.45, 0.85, '(' + str(mm) + ', ' + str(round(mi, 2)) + ')')
    fig.text(0.45, 0.85, '(' + str(th) + ', ' + str(round(rdf[th], 2)) + ')')
    plt.plot(range(l,h), rdf[l:h])
    plt.scatter(th, rdf[th])
    plt.xlabel('Distance(px)')
    plt.ylabel('Radial Distribution')
    if not file_name is None:
        plt.savefig(file_name)
    plt.show()

def detection_th(rdf):
    fil = [0.5,0.5]
    rdf_fil = np.convolve(rdf, fil, mode='same')

    peak_max_list = peak(rdf_fil,mode='max')
    peak_min_list = peak(rdf_fil,mode='min')
    peak_max_list.sort()
    peak_min_list.sort()
    
    ma=0
    m = np.mean(rdf)
    for x in peak_max_list:
        if rdf[x] > m:
            ma = x
            break
    mi=0
    for x in peak_min_list:
        if ma < x:
            mi = x
            break
            
    return mi


def peak(res, mode='max'):
    mlen = len(res)
    dif = np.diff(res,n=1)
    ddif = dif[1:(mlen-1)]*dif[0:(mlen-2)]
    
    ppos = []
    for i in range(len(ddif)):
        if mode == 'max':
            if ddif[i] < 0 and dif[i+1] < 0:
                ppos.append(i+1)
        elif mode == 'min':
            if ddif[i] < 0 and dif[i+1] > 0:
                ppos.append(i+1)
        else:
            return None

    return ppos

def first_neighbors(distances, indices, thres):
    # #第1近接原子の計数
    co_list = []
    for d,i in zip(distances,indices):
        d = d[np.where(d < thres)][1:]
        co_list.append(d.shape[0])
        
    return co_list

def show_co_list(co_list, file_name=None):
    fig = plt.figure(dpi=150,facecolor='white')
    plt.hist(co_list,
             bins=np.linspace(np.min(co_list)-0.5,np.max(co_list)+0.5,np.max(co_list)-np.min(co_list)+2),
             rwidth=0.9)
    plt.xlabel('Coordination number for 1st neighbors')
    plt.ylabel('Density')
    if not file_name is None:
        plt.savefig(file_name)
    plt.show()

def show_first_scatter(co_list, x, y, file_name=None):
    fig = plt.figure(dpi=300,facecolor='white')
    plt.scatter(center_list[1],center_list[0],s=1,c=co_list,cmap='rainbow',alpha=0.7,linewidths=0.1)
    plt.colorbar()
    plt.clim(np.min(co_list), np.argmax(np.bincount(co_list)))
    plt.xlim(0,x)
    plt.ylim(y,0)
    if not file_name is None:
        plt.savefig(file_name)
    plt.show()
    
def standardization(X):
    y = []
    m = np.mean(X)
    s = np.std(X)
    for x in X:
        y.append((x - m)/s)
    return y

def minmax(X):
    return (X - min(X)) / (max(X) - min(X))


if __name__ == "__main__":

    img_list = os.listdir('komon_data')

    hist_vector_list = []

    for img_name in img_list:
        
        #画像読み込み
        img = cv2.imread('komon_data/' + img_name, 0)
        #画像の高さ, 幅を取得
        h, w = img.shape[:2]
        
        #点を検出
        #center_list = detection_point(img)
        
        #点の座標を保存
        #np.savetxt('center_list_data/' + img_name.replace('.png', '') + '.csv', center_list.T,  delimiter=',')
        #データを読み込む
        center_list = np.loadtxt('center_list_data/' + img_name.replace('.png', '') + '.csv', delimiter=',').T
        
        #点を表示
        show_point_scatter(center_list, 500, 500)
        #show_point_scatter(center_list, 500, 500, file_name='komon_result/' + img_name)
        
        
        #近接原子の位置と距離を100個まで計算
        nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(center_list.T)
        distances, indices = nbrs.kneighbors(center_list.T)

        #RDFの計算
        rdf = cal_rdf(distances, indices, th=100)
        #rdf = standardization(rdf)
        
        thres = detection_th(rdf)
        #RDF表示
        show_rdf(rdf,thres,0,100)
        #show_rdf(rdf, thres, 0, 100, file_name='rdf/' + img_name)

        # #第1近接原子の計数
        co_list = first_neighbors(distances, indices, thres)

        show_co_list(co_list)
        show_first_scatter(co_list, w, h)
        #show_co_list(co_list, file_name='histogram/' + img_name)
        #show_first_scatter(co_list, w, h, file_name='color/' + img_name)
        
        hist_vector_list.append(standardization(np.histogram(co_list,bins=16,range=(-0.5,15.5))[0]))
        
        print(img_name)

        

    # with open('hist_vector_list.csv','w',newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(hist_vector_list)
    # with open('hist_vector_list_histogram.csv','w',newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows()

    np.savetxt('hist_vector_list.csv', hist_vector_list, delimiter=',')
    # hist_vector_list = np.loadtxt('hist_vector_list', delimiter=',')

    digits  = hist_vector_list
    colors =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink","lime"]

    tsne = TSNE(random_state=42)
    digits_tsne = tsne.fit_transform(hist_vector_list)

    plt.figure(figsize=(10,10))
    plt.xlim(digits_tsne[:,0].min(), digits_tsne[:,0].max()+1)
    plt.ylim(digits_tsne[:,1].min(), digits_tsne[:,1].max()+1)

    for i in range(len(digits)):
        plt.scatter(digits_tsne[i, 0], digits_tsne[i, 1])
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    plt.savefig('tsne.png')    
        
        
    
    




