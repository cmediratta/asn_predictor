# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:24:22 2022

@author: William
"""
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

TYPE = ["PRESOCIAL", "SUBSOCIAL", "SOLITARY BUT SOCIAL", "PARASOCIAL", "EUSOCIAL"]

def graph_PCA(dataloader,model):
    
    fig,ax = plt.subplots()
    
    pca = PCA(n_components=2)
    
    x_data = np.stack(np.array(dataloader.dataset)[:,0])
    y_data = dataloader.dataset[:][1]
    pca = PCA(n_components=2)
    
    pca.fit(x_data)
    
    transformed = pca.transform(x_data)
    
    scat = ax.scatter(transformed[:,0]-transformed[:,0].min(), transformed[:,1]-transformed[:,1].min(), c=y_data)   
    
    
    legend1 = ax.legend(*scat.legend_elements())
    ax.add_artist(legend1)
    
    plt.xscale('log')
    plt.xlim([.1,100])
    
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    #plt.yscale('log')
    
    plt.savefig('PCA.png',bbox_inches="tight")

    plt.clf()
    
def confusion_matrix(dl, model):
    
    matrix = np.zeros([5,5])
    
    model.eval()
    
    for X,y in dl:
        yhat = model.forward(X).argmax(1)
        #print(yhat,y.long())
        matrix[y.long(), yhat] += 1
        
    df = pd.DataFrame(matrix, index=TYPE, columns=TYPE).astype(int)
    heatmap = sns.heatmap(df, annot=True, fmt='d')    
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig('confusion.png',bbox_inches="tight")

    plt.clf()
        
    return matrix