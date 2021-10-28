#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:45:41 2021

@author: amir
"""

# Imports PIL module 
from PIL import Image
import enum
import numpy as np
import pandas as pd

from numba import jit
from numba import cuda, float32

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mes
from sklearn.metrics import accuracy_score as acc


image = Image.open('input.jpg')

class Pooling(enum.Enum):
   Max = 1
   Min = 2
   Avrage = 3
   Xmode = 4


class Filter(enum.Enum):
   Simpler_box_blur = 1
   Gaussian_blur = 2
   Line_detection_horizontal  = 3
   Line_detection_vertical = 4
   Line_detection_45D = 5
   Line_detection_135D = 6
   Line_detection = 7
   Sobel_edge_horizontal = 8
   Sobel_edge_vertical = 9
   Laplacian_operator = 10
   Laplacian = 11
   Laplacian_gaussian = 12



def padding (img,pad) :
    
    width, height = img.size
      
    new_width = width + pad + pad
    new_height = height + pad + pad
    
    if img.mode == "L" : 
        result = Image.new(img.mode, (250, 250), (0, 0, 0))
    else :
        result = Image.new(img.mode, (new_width, new_height), (0, 0, 0))
        
    result.paste(img, (pad, pad))

    return result


def greyscale(img):
    
    return img.convert('L')


def Avragepooling(img,y,x,Pool):

    pixels = []
            
    for i in reversed(range(1,Pool)):
        for j in reversed(range(1,Pool)):
            #print(-i ,-j)
            #print(-i, j)
            _score = img[y-i][x-j]
            pixels.append(_score)
            _score = img[y-i][x+j]
            pixels.append(_score)
            
    
    for i in range(0,5):
        for j in range(0,5):      
            #print(i ,-j)
            #print(i, j)
            _score = img[y+i][x+j]
            pixels.append(_score)
            _score = img[y+i][x-j]
            pixels.append(_score)
          
    
    maxpool = (sum(pixels)/len(pixels))
    
    return maxpool


def Xpooling(img,y,x):
    

    pixal2=img[y-1][x]

    
    pixal4=img[y][x-1]
    pixal5=img[y][x]
    pixal6=img[y][x+1]

    pixal8=img[y+1][x]

    
    Xpool = (pixal2+pixal4+pixal5+pixal6+pixal8)/5
        
    return Xpool


    
    
    
def conv2dPadding(img,pooling,Padding):
    print("AMMMMMMMMMMMMMMMMMMMMMMMMMMMMm",img.mode)
    if img.mode == "L" : 
        img = img.convert("RGB")
    print("AMMMMMMMMMMMMMMMMMMMMMMMMMMMMm",img.mode)
    if Padding > 0 :
        a = padding(img,Padding)
    b = greyscale(a)
    c = np.asarray(b, np.integer)
    
    Y,X = image.size
    ex = np.zeros((X,Y))
    
    
    ListClum = []
    ListRow = []
    ListDeClum  = []
    ListDeRow = []


    for i in range(Padding,X,Padding):
        if i not in ListRow:
            ListRow.append(i)
        for j in range(Padding,Y,Padding):
            if i > 0 and j > 0 and i < X-1 and j < Y-1 :
                #print(j)
                if j not in ListClum:
                    ListClum.append(j)
                if pooling == Pooling.Avrage :
                    ex[i,j] = Avragepooling(c,i,j,Padding)
                if pooling == Pooling.Xmode :
                    ex[i,j] = Xpooling(c,i,j)
                    
    for i in range(Y):
        if  i < Y-1   :
            if i not in ListClum:
                if i not in ListDeClum:
                    ListDeClum.append(i)
                    
    for i in range(X):
        if  i < X-1   :
            if i not in ListRow:
                if i not in ListDeRow:
                    ListDeRow.append(i)
                    
    for i in range(0,1):
        if i not in ListClum:
            if i not in ListDeClum:
                ListDeClum.append(i)
        if i not in ListDeRow:
                ListDeRow.append(i)
                
    for i in range(Y-1,Y-Padding,-1):
        if i not in ListClum:
            if i not in ListDeClum:
                ListDeClum.append(i)
                
    for i in range(X-1,X-Padding,-1):
        if i not in ListDeRow:
            ListDeRow.append(i)
                
    
    a_del = np.delete(ex,np.array(ListDeClum), 1)

    a_del = np.delete(a_del,np.array(ListDeRow), 0)
    
    out = np.asarray(a_del, np.uint8)
    
    mat = np.reshape(out,out.shape)
    
    newimage = Image.fromarray( mat , 'L')

    return newimage






#Simple box blur

    
def conv2dFiltter(img,filtring,Padding):
    print("StartFilter",img.mode)
    if img.mode == "L" : 
        img = img.convert("RGB")
    print("Change to RGB",img.mode)
    if Padding > 1 :
        a = padding(img,Padding)
        b = greyscale(a)
    else:
        b = greyscale(img)
    c = np.asarray(b, np.integer)
    
    Y,X = img.size
    ex = np.zeros((X,Y))
    
    

    ListDeClum  = []
    ListDeRow = []


    for i in range(Padding,X):
        #print(i,len(ex))
        for j in range(Padding,Y-1):
            if i > 0 and j > 0 and i < X-1 and j < Y-1 :
                

                if filtring == Filter.Simpler_box_blur :
                    ex[i,j] = Simple_box_blur(c,i,j)
                    
                if filtring == Filter.Line_detection_horizontal :
                    ex[i,j] = Line_detection_horizontal(c,i,j)
                if filtring == Filter.Line_detection_vertical :
                    ex[i,j] = Line_detection_vertical(c,i,j)
                if filtring == Filter.Line_detection_45D :
                    ex[i,j] = Line_detection_45D(c,i,j)
                if filtring == Filter.Line_detection_135D :
                    ex[i,j] = Line_detection_135D(c,i,j)                    
                if filtring == Filter.Line_detection :
                    ex[i,j] = Line_detection(c,i,j)
                    
                if filtring == Filter.Sobel_edge_horizontal :
                    ex[i,j] = Sobel_Edge_horizontal(c,i,j)
                if filtring == Filter.Sobel_edge_vertical :
                    ex[i,j] = Sobel_Edge_vertical(c,i,j)
                    
                if filtring == Filter.Laplacian :
                    ex[i,j] = laplacian(c,i,j)
                    
                    
    ListDeRow.append(0)
    ListDeRow.append(X-1)
    ListDeClum.append(0)
    ListDeClum.append(Y-1)
    
    a_del = np.delete(ex,np.array(ListDeClum), 1)

    a_del = np.delete(a_del,np.array(ListDeRow), 0)
    
    out = np.asarray(a_del, np.uint8)
    
    mat = np.reshape(out,out.shape)
    
    newimage = Image.fromarray( mat , 'L')

    return newimage



def conv2dFilttercustom (img,Padding,n1,n2,n3,n4,n5,n6,n7,n8,n9):
    print("StartFilter",img.mode)
    if img.mode == "L" : 
        img = img.convert("RGB")
    print("Change to RGB",img.mode)
    if Padding > 1 :
        a = padding(img,Padding)
        b = greyscale(a)
    else:
        b = greyscale(img)
    c = np.asarray(b, np.integer)
    
    Y,X = img.size
    ex = np.zeros((X,Y))

    ListDeClum  = []
    ListDeRow = []

    for i in range(Padding,X):
        #print(i,len(ex))
        for j in range(Padding,Y-1):
            if i > 0 and j > 0 and i < X-1 and j < Y-1 :
                ex[i,j] = custom(c,i,j,n1,n2,n3,n4,n5,n6,n7,n8,n9)
                    
    ListDeRow.append(0)
    ListDeRow.append(X-1)
    ListDeClum.append(0)
    ListDeClum.append(Y-1)
    
    a_del = np.delete(ex,np.array(ListDeClum), 1)

    a_del = np.delete(a_del,np.array(ListDeRow), 0)
    
    out = np.asarray(a_del, np.uint8)
    
    mat = np.reshape(out,out.shape)
    
    newimage = Image.fromarray( mat , 'L')

    return newimage






def Line_detection_135D(img,y,x):
    
    FILTER =np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
    
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :]) 
    
    return SubColor

def Line_detection_45D(img,y,x):
    
    FILTER =np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
    
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :]) 
    
    return SubColor

def Line_detection_vertical(img,y,x):
    
    FILTER =np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
    
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :]) 
    
    return SubColor

def Line_detection_horizontal(img,y,x):
    
    FILTER =np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])
    
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :]) 
    
    return SubColor

def Line_detection(img,y,x):
    
    FILTER = np.array([[-1,-1,-1],
                       [-1,8,-1],
                       [-1,-1,-1]])
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :]) 
        
    return SubColor

def Sobel_Edge_horizontal(img,y,x):
    
    FILTER = np.array([[-1,-2,-1],
                       [0,0,0],
                       [1,2,1]])
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :])
        
    return SubColor

def Sobel_Edge_vertical(img,y,x):
    
    FILTER = np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]])
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :]) 
        
    return SubColor

def laplacian(img,y,x):
    
    FILTER = np.array([[-1,-1,1],
                       [-1,8,-1],
                       [-1,-1,1]])
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :])
        
    return SubColor

def Simple_box_blur(img,y,x):
    
    FILTER = np.array([[0.1111111111111111,0.1111111111111111,0.1111111111111111],
                       [0.1111111111111111,0.1111111111111111,0.1111111111111111],
                       [0.1111111111111111,0.1111111111111111,0.1111111111111111]])
    
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :])
    
    return SubColor



def custom(img,y,x,n1,n2,n3,n4,n5,n6,n7,n8,n9):
    
    FILTER = np.array([[n1,n2,n3],
                       [n4,n5,n6],
                       [n7,n8,n9]])
    
    COL  = np.array([[img[y-1][x-1],img[y-1][x],img[y-1][x+1]],
                     [img[y][x-1],img[y][x],img[y][x+1]],
                     [img[y+1][x-1],img[y+1][x],img[y+1][x+1]]]  )  
    
    filtermirorr = np.flip(FILTER, (0,1))
    SubColor = np.sum(FILTER[:, :] * COL[: , :])
    
    return SubColor



def Do(Nwhile,n1,n2,n3,n4,n5,n6,n7,n8,n9) :
    
    pic = image
    for i in range(Nwhile):        
        pic = conv2dFilttercustom(pic,1,n1,n2,n3,n4,n5,n6,n7,n8,n9)
    #pic.show()
    pic.save('output.jpg')
    #return pic




#نمونه



#layer1 = conv2d(image,Pooling.Xmode,1)
#layer2 = conv2d(layer1,Pooling.Xmode,1)
#layer3 = conv2d(layer2,Pooling.Avrage,5)


#layer3.show()

