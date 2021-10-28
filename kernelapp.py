#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 00:54:27 2021

@author: amir
"""
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import kernel 
import cnn_app

class MainWindow(QtWidgets.QMainWindow, kernel.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.ui = kernel.Ui_MainWindow() 
        
def dosomtion(text):

    cnn_app.Do(int(form.Mrip.text()),float(form.m1_1.text()),float(form.M1_2.text()),float(form.M1_3.text()),
           float(form.M2_1.text()),float(form.M2_2.text()),float(form.M2_3.text()),
           float(form.M3_1.text()),float(form.M3_2.text()),float(form.M3_3.text()))

    pixmap =  QtGui.QPixmap('output.jpg')
    form.picoutput.setPixmap(pixmap)
    form.picoutput.show()


app = QApplication(sys.argv)
form = MainWindow()
form.apply.clicked.connect(dosomtion)

form.show()
app.exec_()


