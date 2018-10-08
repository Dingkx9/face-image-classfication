#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:19:51 2018

@author: MyReservoir
"""

import matplotlib.pyplot as plt
units=[30,60,90,110,150,200]
train=[86, 90.4,91.06, 91.56, 91.3, 91.3]
validation=[84.6, 88.79,89.53,89.53,89.53,89.50]


plt.figure()
plt.plot(units, train, label= 'Training')
plt.plot(units, validation, label = 'Validation')
plt.xlabel('Number of Hidden Units')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Number of Hidden Units vs Accuracy')
plt.show()



alpha=[.0005,.001,.0025,.005,.01,.1,1]
train=[88.6,91.23,93.96,96.26,97.6,82.2,41.1]
validation=[87.68,89.61,90.29,89.72,89.39,80.13,54.30]

plt.figure()
plt.plot(alpha, train, label= 'Training')
plt.plot(alpha, validation, label = 'Validation')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Rate vs Accuracy')
plt.show()