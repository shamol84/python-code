# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:34:28 2015

@author: mah228
"""

serial = pd.unique(data.sn.values.ravel())
#data1=data[data['sn']==serial[0]]
#data.rename(columns=lambda x: str(serial[1])+x[0:], inplace=True)
data4=pd.DataFrame()
for f in range(0,len(serial)):
    data1=data[data['sn']==serial[f]]
    data4=data4.append(data1)
    