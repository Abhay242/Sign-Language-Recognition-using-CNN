
# coding: utf-8

# In[1]:


import imageio
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import random
get_ipython().run_line_magic('matplotlib', 'inline')
img_files = [f for f in os.walk('C:\\Users\\Legion\\Desktop\\project\\asl_alphabet_train\\asl_alphabet_train\\')] #if f.endswith('.jpg')]
print(len(img_files[1][2]))
print(img_files[0][1])


# In[2]:


for i in range(1,30):
    file_no=2000
    print(i)
    for image in img_files[i][2]:
        if(file_no==0):
            break
        file_no=file_no-1
        pic = imageio.imread(img_files[i][0]+'\\'+image)
        gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
        gray = gray(pic)
        gray=np.uint8(np.reshape(gray,(1,40000)))
        gray=[np.insert(gray[0],0,i)]
        with open(str(img_files[0][1][i-1])+'_train.csv', 'a' ,newline='') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(gray)

