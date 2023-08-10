#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
s1=pd.Series([4,6,8,10])
print('Series object 1:')
print(s1)


# In[3]:


import pandas as pd
s1=pd.Series([11,21,31,41])
print('Series object 2:')
print(s1)


# In[5]:


import pandas as pd
obj1=pd.Series(range(5))
print(obj1)


# In[6]:


import pandas as pd
obj1=pd.Series(['o','h','o'])
print('Series object:')
print(obj1)


# In[7]:


import pandas as pd
print('Series object is:')
s4=pd.Series("So funny")
print(s4)


# In[8]:


import pandas as pd
s5=pd.Series(["I","am","laughing"])
print("Series object:")
print(s5)


# In[12]:


import pandas as pd
s6=pd.Series(np.linspace(24,64,5))
print(s6)


# In[13]:


import pandas as pd
s7=pd.Series(np.tile([3,5],2))
print(s7)


# In[14]:


import pandas as pd
obj5=pd.Series({'Jan':31,'Feb':28,'Mar':31})
print(obj5)


# In[15]:


import pandas as pd
stu={'A':39,'B':41,'C':42,'D':44}
s8=pd.Series(stu)
print(s8)


# In[17]:


import pandas as pd
medalsWon=pd.Series(10,index=range(0,1))
medals2=pd.Series(15,index=range(1,6,2))
ser2=pd.Series('Yet to start',index=['Indore','Delhi','Shimla'])
print(medalsWon)
print(medals2)
print(ser2)


# In[18]:


import pandas as pd
s10=pd.Series(200,index=range(2020,2029,2))
print(s10)


# In[20]:


import pandas as pd
arr=["Jan","feb","Mar","Apr"]
mon=[31,56,54,65]
obj3=pd.Series(data=arr,index=mon)
print(obj3)


# In[21]:


import pandas as pd
arr=["Jan","feb","Mar","Apr"]
mon=[31,56,54,65]
obj3=pd.Series(data=mon,index=arr)
print(obj3)


# In[22]:


import pandas as pd
obj4=pd.Series(data=[32,34,35],index=['A','B','C'])
print(obj4)


# In[23]:


import pandas as pd
obj4=pd.Series([6.5,np.NaN,2.34])
print(obj4)


# In[24]:


import pandas as pd
s1=pd.Series(range(1,15,3),index=[x for x in 'abcde'])
print(s1)


# In[25]:


import pandas as pd
section=['A','B','C','D']
contri=[6700,5600,5000,5200]
s11=pd.Series(data=contri,index=section)
print(s11)


# In[29]:


import pandas as pd
a=np.arange(9,13)
print(a)
obj7=pd.Series(index=a,data=a*2)
print(obj7)
obj8=pd.Series(index=a,data=a**2)
print(obj8)
lst=[9,10,11,12]
obj8=pd.Series(data=(2*lst))
print(obj8)


# In[30]:


import pandas as pd
import numpy as np
section=['A','B','C','D','E']
contri1=np.array([6700,5600,5000,5200,np.NaN])
s12=pd.Series(data=contri1*2,index=section,dtype=np.float32)
print(s12)


# In[32]:


import pandas as pd
import numpy as np
section=['A','B','C','D','E']
contri1=np.array([6700,5600,5000,5200,np.NaN])
s12=pd.Series(data=contri1*2,index=section,dtype=np.float32)
print(s12)


# In[36]:


import pandas as pd
arr=[31,28,31,30]
mon=['Jan','Feb','Mar','Apr']
obj5=pd.Series(data=arr,index=mon,dtype=np.float64)
print(obj5)
print(obj5.index)
print(obj5.values)


# In[7]:


import pandas as pd
import numpy as np
obj2=pd.Series([3.5,5.,6.5,8.])
print(obj2)
obj3=pd.Series([6.5,np.NaN,2.34])
print(obj3)
print(obj3[1: ])
print(obj3[2:5])
print(obj3[0: :2])
print(obj3[: :-1])
print(obj2.dtype)
print(type(obj2))
print(obj2.ndim)
print(obj2.size,obj3.size)
print(obj2.nbytes,obj3.nbytes)
print(obj2.hasnans)
print(obj2.count())
print(len(obj3))


# In[12]:


import pandas as pd
section=['A','B','C','D']
contri=[6700,5600,5000,5200]
s11=pd.Series(data=contri,index=section)
print(s11)
print(s11[:2]*100)
print(s11[0])


# In[18]:


import pandas as pd
s1=pd.Series([11,21,31,41])
print('Series object 2:')
print(s1)
s1[0]=54
print(s1)
s1[2:4]=59
print(s1)
s1.index=['a','b','c','d']
print(s1)


# In[22]:


import pandas as pd
s1=pd.Series([11,21,31,41,98,65,34,61,23,98])
print('Series object 2:')
print(s1)
print(s1.head())
print(s1.head(7))
print(s1.tail())
print(s1.tail(7))


# In[23]:


import pandas as pd
s1=pd.Series([11,21,31,41,98,65,34,61,23,98])
print('Series object 2:')
print(s1)
print(s1+2)
print(s1*3)
print(s1>15)
print(s1**2)


# In[24]:


import pandas as pd
s1=pd.Series([11,21,31,41,98,65,34,61,23,98])
s2=pd.Series([10,20,30,40,50,60,70,80,90,100])
print(s1)
print(s2)
print(s1+s2)


# In[25]:


import pandas as pd
c11=pd.Series(data=[30,40,50],index=['Science','Commerce','Humanities'])
c12=pd.Series(data=[35,45,55],index=['Hindi','Maths','english'])
print("Total no. of students")
print(c11+c12)


# In[26]:


import pandas as pd
population=pd.Series([15726342176,7386482,97382547,97328354],
index=['Delhi','Mumbai','Sikkim','Jabalpur'])
avgincome=pd.Series([9278346,96472937487,927285398,38645745],
index=['Delhi','Mumbai','Kolkata','Chennai'])
perCapita=avgincome/population
print('Population in four metro cities')
print(population)
print("avg. income in four metro cities")
print(avgincome)
print("per capita  income in four metro cities")
print(perCapita)


# In[27]:


import pandas as pd
info=pd.Series(data=[31,41,51])
print(info)
print(info>40)
print(info[info>40])


# In[28]:


import pandas as pd
print("Contribution> 5500 by:")
print(s11[s11>5500])


# In[31]:


import pandas as pd
stu={'A':39,'B':41,'C':42,'D':44}
s8=pd.Series(stu)
s8.sort_values()
print(s8)
s8.sort_values(ascending=False)
print(s8)
s8.sort_index()
print(s8)


# In[34]:


import pandas as pd
stu={'A':39,'B':41,'C':42,'D':44}
s8=pd.Series(stu)
s9=s8.reindex(['e','d','c','b',])
print(s9)


# In[ ]:





# In[39]:


import pandas as pd
sales={'yr1':{'qtr1':34500,'qtr2':56000,'qtr3':47000,'qtr4':49000},
      'yr2':{'qtr1':44900,'qtr2':46100,'qtr3':57000,'qtr4':59000}}
dfsales=pd.DataFrame(sales)
print(dfsales)


# In[40]:


import pandas as pd
narr1=np.array([[1,2,3],[4,5,6]],np.int32)
narr1.shape


# In[1]:


import pandas as pd
import numpy as np
arr1=np.array([[11,12],[13,14],[15,16]],np.int32)
dtf2=pd.DataFrame(arr1)
print(dtf2)


# In[3]:


narr1=np.array([[1,2,3],[4,5,6]],np.int32)
narr1.shape
dtf1=pd.DataFrame(narr1)
print(dtf1)


# In[9]:


narr2=np.array([[11.5,21.2,33.8],[40,50,60],[212.3,301.5,405.2]])
dtf3=pd.DataFrame(narr2,columns=['First','Second','Third'],index=['A','B','C'])
print(dtf3)
narr3=np.array([[101.5,201.5],[400,500,600,700],[212.3,301.5,405.2]])
print(narr3)


# In[11]:


staff=pd.Series([20,36,44])
salaries=pd.Series([166000,246000,563000])
school={'people':staff,'Amount':salaries}
dtf4=pd.DataFrame(school)
print(dtf4)


# In[15]:


narr2=np.array([[11.5,21.2,33.8],[40,50,60],[212.3,301.5,405.2]])
dtf3=pd.DataFrame(narr2,columns=['First','Second','Third'],index=['A','B','C'])
print(dtf3)
narr3=np.array([[101.5,201.5],[400,500,600,700],[212.3,301.5,405.2]])
print(narr3)
print(dtf3.index)
print(dtf3.columns)
print(dtf3.axes)
print(dtf3.dtypes)
print(dtf3.size)
print(dtf3.shape)
print(dtf3.ndim)
print(dtf3.empty)


# In[18]:


narr2=np.array([[11.5,21.2,33.8],[40,50,60],[212.3,301.5,405.2]])
dtf3=pd.DataFrame(narr2,columns=['First','Second','Third'],index=['A','B','C'])
print(dtf3)
narr3=np.array([[101.5,201.5],[400,500,600,700],[212.3,301.5,405.2]])
print(narr3)
print(dtf3.count)
print(dtf3.count(axis='index'))
print(dtf3.count(1))
print(dtf3.count(axis='columns'))


# In[20]:


import pandas as pd
df=pd.DataFrame({'Weight':[42,75,66],'Name':['Arnav','Charles','Guru'],
                                     'Age':[15,22,35]})
print('original dataframe')
print(df)
print('Transpose')
print(df.T)


# In[25]:


narr2=np.array([[11.5,21.2,33.8],[40,50,60],[212.3,301.5,405.2]])
dtf3=pd.DataFrame(narr2,columns=['First','Second','Third'],index=['A','B','C'])
print(dtf3)
narr3=np.array([[101.5,201.5],[400,500,600,700],[212.3,301.5,405.2]])
print(dtf3.values)//numpy representation


# In[27]:


narr2=np.array([[11.5,21.2,33.8],[40,50,60],[212.3,301.5,405.2]])
dtf3=pd.DataFrame(narr2,columns=['First','Second','Third'],index=['A','B','C'])
print(dtf3)
narr3=np.array([[101.5,201.5],[400,500,600,700],[212.3,301.5,405.2]])
print(dtf3['First'])
print(dtf3.First)


# In[30]:


narr2=np.array([[11.5,21.2,33.8],[40,50,60],[212.3,301.5,405.2]])
dtf3=pd.DataFrame(narr2,columns=['First','Second','Third'],index=['A','B','C'])
print(dtf3)
narr3=np.array([[101.5,201.5],[400,500,600,700],[212.3,301.5,405.2]])
print(dtf3[['First','Second']])
print(dtf3[['Second','First']])


# In[38]:


narr2=np.array([[11.5,21.2,33.8],[40,50,60],[212.3,301.5,405.2]])
dtf3=pd.DataFrame(narr2,columns=['First','Second','Third'],index=['A','B','C'])
print(dtf3)
narr3=np.array([[101.5,201.5],[400,500,600,700],[212.3,301.5,405.2]])
print(dtf3.loc['A',:])
print(dtf3.loc['A':'B',:])
print(dtf3.loc['A':'C',:])


# In[15]:


import pandas as pd
Sales={'yr1':{'qtr1':34500,'qtr2':56000,'qtr3':47000,'qtr4':49000},
      'yr2':{'qtr1':44900,'qtr2':46100,'qtr3':57000,'qtr4':59000}}
dfsales=pd.DataFrame(Sales)
print(dfsales)
print(dfsales.iloc[0:2,1:3])
print(dfsales.yr1['qtr2'])
print(dfsales.iat[1,1])


# In[19]:


import pandas as pd
Sales={'yr1':{'qtr1':34500,'qtr2':56000,'qtr3':47000,'qtr4':49000},
      'yr2':{'qtr1':44900,'qtr2':46100,'qtr3':57000,'qtr4':59000}}
dfsales=pd.DataFrame(Sales)
print(dfsales)
del dfsales['yr2']
print(dfsales)


# In[25]:


import pandas as pd
Sales={'yr1':{'qtr1':34500,'qtr2':56000,'qtr3':47000,'qtr4':49000},
      'yr2':{'qtr1':44900,'qtr2':46100,'qtr3':57000,'qtr4':59000}}
dfsales=pd.DataFrame(Sales)
print(dfsales)


# In[26]:


import pandas as pd
dfsales=pd.DataFrame(dfsales)
del dfsales['yr1']
dfsales=dfsales.drop(['qtr3'])
print(dfsales)


# In[27]:


import pandas as pd
disales={'yr1':{'qtr1':34500,'qtr2':56000,'qtr3':47000,'qtr4':49000},
         'yr2':{'qtr1':44900,'qtr2':46100,'qtr3':57000,'qtr4':59000},
         'yr3':{'qtr1':54500,'qtr2':46100,'qtr3':57000,'qtr':58500}}
df1=pd.DataFrame(disales)
for (row,rowSeries) in df1.iterrows():
    print("row index:",row)
    print("containing:")
    i=0
    for val in rowSeries:
        print("At",i,"position:",val)
        i=i+1
         


# In[29]:


import pandas as pd
disales={'yr1':{'qtr1':34500,'qtr2':56000,'qtr3':47000,'qtr4':49000},
         'yr2':{'qtr1':44900,'qtr2':46100,'qtr3':57000,'qtr4':59000},
         'yr3':{'qtr1':54500,'qtr2':46100,'qtr3':57000,'qtr':58500}}
df1=pd.DataFrame(disales)
for (col,colSeries) in df1.iteritems():
    print("col index:",col)
    print("containing:")
    print(colSeries)


# In[32]:


import pandas as pd
dict={'Name':["Ram","Pam","Sam"],
     'Marks':[70,95,80]}
df=pd.DataFrame(dict,index=['Rno.1','Rno.2','Rno.3'])
for i,j in df.iterrows():
    print(j)
    print("----------------")


# In[33]:


import pandas as pd
dict={'Name':["Ram","Pam","Sam"],
     'Marks':[70,95,80]}
df=pd.DataFrame(dict,index=['Rno.1','Rno.2','Rno.3'])
for i,j in df.iteritems():
    print(j)
    print("----------------")


# In[34]:


import pandas as pd
dict={'Name':["Ram","Pam","Sam"],
     'Marks':[70,95,80]}
df=pd.DataFrame(dict,index=['Rno.1','Rno.2','Rno.3'])
for r,row in df.iterrows():
    print(row['Marks'])
    print("---------")


# In[36]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df=pd.DataFrame(people)
print(df)
people={'Sales':{1:76,2:457,3:87},
        'Marketing':{1:65,2:12,3:54}}
df=pd.DataFrame(people)
print(df)


# In[38]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,3:87},
        'Marketing':{1:65,2:12,3:54}}
df2=pd.DataFrame(people)
print(df2)
df3=df1+df2
print(df3)


# In[39]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,},
        'Marketing':{1:65,2:12,}}
df2=pd.DataFrame(people)
print(df2)
df3=df1+df2
print(df3)


# In[40]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,3:87},
        'Marketing':{1:65,2:12,3:54}}
df2=pd.DataFrame(people)
print(df2)
print(df1.add(df2))


# In[41]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,3:87},
        'Marketing':{1:65,2:12,3:54}}
df2=pd.DataFrame(people)
print(df2)
print(df1.radd(df2))


# In[42]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,3:87},
        'Marketing':{1:65,2:12,3:54}}
df2=pd.DataFrame(people)
print(df2)
df3=df1-df2
print(df3)


# In[43]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,},
        'Marketing':{1:65,2:12,}}
df2=pd.DataFrame(people)
print(df2)
df3=df1-df2
print(df3)


# In[44]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,},
        'Marketing':{1:65,2:12,}}
df2=pd.DataFrame(people)
print(df2)
print(df1.sub(df2))


# In[45]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,},
        'Marketing':{1:65,2:12,}}
df2=pd.DataFrame(people)
print(df2)
print(df1.rsub(df2))


# In[46]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,},
        'Marketing':{1:65,2:12,}}
df2=pd.DataFrame(people)
print(df2)
df3=df1*df2
print(df3)


# In[47]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,},
        'Marketing':{1:65,2:12,}}
df2=pd.DataFrame(people)
print(df2)
print(df1.mul(df2))


# In[48]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,},
        'Marketing':{1:65,2:12,}}
df2=pd.DataFrame(people)
print(df2)
df3=df1/df2
print(df3)


# In[49]:


people={'Sales':{1:13,2:78,3:23},
        'Marketing':{1:87,2:98,3:20}}
df1=pd.DataFrame(people)
print(df1)
people={'Sales':{1:76,2:457,},
        'Marketing':{1:65,2:12,}}
df2=pd.DataFrame(people)
print(df2)
print(df1.div(df2))


# In[52]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.min())


# In[53]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.max())


# In[54]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.min(axis=1))


# In[55]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.max(axis=1))


# In[56]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.min(skipna=True,numeric_only=True))


# In[57]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.max(skipna=True,numeric_only=True))


# In[58]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.max(axis=1,skipna=True))


# In[59]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.idxmin())


# In[60]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.idxmax())


# In[61]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.mode(axis=1))


# In[62]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.median(axis=1))


# In[63]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.mode(axis=1))


# In[64]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.mode())


# In[65]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.mode(axis=1))


# In[66]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.median(axis=1))


# In[67]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.mean(axis=1))


# In[68]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.count(axis=1))


# In[69]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.sum(axis=1))


# In[70]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.mode())


# In[71]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.sum())


# In[72]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.quantile([0.25,0.5,0.75,1.0]))


# In[73]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.quantile([0.25,0.5,0.75,1.0],axis=1))


# In[74]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.std(axis=1))


# In[75]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.std())


# In[76]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.var(axis=1))


# In[77]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.var())


# In[78]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.describe())


# In[79]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.describe(include='all'))


# In[80]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.info())


# In[81]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.head())


# In[82]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.tail())


# In[83]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.head(n=3))


# In[84]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.tail(n=4))


# In[85]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.cumsum())


# In[86]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.cumsum(axis='rows'))


# In[87]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.cumsum(axis='columns'))


# In[89]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf['Wheat'].min())


# In[91]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf['Wheat'].count())


# In[92]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf[['Wheat','Rice']].count())


# In[93]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.loc[:,:].count(axis=1))


# In[94]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.iloc[:,2:].count())


# In[95]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.loc[:,:].count())


# In[96]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.iloc[2:,:].count(axis=1))


# In[97]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.iloc[2:4,:].max(axis=1))


# In[105]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.pivot(index='Country',columns='Tutor',values='Classes'))


# In[106]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.pivot(index='Tutor',columns='Country',values='Classes'))


# In[107]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.pivot(index='Tutor',columns='Country'))


# In[108]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.pivot(index='Tutor',columns='Country').fillna(0))


# In[113]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1)


# In[114]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.pivot_table(index='Tutor',values='Classes',aggfunc='sum'))


# In[115]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.pivot_table(index='Tutor',values='Classes',aggfunc='count'))


# In[116]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.pivot_table(index=['Tutor','Country'],values=['Classes'],aggfunc='sum'))


# In[117]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.pivot_table(index=['Tutor','Country'],values=['Classes'],aggfunc='count'))


# In[120]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.sort_values('Country'))


# In[121]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.sort_values('Tutor'))


# In[123]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.sort_values(['Country','Tutor']))


# In[124]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.sort_values(by=['Tutor','Country']))


# In[125]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.sort_values(by=['Tutor','Country'],ascending=False))


# In[126]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.sort_index(ascending=False))


# In[127]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.mad())


# In[128]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.hist())


# In[129]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.hist(column="Classes"))


# In[132]:


oututD={'Tutor':['Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                 'Tahira','Gurjyot','Anusha','Jacob','Venkat',
                ],'Classes':[23,54,23,78,97,56,34,56,76,6,45,56,87,44,23,4,6,87,34,21],
                 'Quarter':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4],
       'Country':['USA','UK','Japan','USA','Brazil','USA','Japan','Brazil','USA','UK','Brazil','USA','UK'
                 ,'Brazil','USA','Japan','Japan','Brazil','UK','USA']
       }
df1=pd.DataFrame(oututD)
print(df1.hist(column='Quarter'))


# In[133]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.isnull())


# In[134]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
print(pdf)
print(pdf.notnull())


# In[135]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
pdf1=pdf.dropna()
print(pdf1)


# In[136]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
pdf1=pdf.dropna(how='all')
print(pdf1)


# In[137]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
pdf1=pdf.dropna(axis=1)
print(pdf1)


# In[145]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
pdf1=pdf.fillna(fillValues)
print(pdf1)


# In[146]:


fillValues={
'Tutor':'Punjab',
'Classes':87.5,
'Country':'Russia',

}
iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
print(pdf)
pdf1=pdf.fillna(fillValues)
print(pdf1)


# In[148]:


iprod={'Tutor':['Andhra Pradesh','Gujarat','Goa','Kerala','Sikkim','Rajasthan','Tripura'],
      'Classes':[76.9,987,98,987,98,43,98],
      'Country':['USA','UK','Japan','USA','Brazil','Canada','France']}
pdf=pd.DataFrame(iprod)
print(pdf)
pdf1=pdf.fillna({'Classes':76,'Country':'Russia'})
print(pdf1)


# In[151]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
iprod1={'Rice':{'Andhra Pradesh':365,'Gujarat':871,'Goa':81.0,'Kerala':76,'Sikkim':84,'Rajasthan':83,'Tripura':65},
      'Wheat':{'Andhra Pradesh':7.9,'Gujarat':7,'Goa':8,'Kerala':97,'Sikkim':8,'Rajasthan':3,'Tripura':8},
      'Maize':{'Andhra Pradesh':5,'Gujarat':6,'Goa':8,'Kerala':717,'Sikkim':837,'Rajasthan':86,'Tripura':97}}
pdf1=pd.DataFrame(iprod1)
print(pdf1)
pdf3=pd.concat([pdf,pdf1])
print(pdf3)


# In[152]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
iprod1={'Rice':{'Andhra Pradesh':365,'Gujarat':871,'Goa':81.0,'Kerala':76,'Sikkim':84,'Rajasthan':83,'Tripura':65},
      'Wheat':{'Andhra Pradesh':7.9,'Gujarat':7,'Goa':8,'Kerala':97,'Sikkim':8,'Rajasthan':3,'Tripura':8},
      'Maize':{'Andhra Pradesh':5,'Gujarat':6,'Goa':8,'Kerala':717,'Sikkim':837,'Rajasthan':86,'Tripura':97}}
pdf1=pd.DataFrame(iprod1)
print(pdf1)
print(pdf.combine_first(pdf1))


# In[154]:


iprod={'Rice':{'Andhra Pradesh':35.8,'Gujarat':865,'Goa':87.0,'Kerala':876,'Sikkim':74,'Rajasthan':87,'Tripura':765},
      'Wheat':{'Andhra Pradesh':76.9,'Gujarat':987,'Goa':98,'Kerala':987,'Sikkim':98,'Rajasthan':43,'Tripura':98},
      'Maize':{'Andhra Pradesh':75,'Gujarat':86,'Goa':86,'Kerala':77,'Sikkim':87,'Rajasthan':866,'Tripura':97}}
pdf=pd.DataFrame(iprod)
print(pdf)
iprod1={'Rice':{'Andhra Pradesh':365,'Gujarat':871,'Goa':81.0,'Kerala':76,'Sikkim':84,'Rajasthan':83,'Tripura':65},
      'Wheat':{'Andhra Pradesh':7.9,'Gujarat':7,'Goa':8,'Kerala':97,'Sikkim':8,'Rajasthan':3,'Tripura':8},
      'Maize':{'Andhra Pradesh':5,'Gujarat':6,'Goa':8,'Kerala':717,'Sikkim':837,'Rajasthan':86,'Tripura':97}}
pdf0=pd.DataFrame(iprod1)
pdf1=pd.concat([pdf,pdf0],ignore_index=True)
print(pdf1)


# In[ ]:




