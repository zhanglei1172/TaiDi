#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import _read_csv
from preprocessing import *
import matplotlib
from skimage import data, io, filters
from config import *
sns.set_style('white', {'font.sans-serif': ['YaHeiMonacoHybrid', 'Arial']})
 # 用来正常显示中文标签
myfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/Consolas+YaHei+hybrid.ttf", size=12)
plt.rcParams['font.sans-serif']=['sans-serif']
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
p = sns.color_palette()
phase = 'venous phase'
patient_sizes = [len(os.listdir(DATA_PATH +'数据集1/' + d + '/'+phase)) for d in os.listdir(DATA_PATH+'数据集1/')]
# plt.hist(patient_sizes, color=p[2], normed=)
sns.distplot(patient_sizes)
plt.ylabel('病人数量', fontproperties=myfont)
plt.xlabel('DICOM 文件数', fontproperties=myfont)
plt.title('每个病人DICOM文件数量({})'.format(phase), fontproperties=myfont)
# plt.show()
# plt.savefig()
plt.savefig('dist.svg', dpi=600, format='svg')
plt.show()
df = _read_csv('临床数据.csv')
df.describe()
df.groupby(['阴性/阳性']).count()
df.groupby(['阴性/阳性']).count()
# sns.set_style('white')
plt.figure()
sns.distplot(df['年龄'].str.slice(0, -1).astype(int))
plt.xlabel('年龄', fontproperties=myfont)
plt.title('年龄分布', fontproperties=myfont)
plt.savefig('年龄.svg', dpi=600, format='svg')
df['性别'].count()
# sns.distplot(df['性别'])
fig, ax = plt.subplots()
sns.countplot(df['性别'], ax=ax)
ax.set_xlabel('性别', fontproperties=myfont)

plt.savefig('性别.svg', dpi=600, format='svg')