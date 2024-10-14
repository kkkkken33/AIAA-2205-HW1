import pandas as pd

# 读取三个.csv文件
df1 = pd.read_csv('mfcc-50.LR.csv', header=None)
df2 = pd.read_csv('mfcc-50.mlp.csv', header=None)
df3 = pd.read_csv('mfcc-50.svm.multiclass.csv', header=None)

data = {
    'ID': [],
    'Category': []
}
# 循环遍历三个文件的每一行
for i in range(len(df1)):
    # 三个文件的每一行的第一个元素是视频的id
    id1 = df1.iloc[i, 0]
    id2 = df2.iloc[i, 0]
    id3 = df3.iloc[i, 0]
    # 如果三个文件的id不相等，说明三个文件的顺序不一样，直接报错
    if id1 != id2 or id1 != id3:
        print('Error: id not equal')
        break
    category1 = df1.iloc[i, 1]
    category2 = df2.iloc[i, 1]
    category3 = df3.iloc[i, 1]
    if category1 == category2 == category3:
        data['ID'].append(id1)
        data['Category'].append(category1)
    if category1 == category2 != category3:
        data['ID'].append(id1)
        data['Category'].append(category1)
    if category1 == category3 != category2:
        data['ID'].append(id1)
        data['Category'].append(category1)
    if category2 == category3 != category1:
        data['ID'].append(id2)
        data['Category'].append(category2)
    if category1 != category2 and category2 != category3 and category1 != category3:
        data['ID'].append(id1)
        data['Category'].append(category1)        
df = pd.DataFrame(data)
df.to_csv('result.csv', index=False, header=False)
