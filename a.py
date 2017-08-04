
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

df = pd.read_csv(r'C:\py_file\consumption_data.csv')
#print(df)
df = df.ix[:,['R','F','M']]
#print(df)
df = df[(df.F<45)&(df.M<30000)]
#print(df.describe())
df2 = (df - df.mean())/df.std()
#print(df2)
model = KMeans(n_clusters=5,max_iter=500)
model.fit(df2)

r1 = pd.Series(model.labels_).value_counts()
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2,r1],axis=1)
r.columns = list(df2.columns)+[u'类别数目']
print(r)


y_pred = model.predict(df2)
sd = plt.figure().add_subplot(111,projection='3d')
sd.scatter(df.R,df.F,df.M,c=y_pred)
sd.set_xlabel('R')
sd.set_ylabel('F')
sd.set_zlabel('M')
plt.show()