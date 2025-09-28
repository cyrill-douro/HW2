import matplotlib.pyplot as plt
import pandas as pd
def visual(df, x1, x2, x3, y1, y2, y3, z2, label3):
  fig = plt.figure(figsize=(20, 7))
  plt.subplot(1, 3, 1)
  plt.scatter(df[x1], df[y1], alpha=0.5)
  plt.xlabel(x1)
  plt.ylabel(y1)
  plt.title(x1,y1)

  ax = fig.add_subplot(111, projection = '3d')
  x = df[x2]
  y = df[y2]
  z = df[z2]
  ax.scatter(x, y, z)
  ax.set_xlabel(x2)
  ax.set_ylabel(y2)
  ax.set_zlabel(z2)
  plt.show()
  
  grouped_data = df.groupby(x3)[y3].sum().sort_values(ascending=False)
  total = grouped_data.sum()
  percentages = (grouped_data / total) * 100
  
  main_categories = grouped_data[percentages >= 2]
  small_categories = grouped_data[percentages < 2]
  if len(small_categories) > 0:
        etc_sum = small_categories.sum()
        etc_percentage = (etc_sum / total) * 100
  final_data = pd.concat([main_categories, pd.Series([etc_sum], index=['Etc.'])])
  plt.figure(figsize=(10, 10))
  explode = ([0.1] * len(final_data))
  plt.pie(final_data.values, explode=explode, labels=final_data.index, autopct='%1.1f%%', shadow=True, startangle=90)
  plt.title(label3, fontsize=20, fontweight='bold')
  plt.axis('equal')
  
  plt.show()