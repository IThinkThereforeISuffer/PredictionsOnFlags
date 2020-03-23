import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

flags = pd.read_csv("flags.csv", header = 0)
labels = flags[["Landmass"]]
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange", "Circles", "Crosses", "Saltires", "Quarters", "Sunstars", "Crescent", "Triangle"]]

best_accuracy = 0
best_depth = 0
accuracies = []
for i in range(1, 21):
  train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)
  tree = DecisionTreeClassifier(random_state = 1, max_depth = i)
  tree.fit(train_data, train_labels)
  accuracy = tree.score(test_data, test_labels)
  accuracies.append(accuracy)
  if accuracy > best_accuracy:
    best_accuracy, best_depth = accuracy, i

print("best accuracy is", best_accuracy)
print("It happened at depth", best_depth)

x_axis = range(1, 21)
y_axis = accuracies
plt.plot(x_axis, y_axis)
plt.show()
