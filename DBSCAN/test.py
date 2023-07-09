
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter
from dbscan import dbscan

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)


X = StandardScaler().fit_transform(X)


# Run my DBSCAN implementation.
print('Моя программная реализация')
my_labels = dbscan(X, eps=0.3, MinPts=10)

counted = Counter(my_labels)
centroids = dict(zip(counted.keys(), [(0, 0) for _ in range(len(counted.keys()))]))
sort_by_cluster = dict(zip(counted.keys(), [[] for _ in range(len(counted.keys()))]))
for label, point in zip(my_labels, X):
    sort_by_cluster[label].append(tuple(point))
    centroids[label] += point

for k in centroids.keys():
    centroids[k] = centroids[k] / len(sort_by_cluster[k])

del centroids[-1]

plt.scatter(*list(zip(*X)))
plt.scatter(*list(zip(*centroids.values())), color='red')
plt.show()

print('Scikit-learn реализация')
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
skl_labels = db.labels_

for i in range(0, len(skl_labels)):
    if not skl_labels[i] == -1:
        skl_labels[i] += 1


print(list(skl_labels))

num_disagree = 0

# Go through each label and make sure they match (print the labels if they do not)
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == my_labels[i]:
        print('Scikit learn:', skl_labels[i], 'mine:', my_labels[i])
        num_disagree += 1

if num_disagree == 0:
    print('Тест пройден, все метки кластеров совпали')
else:
    print('FAIL -', num_disagree, 'labels don\'t match.')
