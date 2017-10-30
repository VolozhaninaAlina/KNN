print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# Задаем количество соседей.
n_neighbors = 15

#Импортируем данные выборки Ирисов для работы с ними
iris = datasets.load_iris()

# Принимаем только первые две координаты(характеристики) Ирисов 
X = iris.data[:, :2]
y = iris.target

h = .02  # Задаем размер шага

# Задаем цветовые характеристики карт 
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) #цвет областей
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) #цвет Ирисов из выборки 

for weights in ['uniform', 'distance']:
    # # Создаем экземпляр Классификатора соседей, передаем ему параметры: 
	#Х -выборку без классов, У - классы для каждого элемента выборки и тип классификации ['uniform', 'distance']
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Выделяем границу решения, назначая цвет каждой точке [x_min, x_max] x [y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    all_points = np.c_[xx.ravel(), yy.ravel()] 
    Z = clf.predict(all_points)

    # Отображаем результат 
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Отображаем исходную выборку 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=10)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()