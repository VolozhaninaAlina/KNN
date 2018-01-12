import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import cross_val_score

#Импортируем данные выборки Ирисов для работы с ними
iris = datasets.load_iris()

# Принимаем только первые две координаты(характеристики) Ирисов 
X = iris.data[:, :2]
y = iris.target

kVal = list(range(1,50))
h = .02  # Задаем размер шага

# Задаем цветовые характеристики карт 
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Выделяем границу решения, назначая цвет каждой точке [x_min, x_max] x [y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

for weights in ('uniform', 'distance'):
    cv_scores = []
    for k in kVal:
	# Создаем экземпляр Классификатора соседей, передаем ему параметры: 
    #Х -выборку без классов, У - классы для каждого элемента выборки и тип классификации ['uniform', 'distance']
        clf = neighbors.KNeighborsClassifier(n_neighbors = k,
                                             weights=weights,
                                             algorithm='auto')
        clf.fit(X, y)
        scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
        
    # Измеряем ошибку
    ERR = [1 - x for x in cv_scores]
	
	#Рисуем LOO ( количество соседей и ошибка алгоритма при данном количестве соседей)
    plt.plot(kVal, ERR)
    plt.xlabel('Neighbors K')
    plt.ylabel('Error')

    # Определаем оптимальное k( количество ближайших соседей)
    optimal_k = kVal[ERR.index(min(ERR))]
    min_err = min(ERR)
    print("The optimal number of neighbors for weights = '%s' is %d and error: %f "  
          % (weights, optimal_k, min_err))
	
    kNN = neighbors.KNeighborsClassifier(n_neighbors = optimal_k, 
                                         weights=weights,
                                         algorithm='auto')
    kNN.fit(X, y)

    Z = kNN.predict(np.c_[xx.ravel(), yy.ravel()])

    # Отображаем результат 
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Отображаем исходную выборку 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
              edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s'))"
              % (optimal_k, weights))
			  

plt.show()