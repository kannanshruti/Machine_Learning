'''
Author: kshruti
Description: Runs PCA to convert data to a 2D plane, then performs k-NN over a range or a specific value to get output labels
Keywords: PCA, k-nearest-neighbors, iris_dataset, classification
Iris: 98% accuracy
HSV_comma_sep: 
'''
import numpy as np
import csv
import pandas as pd
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def get_csv(file_name):
	''' Returns the contents of the "file_name.csv" as a pandas dataframe "data"
	'''
	data = pd.read_csv(file_name)
	return data


def get_XY(data, output_col):
	''' Return a new dataframe "output_data" with columns in "col_exclude"
	 removed
	'''
	cols = [col for col in data.columns if col != output_col]
	xdata = data[cols]
	ydata = data[output_col]
	return xdata, ydata


def get_colors(number_colors):
	colors = []
	for i in range(number_colors):
		c = [x/255.0 for x in list(np.random.choice(range(256), size=3))]
		colors.append(c)
	return colors

def show_fig(xdata, ydata, output_col):
	''' Creates a plot of the "data", column labels specified as 'x' and 'y'
	'''
	# ax = data.plot.scatter(x='PCA1', y='PCA2', c='b', label='TP1')
	colors = get_colors(len(np.unique(ydata)))	
	color_idx = 0
	xdata[output_col] = ydata
	groups = xdata.groupby(output_col)
	for name, group in groups:
		plt.scatter(group['PCA1'], group['PCA2'], color = colors[color_idx], label=name)
		color_idx += 1
	plt.xlabel('PCA1')
	plt.ylabel('PCA2')
	plt.title('PCA Scatterplot')
	plt.grid()
	plt.legend()
	plt.show()


def get_PCA(data, num_components, column_names):
	''' Performs PCA on "data" as per the components specified by 
	"num_components", then converts the resulting PCA components to a 
	dataframe
	'''
	pca = decomposition.PCA(n_components=num_components) # Framework
	pca.fit(data) # Applying PCA
	data = pca.transform(data) # Converting to the coordinates
	data = pd.DataFrame(data = data, columns = column_names)
	return data


def get_knn(xdata, ydata, num_neighbors):
	''' Runs k-Nearest neighbors on "pca_data1" using the labels from 
	"data2" for "num_neighbors".
	Returns predicted labels "ypred" and true labels "ytest" as lists
	'''
	knn = KNeighborsClassifier(n_neighbors=num_neighbors)
	ypred = []
	for i in range(len(xdata.index)):
		# Train set is TP1 data and TP2 labels without the current row
		# Test set data is the current row in TP1 data
		xtrain = xdata.drop(i) 
		ytrain = ydata.drop(i) 
		xtest = xdata.iloc[[i]]		
		knn.fit(xtrain, ytrain)
		ypred.append(knn.predict(xtest)[0])
	ytest = ydata.tolist() # Test set labels are TP2 labels
	return ypred, ytest

def run_knn(xdata, ydata, range_neighbors, plot_status=True):
	''' Runs kNN on "xdata" and "ydata" over a range of neighbors "range_neighbors"
	If plot of performance over neighbors is to be viewed, set "plot_status" to True
	'''
	performance_list = []
	for i in range_neighbors: # To find best accuracy from a range of neighbors
		num_neighbors = i
		ypred, ytest = get_knn(xdata, ydata, num_neighbors)

		# Performance metrics for current 'i'
		cm = confusion_matrix(ytest, ypred)
		accuracy = float(np.trace(cm)) / float(np.sum(np.sum(cm)))
		performance_list.append(accuracy*100)
	if plot_status:
		plt.xticks(range_neighbors)
		plt.plot(range_neighbors, performance_list)
		plt.xlabel('#Neighbors')
		plt.ylabel('Accuracy (%)')
		plt.title('Accuracy over a range of neighbors')
		plt.grid()
		plt.show()
	return performance_list


if __name__ == '__main__':

	# Modify the filenames accordingly
	file_name = 'iris.csv'	
	output_col = 'species'

	# Set parameters for PCA and kNN
	num_components = 2 # Specifies how many PCA components
	range_neighbors = range(1,30) # Specifies start-(end-1) range over which k-NN is to be run

	# Read data
	data = get_csv(file_name)
	print 'Data read:', data.shape
	
	# Get the data for PCA
	xdata, ydata = get_XY(data, output_col)

	# Perform PCA 
	xdata_pca = get_PCA(xdata, num_components, ['PCA1', 'PCA2'])

	# Uncomment to view PCA scatter plot
	show_fig(xdata_pca, ydata, output_col)

	# Predict TP2 labels by applying kNN
	performance_list = run_knn(xdata, ydata, range_neighbors, plot_status=True)

	best_accuracy = np.argmax(performance_list) 
	print 'Best Accuracy is:', performance_list[best_accuracy], 'at neighbors:', best_accuracy
