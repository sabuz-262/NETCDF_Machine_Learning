from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import Constants
import xarray as xr
import sklearn.cluster
import pandas
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt



# Acerage distance of data points to cluster centriods
def k_mean_distance(data, cx, cy, ca, cb, i_centroid, cluster_labels):
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (a - ca) ** 2 + (b - cb) ** 2 ) for (x, y, a, b) in data[cluster_labels == i_centroid]]
    return np.array(distances).mean()


def apply_k_means(data, K):
    data_values = data.values
    km = sklearn.cluster.KMeans(n_clusters=K)
    km.fit(data_values)
    clusters = km.fit_predict(data_values)
    labels = km.labels_
    centroids = km.cluster_centers_
    distances = []
    for i, (cx, cy, ca, cb) in enumerate(centroids):
        mean_distance = k_mean_distance(data_values, cx, cy, ca, cb, i, clusters)
        distances.append(mean_distance)
    return np.array(distances).mean()


def find_K(dataframe):
    K_array = []
    distance_array = []
    for i in range(2,100,1):
        distance = apply_k_means(dataframe, i)
        K_array.append(i)
        distance_array.append(distance)
    print(K_array)
    print(distance_array)
    plt.plot(K_array, distance_array)
    plt.legend(loc='best')
    plt.xlabel("K(numebr of clusters)")
    plt.ylabel("Average within cluster distance to centroid")
    #plt.show()
    plt.savefig('./elbow_point.pdf', bbox_inches='tight')
    plt.close()


def store_data(dataframe):
    dataset = xr.Dataset.from_dataframe(dataframe)
    dataset.to_netcdf(Constants.output_file_name)


def rename_column(dataframe):
    dataframe.rename(columns={Constants.old_atm_pressure_name: Constants.new_atm_pressure_name, Constants.old_rh_name: Constants.new_rh_name,
                       Constants.old_temp_name: Constants.new_temp_name}, inplace=True)

    #resample the data with 5min
    resampled_all_data = dataframe.resample('5T')
    resampled_data = resampled_all_data[[Constants.new_atm_pressure_name, Constants.new_rh_name, Constants.new_temp_name]].mean()
    #calcualte the derivative of temperature

    #resampled_data[Constants.temperature_derivative] = (resampled_data[Constants.new_temp_name].shift(-1) - resampled_data[Constants.new_temp_name].shift(1)) / 2

    temprature_data = resampled_data[Constants.new_temp_name]
    resampled_data[Constants.temperature_derivative] = pandas.Series(np.gradient(temprature_data.values), temprature_data.index)


    return resampled_data

def read_cdf_file(filename):
    dataset = xr.open_dataset(filename)
    dataframe = dataset.to_dataframe()
    return dataframe

def main():

    for i in range(1,10,1):
        file_name = Constants.data_folder + Constants.file_name_init + str(i) + Constants.file_name_last
        if i == 1:
            dataframe = read_cdf_file(filename=file_name)
        else:
            df = read_cdf_file(filename=file_name)
            dataframe = dataframe.append(df)


    dataframe = rename_column(dataframe)
    #print(dataframe)
    store_data(dataframe)
    #dataframe = read_cdf_file(Constants.output_file_name)
    #print(dataframe)
    find_K(dataframe)

if __name__ == '__main__':
    main()