from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import Constants
import xarray as xr
from sklearn.cluster import KMeans
import pandas
import numpy as np
import matplotlib.pyplot as plt


def apply_k_means(data, K):
    data_values = data.values
    km = KMeans(n_clusters=K)
    km.fit(data_values)
    return km.inertia_


def apply_k_means_with_elbw_point(data, K):
    data_values = data.values
    km = KMeans(n_clusters=K)
    km.fit(data_values)
    labels = km.labels_
    results = pandas.DataFrame([data.index, labels]).T
    return results

def find_K(dataframe):
    K_array = []
    distance_array = []
    for i in range(2,Constants.highest_K,1):
        distance = apply_k_means(dataframe, i)
        K_array.append(i)
        distance_array.append(distance)
    #print(K_array)
    #print(distance_array)
    plt.plot(K_array, distance_array)
    plt.legend(loc='best')
    plt.xlabel("K(numebr of clusters)")
    plt.ylabel("Inertia")
    #plt.show()
    plt.savefig('./elbow_point.pdf', bbox_inches='tight')
    plt.close()


def store_resample_data(dataframe):
    dataset = xr.Dataset.from_dataframe(dataframe)
    dataset.to_netcdf(Constants.output_file_name)


def rename_column(dataframe):
    dataframe.rename(columns={Constants.old_atm_pressure_name: Constants.new_atm_pressure_name, Constants.old_rh_name: Constants.new_rh_name,
                       Constants.old_temp_name: Constants.new_temp_name}, inplace=True)

    #resample the data with 5min
    resampled_all_data = dataframe.resample(Constants.resample_time)
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


def read_all_files():
    for i in range(1,Constants.total_number_file,1):
        file_name = Constants.data_folder + Constants.file_name_init + str(i) + Constants.file_name_last
        if i == 1:
            dataframe = read_cdf_file(filename=file_name)

            #print((dataframe))
        else:
            df = read_cdf_file(filename=file_name)
            #print(len(df))
            dataframe = dataframe.append(df)

    return dataframe

def main():

    dataframe = read_all_files()
    #print(len(dataframe))
    dataframe = rename_column(dataframe)
    print(dataframe.shape[1])
    store_resample_data(dataframe)
    find_K(dataframe)
    results = apply_k_means_with_elbw_point(dataframe, 7)
    results.to_csv("result.csv")

if __name__ == '__main__':
    main()