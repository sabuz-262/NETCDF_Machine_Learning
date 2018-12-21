import Constants
import xarray as xr
import sklearn.cluster
import pandas
from scipy.spatial.distance import pdist


def apply_k_means(data, K):
    mat = data.values
    mat[0][3] = 0
    mat[-1][3] = 0
    km = sklearn.cluster.KMeans(n_clusters=K)
    km.fit(mat)
    labels = km.labels_
    centroids = km.cluster_centers_
    dists = pdist(centroids, metric='euclidean')
    results = pandas.DataFrame([data.index, labels]).T
    return dists, results




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
    resampled_data[Constants.temperature_derivative] = (resampled_data[Constants.new_temp_name].shift(-1) - resampled_data[Constants.new_temp_name].shift(1)) / 2

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
    dataframe = read_cdf_file(Constants.output_file_name)
    print(dataframe)

if __name__ == '__main__':
    main()