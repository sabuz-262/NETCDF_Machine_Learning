import unittest
import NETCDF
import Constants


class NETSCDFTest(unittest.TestCase):
    # Returns True or False.
    def test_read_all_file(self):
        dataframe = NETCDF.read_all_files()
        assert len(dataframe) == 1440*9

    def test_resample(self):
        dataframe = NETCDF.read_all_files()
        dataframe2 = NETCDF.rename_column(dataframe)
        assert len(dataframe) == len(dataframe2)*5

    def test_resample_rename(self):
        dataframe = NETCDF.read_all_files()
        dataframe = NETCDF.rename_column(dataframe)
        dataframe = NETCDF.rename_column(dataframe)
        assert dataframe.shape[1] == 4

    def test_store_resample_data(self):
        dataframe = NETCDF.read_all_files()
        dataframe = NETCDF.rename_column(dataframe)
        dataframe = NETCDF.rename_column(dataframe)
        NETCDF.store_resample_data(dataframe)
        dataframe2 = NETCDF.read_cdf_file(Constants.output_file_name)
        assert len(dataframe) == len(dataframe2)

if __name__ == '__main__':
    unittest.main()