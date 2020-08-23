import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist, pdist

sys.path.insert(0, '../main')
import config


class ETL_data:
    def __init__(self, path):
        self.path = path

    def distance_on_sphere_numpy(self, coordinate_df):
        """
        Compute a distance matrix of the coordinates using a spherical metric.
        :param coordinate_array: numpy.ndarray with shape (n,2); latitude is in 1st col, longitude in 2nd.
        :returns distance_mat: numpy.ndarray with shape (n, n) containing distance in km between coords.
        """
        # Radius of the earth in km (GRS 80-Ellipsoid)
        EARTH_RADIUS = 6371.007176
        km2mile_ratio = 0.62137

        # Unpacking coordinates
        latitudes = coordinate_df.loc[:, 'latitude']
        longitudes = coordinate_df.loc[:, 'longitude']
        # Convert latitude and longitude to spherical coordinates in radians.
        degrees_to_radians = np.pi / 180.0
        phi_values = (90.0 - latitudes) * degrees_to_radians
        theta_values = longitudes * degrees_to_radians
        # Expand phi_values and theta_values into grids
        theta_1, theta_2 = np.meshgrid(theta_values, theta_values)
        theta_diff_mat = theta_1 - theta_2
        phi_1, phi_2 = np.meshgrid(phi_values, phi_values)
        # Compute spherical distance from spherical coordinates
        angle = (np.sin(phi_1) * np.sin(phi_2) * np.cos(theta_diff_mat) + np.cos(phi_1) * np.cos(phi_2))
        arc = np.arccos(angle)
        # Multiply by earth's radius to obtain distance in km
        return np.nan_to_num(arc * EARTH_RADIUS * km2mile_ratio)

    def col_name(self, file):
        """
        this is to trim the data_frame column names to a unique format:
        all case, replace space to underscore, remove parentheses
        param df:
            raw from share drive for
        return:
            polished data set with new column names
        """
        df = pd.read_csv(os.path.join(self.path, file), low_memory=False)
        df.columns = df.columns.str.strip().str.lower().str.replace('-', '').str.replace(' ', '_').str.replace('(', ''). \
            str.replace(')', '').str.replace('"', '')
        return df

    def str2time(self, x):
        """
        paramter
            x: string formated time
        return:
            timeStamp with mm/dd/yyyy format
        """
        try:
            return datetime.strptime(x, '%b %d, %Y')
        except:
            return '0000-00-00'

    def logic_and_3_condition(self, x1, x2, x3):
        return np.logical_and(np.logical_and(x1, x2), x3)

    def clean_zip(self, source_file):
        """
        this is to clean the zip file:
            - corrdinate format
            - save file as feather format
        parameter:
            df: original zipcode msater file
            file_path: zipcode file directory
        return:
            zipcode feather format
        """

        # change zipcodes which contain alphabix letter to 0 (outside of USA)
        def to_string(x):
            try:
                return str(x)
            except:
                return 0

        zipcode = self.col_name(source_file)
        zipcode['zip_code'] = zipcode['zip_code'].apply(lambda x: to_string(x))
        return zipcode

    def get_cluster(self, state):
        if state in ['WI', 'IL', 'MI', 'IN']:
            return 'mid_west'
        elif state in ['PA', 'NY', 'MD']:
            return 'north_east'
        elif state in ['NC', 'SC', 'GA']:
            return 'south_east'
        return 'other'

    def clean_cass(self, source_file, save_file, source_state=None, dest_zip='54942',
                   shipping_date_start='2019-01-01', shipping_window=7, truck_mode='LT', inbound_indicator='INBOUND'):
        """
        parameter:
            df: dataFrame, original dataset downloaded from cass
            source_state: str, comma needed, default 'WI' as the largest shipping from
            dest_zip: str, default 54942 as greenville
            shipping_date_start: str, starting shipping schedule cut-off date
            shipping_window: int, days out as shipping period window
            truck_mode: str, no comma needed, default LT for less than truck and full truck both inclusive

        return:
            df: cleaned dataFrame
        """
        shipping_date_start = datetime.strptime(shipping_date_start, '%Y-%m-%d')

        df = self.col_name(source_file)
        df = df[['source_weekend_date', 'mode', 'shipper_name', 'shipper_address',
                 'shipper_city', 'shipper_zip', 'shipper_state', 'destination_city',
                 'destination_zip', 'destination_state', 'bill_of_lading_number',
                 'ship_weight', 'miles', 'billed_amount', 'inbound_outbound_indicator']]
        df = df[df.inbound_outbound_indicator == inbound_indicator]
        df = df[df.destination_zip == dest_zip]
        df['ship_weight'] = df.ship_weight.apply(lambda x: x.replace(',', '')).astype('int')

        if source_state and source_state is not 'all':
            states = source_state.split(',')
            df = df[np.logical_and(df.destination_zip == dest_zip, df.shipper_state.isin(states))]

        df['shipping_date'] = df.source_weekend_date.apply(lambda x: self.str2time(x))
        df = df[self.logic_and_3_condition((df.shipping_date >= shipping_date_start),
                                           (df.shipping_date <= shipping_date_start + dt.timedelta(shipping_window)),
                                           (df['mode'].isin(list(truck_mode))))]
        df['miles'] = df.miles.apply(lambda x: x.replace(',', '')).astype('int')
        df['billed_amount'] = df.billed_amount.apply(lambda x: x.replace(',', '')).astype('float')

        # change zipcodes which contain alphabix letter to 0 (outside of USA)
        def to_string(x):
            try:
                return str(x)
            except:
                return 0

        df['shipper_zip'] = df['shipper_zip'].apply(lambda x: to_string(x))
        df['cluster'] = df.shipper_state.apply(self.get_cluster)
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(self.path, save_file), index=False)
        return df

    def cass_merge_zip(self, df_cass, df_zip, save_file):
        """
        parameter:
            df1: cleaned cass dataset
            df2: original zipcode matrix
        return:
            merged dataFrame contains longitude and latitude
        """
        df = pd.merge(df_cass, df_zip[['zip_code', 'longitude', 'latitude']], how='left', left_on='shipper_zip',
                      right_on='zip_code')
        df = df.groupby(['zip_code', 'longitude', 'latitude', 'cluster', 'shipper_name', 'shipper_state'])[
            ['ship_weight', 'miles', 'billed_amount']].sum()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df.to_csv(os.path.join(self.path, save_file), index=False)
        return df

    def load_riding_distance_matrix(self, file):
        """
        this is to read PMI_riding distance
        param df:
            VBA modified distance matrix from share drive
        return:
            distance matrix with riding distance
        """
        df = pd.read_excel(os.path.join(self.path, file)).set_index('zipcode')
        df.columns = df.columns.astype('str')
        df.index = df.index.astype('str')
        return df

    def riding_distance(self, riding_distance_matrix, geo):
        """
        Compute a distance matrix of the coordinates using a spherical metric.
        :param
            coordinate_df: numpy.ndarray with shape (n,n); riding_distance_matri: dataframe, col & index type: str
            geo_zipcode: Data.Series, element type: str
        returns:
            distance_mat: numpy.ndarray with shape (n, n) containing distance in km between coords.
        """

        d_matrix = []
        zipcodes = geo['zip_code'].astype('str')

        for i in zipcodes:
            d_row = []
            for j in zipcodes:

                d_row.append(riding_distance_matrix.loc[i, j])
            d_matrix.append(d_row)
        return np.asarray(d_matrix)


class ClusterModel:
    """ to complete this, complete ipynb to py """

    def __init__(self, path):
        self.path = path

    def load_data(self, file):
        df = pd.read_csv(os.path.join(self.path, file))
        return df

    def kcluster(self, file, k_range=20):
        """
        :param file: cass_merge_zip generated file
        :param k_range: initialize number of max clustering upper bound
        :return:
        """
        cass_zip = self.load_data(file=file)
        geo = cass_zip.loc[:, ['longitude', 'latitude']]
        distortions = []
        K = range(1, k_range)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(geo)
            distortions.append(sum(np.min(cdist(geo, kmeans.cluster_centers_, 'euclidean'), axis=1)) / geo.shape[0])

        cass_zip['label'] = kmeans.labels_
        cass_zip.to_csv(os.path.join(self.path, config.OUTPUT_CASSZIP_FILE_KLABELS), index=False)

        df_distortion = pd.DataFrame(distortions, columns=['sum_square_distances_to_center'])
        df_distortion['distortion_moving_diff'] = df_distortion.sum_square_distances_to_center.diff(-1)
        df_distortion.to_csv(os.path.join(self.path, config.OUTPUT_K_SELECTION), index='No_cluster')

    def dbscan(self, file, cluster_states):
        if not isinstance(cluster_states, list):
            cluster_states = cluster_states.split(',')
        cass_zip = self.load_data(file=file)
        geo = cass_zip[cass_zip.shipper_state.isin(cluster_states)].reset_index(drop=True)

        etl = ETL_data(self.path)

        riding_distance_matrix = etl.load_riding_distance_matrix(file=config.INPUT_RIDING_DISTANCE)

        distance_matrix = etl.riding_distance(riding_distance_matrix=riding_distance_matrix, geo=geo)

        db = DBSCAN(eps=config.EPS, min_samples=config.MIN_SAMPLES, metric=config.METRIC, leaf_size=config.LEAF_SIZE)
        db.fit(distance_matrix)

        geo['label'] = db.labels_
        geo.to_csv(os.path.join(self.path, config.OUTPUT_SUPPLIER_CLUSTER_FILE), index=False)
        return None


def main_preprocess():
    etld = ETL_data(path=config.PATH)
    zipcode = etld.clean_zip(config.INPUT_ZIPCODE_FILE)
    dfcass = etld.clean_cass(source_file=config.INPUT_CASS_FILE,
                             save_file=config.OUTPUT_SUPPLIER_CLUSTER_FILE,
                             dest_zip=config.DESTINATION_DEPORT_ZIP,
                             source_state=config.SOURCE_STATE,
                             shipping_date_start=config.SHIPPING_WINDOW_START,
                             shipping_window=config.SHIPPING_WINDOW_SPAN,
                             truck_mode=config.TRUCK_MODE)
    cass_zip = etld.cass_merge_zip(dfcass, zipcode, save_file=config.OUTPUT_CASSZIP_FILE)


def main_cluster():
    KMM = ClusterModel(path=config.PATH)
    KMM.kcluster(file=config.OUTPUT_CASSZIP_FILE, k_range=config.K_MAX)
    KMM.dbscan(file=config.OUTPUT_CASSZIP_FILE, cluster_states=config.CLUSTER_STATES)


if __name__ == "__main__":
    _list = list(np.random.randint(1,9,4))
    print(f'###### program is running......\n### Play a 24 Game using [+ - * /] to get 24\n>>> {_list}\n')
    main_preprocess()
    main_cluster()
    print('###### kmeans & dbscan runs successfully #######')