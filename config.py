from collections import namedtuple
# ========================== PATHS ==========================
PATH = 'C:\\Users\\u279014\\Documents\\H_Drive\\7.AA Models\\12.Logistic_Optimization\\data'

# Share_drive_path data source:
SHARE_DRIVE_PATH = 'S:\\OSK-Share\\DEPT\\LOGISTICS\\LC3\\Freight Payment and Reports\\Cognos Reports'

INPUT_ZIPCODE_FILE = 'zipcode_Lon_Lat.csv'
INPUT_CASS_FILE = 'Cass FY19 Invoice Detail.csv'
INPUT_RIDING_DISTANCE = 'riding_distance_matrix.xlsx'
INPUT_TMC = 'TMC_freight_rate.xlsx'

# change following output to SQL server in the future
OUTPUT_SUPPLIER_CLUSTER_FILE = 'cass_zip_cluster.csv'
OUTPUT_CASSZIP_FILE = 'suppliers_geo.csv'
OUTPUT_CASSZIP_FILE_KLABELS = 'geo_cass.csv'
OUTPUT_K_SELECTION = 'k_selection.csv'
OUTPUT_EXCEED_VEHICLE_LIMIT = 'shipment_weight_over_truck_limit.csv'
OUTPUT_ROUTE_IN_WEIGHT = 'route_in_weight.csv'

# ========================== CASS SELECTION PARAMETERS ==========================
# CLUSTERING PARAMETER
SOURCE_STATE = 'all'
DESTINATION_DEPORT_ZIP = '54942'
SHIPPING_WINDOW_START = '2019-01-01'
SHIPPING_WINDOW_SPAN = 20

# VRP PARAMETERS
TRUCK_MODE = 'L'
CLUSTER_STATES = 'WI,IL,MI,IN'
HUB_LIST = {'GREENVILLE': ['54942', -88.53557, 44.293820, 'mid_west', 'GREENVILLE_WH', 'WI']}
HUB_NAME = 'GREENVILLE'
OSK_DESTINATION_LIST = ['MILWAUKEE', 'OSHKOSH', 'GREENVILLE']
SHEET_NAMES = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']

# ========================== MODEL SELECTION HYPER-PARAMETER ==========================
# CLUSTERING HYPER-PARAMETERS
K_MAX = 8
EPS = 25
MIN_SAMPLES = 3
METRIC = 'precomputed'
LEAF_SIZE = 30

# VRP HYPER-PARAMETERS
VEHICLE_CAPACITY = 45000
CLUSTER_RANK = 1
VEHICLE_COUNTS = 20
VEHICLE_STOPS = 7
ROUTE_LOCATION_COUNTS = 5
PENALTY = 5000
