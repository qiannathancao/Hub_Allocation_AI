import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, '../main')

import clustering_main as cm
import config


class VrpViz:

    def __init__(self, path):
        self.path = path

    def distance_index(self, df, x):
        """
        param:
            df: distance matrix with UNIQUE index & columns
            x: truck location source and truck location next-stop
        return:
            DataFrame: distance matrix
        """
        try:
            return df.loc[x[0], x[1]]
        except:
            return 0

    def route_weight_output(self, route_schedule, cass_zip_100, df_unique_distance_matrix, tmc, file):
        route_in_weight = route_schedule.merge(cass_zip_100, left_on='pick_node',right_index=True,how='left')
        route_in_weight['next_zip_code'] = route_in_weight.groupby(['truck_number'])['zip_code'].shift(-1)
        route_in_weight['next_shipper_name'] = route_in_weight.groupby(['truck_number'])['shipper_name'].shift(-1)
        route_in_weight['milk_run_distance'] = route_in_weight[['zip_code', 'next_zip_code']].\
            apply(lambda x: round(VrpViz(self.path).distance_index(df_unique_distance_matrix, x)), axis=1)
        route_in_weight['stop_number'] = route_in_weight.groupby('truck_number').cumcount()
        route_in_weight['milk_run_cost'] = 0
        truck_load_cost = np.max(tmc.freight_cost)
        route_in_weight.loc[route_in_weight.groupby('truck_number').tail(1).index, 'milk_run_cost'] = truck_load_cost
        route_in_weight.to_csv(os.path.join(self.path, file), index=False)
        return route_in_weight

    def load_data(self, file, sheet_name=None):
        df = pd.read_excel(os.path.join(self.path, file), sheet_name=sheet_name)
        df = pd.concat(df[frame] for frame in df.keys())
        df.reset_index(drop=True, inplace=True)
        return df

    def clean_tmc(self, df, sink_state='WI', source_states='IL'):
        """
        parameter:
            df: original TMC dataset
            sink_state: destination warehouse, only one locations allowed
            source_states: shipping states, allowing multiple states as source state
        return:
            cleaned TMC including freight_cost from all states to sink_state
        """
        df.columns = df.columns.str.strip().str.lower().str.replace('-', '').str.replace(' ', '_').\
            str.replace('(', ''). str.replace(')', '').str.replace('"', '')

        # drop rows if all cols are nan
        df.dropna(how='all', subset=['market_rate_over_quarter_decmar',
                                     'market_rate_over_jan_2019mar_2020',
                                     'market_rate_all_offers_jan_2019_mar_2020_no_fb',
                                     'market_rate_all_offers_jan_2019_mar_2020_with_fb'], inplace=True)

        # generate freight_cost = market_rate_all_offers_jan_2019_mar_2020_no_fb or max of all
        df['freight_cost'] = np.round(np.where(df.market_rate_all_offers_jan_2019_mar_2020_no_fb.isnull(),
                                               np.max(df, axis=1),
                                               df.market_rate_all_offers_jan_2019_mar_2020_no_fb), 2)
        df['source_state'] = df.lane.apply(lambda x: x[:2])  # find source state short code
        df['sink_state'] = df.lane.apply(lambda x: x[-2:])  # find sink state short code

        df = df[df.source_state.isin(source_states)]  # slice only source state
        df = df[df.sink_state.str.contains(sink_state)]  # slice to include destination state only
        df = df.groupby(['source_state', 'sink_state'])[
            'freight_cost'].mean().reset_index()  # average duplicated states to same destination,
        return df
#
#
# if __name__ == "__main__":
#     vrp_finance = VrpViz(path=config.PATH)
#     vrp_prep = vmm.VrpPrep(path=config.PATH)
#     osk_hub_dict = vrp_prep.hub_dict(destination_list=config.OSK_DESTINATION_LIST, file=config.INPUT_CASS_FILE)
#
#     cass_zip_cluster_copy, source_states = vrp_prep.choose_nth_supplier_cluster(file=config.OUTPUT_SUPPLIER_CLUSTER_FILE,
#                                                                                 rank=config.CLUSTER_RANK)
#
#     df_tmc = vrp_finance.load_data(file=config.INPUT_TMC, sheet_name=config.SHEET_NAMES)
#
#     sink_state = config.HUB_LIST[config.HUB_NAME][-1]
#     source_states_unique = source_states.shipper_state.unique()
#
#     tmc = vrp_finance.clean_tmc(df=df_tmc, sink_state=sink_state, source_states=source_states_unique)
#
#     route_in_weight = vrp_finance.route_weight_output()
















