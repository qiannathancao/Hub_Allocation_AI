import numpy as np
import pandas as pd
import os
import sys
from collections import Counter
from collections import defaultdict
sys.path.insert(0, '../main')

import config
import clustering_main as cm
import VRP_finAnalysis as vfa

# from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class VrpPrep:
    """ to complete convert ipynb to py """
    def __init__(self, path):
        self.path = path

    def riding_distance(self, riding_distance_matrix, geo):
        """
        apply riding distance matrix to compute riding distance for actual pickup locations(geo)
        :param
            riding_distance_matrix: dataframe, col & index type: str
            geo: Data.Series, element type: str
        :returns distance_matrix
        """
        d_matrix = []
        zipcodes = geo['zip_code']
        for i in zipcodes:
            d_row = []
            for j in zipcodes:
                d_row.append(riding_distance_matrix.loc[i, j])
            d_matrix.append(d_row)
        return np.asarray(d_matrix)

    def load_riding_distance_matrix(self, file):
        """
        Compute a distance matrix of the coordinates using a spherical metric.
        :param
            file: riding matrix generated from PCMiller
        :returns distance_matrix
        """
        riding_distance_matrix = pd.read_excel(os.path.join(self.path, file)).set_index('zipcode')
        riding_distance_matrix.columns = riding_distance_matrix.columns.astype('str')
        riding_distance_matrix.index = riding_distance_matrix.index.astype('str')
        print('riding distance matrix completed')
        return riding_distance_matrix

    def hub_dict(self, file, destination_list, inbound_indicator='INBOUND'):
        """
        param:
            file: Cass FY19 Invoice Detail.csv
            inbound_indicator: str
            destination_list: list
        return:
            osk_hub_dict: dictionary, {supplier_name: [osk_warehouses...]
        """
        _data = cm.ETL_data(path=config.PATH).col_name(file=file)
        _data = _data[_data.inbound_outbound_indicator == inbound_indicator]
        hub_dict = defaultdict(set)
        for sn, dc in zip(_data.shipper_name, _data.destination_city):
            if dc in destination_list:
                hub_dict[sn].add(dc)
            else:
                pass
        print('hub_dict completed')
        return hub_dict

    def load_data(self, file):
        """
        :param
            file: to load OUTPUT_SUPPLIER_CLUSTER_FILE = 'cass_zip_cluster.csv'
        :return:
            dataframe: excluding cluster = -1
        """
        df = pd.read_csv(os.path.join(self.path, file))
        df_copy = df.copy()
        df_copy = df_copy[df_copy.label != -1]  # drop label(cluster)=-1, which do not belong to any group
        df_copy['zip_code'] = df_copy['zip_code'].astype('str')
        df_copy['shipping_date'] = config.SHIPPING_WINDOW_START

        df_exceeds_capacity = df_copy[df_copy['ship_weight'] >= config.VEHICLE_CAPACITY].reset_index(drop=True)
        df_exceeds_capacity.to_csv(os.path.join(self.path, config.OUTPUT_EXCEED_VEHICLE_LIMIT), index=False)

        df_copy = df_copy[df_copy['ship_weight'] < config.VEHICLE_CAPACITY].reset_index(drop=True)
        print('load data completed')
        return df_copy, df_exceeds_capacity

    def choose_nth_supplier_cluster(self, file, rank=1):
        """
        :param file: str
            output from load_data module
        :param rank: int
            choose cluster number based on popularity rank
        :return: dataframe
            selected dataframe per the rank
        """

        df_copy, _ = VrpPrep(self.path).load_data(file)
        label_no = Counter(df_copy.label).most_common()[rank-1][0]
        cluster = df_copy[df_copy.label == label_no]
        hub = pd.DataFrame([[config.HUB_LIST[config.HUB_NAME][0], config.HUB_LIST[config.HUB_NAME][1],
                            config.HUB_LIST[config.HUB_NAME][2], config.HUB_LIST[config.HUB_NAME][3],
                            config.HUB_LIST[config.HUB_NAME][4], config.HUB_LIST[config.HUB_NAME][5],
                            0, 0, 0, 999, config.SHIPPING_WINDOW_START]], columns=cluster.columns)
        result = hub.append(cluster).reset_index(drop=True)
        print('choose nth cluster completed')
        return result, cluster

    def limited_distance_matrix(self, riding_distance_matrix, cass_zip_cluster_copy):
        cass_zip_100 = cass_zip_cluster_copy[:100]
        distance_matrix_100 = VrpPrep(self.path).riding_distance(riding_distance_matrix=riding_distance_matrix,
                                                                 geo=cass_zip_100)
        unique_cass_zip_100 = cass_zip_100.drop_duplicates(subset=['zip_code'])
        unique_distance_matrix_100 = VrpPrep(self.path).riding_distance(riding_distance_matrix=riding_distance_matrix,
                                                                        geo=unique_cass_zip_100)

        df_unique_distance_matrix = pd.DataFrame(unique_distance_matrix_100,
                                                 index=unique_cass_zip_100['zip_code'],
                                                 columns=unique_cass_zip_100['zip_code'])

        ship_weight_list = cass_zip_100['ship_weight'].tolist()
        print(f'total shipping weight: {sum(ship_weight_list)} lbs')
        return cass_zip_100, distance_matrix_100, unique_cass_zip_100, unique_distance_matrix_100, \
               df_unique_distance_matrix, ship_weight_list


class VrpModel:

    def __init__(self, path):
        self.path = path

    def create_data_model(self,
                          distance_matrix=None,
                          ship_weight_list=None,
                          each_vehicle_capacity=45000,
                          num_vehicles=30,
                          nrlocations=9):

        _data = defaultdict()
        _data['distance_matrix'] = distance_matrix
        _data['demands'] = ship_weight_list
        _data['vehicle_capacities'] = [each_vehicle_capacity] * num_vehicles
        _data['num_vehicles'] = num_vehicles
        _data['depot'] = 0
        _data['nrLocations'] = nrlocations
        return _data

    def print_solution(self, data, manager, routing, assignment):
        """Prints assignment on console."""
        total_distance = 0
        total_load = 0

        vehicle_routes = defaultdict()  # for list out the same truck pick zip codes

        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            plan_output_backward = 'Route for vehicle {}:\n'.format(vehicle_id)  # if backward is shorter path
            route_distance = 0
            route_load = 0
            edge_distance = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
                plan_output_backward += ' {0} Load({1}) <- '.format(node_index,
                                                                    route_load)  # if backward is shorter path
                previous_index = index
                index = assignment.Value(routing.NextVar(index))

                if vehicle_id in vehicle_routes:
                    vehicle_routes[vehicle_id].append(node_index)  # adding zip codes to same truck
                else:
                    vehicle_routes[vehicle_id] = [node_index]

                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
                edge_distance.append(routing.GetArcCostForVehicle(previous_index, index, vehicle_id))

            # adding destination to entire route

            """ this situation is Fudging Headacheeeeeeee"""
            # distance from Greenville to first supplier is larger than last supplier to Greenville,
            # truck starts from first supplier, remove first span of driving from VRP
            if edge_distance[0] >= edge_distance[-1]:
                vehicle_routes[vehicle_id].append(0)
                vehicle_routes[vehicle_id].pop(0)
                route_distance = route_distance - edge_distance[0]
                plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index), route_load)
                plan_output += 'Distance of the route: {} miles\n'.format(route_distance)
                plan_output += 'Load of the route: {}\n'.format(route_load)
                # print(plan_output.replace('0 Load(0) ->  ', ''))
                total_distance += route_distance
                total_load += route_load

            # truck starts form last supplier,remove last span of driving from VRP
            else:
                route_distance = route_distance - edge_distance[-1]
                vehicle_routes[vehicle_id] = vehicle_routes[vehicle_id][::-1]
                plan_output_backward += ' {0} Load({1})\n'.format(manager.IndexToNode(index), route_load)
                plan_output_backward += 'Distance of the route: {} miles\n'.format(route_distance)
                plan_output_backward += 'Load of the route: {}\n'.format(route_load)
                # print(plan_output_backward)
                total_distance += route_distance
                total_load += route_load
        # print('Total distance of all routes: {} miles'.format(total_distance))
        # print('Total load of all routes: {}'.format(total_load))

        """
        generate route schedule as a readable format: {truck: supplier_index}
        """
        df = pd.DataFrame()
        for k in vehicle_routes.keys():
            if len(vehicle_routes[k]) == 1:  # this step eliminate dummy trucks like #0,#1 trucks doing nothing
                continue
            for v in vehicle_routes[k]:
                df = df.append(pd.DataFrame({'truck_number': [k], 'pick_node': [v]}))
        route_schedule = df.reset_index(drop=True)
        return route_schedule


def vrp_main(model, _data=None):

    # Register transit callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return _data['distance_matrix'][from_node][to_node]

    # Add Capacity constraint
    def demand_callback(from_index):
        from_code = manager.IndexToNode(from_index)
        return _data['demands'][from_code]

    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(len(_data['distance_matrix']), _data['num_vehicles'], _data['depot'])

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Define cost of each arch
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add count_stops constraint
    count_stop_callback = routing.RegisterUnaryTransitCallback(lambda index: 1)
    dimension_name = 'Counter'
    routing.AddDimension(evaluator_index=count_stop_callback, slack_max=0, capacity=config.VEHICLE_CAPACITY,
                         fix_start_cumul_to_zero=True, name='Counter')

    # Add solver to count stop numbers
    counter_dimension = routing.GetDimensionOrDie(dimension_name)
    for vehicle_id in range(config.VEHICLE_COUNTS):
        index = routing.End(vehicle_id)
        solver = routing.solver()
        solver.Add(counter_dimension.CumulVar(index) <= config.VEHICLE_STOPS)

    # Add Capacity constraint
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(evaluator_index=demand_callback_index,
                                            slack_max=0,
                                            vehicle_capacities=_data['vehicle_capacities'],
                                            fix_start_cumul_to_zero=True,
                                            name='Capacity')

    # Adding penalty for loading weight exceeds truck capacity
    penalty = config.PENALTY
    for node in range(1, len(_data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Solve the problem_applying solver
    assignment = routing.SolveWithParameters(search_parameters)

    if assignment:
        res = model.print_solution(_data, manager, routing, assignment)
    return res


if __name__ == "__main__":

    vrp_prep = VrpPrep(path=config.PATH)
    vrp_finance = vfa.VrpViz(path=config.PATH)

    osk_hub_dict = vrp_prep.hub_dict(destination_list=config.OSK_DESTINATION_LIST, file=config.INPUT_CASS_FILE)

    cass_zip_cluster_copy, source_states = vrp_prep.choose_nth_supplier_cluster(file=config.OUTPUT_SUPPLIER_CLUSTER_FILE,
                                                                                 rank=config.CLUSTER_RANK)

    riding_distance_matrix = vrp_prep.load_riding_distance_matrix(file=config.INPUT_RIDING_DISTANCE)

    cass_zip_100, distance_matrix_100, unique_cass_zip_100, \
        unique_distance_matrix_100, df_unique_distance_matrix, \
        ship_weight_list = vrp_prep.limited_distance_matrix(riding_distance_matrix, cass_zip_cluster_copy)

    vrp_model = VrpModel(path=config.PATH)

    _data = vrp_model.create_data_model(distance_matrix=distance_matrix_100, ship_weight_list=ship_weight_list,
                                        each_vehicle_capacity=config.VEHICLE_CAPACITY,
                                        num_vehicles=config.VEHICLE_COUNTS,
                                        nrlocations=config.ROUTE_LOCATION_COUNTS)

    route_schedule = vrp_main(model=vrp_model, _data=_data)

    df_tmc = vrp_finance.load_data(file=config.INPUT_TMC, sheet_name=config.SHEET_NAMES)

    sink_state = config.HUB_LIST[config.HUB_NAME][-1]
    source_states_unique = source_states.shipper_state.unique()

    tmc_output = vrp_finance.clean_tmc(df=df_tmc, sink_state=sink_state, source_states=source_states_unique)

    route_in_weight = vrp_finance.route_weight_output(route_schedule=route_schedule,
                                                      cass_zip_100=cass_zip_100,
                                                      file=config.OUTPUT_ROUTE_IN_WEIGHT,
                                                      tmc=tmc_output,
                                                      df_unique_distance_matrix=df_unique_distance_matrix)




















