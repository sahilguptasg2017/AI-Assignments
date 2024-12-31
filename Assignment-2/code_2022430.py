# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque
# import time
# import tracemalloc
## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------
# print(df_stops , df_routes , df_stop_times , df_fare_attributes , df_trips , df_fare_rules)
# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping

    # Map route_id to a list of stops in order of their sequence

    # Ensure each route only has unique stops
    
    # Count trips per stop

    # Create fare rules for routes
    df_stop_times['arrival_time'] = pd.to_datetime(df_stop_times['arrival_time'], format='%H:%M:%S' , errors='coerce').dt.time
    # Merge fare rules and attributes into a single DataFrame
    for u , row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']
    sequence = defaultdict(list)
    for u , row in df_stop_times.iterrows():
        route = trip_to_route[row['trip_id']]
        stop = row['stop_id']
        seq = row['stop_sequence']
        sequence[route].append((seq , stop))
    for route , stop in sequence.items():
        stop = sorted(stop)
        lstop = [st for seq , st in stop]
        route_to_stops[route] = list(dict.fromkeys(lstop))
    tempStop = {}
    for u , row in df_stop_times.iterrows():
        if tempStop.get(row['stop_id']) is None:
            tempStop[row['stop_id']] = [row['trip_id']]
        else:
            # if row['trip_id'] not in tempStop[row['stop_id']]: # remove if we dont have to check for unique trips
            tempStop[row['stop_id']].append(row['trip_id'])
    for stop , trips in tempStop.items():
        stop_trip_count[stop] = len(trips)
                
    fareprice = {row['fare_id'] : row['price'] for u , row in df_fare_attributes.iterrows()}
    for u , row in df_fare_rules.iterrows():
        fareid = row['fare_id'] ; route = row['route_id'] ; origin = row['origin_id'] ; dest = row['destination_id'] ; price = fareprice.get(fareid)
        fare_rules[route] = {'fare_id' : fareid , 'price' : price , 'origin_id' : origin , 'destination_id' : dest}
    merged_fare_df = pd.merge(df_fare_rules , df_fare_attributes , on='fare_id' , how='left')
    
    

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    # Implementation here
    routeTripcount = defaultdict(int)
    for trip , route in trip_to_route.items() :
        routeTripcount[route] += 1
    # print(routeTripcount[789])
    busieintoutes = sorted(routeTripcount.items(), key = lambda x:x[1] , reverse = True)[:5]
    return busieintoutes
    
    
# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    # Implementation here
    mostFrequentStops = sorted(stop_trip_count.items(), key = lambda x:x[1] , reverse = True)[:5]
    return mostFrequentStops

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    # Implementation here
    stopRouteCount = defaultdict(list)
    for routeID in route_to_stops:
        for stopID in route_to_stops[routeID]:
            if routeID not in stopRouteCount[stopID]:
                stopRouteCount[stopID].append(routeID)
    top5BusiestStops = sorted(stopRouteCount.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    top5BusiestStops = [(stop , len(route)) for stop , route in top5BusiestStops]
    return top5BusiestStops

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
   # Implementation here
    # print(route_to_stops[1433])
    directRoute = {}
    for route , stops in route_to_stops.items():
        for i in range(len(stops) -1):
            pr = (stops[i] , stops[i+1])
            if pr not in directRoute:
                directRoute[pr] = [route]
            else:
                directRoute[pr].append(route)
    ans = []
    for x in directRoute:
        if len(set(directRoute[x])) == 1:
            ans.append((x , directRoute[x][0]))
    ans = sorted(ans ,key =  lambda x : stop_trip_count[x[0][0]] + stop_trip_count[x[0][1]] , reverse= True)
    return ans[:5]
   
# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # Implementation here
    graphG = nx.Graph()
    for routeID in route_to_stops:
        for stops in route_to_stops[routeID]:
            graphG.add_node(stops)
        for i in range(len(route_to_stops[routeID])-1):
            graphG.add_edge(route_to_stops[routeID][i], route_to_stops[routeID][i+1] , route = routeID)
    layout = nx.spring_layout(graphG)
    edgeX = [] ; edgeY = []
    for e in graphG.edges():
        x0, y0 = layout[e[0]]
        x1 , y1 = layout[e[1]]
        edgeX.extend([x0, x1, None])    
        edgeY.extend([y0, y1, None])
    traceofG = go.Scatter(x = edgeX, 
            y = edgeY, 
            line = dict(width = 0.5, color = '#888'),
            hoverinfo = 'none', 
            mode = 'lines')
    nodeX = [] ; nodeY = []
    for node in layout:
        x, y = layout[node]
        nodeX.append(x)
        nodeY.append(y)
    nodeAdjacencies = [len(graphG[node]) for node in graphG.nodes()]
    nodeText = ["Stop ID is: " + str(node) + "Number of routes passing through this stop is: " + str(len(graphG[node])) for node in graphG.nodes()]
    traceofNode = go.Scatter(
        x=nodeX, y=nodeY,
        mode='markers',
        hoverinfo='text',
        text=nodeText,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=nodeAdjacencies,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        )
    )
    # for n in graphG.nodes():
    #     nodeAdjacencies.append(len(graphG[n]))
    #     nodeText.append("Stop ID is" + int(n) + "Number of routes passing through this stop is" + int(len(graphG[n])))
    traceofNode.text = nodeText
    fig = go.Figure(data = [traceofG, traceofNode], 
                    layout=go.Layout(showlegend = False,
                                    hovermode = 'closest',
                                    margin = dict(b = 20, l = 5, r = 5, t = 40),
                                    xaxis = dict(showgrid = False, zeroline = False, showticklabels = False), 
                                    yaxis = dict(showgrid = False, zeroline = False, showticklabels = False))
                    )
            
            
    
    fig.show()
# create_kb()
# visualize_stop_route_graph_interactive(route_to_stops)

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.cl
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    # tracemalloc.start()
    # startTime = time.time()
    ans = []
    for route , stop in route_to_stops.items():
        flag = False
        for i in range(len(stop)):
            for j in range(len(stop)):
                if i != j:
                    if stop[i] == start_stop and stop[j] == end_stop:
                        # flag = True
                        ans.append(route)
    # endTime = time.time()
    # curr , peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # print(f"Time taken for brute-force direct route is:  {endTime - startTime:.6f} sec")
    # print(f"Current memory usage for brute-force direct route is:  {curr/(1024*1024):.6f} MB")
    # print(f"Peak memory usage for brute-force direct route is:  {peak/(1024*1024):.6f} MB")
    return ans

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  

def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates
    create_kb()  # Populate the knowledge base 
    add_route_data(route_to_stops)  # Add route data to Datalog
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # Implementation here         
    for r , s in route_to_stops.items():
        for st in s:
            +RouteHasStop(r,st)
# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.
    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (int) connecting the two stops.
    """
    DirectRoute(X, Y) <= (RouteHasStop(R, X) & RouteHasStop(R, Y))
    # startTime = time.time()
    # tracemalloc.start()
    ans = (DirectRoute(start , end) & (RouteHasStop(R,start) & RouteHasStop(R,end))).data
    res = [a[0] for a in ans]
    # endTime = time.time()
    # curr , peak = tracemalloc.get_traced_memory()
    # print(f"Time taken for FOL-based direct route is:  {endTime - startTime:.6f} sec")
    # print(f"Current memory usage for FOL-based direct route is:  {curr/(1024*1024):.6f} MB")
    # print(f"Peak memory usage for FOL-based direct route is:  {peak/(1024*1024):.6f} MB")
    return sorted(list(set(res)))

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers=1):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    
    # Define the forward chaining logic
    OptimalRoute(X, Y, Z, R1, R2) <= (RouteHasStop(R1, X) & RouteHasStop(R1, Y) & RouteHasStop(R2, Y) & RouteHasStop(R2, Z) & (R1 != R2))
    # startTime = time.time()
    # tracemalloc.start()
    res = OptimalRoute(start_stop_id, stop_id_to_include, end_stop_id, R1, R2).data
    res = [(r1, stop_id_to_include, r2) for (r1, r2) in res]
    # endTime = time.time()
    # curr , peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # print(f"Time taken for FOL-based forward chaining is:  {endTime - startTime:.6f} sec")
    # print(f"Current memory usage for FOL-based forward chaining is:  {curr/(1024*1024):.6f} MB")
    # print(f"Peak memory usage for FOL-based forward chaining is:  {peak/(1024*1024):.6f} MB")
    return res
       
                                    
    

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    # Implementation here
    # startTime = time.time()
    # tracemalloc.start()
    res = OptimalRoute(end_stop_id, stop_id_to_include, start_stop_id, R1, R2).data
    res = [(r1, stop_id_to_include, r2) for (r1, r2) in res]
    # endTime = time.time()
    # curr , peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # print(f"Time taken for FOL-based backward chaining is:  {endTime - startTime:.6f} sec")
    # print(f"Current memory usage for FOL-based backward chaining is:  {curr/(1024*1024):.6f} MB")
    # print(f"Peak memory usage for FOL-based backward chaining is:  {peak/(1024*1024):.6f} MB")
    return res

    
pyDatalog.create_terms('Board' , 'Transfer' , 'OptimalRouteNew')
# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    # Implementation here
    # Define the PDDL-style planning logic
    Board(R , X , Y) <= (RouteHasStop(R,X) & RouteHasStop(R,Y))
    Transfer(R1 , R2 , Y , Z) <= (RouteHasStop(R1 , Y) & RouteHasStop(R2 , Y) & RouteHasStop(R2 , Z) & (R1 != R2))
    OptimalRouteNew(X , Y , Z , R1 , R2) <= (Board(R1 , X , Y) & Transfer(R1 , R2 , Y , Z))
    # startTime = time.time()
    # tracemalloc.start()
    res = OptimalRouteNew(start_stop_id , stop_id_to_include , end_stop_id , R1 , R2).data
    ans = []
    for (r1 , r2) in res:
        ans.append((r1 , stop_id_to_include , r2))
        # print(f'Current State: Route from {r1} to {r2} via {stop_id_to_include}') 
    # endTime = time.time()
    # curr , peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # print(f"Time taken for pddl is:  {endTime - startTime:.6f} sec")
    # print(f"Current memory usage for pddl is:  {curr/(1024*1024):.6f} MB")
    # print(f"Peak memory usage for pddl is:  {peak/(1024*1024):.6f} MB")
    return ans
    

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    # Implementation here
    ans = merged_fare_df[merged_fare_df['price'] <= initial_fare]
    return ans
# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    # Implementation here
    sm = {}
    for r in pruned_df['route_id'].unique():
        rData = pruned_df[pruned_df['route_id'] == r]
        minPrice = rData['price'].min()
        s = set(route_to_stops[r])
        sm[r] = {'min_price' : minPrice , 'stops' : s}
    return sm

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    # Implementation here
    que = deque([(start_stop_id , [] , -1 , initial_fare)])
    vis = set()
    while que:
        curr , ans , transfer , rem = que.popleft()
        if transfer > max_transfers or rem < 0:
            continue
        if curr == end_stop_id:
            return ans
        for r, rinfo in route_summary.items():
            if curr in rinfo['stops'] and rinfo['min_price'] <= rem:
                for next in rinfo['stops']:
                    if next != curr and (next , r) not in vis:
                        vis.add((next , r))
                        que.append((next , ans + [(r, next)] , transfer + 1 , rem - rinfo['min_price']))
    return []   
