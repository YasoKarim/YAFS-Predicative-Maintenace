from matplotlib import pyplot as plt
from yafs.application import Application

import random
import networkx as nx
import argparse
from pathlib import Path
import time
import numpy as np

from yafs.core import Sim
from yafs.application import Application,Message

from yafs.population import *
from yafs.topology import Topology

from simpleSelection import MinPath_RoundRobin, MinimunPath
from simplePlacement import CloudPlacement
from yafs.stats import Stats
from yafs.distribution import deterministic_distribution
from yafs.application import fractional_selectivity

RANDOM_SEED = 1

def create_application():

    app = Application(name="Predictive Maintenance")

    # Set the modules for your application
    app.set_modules([
            {"Sensor": {"Type": Application.TYPE_SOURCE}},
            {"ServiceA": {"RAM": 10, "Type": Application.TYPE_MODULE}},
            {"Actuator": {"Type": Application.TYPE_SINK}},
        ])
    
    # Define messages (modify as needed)
    m_a = Message("M.A", "Sensor", "ServiceA", instructions=50*10**6, bytes=500)
    m_b = Message("M.B", "ServiceA", "Actuator", instructions=30*10**6, bytes=500,broadcasting=True)
    # Add source messages
    app.add_source_messages(m_a)
   
    # Define service module 
    app.add_service_module("ServiceA", m_a, m_b, fractional_selectivity, threshold=1.0)
    
    return app

# Create topology for the simulation
def create_topology():
    """
    Creates a topology for the predictive maintenance simulation.
    Includes 3 sensors, 3 actuators, and 2 computing devices.
    """
    ## MANDATORY FIELDS
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []
    
    '''
    We have 3 sensors, 1 service, and 3 actuators
    3 sensors are sources and 3 actuators are sinks while the service is a module and it has 10 RAM units
    Connection: Sensor1 -> ServiceA -> Actuator1 (It is a temprature sensor, and it is connected to a temprature actuator)
    Connection: Sensor2 -> ServiceA -> Actuator2 (It is a vibration sensor, and it is connected to a vibration actuator)
    Connection: Sensor3 -> ServiceA -> Actuator3 (It is a pressure sensor, and it is connected to a pressure actuator)
    '''
    # Entities (Nodes)
    cloud_dev = {"id": 0, "model": "cloud", "mytag": "cloud", "IPT": 5000 * 10 ** 6, "RAM": 40000, "COST": 3, "WATT": 20.0}
    fog_dev1 = {"id": 1, "model": "fog-device", "IPT": 1000 * 10 ** 6, "RAM": 8000, "COST": 3, "WATT": 10.0}
    fog_dev2 = {"id": 2, "model": "fog-device", "IPT": 1000 * 10 ** 6, "RAM": 8000, "COST": 3, "WATT": 10.0}
    sensor1 = {"id": 3, "model": "sensor-device-1", "IPT": 50 * 10 ** 6, "RAM": 4000, "COST": 1, "WATT": 40.0}
    sensor2 = {"id": 4, "model": "sensor-device-2", "IPT": 50 * 10 ** 6, "RAM": 4000, "COST": 1, "WATT": 40.0}
    sensor3 = {"id": 5, "model": "sensor-device-3", "IPT": 50 * 10 ** 6, "RAM": 4000, "COST": 1, "WATT": 40.0}
    actuator1 = {"id": 6, "model": "actuator-device-1", "IPT": 50 * 10 ** 6, "RAM": 4000, "COST": 1, "WATT": 40.0}
    actuator2 = {"id": 7, "model": "actuator-device-2", "IPT": 50 * 10 ** 6, "RAM": 4000, "COST": 1, "WATT": 40.0}
    actuator3 = {"id": 8, "model": "actuator-device-3", "IPT": 50 * 10 ** 6, "RAM": 4000, "COST": 1, "WATT": 40.0}

    # Links (Connections)
    link1 = {"s": 0, "d": 3, "BW": 1, "PR": 10}  # Cloud -> Sensor1
    link2 = {"s": 0, "d": 4, "BW": 1, "PR": 10}  # Cloud -> Sensor2
    link3 = {"s": 0, "d": 5, "BW": 1, "PR": 10}  # Cloud -> Sensor3
    
    link4 = {"s": 1, "d": 3, "BW": 1, "PR": 5}   # Sensor1 -> Fog Device 1
    link5 = {"s": 1, "d": 4, "BW": 1, "PR": 5}   # Sensor2 -> Fog Device 1
    link6 = {"s": 1, "d": 5, "BW": 1, "PR": 5}   # Sensor3 -> Fog Device 1
    
    link7 = {"s": 0, "d": 6, "BW": 1, "PR": 5}   # cloud -> Actuator1
    link8 = {"s": 0, "d": 7, "BW": 1, "PR": 5}   # cloud -> Actuator2
    link9 = {"s": 0, "d": 8, "BW": 1, "PR": 5}   # cloud -> Actuator3
    
    link10 = {"s": 1, "d": 6, "BW": 1, "PR": 5}  # Fog Device 1 -> Actuator1
    link11 = {"s": 1, "d": 7, "BW": 1, "PR": 5}  # Fog Device 1 -> Actuator2
    link12 = {"s": 1, "d": 8, "BW": 1, "PR": 5}  # Fog Device 1 -> Actuator3
    
    link13 = {"s": 0, "d": 1, "BW": 1, "PR": 5}  # Fog Device 1 -> Cloud
    link14 = {"s": 0, "d": 2, "BW": 1, "PR": 5}  # Fog Device 2 -> Cloud
    
    link15 = {"s": 2, "d": 6, "BW": 1, "PR": 5}  # Fog Device 2 -> Actuator1
    link16 = {"s": 2, "d": 7, "BW": 1, "PR": 5}  # Fog Device 2 -> Actuator2
    link17 = {"s": 2, "d": 8, "BW": 1, "PR": 5}  # Fog Device 2 -> Actuator3
    
    link18 = {"s": 2, "d": 1, "BW": 1, "PR": 5}  # Fog Device 2 -> Fog Device 1
    
    link19 = {"s": 3, "d": 2, "BW": 1, "PR": 5}  # Sensor1 -> Fog Device 2
    link20 = {"s": 4, "d": 2, "BW": 1, "PR": 5}  # Sensor2 -> Fog Device 2
    link21 = {"s": 5, "d": 2, "BW": 1, "PR": 5}  # Sensor3 -> Fog Device 2
    
    # Add entities and links to the topology
    topology_json["entity"].append(cloud_dev)
    topology_json["entity"].append(fog_dev1)
    topology_json["entity"].append(fog_dev2)
    topology_json["entity"].append(sensor1)
    topology_json["entity"].append(sensor2)
    topology_json["entity"].append(sensor3)
    topology_json["entity"].append(actuator1)
    topology_json["entity"].append(actuator2)
    topology_json["entity"].append(actuator3)
    topology_json["link"].append(link1)
    topology_json["link"].append(link2)
    topology_json["link"].append(link3)
    topology_json["link"].append(link4)
    topology_json["link"].append(link5)
    topology_json["link"].append(link6)
    topology_json["link"].append(link7)
    topology_json["link"].append(link8)
    topology_json["link"].append(link9)
    topology_json["link"].append(link10)
    topology_json["link"].append(link11)
    topology_json["link"].append(link12)
    topology_json["link"].append(link13)
    topology_json["link"].append(link14)
    topology_json["link"].append(link15)
    topology_json["link"].append(link16)
    topology_json["link"].append(link17)
    topology_json["link"].append(link18)
    topology_json["link"].append(link19)
    topology_json["link"].append(link20)
    topology_json["link"].append(link21)
    
    
    return topology_json

 
    # In predictive maintenance, we need to consider additional factors related to data collection and decision-making.

    # 1. Data Collection:
    # - Collect data from sensors (e.g., temperature, vibration, pressure).
    # - Assume you have real-time data available for each sensor.

    # 2. Anomaly Detection:
    # - Implement anomaly detection algorithms (e.g., statistical methods, machine learning) to identify deviations.
    # - For example, if temperature suddenly spikes or vibration exceeds a threshold, it could indicate a potential issue.

    # 3. Decision Thresholds:
    # - Set thresholds for triggering maintenance actions.
    # - For instance, if temperature exceeds a certain value, schedule maintenance.

    # 4. Integration with Selector Algorithm:
    # - Combine shortest path information with maintenance routes.
    # - When an anomaly is detected, determine the most efficient path for maintenance personnel.

def main(simulated_time):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    folder_results = Path("results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)+"/"

    """
    TOPOLOGY from a json
    """
    t = Topology()
    t_json = create_topology()
    t.load(t_json)
    nx.write_gexf(t.G,folder_results+"graph_predicitive_Maintaence") # you can export the Graph in multiples format to view in tools like Gephi, and so on.

    G = nx.read_gexf(folder_results+"graph_predicitive_Maintaence")
    nx.draw(G, with_labels=True, font_weight='bold')
    #plt.draw()
    
    """
    APPLICATION
    """
    app = create_application()
    
    """
    PLACEMENT algorithm
    """
    placement = CloudPlacement("onCloud") # it defines the deployed rules: module-device
    placement.scaleService({"ServiceA": 3})
    #placement.scaleService({"ServiceB": 3})
    
    """
    POPULATION algorithm
    """

    pop = Statical("Statical")

    # Set up sink (control module) deployment:
    # - One control sink linked to the "actuator-device" model
    # - This sink receives messages from the predictive maintenance service
    pop.set_sink_control({"model": "actuator-device-1", "number": 1, "module": app.get_sink_modules()})
    pop.set_sink_control({"model": "actuator-device-2", "number": 1, "module": app.get_sink_modules()})
    pop.set_sink_control({"model": "actuator-device-3", "number": 1, "module": app.get_sink_modules()})
    
    # Set up source (sensor) deployment:
    # - One sensor linked to the "sensor-device" model
    # - Use a deterministic distribution (you can customize this based on real data)
    
    dDistribution = deterministic_distribution(name="Deterministic", time=100)
    pop.set_src_control({"model": "sensor-device-1", "number": 1, "message": app.get_message("M.A"), "distribution": dDistribution})
    pop.set_src_control({"model": "sensor-device-2", "number": 1, "message": app.get_message("M.A"), "distribution": dDistribution})
    pop.set_src_control({"model": "sensor-device-3", "number": 1, "message": app.get_message("M.A"), "distribution": dDistribution})
    
    """
    SELECTOR algorithm
    """
    # We'll leverage shortest paths for maintenance routes.
    selectorPath = MinPath_RoundRobin()

    """
    SIMULATION ENGINE
    """

    stop_time = simulated_time
    s = Sim(t, default_results_path=folder_results+"sim_predictive_maintenance_trace")
    s.deploy_app2(app, placement, pop, selectorPath)

    """
    RUNNING - last step
    """
    s.run(stop_time, show_progress_monitor=False)  # To test deployments put test_initial_deploy a TRUE
    s.print_debug_assignaments()

if __name__ == '__main__':
    import logging.config
    import os

    logging.config.fileConfig(os.getcwd()+'/logging.ini')

    start_time = time.time()
    main(simulated_time=2000)

    print("\n--- %s seconds ---" % (time.time() - start_time))

    ### Finally, you can analyse the results:
    '''
    print("-"*20)
    print("Results:")
    print("-" * 20)
    m = Stats(defaultPath="Results") #Same name of the results
    time_loops = [["M.A", "M.B"]]
    m.showResults2(1000, time_loops=time_loops)
    print("\t- Network saturation -")
    print("\t\tAverage waiting messages : %i" % m.average_messages_not_transmitted())
    #print("\t\tPeak of waiting messages : %i" % m.peak_messages_not_transmitted()PartitionILPPlacement)
    print("\t\tTOTAL messages not transmitted: %i" % m.messages_not_transmitted())
    '''
    
"""
Calculating Basic Statistics
"""
import pandas as pd


# Get the current working directory
base_dir = os.getcwd()

# Construct the full file paths using os.path.join()
task_execution_file = os.path.join(base_dir, 'results', 'sim_predictive_maintenance_trace.csv')
network_transmission_file = os.path.join(base_dir,'results', 'sim_predictive_maintenance_trace_link.csv')

# Load task execution results
task_df = pd.read_csv(task_execution_file)

# CSV columns are named as follows:
# - 'module' for module or service name
# - 'service' for service time
# - 'time_out' and 'time_emit' for timestamps

# Calculate Service Time Spent Processing at Each Node
task_df['ServiceTime'] = task_df['time_out'] - task_df['time_emit']

# Calculate Total Service Time for a single module
total_service_time = task_df.groupby('module')['ServiceTime'].sum()

# Calculate Node Utilization (NU)
total_simulation_time = task_df['time_out'].max()  # Total simulation time
#print("total_simulation_time",total_simulation_time)
#print("NU",task_df['NU'])

# Initialize NU column to zero for all rows
task_df['NU'] = 0

# Calculate NU for each module
for module in total_service_time.index:
    # Get the total service time for the module
    module_service_time = total_service_time[module]
    
    # Calculate NU for the module
    if total_simulation_time > 0:
        module_nu = module_service_time / total_simulation_time
    else:
        module_nu = 0  # Handle the case where total_simulation_time is 0
    
    # Assign NU to the rows where the module is the same as the current one
    task_df.loc[task_df['module'] == module, 'NU'] = module_nu

#print("NU", task_df['NU'])

# Calculate Throughput
total_messages_received_by_sink = task_df[task_df['type'] == 'SINK_M'].shape[0]
throughput = total_messages_received_by_sink / total_simulation_time

# Load network transmission results
network_df = pd.read_csv(network_transmission_file)

# Assuming your network transmission CSV columns are named as follows:
# - 'latency' for time taken to transmit the message between both nodes
# - 'size' for size of the message
# Calculate Latency for each message
network_df['Latency'] = network_df['size'] / network_df['latency']

# Calculate Average Link Latency
average_link_latency = network_df['Latency'].mean()

# Print the results
print(f"Average Service Time: {task_df['ServiceTime'].mean():.6f} seconds")
print(f"Average Link Latency: {average_link_latency:.6f} seconds")
print(f"Node Utilization: {task_df['NU'].mean():.4f}")
print(f"Throughput: {throughput:.4f} messages per second")


'''
Plotting of the Results
'''
# Plot Total Service Time for each module
plt.figure(figsize=(10, 6))
total_service_time.plot(kind='bar', color='skyblue')
plt.title('Total Service Time per Module')
plt.xlabel('Module')
plt.ylabel('Total Service Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot Node Utilization for each module
nu_values = task_df.groupby('module')['NU'].mean()  # Assuming 'NU' is already calculated as shown above

plt.figure(figsize=(10, 6))
nu_values.plot(kind='bar', color='lightgreen')
plt.title('Node Utilization per Module')
plt.xlabel('Module')
plt.ylabel('Node Utilization (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot the Frequency Distribution of Service Time
plt.figure(figsize=(10, 6))
plt.hist(task_df['ServiceTime'], bins=20, color='purple')
plt.title('Frequency Distribution of Service Time')
plt.xlabel('Service Time (seconds)')
plt.ylabel('Frequency')
plt.show()
