# Jack Shipway, 6/10/16, UCL GCRF Project
#
# Compute various centrality metrics for each graph at a particular temporal granluarity
#
# Input: csv files generated from 'temporalDataGeneration.py'
# Output: plots of mean/variance of centrality through time

import pandas as pd
import numpy as np
import arrow as ar
import math
import time
import networkx as nx
import csv
import matplotlib.pyplot as plt
import sys

# Define constants - given in data description
CELL_TOWERS = 1238
MONTHS = 5
WEEKS = 20
DAYS = 140
HOURS = 3360

if __name__ == "__main__":

    # Hour 23, DS9
    data = pd.DataFrame(pd.read_csv("IvoryCoastData/CDR/ICS1/ICS1AntToAnt_9.TSV", sep='\t', header=None))
    data.columns=['datetime', 'source', 'target', 'weight', 'duration']

    for i in range(178):
        data[:][16644011:].to_csv("IvoryCoastData/CDR/TemporalData/Month_4/Week_3/Day_6/Hour_23/graph.csv", sep=',', index=None)







    # # Array to store monthly bc values per node
    # monthly_bc = np.zeros((MONTHS, CELL_TOWERS))
    #
    # # Monthly level
    # for i in range(MONTHS):
    #
    #     # Data split into fortnights - two fortnights (a & b) make a month
    #     print "Reading data set %s" % (2 * i)
    #     fortnight_a = pd.DataFrame(pd.read_csv("IvoryCoastData/CDR/ICS1/ICS1AntToAnt_%s.TSV" % (2 * i), sep='\t', header=None), index=None)
    #     fortnight_a.columns = ['datetime', 'source', 'target', 'weight', 'duration']
    #     fortnight_a = fortnight_a[['source', 'target', 'weight']]
    #     print "Reading data set %s" % ((2 * i) + 1)
    #     fortnight_b = pd.DataFrame(pd.read_csv("IvoryCoastData/CDR/ICS1/ICS1AntToAnt_%s.TSV" % ((2 * i) + 1), sep='\t', header=None), index=None)
    #     fortnight_b.columns = ['datetime', 'source', 'target', 'weight', 'duration']
    #     fortnight_b = fortnight_b[['source', 'target', 'weight']]
    #     month = pd.DataFrame(pd.concat([fortnight_a, fortnight_b]), index=None)
    #     del fortnight_a, fortnight_b
    #
    #     month.columns = ['source', 'target', 'weight']
    #     month = month[~(month['source'] == -1)]
    #     month = month[~(month['target'] == -1)]
    #
    #     # Aggregate call volume (weight) at monthly level
    #     month = month.groupby(['source', 'target'], as_index=False)['weight'].sum()
    #     month.columns = ['source', 'target', 'weight']
    #
    #     # Get IDs of cell towers present at each aggregated level
    #     source = month.source.unique()
    #     target = month.target.unique()
    #     nodes = np.union1d(source, target)
    #     num_nodes = len(nodes)
    #
    #     # Initialise networkx graph
    #     G = nx.Graph()
    #     # Add nodes
    #     for j in range(num_nodes):
    #         G.add_node(nodes[j], color='red')
    #     # color nodes
    #     color_map = []
    #     for n in G.nodes():
    #         color_map.append(G.node[n]['color'])
    #
    #     # Add links to graph
    #     num_links = len(month)
    #     # change t0 num_links
    #     for k in range(10000):
    #         G.add_edge(month['source'][k], month['target'][k])
    #
    #     del month
    #     # Compute betweenness centrality - any others??
    #     month_i_bc = nx.betweenness_centrality(G)
    #     keys = month_i_bc.keys()
    #     values = month_i_bc.values()
    #
    #     for j in range(num_nodes):
    #         monthly_bc[i][keys[j]] = values[j]
    #
    # #     del G, month_i_bc, keys, values
    #
    #
    # # Array to store weekly bc values per node
    # weekly_bc = np.zeros((WEEKS, CELL_TOWERS))
    #
    # # Weekly index splits
    # week_split = [11263389, 8786057, 8699807, 9051285, 10325885, 7091547, 8535557, 5903162, 8424655, 8038527]
    #
    # # Weekly level
    # for i in range(WEEKS/2):
    #
    # # Data split into fortnights - there are two weeks per data set
    #     print "Reading data set %s" % i
    #     data = pd.DataFrame(pd.read_csv("IvoryCoastData/CDR/ICS1/ICS1AntToAnt_%s.TSV" % i,
    #                                       sep='\t', header=None), index=None)
    #     data.columns = ['datetime', 'source', 'target', 'weight', 'duration']
    #
    #     # Get week a)
    #     print "Getting week a"
    #     week_a = data[['source', 'target', 'weight']][0:week_split[i]]
    #     print "len week b is : ", len(week_a)
    #     # getting week b
    #     print "getting week b"
    #     week_b = data[['source', 'target', 'weight']][week_split[i]:]
    #     print "len week b is : ", len(week_a)
    #
    #     print "Removing -1 data: a"
    #     week_a = week_a[~(week_a['source'] == -1)]
    #     print "Removing -1 data: b"
    #     week_b = week_b[~(week_b['target'] == -1)]
    #     print "Removing -1 data: a"
    #     week_a = week_a[~(week_a['source'] == -1)]
    #     print "Removing -1 data: b"
    #     week_b = week_b[~(week_b['target'] == -1)]
    #
    #     # Aggregate call volume (weight) at monthly level
    #     print "Aggregating data: a"
    #     week_a = week_a.groupby(['source', 'target'], as_index=False)['weight'].sum()
    #     week_a.columns = ['source', 'target', 'weight']
    #     week_b = week_b.groupby(['source', 'target'], as_index=False)['weight'].sum()
    #     print "Aggregating data: a"
    #     week_b.columns = ['source', 'target', 'weight']
    #
    #     # Get IDs of cell towers present at each aggregated level
    #     print "getting IDs: a"
    #     source_a = week_a.source.unique()
    #     target_a = week_a.target.unique()
    #     nodes_a = np.union1d(source_a, target_a)
    #     num_nodes_a = len(nodes_a)
    #
    #     print "getting IDs: b"
    #     source_b = week_b.source.unique()
    #     target_b = week_b.target.unique()
    #     nodes_b = np.union1d(source_b, target_b)
    #     num_nodes_b = len(nodes_b)
    #
    #     # Initialise networkx graph
    #     G = nx.Graph()
    #     # Add nodes
    #     for j in range(num_nodes_a):
    #         G.add_node(nodes_a[j], color='red')
    #     # color nodes
    #     color_map = []
    #     for n in G.nodes():
    #         color_map.append(G.node[n]['color'])
    #     # Add links to graph
    #     num_links_a = len(week_a)
    #     # change t0 num_links
    #     for k in range(num_links_a):
    #         G.add_edge(week_a['source'][k], week_a['target'][k])
    #     # Compute betweenness centrality - any others??
    #     print "computing betweenness of week %s" % (2*i)
    #     week_a_bc = nx.betweenness_centrality(G)
    #     keys_a = week_a_bc.keys()
    #     values_a = week_a_bc.values()
    #
    #     for j in range(num_nodes_a):
    #         weekly_bc[(2 * i)][keys_a[j]] = values_a[j]
    #
    #     # Initialise networkx graph
    #     H = nx.Graph()
    #     # Add nodes
    #     for j in range(num_nodes_b):
    #         H.add_node(nodes_b[j], color='red')
    #     # color nodes
    #     color_map = []
    #     for n in H.nodes():
    #         color_map.append(H.node[n]['color'])
    #     # Add links to graph
    #     num_links_b = len(week_b)
    #     # change t0 num_links
    #     for k in range(num_links_b):
    #         H.add_edge(week_b['source'][k], week_b['target'][k])
    #
    #     print "computing betweenness of week %s" % ((2*i) + 1)
    #     week_b_bc = nx.betweenness_centrality(H)
    #     keys_b = week_b_bc.keys()
    #     values_b = week_b_bc.values()
    #
    #     for j in range(num_nodes_b):
    #         weekly_bc[(2 * i) + 1][keys_b[j]] = values_b[j]
    #
    #     print "First two weeks of the bc data: "
    #     print weekly_bc
    #
    # # Array to store weekly bc values per node
    # daily_bc = np.zeros((DAYS, CELL_TOWERS))
    #
    # # Daily level
    # for i in range(DAYS):
    #
    # # Data split into days - there are 28 days per data set
    #     print "Reading data set %s" % i
    #     data = pd.DataFrame(pd.read_csv("IvoryCoastData/CDR/ICS1/ICS1AntToAnt_%s.TSV" % i,
    #                                       sep='\t', header=None), index=None)
    #     data.columns = ['datetime', 'source', 'target', 'weight', 'duration']
    #
    #     # Get week a)
    #     print "Getting week a"
    #     week_a = data[['source', 'target', 'weight']][0:week_split[i]]
    #     print "len week b is : ", len(week_a)
    #     # getting week b
    #     print "getting week b"
    #     week_b = data[['source', 'target', 'weight']][week_split[i]:]
    #     print "len week b is : ", len(week_a)
    #
    #     print "Removing -1 data: a"
    #     week_a = week_a[~(week_a['source'] == -1)]
    #     print "Removing -1 data: b"
    #     week_b = week_b[~(week_b['target'] == -1)]
    #     print "Removing -1 data: a"
    #     week_a = week_a[~(week_a['source'] == -1)]
    #     print "Removing -1 data: b"
    #     week_b = week_b[~(week_b['target'] == -1)]
    #
    #     # Aggregate call volume (weight) at monthly level
    #     print "Aggregating data: a"
    #     week_a = week_a.groupby(['source', 'target'], as_index=False)['weight'].sum()
    #     week_a.columns = ['source', 'target', 'weight']
    #     week_b = week_b.groupby(['source', 'target'], as_index=False)['weight'].sum()
    #     print "Aggregating data: a"
    #     week_b.columns = ['source', 'target', 'weight']
    #
    #     # Get IDs of cell towers present at each aggregated level
    #     print "getting IDs: a"
    #     source_a = week_a.source.unique()
    #     target_a = week_a.target.unique()
    #     nodes_a = np.union1d(source_a, target_a)
    #     num_nodes_a = len(nodes_a)
    #
    #     print "getting IDs: b"
    #     source_b = week_b.source.unique()
    #     target_b = week_b.target.unique()
    #     nodes_b = np.union1d(source_b, target_b)
    #     num_nodes_b = len(nodes_b)
    #
    #     # Initialise networkx graph
    #     G = nx.Graph()
    #     # Add nodes
    #     for j in range(num_nodes_a):
    #         G.add_node(nodes_a[j], color='red')
    #     # color nodes
    #     color_map = []
    #     for n in G.nodes():
    #         color_map.append(G.node[n]['color'])
    #     # Add links to graph
    #     num_links_a = len(week_a)
    #     # change t0 num_links
    #     for k in range(num_links_a):
    #         G.add_edge(week_a['source'][k], week_a['target'][k])
    #     # Compute betweenness centrality - any others??
    #     print "computing betweenness of week %s" % (2*i)
    #     week_a_bc = nx.betweenness_centrality(G)
    #     keys_a = week_a_bc.keys()
    #     values_a = week_a_bc.values()
    #
    #     for j in range(num_nodes_a):
    #         weekly_bc[(2 * i)][keys_a[j]] = values_a[j]
    #
    #     # Initialise networkx graph
    #     H = nx.Graph()
    #     # Add nodes
    #     for j in range(num_nodes_b):
    #         H.add_node(nodes_b[j], color='red')
    #     # color nodes
    #     color_map = []
    #     for n in H.nodes():
    #         color_map.append(H.node[n]['color'])
    #     # Add links to graph
    #     num_links_b = len(week_b)
    #     # change t0 num_links
    #     for k in range(num_links_b):
    #         H.add_edge(week_b['source'][k], week_b['target'][k])
    #
    #     print "computing betweenness of week %s" % ((2*i) + 1)
    #     week_b_bc = nx.betweenness_centrality(H)
    #     keys_b = week_b_bc.keys()
    #     values_b = week_b_bc.values()
    #
    #     for j in range(num_nodes_b):
    #         weekly_bc[(2 * i) + 1][keys_b[j]] = values_b[j]
    #
    #     print "First two weeks of the bc data: "
    #     print weekly_bc
    #
    #
    # print weekly_bc, weekly_bc.shape
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # # # Initialise matrix to store betweenness centrality of each node
    # # hourly_bc = np.zeros((2500, 1237))
    # #
    # # # Get directory corresponding to timestep
    # # hour, day, week, month = 0, 0, 0, 0
    # #
    # # # Hourly granularity
    # # for i in range(36):
    # #
    # #     # Load graph data
    # #     print "Loading data set %s" % i
    # #     data = pd.DataFrame(pd.read_csv("IvoryCoastData/CDR/TemporalData/Month_%s/Week_%s/Day_%s/Hour_%s/graph.csv" % (month, week, day, hour),
    # #                                     sep=','), index=None)
    # #     data.columns = ['source', 'target', 'weight']
    # #
    # #     # Get unique node ID numbers and number of nodes
    # #     source = data.source.unique()
    # #     target = data.target.unique()
    # #     nodes = np.union1d(source, target)
    # #     nodes = nodes[nodes > 0]
    # #     num_nodes = len(nodes)
    # #
    # #     # Initialise networkx graph per timestep
    # #
    # #
    # #     # Compute betweenness centrality - any others??
    # #     print "Computing BC for Month: %s, Week: %s, Day: %s, Hour: %s" % (month, week, day, hour)
    # #     hour_i_bc = nx.betweenness_centrality(G)
    # #     keys = hour_i_bc.keys()
    # #     values = hour_i_bc.values()
    # #
    # #     for j in range(num_nodes):
    # #         hourly_bc[i][keys[j]] = values[j]
    # #
    # #     #nx.draw_networkx(G, node_color=color_map, with_labels=True, node_size=500)
    # #     #plt.show()
    # #
    # #     # Update timestep
    # #     hour = int(math.fmod(hour + 1, 24))
    # #     if hour == 0:
    # #         day = int(math.fmod(day + 1, 7))
    # #     if hour == 0 and day == 0:
    # #         week = int(math.fmod(week + 1, 4))
    # #     if hour == 0 and day == 0 and week == 0:
    # #         month = int(math.fmod(month + 1, 5))
    # #
    # # # Compute the standard deviation and mean of each node's betwenness centrality as it changes through time
    # # std = np.std(hourly_bc, axis = 0)
    # # avg = np.mean(hourly_bc, axis = 0)
    # # print std
    # # print avg
    # #
    # # plt.bar(range(1237), avg)
    # # plt.show()
    # # plt.bar(range(1237), std)
    # # plt.show()
    #
    # #
    # #
    # #
    #
    #
    #
    #
