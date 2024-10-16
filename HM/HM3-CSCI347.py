import matplotlib.pyplot as plt
import networkx as nx
def genUndirectedGraph():
    g = nx.Graph()
    #adds 1-12 as nodes
    g.add_nodes_from(range(1,13))
    #add edges (realized i made some dupes)
    g.add_edges_from([(1,2),(1,3)])
    g.add_edges_from([(2,1),(2,3)])
    g.add_edges_from([(3,1),(3,2),(3,4),(3,5),(3,12)])
    g.add_edges_from([(4,3),(4,5)])
    g.add_edges_from([(5,3),(5,4), (5,11)])
    g.add_edges_from([(6,7),(6,12)])
    g.add_edges_from([(7,6),(7,12)])
    g.add_edges_from([(8,12)])
    g.add_edges_from([(9,12)])
    g.add_edges_from([(10,12)])
    g.add_edges_from([(11,12)])
    return g
#betweenness centrality
def question3(g):
    bc = nx.betweenness_centrality(g)
    print("Question 3:")
    print("betweenness centrality of vertice 3:",bc[3])
    print("betweenness centrality of vertice 12:",bc[12],"\n")
#eigenvector centrality
def question4(g):
    ev = nx.eigenvector_centrality(g)
    print("Question 4:")
    print("Eigenvector Centrality of vertice 3:",ev[3])
    print("Eigenvector Centrality of vertice 12:",ev[12],"\n")

#average length of shortest path
def question5(g):
    avgPath = nx.average_shortest_path_length(g)
    print("Question 5:")
    print("Average length of shortest path:",avgPath)
    
#degree distribution
def question6(g):
    distribution=[]
    #generates degree distribution
    for i in range(1,13):
        distribution.append(g.degree[i])

    fig, ax = plt.subplots()
    xAxis = [str(i) for i in range(1,13)]
    ax.bar(xAxis, height=distribution)
    ax.set_xlabel('Vertex')
    ax.set_ylabel('Number of Degrees')
    ax.set_title('Degree Distribution Among Vertices')

    plt.show()

def question9():
    #generates the plots and gets betweenness centrality
    g = nx.erdos_renyi_graph(n=200, p=0.1)
    bc = nx.betweenness_centrality(g)
    degree = dict(g.degree())
    #multiplier scales size of vertex so they are actually visible
    multiplier = 5000
    nodeSize = [v*multiplier for v in bc.values()]
    nodeColors = [degree[node] for node in g.nodes()]

    nx.draw(g, node_size= nodeSize, node_color = nodeColors)
    plt.title('Erdos-Renyi Random Graph')
    plt.show()

def main():
    g = genUndirectedGraph()
    #question3(g)
    #question4(g)
    #question5(g)
    #question6(g)
    question9()

main()
