import networkx as nx
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import ndlib.models.ModelConfig as mc
import ndlib.models.CompositeModel as gc
import ndlib.models.compartments as cpm



class Network():
    """
    Represents a network structure used to model disease transmission.

    Parameters:
    ----------
    network_type : str, optional
        The type of network to generate ('erdos_renyi', 'barabasi_albert', or 'watts_strogatz').
    N : int, optional
        The number of nodes in the network.
    connection_prob : float, optional
        The probability of connections between nodes for Erdős-Rényi and Watts-Strogatz networks.
    avg_degree : int, optional
        The average degree for Barabási-Albert and Watts-Strogatz networks.
    clustering_coeff : float, optional
        Desired clustering coefficient for Watts-Strogatz network.
    """

    def __init__(self, network_type=None, N=None, connection_prob=None, avg_degree=None, clustering_coeff=None):
        self.network = self.generate_network(
            network_type, N, connection_prob, avg_degree, clustering_coeff)
        attr = {n: {"Tested": False, "Vaccinated": "False"}
                for n in self.network.nodes()}

        self.network.remove_nodes_from(list(nx.isolates(self.network)))
        nx.set_node_attributes(self.network, attr)

    def init_data(self):
        """
        Initializes data from a CSV file for creating a network.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame representing the adjacency matrix for the network.
        """
        data = pd.read_csv("data/transmission_network.csv", sep=";")
        data = data.set_index("Unnamed: 0")
        data.index.name = None
        data.columns = data.columns.astype(int)
        return data

    def generate_network(self, network_type=None, N=None, connection_prob=None, avg_degree=None, clustering_coeff=None):
        """
        Generates a network based on the specified type and parameters.
        If no network type is provided, the network is generated from the provided data.

        Parameters:
        ----------
        network_type : str, optional
            The type of network to generate ('erdos_renyi', 'barabasi_albert', or 'watts_strogatz').
        N : int, optional
            The number of nodes in the network.
        connection_prob : float, optional
            The probability of connections for Erdős-Rényi and Watts-Strogatz networks.
        avg_degree : int, optional
            The average degree for Barabási-Albert and Watts-Strogatz networks.
        clustering_coeff : float, optional
            Desired clustering coefficient for Watts-Strogatz networks.

        Returns:
        -------
        networkx.Graph
            The generated network based on the provided parameters.
        """
        if network_type == 'erdos_renyi':
            if connection_prob is None:
                raise ValueError(
                    "Connection probability is required for Erdős-Rényi network.")
            G = nx.erdos_renyi_graph(N, connection_prob)

        elif network_type == 'barabasi_albert':
            if avg_degree is None or avg_degree < 1:
                raise ValueError(
                    "Average degree must be >= 1 for Barabási-Albert network.")
            m = avg_degree // 2
            G = nx.barabasi_albert_graph(N, m)

        elif network_type == 'watts_strogatz':
            if avg_degree is None or connection_prob is None:
                raise ValueError(
                    "Both average degree and connection probability are required for Watts-Strogatz network.")
            G = nx.watts_strogatz_graph(N, avg_degree, connection_prob)

            if clustering_coeff is not None:
                current_cc = nx.average_clustering(G)
                if not np.isclose(current_cc, clustering_coeff, atol=0.01):
                    print(
                        f"Warning: Desired clustering coefficient not achieved. Current: {current_cc}, Target: {clustering_coeff}")

        elif network_type == None:
            data = self.init_data()
            G = nx.from_pandas_adjacency(data)

        else:
            raise ValueError("Invalid network type.")

        return G

    def create_model(self, beta, gamma, num_infected, infect_hubs_only=False):
        """
        Creates a disease transmission model based on the network.

        Parameters:
        ----------
        beta : float
            Transmission rate between individuals.
        gamma : float
            Recovery rate of infected individuals.
        num_infected : int
            Initial number of infected individuals in the network.
        infect_hubs_only : bool, optional
            If True, only the most connected nodes are infected.

        Returns:
        -------
        gc.CompositeModel
            The model configured for disease transmission and recovery.
        """
        model = gc.CompositeModel(self.network)
        model.add_status("Susceptible")
        model.add_status("Infected")
        model.add_status("Removed")

        c_beta = cpm.NodeStochastic(
            beta, triggering_status="Infected")  # Susceptible - Infected
        c_gamma = cpm.NodeStochastic(gamma)  # Infected - Removed
        c_tested = cpm.NodeCategoricalAttribute(
            "Vaccinated", 'True', probability=1)

        model.add_rule("Susceptible", "Infected", c_beta)
        model.add_rule("Infected", "Removed", c_gamma)
        model.add_rule("Susceptible", "Removed",
                       c_tested)  # TODO change to removed

        config = mc.Configuration()
        frac_infected = num_infected / len(self.network)

        if infect_hubs_only:
            degrees = sorted(self.network.degree, key=lambda x: x[1], reverse=True)
            hubs = [node for node, degree in degrees[:num_infected]]
            for hub in hubs:
                config.add_node_status(hub, "Infected")
        else:
            frac_infected = num_infected / len(self.network)
            config.add_model_parameter('fraction_infected', frac_infected)
            model.set_initial_status(config)

        return model

    def run_simulation(self, t, beta, gamma, num_infected):
        """
        Runs the disease transmission simulation over a specified number of time steps.

        Parameters:
        ----------
        t : int
            Number of time steps for the simulation.
        beta : float
            Transmission rate.
        gamma : float
            Recovery rate.
        num_infected : int
            Initial number of infected individuals.

        Returns:
        -------
        tuple
            A tuple containing the final counts of Susceptible (S), Infected (I),
            Removed (R) individuals, and the number of vaccinated individuals.
            Format: (S, I, R, vacc_count)
        """
        if beta < 0 or gamma < 0:
            raise ValueError("Transmission and recovery rates must be >= 0.")
        if num_infected < 0:
            raise ValueError("Number of infected individuals must be >= 0.")

        history = [None] * t
        model = self.create_model(beta, gamma, num_infected)
        self.vaccinated_network = self.network.copy()

        for t_delta in range(t):
            tested = None
            if t_delta % self.tests_inter == 0:
                tested = self.testing_program()
            if t_delta % self.vacc_inter == 0 and tested is not None:
                self.vaccinate_nodes(tested)

            history[t_delta] = model.iteration()

        trends = model.build_trends(history)
        S = trends[0]['trends']['node_count'][0]
        I = trends[0]['trends']['node_count'][1]
        R = trends[0]['trends']['node_count'][2]
        vacc_count = self.vacc_count

        return S, I, R, vacc_count

    def vaccination_setup(self, vacc_total, vacc_num, vacc_inter, tests_total, tests_num, tests_inter, test_accuracy, strategy):
        """
        Sets up the vaccination and testing strategies for the simulation.
        If no vaccinations are allowed, no strategy is applied.

        Parameters:
        ----------
        vacc_total : int
            Total number of vaccinations allowed.
        vacc_num : int
            Number of individuals vaccinated per round.
        vacc_inter : int
            Interval between vaccination rounds.
        tests_total : int
            Total number of tests allowed.
        tests_num : int
            Number of individuals tested per round.
        tests_inter : int
            Interval between testing rounds.
        test_accuracy : float
            Accuracy of the testing method (0 to 1).
        strategy : str
            The vaccination strategy (e.g., 'random', 'degree', 'betweenness', 'closeness').
        """
        if vacc_total < 0 or vacc_num < 0 or vacc_inter < 0:
            raise ValueError("Vaccination parameters must be >= 0.")
        if tests_total < 0 or tests_num < 0 or tests_inter < 0:
            raise ValueError("Testing parameters must be >= 0.")
        if strategy not in ['random', 'degree', 'betweenness', 'closeness']:
            raise ValueError(
                "Invalid vaccination strategy. Choose from 'random', 'degree', 'betweenness', 'closeness'.")

        self.strategy = strategy
        self.vacc_total = vacc_total
        self.vacc_num = vacc_num
        self.vacc_inter = vacc_inter
        self.tests_total = tests_total
        self.tests_num = tests_num
        self.tests_inter = tests_inter
        self.test_accuracy = test_accuracy

        self.vacc_count = 0

    def test_nodes(self, nodes):
        """
        Helper function for testing_program().
        Tests a subset of nodes in the network.

        Parameters:
        ----------
        nodes : list
            List of nodes to be tested for infection.
        """
        for node in nodes:
            if random.random() < self.test_accuracy:
                self.network.nodes[node]['Tested'] = True

    def vaccinate_nodes(self, nodes):
        """
        Helper function for run_simulation().
        Vaccinates a subset of nodes based on the current strategy and testing results.

        Parameters:
        ----------
        nodes : list
            List of nodes to be vaccinated.
        """
        if self.vacc_count >= self.vacc_total:
            return

        vaccinated = nodes[:self.vacc_num]
        for node in vaccinated:
            if self.network.nodes[node]['Tested'] == True:
                self.network.nodes[node]['Vaccinated'] = 'True'
                self.vacc_count += 1

    def test_pool(self, nodes, tests_num):
        """
        Selects nodes for testing based on the current strategy.

        Parameters:
        ----------
        nodes : list
            List of available nodes for testing.
        tests_num : int
            Number of nodes to test.

        Returns:
        -------
        list
            A list of nodes selected for testing.
        """
        if self.strategy == 'random':
            tested = random.sample(nodes, tests_num)

        elif self.strategy == 'degree':
            degrees = dict(self.network.degree(nodes))
            tested = sorted(degrees, key=degrees.get, reverse=True)[:tests_num]

        elif self.strategy == 'closeness':
            closeness = nx.closeness_centrality(self.vaccinated_network)
            tested = sorted(closeness, key=closeness.get,
                            reverse=True)[:tests_num]

        elif self.strategy == 'betweenness':
            betweenness = nx.betweenness_centrality(self.vaccinated_network)
            tested = sorted(betweenness, key=betweenness.get,
                            reverse=True)[:tests_num]

        return tested

    def testing_program(self):
        """
        Conducts the testing program and updates the network with the results.

        Returns:
        -------
        list
            A list of nodes that were tested.
        """
        untested_nodes = [n for n, data in self.network.nodes(
            data=True) if not data.get('Tested', True)]

        tests_num = min(self.tests_num, self.tests_total, len(untested_nodes))
        if tests_num == 0:
            return

        tested = self.test_pool(untested_nodes, tests_num)
        self.test_nodes(tested)
        self.tests_total -= tests_num

        self.vaccinated_network.remove_nodes_from(tested)
        return tested

    def get_network_statistics(self):
        """
        Retrieves various statistics about the current network.

        Returns:
        -------
        tuple
            A tuple containing network statistics: number of nodes, betweenness centrality,
            closeness centrality, degree distribution, average degree,
            eigenvector centrality, and clustering coefficient.
            Format: (N, betweenness, closeness, degrees, average_degree, eigenvalues, clustering)
        """
        N = len(self.network)
        betweenness = nx.betweenness_centrality(self.network)
        closeness = nx.closeness_centrality(self.network)
        degrees = dict(self.network.degree())
        average_degree = np.mean(list(degrees.values()))
        eigenvalues = nx.eigenvector_centrality(self.network)
        clustering = nx.clustering(self.network)

        return N, betweenness, closeness, degrees, average_degree, eigenvalues, clustering

    def plot_network_statistic(self):
        """
        Plots various centrality and network statistics for the current network.
        """
        statistics = self.get_network_statistics()
        N = statistics[0]
        betweenness = statistics[1]
        closeness = statistics[2]
        degrees = statistics[3]
        average_degree = statistics[4]
        eigenvalues = statistics[5]
        clustering = statistics[6]

        centralities = pd.DataFrame({
            'Betweenness Centrality': betweenness,
            'Closeness Centrality': closeness,
            'Eigenvector Centrality': eigenvalues,
            'Clustering': clustering,
            'Degree': degrees
        })

        fig, axs = plt.subplots(1, 5)
        axs = axs.flatten()

        centrality_names = centralities.columns
        for i, metric in enumerate(centrality_names):
            centrality = centralities[metric].sort_values(ascending=True)
            len_cent = len(centrality)
            axs[i].scatter(centrality.values, np.arange(len_cent), marker='o')
            axs[i].set_yticks([])
            axs[i].set_title(metric)
            axs[i].grid()

        plt.tight_layout()
        plt.show()

    def plot_degree_distribution(self):
        """
        Plots the degree distribution of the current network in both linear and log-log scale.
        """
        degrees = dict(self.network.degree())
        degree_values = list(degrees.values())

        unique_degrees, degree_counts = np.unique(
            degree_values, return_counts=True)
        total_nodes = len(degree_values)
        degree_fractions = degree_counts / total_nodes

        fig, axs = plt.subplots(1, 2)

        axs[0].scatter(unique_degrees, degree_fractions)
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_title("Degree Distribution as loglog")
        axs[0].set_xlabel("Degree")
        axs[0].set_ylabel("Fraction of Nodes")

        axs[1].bar(unique_degrees, degree_counts, width=0.8, color='skyblue')
        axs[1].set_title("Degree Distribution")
        axs[1].set_xlabel("Degree")
        axs[1].set_xlim([min(unique_degrees) - 1, max(unique_degrees) + 1])
        axs[1].set_ylabel("Number of Nodes")

        plt.show()

    def plot_network(self):
        """
        Plots the current network with node sizes proportional to degree.

        Parameters:
        ----------
        connection_prob : float, optional
            Connection probability of the network (for plotting annotation).
        avg_degree : int, optional
            Average degree of the network (for plotting annotation).
        clustering_coeff : float, optional
            Clustering coefficient of the network (for plotting annotation).
        """
        pos = nx.spring_layout(self.network, k=1.5, weight='weight', scale=3.0)
        degrees = dict(self.network.degree())
        node_sizes = [degrees[node] * 10 for node in self.network.nodes()]
        nx.draw(self.network, pos, node_size=node_sizes, width=0.1)

        "
        plt.set_title(f"{self.network_type} Network")
        plt.show()


def simulate_spread():
    network = network = Network(
        network_type='erdos_renyi', N=1000, connection_prob=0.01)
    network.plot_network()


if __name__ == "__main__":
    simulate_spread()
    # epochs = 10
    # history_I = [None] * epochs
    # history_V = [None] * epochs
    # intervall = 4

    # time_start = time.time()
    # for strategy in ['random', 'degree', 'betweenness', 'closeness']:
    #     network = Network()
    #     model = network.create_model(beta=0.3, gamma=0.1, num_infected=10)
    #     network.vaccination_setup(vacc_total=0, vacc_num=10, vacc_inter=intervall, tests_total=200,
    #                               tests_num=10, tests_inter=intervall, test_accuracy=0.9, strategy=strategy)
    #     for _ in range(epochs):
    #         S, I, R, vacc_count = network.run_simulation(
    #             t=100, beta=0.3, gamma=0.1, num_infected=10)
    #         history_I[_] = I
    #         history_V[_] = vacc_count

    #     N, _, _, _, _, _ = network.get_network_statistics()
    #     print(f"Strategy: {strategy}")
    #     print(
    #         f"Number of infected: {np.mean([max(I) for I in history_I]):.0f}/{N}")
    #     print(
    #         f"Number of vacinated: {np.mean([V for V in history_V]):.0f}/{N}")
    #     print("-----------------------------")

    # time_end = time.time()
    # print(f"Time elapsed: {time_end - time_start:.2f} seconds")
    # plt.plot(S, label="Susceptible")
    # plt.plot(I, label="Infected")
    # plt.plot(R, label="Removed")
    # plt.plot(V, label="Vaccinated")
    # plt.legend()
    # plt.grid()
    # plt.show()

    temp = Network(network_type='erdos_renyi', N=1000, connection_prob=0.01)
    temp1 = Network(network_type='watts_strogatz', N=1000, connection_prob=0.01, avg_degree=4)