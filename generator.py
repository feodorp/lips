import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from pyomo.environ import *

def generate_random_dag(I, one_out_degree_percentage, initial_nodes_percentage):
    """
    Generates a Directed Acyclic Graph (DAG) with specified properties.

    Args:
        I (list): A list of nodes (operations) to include in the graph.
        one_out_degree_percentage (float): The percentage of non-sink nodes
                                           that should have an out-degree of exactly 1.
        initial_nodes_percentage (float): The percentage of nodes that should be
                                          initial nodes (in-degree of 0).

    Returns:
        networkx.DiGraph: The generated DAG.

    Raises:
        ValueError: If input parameters are invalid or if the graph cannot be
                    generated according to the constraints.
    """
    num_nodes = len(I)
    if num_nodes < 2:
        raise ValueError("Too few nodes to generate a meaningful DAG.")
    if not 0 <= one_out_degree_percentage <= 1 or not 0 <= initial_nodes_percentage <= 1:
        raise ValueError("Percentages must be between 0 and 1.")

    G = nx.DiGraph()
    nodes = list(I)
    G.add_nodes_from(nodes)

    # Step 1: Determine initial nodes and sink node
    initial_nodes_count = max(1, int(initial_nodes_percentage * num_nodes))

    # Ensure initial_nodes_count does not exceed num_nodes - 1 (to leave room for sink)
    if initial_nodes_count >= num_nodes:
        initial_nodes_count = num_nodes - 1
        if initial_nodes_count < 1:
            raise ValueError("Cannot select initial nodes and sink with too few nodes.")

    initial_nodes = random.sample(nodes, initial_nodes_count)

    available_for_sink = [n for n in nodes if n not in initial_nodes]
    if not available_for_sink:
        raise ValueError("No nodes available for sink after selecting initial nodes.")

    sink_node = random.choice(available_for_sink)

    # Ensure sink_node is not in initial_nodes (a safeguard, should be handled by available_for_sink)
    if sink_node in initial_nodes:
        initial_nodes.remove(sink_node)
        initial_nodes_count = len(initial_nodes) # Adjust count if sink was accidentally picked as initial

    remaining_nodes = [n for n in nodes if n not in initial_nodes and n != sink_node]

    # Step 2: Build a main path from one initial node to the sink
    if not initial_nodes:
        raise ValueError("No initial nodes available for main path.")

    main_initial = random.choice(initial_nodes)

    # Adjust main_path_length to be more robust and relative to num_nodes
    main_path_length = random.randint(2, min(num_nodes, max(3, int(num_nodes * 0.5))))

    num_intermediate = max(0, main_path_length - 2)

    # Ensure we don't try to sample more intermediate nodes than available
    if num_intermediate > len(remaining_nodes):
        num_intermediate = len(remaining_nodes)
        main_path_length = num_intermediate + 2

    intermediate_nodes = random.sample(remaining_nodes, num_intermediate)
    main_path = [main_initial] + intermediate_nodes + [sink_node]

    for i in range(len(main_path) - 1):
        u, v = main_path[i], main_path[i + 1]
        if not G.has_edge(u, v): # Avoid duplicate edges
            G.add_edge(u, v)

    # Update remaining_nodes
    remaining_nodes = [n for n in remaining_nodes if n not in intermediate_nodes]

    # Step 3: Add branches from other initial nodes
    for initial in initial_nodes:
        if initial != main_initial and G.out_degree(initial) == 0: # Only if not already connected
            branch_length = random.randint(1, 3)
            branch_nodes = [initial]

            num_branch_nodes = min(branch_length - 1, len(remaining_nodes))
            if num_branch_nodes > 0:
                branch_intermediate = random.sample(remaining_nodes, num_branch_nodes)
                branch_nodes.extend(branch_intermediate)
                remaining_nodes = [n for n in remaining_nodes if n not in branch_intermediate]

            for i in range(len(branch_nodes) - 1):
                u, v = branch_nodes[i], branch_nodes[i + 1]
                # Add edge only if it doesn't create a cycle and respects degree limits
                if not G.has_edge(u, v) and not nx.has_path(G, v, u) and G.out_degree(u) < 3 and G.in_degree(v) < 2:
                    G.add_edge(u, v)

            # Connect the end of the branch to the main path or sink
            if branch_nodes:
                last_branch_node = branch_nodes[-1]
                possible_targets = [n for n in main_path[1:] if n != last_branch_node and not nx.has_path(G, n, last_branch_node) and G.out_degree(last_branch_node) < 3 and G.in_degree(n) < 2]
                if possible_targets:
                    target = random.choice(possible_targets)
                    if not G.has_edge(last_branch_node, target):
                        G.add_edge(last_branch_node, target)
                elif not G.has_edge(last_branch_node, sink_node) and not nx.has_path(G, sink_node, last_branch_node) and G.out_degree(last_branch_node) < 3 and G.in_degree(sink_node) < 2:
                    G.add_edge(last_branch_node, sink_node)

    # Step 4: Incorporate remaining nodes and add more random edges to increase density
    random.shuffle(remaining_nodes)
    for node in remaining_nodes:
        # Try to connect this node from an existing node if it has no incoming edges
        if G.in_degree(node) == 0:
            possible_sources = [n for n in G.nodes if n != node and not nx.has_path(G, node, n) and G.out_degree(n) < 3]
            if possible_sources:
                source = random.choice(possible_sources)
                G.add_edge(source, node)

        # Try to connect this node to an existing node if it has no outgoing edges (and is not the sink)
        if G.out_degree(node) == 0 and node != sink_node:
            possible_targets = [n for n in G.nodes if n != node and not nx.has_path(G, n, node) and G.in_degree(n) < 2]
            if possible_targets:
                target = random.choice(possible_targets)
                G.add_edge(node, target)
            else: # As a last resort, connect to sink if no other option
                if not G.has_edge(node, sink_node) and not nx.has_path(G, sink_node, node) and G.out_degree(node) < 3 and G.in_degree(sink_node) < 2:
                    G.add_edge(node, sink_node)

    # Add more random edges to increase graph density, respecting constraints
    target_edges_count = G.number_of_edges() + int(num_nodes * 1.5) # Aim for more edges
    attempts_limit = num_nodes * 50 # Limit attempts to avoid infinite loops
    attempts = 0

    while G.number_of_edges() < target_edges_count and attempts < attempts_limit:
        u, v = random.sample(nodes, 2)
        attempts += 1
        if u == v: continue

        if not G.has_edge(u, v) and not nx.has_path(G, v, u) and \
           G.out_degree(u) < 3 and G.in_degree(v) < 2:
            G.add_edge(u, v)
            attempts = 0 # Reset attempts after successful edge addition

    # Step 5: Enforce out-degree = 1 for specified percentage of non-sink nodes
    non_sink_nodes = [n for n in G.nodes if n != sink_node]
    num_one_out = int(one_out_degree_percentage * len(non_sink_nodes))

    one_out_nodes = random.sample(non_sink_nodes, min(num_one_out, len(non_sink_nodes)))

    for node in one_out_nodes:
        # If out-degree is > 1, remove extra edges
        successors = list(G.successors(node))
        if len(successors) > 1:
            keep_succ = random.choice(successors)
            for succ in successors:
                if succ != keep_succ:
                    G.remove_edge(node, succ)
        # If out-degree is 0, add one edge
        elif G.out_degree(node) == 0:
            possible_targets = [n for n in G.nodes if n != node and not nx.has_path(G, n, node) and G.in_degree(n) < 2]
            if possible_targets:
                G.add_edge(node, random.choice(possible_targets))
            # Else: This node might remain a sink, which will be caught by final validation.

    # Step 6: Ensure no outgoing edges from the designated sink node
    G.remove_edges_from(list(G.out_edges(sink_node)))

    # Step 7: Final validation and attempts to fix common issues
    if sink_node not in G.nodes:
        raise ValueError(f"Sink node {sink_node} not in graph.")

    # Ensure exactly one sink node
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    if len(sinks) != 1 or sinks[0] != sink_node:
        # Attempt to fix: connect other sinks to the designated sink_node
        for s in sinks:
            if s != sink_node:
                if not G.has_edge(s, sink_node) and not nx.has_path(G, sink_node, s) and G.out_degree(s) < 3 and G.in_degree(sink_node) < 2:
                    G.add_edge(s, sink_node)
        sinks = [n for n in G.nodes if G.out_degree(n) == 0] # Re-check
        if len(sinks) != 1 or sinks[0] != sink_node:
            raise ValueError(f"Failed to ensure exactly one sink node ({sink_node}). Found: {sinks}")

    if not nx.is_directed_acyclic_graph(G):
        # Attempt to break cycles (heuristic: remove one edge from each simple cycle)
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                if len(cycle) > 1:
                    u, v = cycle[0], cycle[1] # Remove the first edge in the cycle
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
            if not nx.is_directed_acyclic_graph(G):
                raise ValueError("Graph is still not a DAG after cycle removal attempt.")
        except nx.NetworkXNoCycle:
            pass
        except Exception as e:
            raise ValueError(f"Error during cycle breaking: {e}")

    if not nx.is_weakly_connected(G):
        # Attempt to connect components
        components = list(nx.weakly_connected_components(G))
        if len(components) > 1:
            main_component = None
            for comp in components:
                if sink_node in comp:
                    main_component = comp
                    break
            if main_component is None: # If sink is isolated, pick largest component
                main_component = max(components, key=len)

            for comp in components:
                if comp == main_component:
                    continue
                connected = False
                for u in comp:
                    for v in main_component:
                        if not G.has_edge(u, v) and not nx.has_path(G, v, u) and \
                           G.out_degree(u) < 3 and G.in_degree(v) < 2:
                            G.add_edge(u, v)
                            connected = True
                            break
                    if connected:
                        break
                if not connected: # If direct connection failed, try connecting to sink from a node in comp
                    for u in comp:
                        if not G.has_edge(u, sink_node) and not nx.has_path(G, sink_node, u) and G.out_degree(u) < 3 and G.in_degree(sink_node) < 2:
                            G.add_edge(u, sink_node)
                            connected = True
                            break
                if not connected:
                    raise ValueError("Failed to make graph weakly connected.")

    # Final degree checks (after all additions/removals)
    if any(G.in_degree(n) > 2 for n in G.nodes):
        raise ValueError("In-degree > 2 detected after generation.")
    if any(G.out_degree(n) > 3 for n in G.nodes):
        raise ValueError("Out-degree > 3 detected after generation.")

    current_initial = [n for n in G.nodes if G.in_degree(n) == 0]
    if len(current_initial) != initial_nodes_count:
        # Attempt to fix: if too many initial nodes, add incoming edges to some of them
        while len(current_initial) > initial_nodes_count:
            non_intended_initial = [n for n in current_initial if n not in initial_nodes]
            if not non_intended_initial:
                break
            node_to_fix = random.choice(non_intended_initial)

            possible_sources = [n for n in G.nodes if n != node_to_fix and not nx.has_path(G, node_to_fix, n) and G.out_degree(n) < 3]
            if possible_sources:
                source = random.choice(possible_sources)
                if G.in_degree(node_to_fix) < 2: # Ensure target doesn't exceed in-degree
                    G.add_edge(source, node_to_fix)
                    current_initial = [n for n in G.nodes if G.in_degree(n) == 0]
            else:
                break

        if len(current_initial) != initial_nodes_count:
            raise ValueError(f"Expected {initial_nodes_count} initial nodes, found {len(current_initial)} after attempts to fix.")

    return G

def generate_fdjssp_instance(n_ops, n_machines, seed=None):
    """
    Generates a Flexible Dynamic Job Shop Scheduling Problem (FDJSSP) instance
    with a complex precedence graph using networkx and defines it as a Pyomo model.

    Args:
        n_ops (int): Number of operations.
        n_machines (int): Number of machines.
        seed (int, optional): Seed for random number generators for reproducibility. Defaults to None.

    Returns:
        pyomo.environ.ConcreteModel: The Pyomo model representing the FDJSSP instance.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Generate sets
    I = list(range(1, n_ops + 1))  # Operations
    J = list(range(1, n_machines + 1))  # Machines

    # Create a directed acyclic graph (DAG) for precedence constraints
    # The DAG generation is complex and might be sensitive to parameters.
    G = generate_random_dag(I, 0.95, 0.15)

    # Extract precedence constraints (B)
    B = list(G.edges)

    # Plotting the generated DAG (optional, for visualization)
    fig = plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=seed) # Use a layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_size=10, font_weight='bold', arrowsize=20)
    plt.title("Generated Precedence Graph (DAG)")
    plt.show()

    # Generate parameters
    M = 10000  # Big-M constant for linearization
    D_o = {i: np.random.randint(50, 65) for i in I}  # Processing times
    D_c = {(i1, i2): np.random.randint(30, 50) for (i1, i2) in B}  # Communication times for precedence

    # D_cj is communication time between different machines
    D_cj_init = {(j1, j2): np.random.randint(100, 120) for j1 in J for j2 in J if j1 != j2}

    # Create Pyomo model
    model = ConcreteModel(name="FDJSSP")

    # Define sets
    model.I = Set(initialize=I, doc="Operations")
    model.J = Set(initialize=J, doc="Machines")
    model.JxJ = Set(dimen=2, initialize=[(j1,j2) for j1 in J for j2 in J], doc="Machines x Machines")
    model.B = Set(dimen=2, initialize=B, doc="Precedence constraints")
    model.C = Set(dimen=2, initialize=[(j1, j2) for j1 in J for j2 in J if j1 != j2], doc="Communication channels (different machines)")

    # Define parameters
    model.D_o = Param(model.I, initialize=D_o, doc="Processing time for operation i")
    model.D_c = Param(model.B, initialize=D_c, doc="Base communication time for precedence (i1, i2)")
    # Corrected: Added the missing D_cj parameter
    model.D_cj = Param(model.C, initialize=D_cj_init, doc="Additional communication time between different machines (j1, j2)")


    # Create Pyomo model

    model = ConcreteModel(name="FDJSSP")


    # Define sets

    model.I = Set(initialize=I, doc="Operations")

    model.J = Set(initialize=J, doc="Machines")

    model.JxJ = Set(dimen=2, initialize=[(j1,j2) for j1 in J for j2 in J], doc="Machines x Machines")

    model.B = Set(dimen=2, initialize=B, doc="Precedence constraints")

    model.C = Set(dimen=2, initialize=[(j1, j2) for j1 in J for j2 in J if j1 != j2], doc="Communication channels")


    # Define parameters

    model.D_o = Param(model.I, initialize=D_o, doc="Processing time")

    model.D_c = Param(model.B, initialize=D_c, doc="Communication time")


    # Define variables

    model.s = Var(model.I, within=NonNegativeReals, doc="Start time")

    model.e = Var(model.I, within=NonNegativeReals, doc="End time")

    model.x = Var(model.I, model.J, within=Binary, doc="Machine assignment")

    model.c = Var(model.B, within=NonNegativeReals, doc="Comm start time")

    model.d = Var(model.B, within=NonNegativeReals, doc="Comm end time")

    model.y = Var(model.I, model.I, within=Binary, doc="Operation order")

    model.z = Var(model.B, model.JxJ, within=Binary, doc="Channel assignment")

    model.w = Var(model.B, model.B, within=Binary, doc="Comm order")

    model.z_var = Var(within=NonNegativeReals, doc="Makespan")


    # 1) Objective: Minimize makespan

    model.obj = Objective(expr=model.z_var, sense=minimize)


    # 2) Makespan constraint

    def makespan_rule(m, i):

        return m.e[i] <= m.z_var

    model.makespan_con = Constraint(model.I, rule=makespan_rule)


    # 3) Operation duration constraint

    def proc_time_rule(m, i):

        return m.e[i] == m.s[i] + m.D_o[i]

    model.proc_time = Constraint(model.I, rule=proc_time_rule)


    # 4) Slack time between operations

    def slack_time_rule(m, i1, i2):

        if (i1, i2) in m.B:

            return m.s[i2] - m.e[i1] >= 0

        return Constraint.Skip

    model.slack_time = Constraint(model.B, rule=slack_time_rule)


    # 6) Machine assignment

    def one_machine_rule(m, i):

        return sum(m.x[i, j] for j in m.J) == 1

    model.one_machine = Constraint(model.I, rule=one_machine_rule)


    # 7) No overlap on machines (part 1)

    def no_overlap_rule1(m, i1, i2, j):

        if i1 != i2:

            return m.s[i2] >= m.e[i1] - M * (3 - m.x[i1, j] - m.x[i2, j] - m.y[i1, i2])

        return Constraint.Skip

    model.no_overlap1 = Constraint(model.I, model.I, model.J, rule=no_overlap_rule1)


    # 8) No overlap on machines (part 2)

    def no_overlap_rule2(m, i1, i2):

        if i1 != i2:

            return m.y[i1, i2] + m.y[i2, i1] == 1

        return Constraint.Skip

    model.no_overlap2 = Constraint(model.I, model.I, rule=no_overlap_rule2)


    # 9) Communication between assigned machines

    def communication_rule1(m, i1, i2, j1, j2):

        if (i1, i2) in m.B and (j1, j2) in m.C:

            return m.z[i1, i2, j1, j2] == m.x[i1, j1] * m.x[i2, j2]

        return Constraint.Skip

    model.communication1 = Constraint(model.B, model.C, rule=communication_rule1)


    # 10) No communication between machines without comm. channel

    def communication_rule2(m, i1, i2, j1, j2):

        if (i1, i2) in m.B and (j1, j2) not in m.C:

            return m.z[i1, i2, j1, j2] <= 0

        return Constraint.Skip

    model.communication2 = Constraint(model.B, model.JxJ, rule=communication_rule2)


    # 11) Communication duration

    def comm_duration_rule(m, i1, i2):

        if (i1, i2) in m.B:

            return m.d[i1, i2] >= m.c[i1, i2] + m.D_c[i1, i2]

        return Constraint.Skip

    model.comm_duration = Constraint(model.B, rule=comm_duration_rule)


    # 12) Precedence with communication

    def precedence_rule(m, i1, i2):

        if (i1, i2) in m.B:

            return m.s[i2] >= m.d[i1, i2]

        return Constraint.Skip

    model.precedence = Constraint(model.I, model.I, rule=precedence_rule)


    # 13) Communication start time

    def comm_start_rule(m, i1, i2):

        if (i1, i2) in m.B:

            return m.c[i1, i2] >= m.e[i1]

        return Constraint.Skip

    model.comm_start = Constraint(model.B, rule=comm_start_rule)


    # 14) No overlap on channels (part 1)

    def no_comm_overlap_rule1(m, i1, i2, i3, i4, j1, j2):

        if ((i1, i2) in m.B and (i3, i4) in m.B and (i1, i2) != (i3, i4)) and (j1, j2) in m.C:

            return m.c[i3, i4] >= m.d[i1, i2] - M * (3 - m.z[i1, i2, j1, j2] - m.z[i3, i4, j1, j2] - m.w[i1, i2, i3, i4])

        return Constraint.Skip

    model.no_comm_overlap1 = Constraint(model.B, model.B, model.C, rule=no_comm_overlap_rule1)


    # 15) No overlap on channels (part 2)

    def no_comm_overlap_rule2(m, i1, i2, i3, i4):

        if (i1, i2) in m.B and (i3, i4) in m.B and (i1, i2) != (i3, i4):

            return  m.w[i1, i2, i3, i4] + m.w[i3, i4, i1, i2] == 1

        return Constraint.Skip

    model.no_comm_overlap2 = Constraint(model.B, model.B, rule=no_comm_overlap_rule2)


    # Export to LP format

    model.write("fdjssp_instance2.lp", io_options={"symbolic_solver_labels": True})

    print(f"Generated FDJSSP instance with {n_ops} operations and {n_machines} machines. Exported to 'fdjssp_instance.lp'.")


    return model


if __name__ == "__main__":

    generate_fdjssp_instance(60, 2, seed=48)
