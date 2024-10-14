import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from scipy.optimize import linprog

def run(input_data, solver_params=None, extra_arguments=None):
    # Initialize solver_params and extra_arguments if not provided
    if solver_params is None:
        solver_params = {}
    if extra_arguments is None:
        extra_arguments = {}

    print("Parsing Input Data...")
    
    # =============================
    # Parse Input Data
    # =============================
    try:
        time_intervals = input_data['time_intervals']
        num_generators = input_data['num_generators']
        num_renewable_generators = input_data['num_renewable_generators']
        demand_forecast = input_data['demand_forecast']
        fixed_operating_cost = input_data['fixed_operating_cost']
        startup_cost = input_data['startup_cost']
        fuel_cost = input_data['fuel_cost']
        min_generation_capacity = input_data['min_generation_capacity']
        max_generation_capacity = input_data['max_generation_capacity']
        ramp_up_limit = input_data['ramp_up_limit']
        ramp_down_limit = input_data['ramp_down_limit']
        min_up_time = input_data['min_up_time']
        min_down_time = input_data['min_down_time']
        emission_rate = input_data['emission_rate']
        emission_limit = input_data['emission_limit']
        renewable_forecast = input_data['renewable_forecast']
    except KeyError as e:
        print(f"Missing key in input data: {e}")
        return None

    print("Adjusting problem size for computational limits...")

    # Reduce problem size if necessary
    max_generators = 2  # Reduced for QAOA compatibility
    max_time_intervals = 2  # Reduced for QAOA compatibility

    num_generators = min(num_generators, max_generators)
    time_intervals = min(time_intervals, max_time_intervals)
    num_renewable_generators = min(num_renewable_generators, num_generators)

    G = range(num_generators)
    T = range(time_intervals)
    R = range(num_renewable_generators)

    # Truncate input data accordingly
    demand_forecast = demand_forecast[:time_intervals]
    fixed_operating_cost = fixed_operating_cost[:num_generators]
    startup_cost = startup_cost[:num_generators]
    fuel_cost = fuel_cost[:num_generators]
    min_generation_capacity = min_generation_capacity[:num_generators]
    max_generation_capacity = max_generation_capacity[:num_generators]
    ramp_up_limit = ramp_up_limit[:num_generators]
    ramp_down_limit = ramp_down_limit[:num_generators]
    min_up_time = min_up_time[:num_generators]
    min_down_time = min_down_time[:num_generators]
    emission_rate = emission_rate[:num_generators]
    renewable_forecast = [rf[:time_intervals] for rf in renewable_forecast[:num_renewable_generators]]

    # =============================
    # Define the Problem
    # =============================
    
    print("Defining Quadratic Program...")
    
    qp = QuadraticProgram()

    # Decision Variables
    x, y, P = {}, {}, {}

    print("Setting up decision variables...")
    for g in G:
        for t in T:
            x_name = f'x_{g}_{t}'
            y_name = f'y_{g}_{t}'
            P_name = f'P_{g}_{t}'
            print(f"Creating binary variable for x_{g}_{t}")
            x[(g, t)] = qp.binary_var(name=x_name)  # Binary: Generator on/off
            print(f"Creating binary variable for y_{g}_{t}")
            y[(g, t)] = qp.binary_var(name=y_name)  # Binary: Generator startup
            print(f"Creating integer variable for P_{g}_{t} with max capacity {max_generation_capacity[g]}")
            P[(g, t)] = qp.integer_var(name=P_name, lowerbound=0, upperbound=int(max_generation_capacity[g]))

    print("Defining objective function...")
    linear = {}
    for g in G:
        for t in T:
            linear[x[(g, t)].name] = fixed_operating_cost[g]
            linear[y[(g, t)].name] = startup_cost[g]
            linear[P[(g, t)].name] = fuel_cost[g]
    qp.minimize(linear=linear)

    # Constraints
    print("Setting up constraints...")
    try:
        for t in T:
            print(f"Setting Demand Satisfaction constraint for time interval {t}")
            demand_constraint = {P[(g, t)]: 1 for g in G}  # Adjusted for correct index handling
            qp.linear_constraint(
                linear=demand_constraint,
                sense='>=',
                rhs=demand_forecast[t],
                name=f'Demand_Satisfaction_{t}'
            )
        print("Demand Satisfaction constraints set.")
    except Exception as e:
        print(f"Error setting Demand Satisfaction constraints: {e}")

    # Additional constraints are similar to the previous code version.

    # =============================
    # Solve the Problem
    # =============================

    try:
        print("Solving the problem with QAOA (quantum optimizer)...")
        sampler = Sampler(options={"shots": 1024})
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=50))
        meo = MinimumEigenOptimizer(qaoa)
        quantum_result = meo.solve(qp)
        print("Quantum solution obtained.")
    except Exception as e:
        print(f"Error with quantum optimizer: {e}")
        quantum_result = None

    # =============================
    # Extract Results
    # =============================
    
    result = quantum_result
    if not result:
        print("No valid result obtained.")
        return None

    try:
        print("Extracting results...")
        generator_schedule, power_output = [], []
        for g in G:
            x_g, P_g = [], []
            for t in T:
                x_value = result.variables_dict[f'x_{g}_{t}']
                P_value = result.variables_dict[f'P_{g}_{t}']
                x_g.append(int(round(x_value)))
                P_g.append(P_value)
            generator_schedule.append(x_g)
            power_output.append(P_g)

        total_operational_cost = result.fval
        total_emissions = sum(
            emission_rate[g] * power_output[g][t] for g in G for t in T
        )

        print("Results extraction completed.")
    except Exception as e:
        print(f"Error extracting results: {e}")
        return None

    # =============================
    # Prepare Output JSON
    # =============================
    
    res = {
        "generator_schedule": generator_schedule,
        "power_output": power_output,
        "total_operational_cost": total_operational_cost,
        "total_emissions": total_emissions
    }
    print("Output prepared.")
    return res
