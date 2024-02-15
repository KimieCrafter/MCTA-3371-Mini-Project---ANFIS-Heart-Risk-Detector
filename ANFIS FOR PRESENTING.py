import math # for all mathematical operations
import random # used for genetic algorithm
import matplotlib.pyplot as plt # used for plotting graphs
import pandas as pd # used to extract data from excel sheets
import time # used to calculate the time taken for training
from numpy.polynomial import Polynomial # used for plotting best fit line

def load_data_from_spreadsheet(filename):
    """
    Load data from Excel file using Pandas.
    """
    data = pd.read_excel(filename)
    return data

def gaussian(x, mu, sigma):
    """
    Calculate the Gaussian function value for a given x, mean, and standard deviation.
    """
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def get_input(prompt, min_val, max_val):
    """
    Function to get and validate user input.
    """
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def tanh(x):
    """
    Tan Sigmoid function used as activation function to get output between 0-1.
    """
    return math.tanh(1 / 400 * x)

def calculate_mse(actual_output, calculated_output):
    """
    Calculate the Mean Squared Error (MSE) between the actual and predicted output.
    """
    return 1 * (actual_output - calculated_output) ** 2\

def fitness(individual, age, blood_pressure, cholesterol, heart_rate, heart_attack_risk):
    """
    Evaluate fitness of population by running through the ANFIS to get MSE.
    """

    sd_values = individual['sd_values']
    a_values = individual['a_values']
    b_values = individual['b_values']
    c_values = individual['c_values']
    d_values = individual['d_values']
    e_values = individual['e_values']

    mse = ANFIS(sd_values, a_values, b_values, c_values, d_values, e_values, age, blood_pressure, cholesterol, heart_rate, heart_attack_risk)
    if mse == 0:
        return 99999
    else:
        return abs(1 / mse)

def initialize_population(size):
    """
    Randomly initialize population within the set range.
    """
    population = []
    for _ in range(size):
        individual = {
            'sd_values': [random.uniform(30, 100) for _ in range(8)],
            'a_values': [random.uniform(0, 1) for _ in range(16)],
            'b_values': [random.uniform(0, 1) for _ in range(16)],
            'c_values': [random.uniform(0, 1) for _ in range(16)],
            'd_values': [random.uniform(0, 1) for _ in range(16)],
            'e_values': [random.uniform(0, 1) for _ in range(16)]
        }
        population.append(individual)
    return population

def tournament_selection(ranked_population, tournament_size=7):
    """
    Selection method used is Tournament Selection.
    """
    new_population = []
    while len(new_population) < len(ranked_population):
        tournament = random.sample(ranked_population, tournament_size)
        winner = sorted(tournament, key=lambda x: x[0], reverse=True)[0][1]  # Select the best individual
        new_population.append(winner)
    return new_population

def crossover(parent1, parent2, crossover_rate=0.8):
    """
    Perform single crossover.
    """
    # Create deep copies of the parents to avoid modifying the original parents
    child1, child2 = parent1.copy(), parent2.copy()

    # Check if crossover should occur
    if random.random() <= crossover_rate:
        # Perform crossover for each list of parameters
        for key in ['sd_values', 'a_values', 'b_values', 'c_values', 'd_values', 'e_values']:
            if len(parent1[key]) > 1:  # Ensure there's something to crossover
                crossover_point = random.randint(1, len(parent1[key]) - 1)
                child1[key] = parent1[key][:crossover_point] + parent2[key][crossover_point:]
                child2[key] = parent2[key][:crossover_point] + parent1[key][crossover_point:]

    return child1, child2

def mutate(population, mutation_rate=0.1):
    """
    Perform Mutation.
    """
    for individual in population:
        if random.random() < mutation_rate:
            param_to_mutate = random.choice(
                ['sd_values', 'a_values', 'b_values', 'c_values', 'd_values', 'e_values'])
            num_mutations = random.randint(1, 3)

            for _ in range(num_mutations):
                # Choose a random index to mutate
                index_to_mutate = random.randint(0, len(individual[param_to_mutate]) - 1)
                # Apply mutation based on the parameter's specific range
                if param_to_mutate == 'sd_values':
                    mutation_value = random.uniform(30, 100)
                else:  # a_values, b_values, c_values, d_values, e_values
                    mutation_value = random.uniform(0, 1)

                individual[param_to_mutate][index_to_mutate] = mutation_value

    return population

def check_convergence(ranked_population, threshold=0.005, generations_to_wait=15, fitness_limit=100):
    """
    If the fitness is converged or reached fitness limit, genetic algorithm loop breaks.
    """
    # `convergence_history` is a list storing the best fitness of the last N generations
    global convergence_history
    current_best_fitness = ranked_population[0][0]
    convergence_history.append(current_best_fitness)

    if len(convergence_history) > generations_to_wait:
        convergence_history.pop(0)
        # Check if the improvement over the last N generations is less than the threshold
        if max(convergence_history) - min(convergence_history) < threshold:
            return True  # Converged

    # Check if fitness limit is reached
    if fitness_limit is not None and current_best_fitness >= fitness_limit:
        return True  # Converged

    return False  # Not converged yet

def genetic_algorithm(row_index, age, blood_pressure, cholesterol, heart_rate, heart_attack_risk, population,
                      elitsm_percentage=0.05, generations=1000):
    """
    Genetic algorithm used to train the ANFIS.
    """

    global convergence_history
    convergence_history = []
    elitism_size = max(1, int(len(population) * elitsm_percentage))

    # Setup plotting
    fitness_values = []
    mse_values = []
    print("\n\n")

    print(f"Row: {row_index + 1}, Inputs: Age: {age}, BP: {blood_pressure}, Chol: {cholesterol}, HR: {heart_rate}, Risk: {heart_attack_risk}")

    for generation in range(generations):
        # Evaluate fitness
        ranked_population = [
            (fitness(individual, age, blood_pressure, cholesterol, heart_rate, heart_attack_risk), individual) for
            individual in population]
        ranked_population.sort(key=lambda x: x[0], reverse=True)  # Assuming higher fitness is better

        # Extract the elite individuals
        elites = [individual for _, individual in ranked_population[:elitism_size]]

        # Extract the best MSE from the current generation
        best_fitness = ranked_population[0][0]
        fitness_values.append(best_fitness)
        mse_for_generation = 1 / best_fitness
        mse_values.append(mse_for_generation)
        print(f"Best fitness at generation {generation}: {best_fitness}")

        # Selection
        selected_population = tournament_selection(ranked_population[elitism_size:])

        # Crossover
        new_population = elites[:]
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # Ensure the population does not exceed the intended size due to addition of elite individuals
        new_population = new_population[:len(population)]

        # Mutation
        population = mutate(new_population)

        # convergence check to break the loop early
        if check_convergence(ranked_population):
            break

    # Return the best solution found
    best_solution = ranked_population[0][1]
    return best_solution, population, fitness_values, mse_values

def ANFIS(sd_values, a_values, b_values, c_values, d_values, e_values, age, blood_pressure, cholesterol, heart_rate,
          heart_attack_risk):
    """
    ANFIS
    """

    # Specified Mean Values
    mean_values = [0, 100, 90, 200, 120, 580, 40, 200]

    gaussian_parameters = {
        'age': {
            'low': {'mu': mean_values[0], 'sigma': sd_values[0]},
            'high': {'mu': mean_values[1], 'sigma': sd_values[1]}
        },
        'blood_pressure': {
            'low': {'mu': mean_values[2], 'sigma': sd_values[2]},
            'high': {'mu': mean_values[3], 'sigma': sd_values[3]}
        },
        'cholesterol': {
            'low': {'mu': mean_values[4], 'sigma': sd_values[4]},
            'high': {'mu': mean_values[5], 'sigma': sd_values[5]}
        },
        'heart_rate': {
            'low': {'mu': mean_values[6], 'sigma': sd_values[6]},
            'high': {'mu': mean_values[7], 'sigma': sd_values[7]}
        }
    }

    # Calculate the membership values
    age_low = gaussian(age, **gaussian_parameters['age']['low'])
    age_high = gaussian(age, **gaussian_parameters['age']['high'])

    bp_low = gaussian(blood_pressure, **gaussian_parameters['blood_pressure']['low'])
    bp_high = gaussian(blood_pressure, **gaussian_parameters['blood_pressure']['high'])

    cholesterol_low = gaussian(cholesterol, **gaussian_parameters['cholesterol']['low'])
    cholesterol_high = gaussian(cholesterol, **gaussian_parameters['cholesterol']['high'])

    heart_rate_low = gaussian(heart_rate, **gaussian_parameters['heart_rate']['low'])
    heart_rate_high = gaussian(heart_rate, **gaussian_parameters['heart_rate']['high'])

    # Calculate weights
    weights = {
        'w1': age_low * cholesterol_low * bp_low * heart_rate_low,
        'w2': age_low * cholesterol_low * bp_low * heart_rate_high,
        'w3': age_low * cholesterol_low * bp_high * heart_rate_low,
        'w4': age_low * cholesterol_low * bp_high * heart_rate_high,
        'w5': age_low * cholesterol_high * bp_low * heart_rate_low,
        'w6': age_low * cholesterol_high * bp_low * heart_rate_high,
        'w7': age_low * cholesterol_high * bp_high * heart_rate_low,
        'w8': age_low * cholesterol_high * bp_high * heart_rate_high,
        'w9': age_high * cholesterol_low * bp_low * heart_rate_low,
        'w10': age_high * cholesterol_low * bp_low * heart_rate_high,
        'w11': age_high * cholesterol_low * bp_high * heart_rate_low,
        'w12': age_high * cholesterol_low * bp_high * heart_rate_high,
        'w13': age_high * cholesterol_high * bp_low * heart_rate_low,
        'w14': age_high * cholesterol_high * bp_low * heart_rate_high,
        'w15': age_high * cholesterol_high * bp_high * heart_rate_low,
        'w16': age_high * cholesterol_high * bp_high * heart_rate_high,
    }

    # Calculate the sum of all weights
    total_weight = sum(weights.values())

    # Return infinity if weight is 0
    if total_weight == 0:
        return float('inf')

    # Normalize the weights
    normalized_weights = {key: value / total_weight for key, value in weights.items()}

    # Calculate O1 to O16 using Sugeno Fuzzy
    O_values = [normalized_weights[f'w{i}'] * (
            a_values[i - 1] * age + b_values[i - 1] * blood_pressure + c_values[i - 1] * cholesterol + d_values[
        i - 1] * heart_rate + e_values[i - 1]) for i in range(1, 17)]

    # Sum up all the output values
    total_output = sum(O_values)

    # Calculate the tanh/tan sigmoid of total_output
    tanh_output = tanh(total_output)

    # Calculate the MSE using the actual and predicted output
    mse = calculate_mse(heart_attack_risk, tanh_output)

    return mse

def main(target=65):
    """
    Main function to glue everything
    Reads data from given Excel Dataset
    Training will stop until it reaches accuracy target per chunk or no inputs left
    Once finished, will return the best parameters for the ANFIS
    """
    # Load Excel file start time
    start_time = time.time()
    population = initialize_population(50)
    filename = r"D:\Pycharm Projects\AI Project\7\2\Heart_Disease_Prediction_Good_Dataset_Test.xlsx"
    data = load_data_from_spreadsheet(filename)

    chunk_size = 10
    total_rows = len(data)
    best_solution = None
    best_accuracy = 0
    chunks_processed = 0
    fitness_scores_all_rows = []
    mse_scores_all_rows = []
    accuracies_all_chunks = []

    for start_row in range(0, total_rows, chunk_size):
        end_row = min(start_row + chunk_size, total_rows)
        print(f"\nProcessing rows {start_row + 1} to {end_row}...")
        chunks_processed += 1

        for index, row in data.iloc[start_row:end_row].iterrows():
            # Extract data from the row
            age, blood_pressure, cholesterol, heart_rate, heart_attack_risk = row["Age"], row["Blood Pressure"], row[
                "Cholesterol"], row["Max HR"], row["Heart Disease"]

            # Calling Genetic Algorithm function
            best_solution, population, fitness_value, mse_value = genetic_algorithm(index, age, blood_pressure,
                                                                                    cholesterol, heart_rate,
                                                                                    heart_attack_risk, population)
            fitness_scores_all_rows.append(fitness_value)
            mse_scores_all_rows.append(mse_value)

        # After updating the model with the current chunk, evaluate it on the entire dataset
        mse_values = []  # Reinitialize mse_values for the entire dataset evaluation
        for index, row in data.iterrows():
            # Calculate MSE for each row in the entire dataset
            mse = ANFIS(best_solution['sd_values'], best_solution['a_values'], best_solution['b_values'],
                     best_solution['c_values'], best_solution['d_values'], best_solution['e_values'], row["Age"],
                     row["Blood Pressure"], row["Cholesterol"], row["Max HR"], row["Heart Disease"])
            mse_values.append(mse)

        # Calculate accuracy for the entire dataset up to this point
        average_mse = sum(mse_values) / len(mse_values)
        current_accuracy = 100 * (1 - average_mse)
        print(f"Current Accuracy after {chunks_processed} chunk(s) over the entire dataset: {current_accuracy}%")

        accuracies_all_chunks.append(current_accuracy)

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy

        # Stop if accuracy target reached
        if current_accuracy > target:
            print(
                f"Achieved desired accuracy after processing {chunks_processed} chunk(s) over the entire dataset. Stopping the training.")
            break

    # Print best solution if it exists and if any accuracy improvement was noted
    if best_solution and best_accuracy > 0:
        print(
            f"\nAchieved Accuracy: {best_accuracy}% after processing {chunks_processed} chunk(s) over the entire dataset")
        print("Best Solution Parameters:")
        for key, value in best_solution.items():
            print(f"{key}:")
            if isinstance(value, list):
                for i, val in enumerate(value):
                    print(f"    {i + 1}: {val}")
            else:
                print(f"    {value}")

    # Print total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal runtime: {total_time} seconds")

    # Plot Accuracy After Each Chunk
    plt.figure(figsize=(12, 6))
    plt.plot(accuracies_all_chunks, marker='o', linestyle='-', color='b')
    plt.xlabel('Chunk Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy After Each Chunk')
    plt.xticks(range(len(accuracies_all_chunks)), [f'Chunk {i + 1}' for i in range(len(accuracies_all_chunks))])
    plt.show()

    plt.figure(figsize=(12, 6))

    # Plotting MSE over generations for each row with best fit line
    plt.figure(figsize=(12, 6))
    for i, mse_scores in enumerate(mse_scores_all_rows):
        plt.plot(mse_scores, label=f'MSE Row {i + 1}')

    # Adding best fit line
    mse_combined = [sum(mse_scores) / len(mse_scores) for mse_scores in mse_scores_all_rows]
    coefficients = Polynomial.fit(range(len(mse_combined)), mse_combined, 1).convert().coef
    best_fit_line = [coefficients[1] * x + coefficients[0] for x in range(len(mse_combined))]
    plt.plot(best_fit_line, label='Best Fit Line', linestyle='--', color='red')

    plt.xlabel('Generation')
    plt.ylabel('MSE')
    plt.title('MSE per Generation for All Rows with Best Fit Line')
    plt.legend()
    plt.show()

    # Plotting Fitness Score over generations for each row with best fit line
    plt.figure(figsize=(12, 6))
    for i, fitness_scores in enumerate(fitness_scores_all_rows):
        plt.plot(fitness_scores, label=f'Row {i + 1}')

    # Adding best fit line
    fitness_combined = [sum(fitness_scores) / len(fitness_scores) for fitness_scores in fitness_scores_all_rows]
    coefficients = Polynomial.fit(range(len(fitness_combined)), fitness_combined, 1).convert().coef
    best_fit_line = [coefficients[1] * x + coefficients[0] for x in range(len(fitness_combined))]
    plt.plot(best_fit_line, label='Best Fit Line', linestyle='--', color='red')

    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.title('Best Fitness Score per Generation for All Rows with Best Fit Line')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
