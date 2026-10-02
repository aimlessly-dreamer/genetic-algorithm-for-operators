import operator
import random

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools


# Search settings.
NUM_GENERATIONS = 100
POPULATION_SIZE = 100
MAX_TERMS = 8  # Maximum number of column references in an expression.
MAX_COLUMN_REPEATS = 3  # Each column may appear this many times per expression.
MAX_COLUMN_MULTIPLIER = 3  # Each term may multiply its column by an integer from 1 to n.
RANDOM_SEED = None  # Different runs; use an integer for reproducible results.

# Random-only test preset: no selection, crossover or mutation; 100% fresh candidates.
# Lower IMMIGRANT_FRACTION and increase mutation/crossover to resume evolution.
CROSSOVER_PROBABILITY = 0.0
MUTATION_PROBABILITY = 0.0
GENE_MUTATION_PROBABILITY = 0.0
IMMIGRANT_FRACTION = 1.0


# Comment out an operator to disable it; uncomment an optional one to enable it.
OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '//': operator.floordiv,  # Floor division.
    '**': operator.pow,  # Power.
}

# Unary transforms apply to raw columns, before integer multipliers and binary operations.
TRANSFORMS = {
    None: lambda values: values,  # Leave the column unchanged.
    'np.log': np.log,  # Natural logarithm; requires positive column values.
}


def create_demo_data(seed=42):
    """Create a reproducible dataset whose target formula is reachable."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.random((10, 5)),
        columns=[f'Column{i}' for i in range(1, 6)],
    )
    df['Result'] = df['Column1'] / 4 + df['Column3'] - df['Column4']
    df.insert(1, 'Four', 4)
    return df


def expression_key(individual):
    """Keep column order and repetition, ignoring only the first incoming operator."""
    selected = list(individual)
    if selected:
        index, _, transform, multiplier = selected[0]
        selected[0] = index, None, transform, multiplier
    return tuple(selected)


def describe_individual(individual, column_names):
    """Return the column terms (including transforms and multipliers) and operators."""
    selected = expression_key(individual)
    columns = []
    for index, _, transform, multiplier in selected:
        column = f'{transform}({column_names[index]})' if transform else column_names[index]
        columns.append(f'({multiplier} * {column})' if multiplier != 1 else column)
    symbols = [symbol for _, symbol, _, _ in selected[1:]]
    return columns, symbols


def format_expression(individual, column_names):
    """Show the expression's exact left-to-right evaluation order."""
    columns, symbols = describe_individual(individual, column_names)
    if not columns:
        return '(empty)'
    expression = columns[0]
    for symbol, column in zip(symbols, columns[1:]):
        expression = f'({expression} {symbol} {column})'
    return expression


def evaluate(individual, data, target):
    """Score the expression encoded by a candidate without randomness."""
    selected = expression_key(individual)
    if not selected:
        return float('inf'),

    if not np.isfinite(target).all():
        return float('inf'),

    result = None
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for index, symbol, transform, multiplier in selected:
            # A log term is multiplier * np.log(column), never np.log(multiplier * column).
            values = multiplier * TRANSFORMS[transform](data[:, index])
            if not np.isfinite(values).all():
                return float('inf'),
            if result is None:
                result = values
                continue
            if symbol in ('/', '//') and np.any(values == 0):
                return float('inf'),
            result = OPERATORS[symbol](result, values)
            if not np.isfinite(result).all():
                return float('inf'),

        mse = np.mean((result - target) ** 2)

    return (float(mse),) if np.isfinite(mse) else (float('inf'),)


def create_gene(num_columns, column_index=None, max_column_multiplier=MAX_COLUMN_MULTIPLIER):
    """Choose a column, incoming operator, transform and positive integer multiplier."""
    if column_index is None:
        column_index = random.randrange(num_columns)
    symbol = random.choice(tuple(OPERATORS))
    transforms = tuple(TRANSFORMS)
    transform = random.choice(transforms) if len(transforms) > 1 else transforms[0]
    multiplier = random.randint(1, max_column_multiplier)
    return column_index, symbol, transform, multiplier


def create_individual(num_columns, max_terms, max_column_repeats,
                      max_column_multiplier=MAX_COLUMN_MULTIPLIER):
    """Create an expression with arbitrary column order and bounded repetition."""
    length = random.randint(1, min(max_terms, num_columns * max_column_repeats))
    counts = [0] * num_columns
    available = list(range(num_columns))
    genes = []
    for _ in range(length):
        column_index = random.choice(available)
        genes.append(create_gene(num_columns, column_index, max_column_multiplier))
        counts[column_index] += 1
        if counts[column_index] == max_column_repeats:
            available.remove(column_index)
    return creator.OperatorIndividual(genes)


def repair_individual(individual, num_columns, max_terms, max_column_repeats,
                      max_column_multiplier=MAX_COLUMN_MULTIPLIER):
    """Keep a nonempty expression within the length and column-repeat limits."""
    counts = [0] * num_columns
    genes = []
    for gene in individual:
        column_index = gene[0]
        if counts[column_index] < max_column_repeats:
            genes.append(gene)
            counts[column_index] += 1
            if len(genes) == max_terms:
                break
    individual[:] = genes or [create_gene(num_columns,
                                         max_column_multiplier=max_column_multiplier)]
    return individual


def mate(first, second, num_columns, max_terms, max_column_repeats,
         max_column_multiplier=MAX_COLUMN_MULTIPLIER):
    """Exchange variable-length expression segments, including single-term parents."""
    first_cut = random.randrange(len(first) + 1)
    second_cut = random.randrange(len(second) + 1)
    first_genes = first[:first_cut] + second[second_cut:]
    second_genes = second[:second_cut] + first[first_cut:]
    first[:] = first_genes
    second[:] = second_genes
    repair_individual(first, num_columns, max_terms, max_column_repeats, max_column_multiplier)
    repair_individual(second, num_columns, max_terms, max_column_repeats, max_column_multiplier)
    return first, second


def mutate(individual, indpb, num_columns, max_terms, max_column_repeats,
           max_column_multiplier=MAX_COLUMN_MULTIPLIER):
    """Change column terms and multipliers; add, remove or reorder terms."""
    for index, (column_index, symbol, transform, multiplier) in enumerate(individual):
        if random.random() < indpb:
            alternatives = tuple(column for column in range(num_columns)
                                 if column != column_index)
            if alternatives:
                column_index = random.choice(alternatives)
        if random.random() < indpb:
            alternatives = tuple(op for op in OPERATORS if op != symbol)
            if alternatives:
                symbol = random.choice(alternatives)
        alternatives = tuple(name for name in TRANSFORMS if name != transform)
        if alternatives and random.random() < indpb:
            transform = random.choice(alternatives)
        if random.random() < indpb:
            alternatives = tuple(value for value in range(1, max_column_multiplier + 1)
                                 if value != multiplier)
            if alternatives:
                multiplier = random.choice(alternatives)
        individual[index] = column_index, symbol, transform, multiplier

    if random.random() < indpb and len(individual) < max_terms:
        individual.insert(random.randrange(len(individual) + 1),
                          create_gene(num_columns, max_column_multiplier=max_column_multiplier))
    if random.random() < indpb and len(individual) > 1:
        del individual[random.randrange(len(individual))]
    if random.random() < indpb and len(individual) > 1:
        first, second = random.sample(range(len(individual)), 2)
        individual[first], individual[second] = individual[second], individual[first]
    repair_individual(individual, num_columns, max_terms, max_column_repeats,
                      max_column_multiplier)
    return individual,


def main(dataframe=None, seed=RANDOM_SEED, population_size=POPULATION_SIZE,
         num_generations=NUM_GENERATIONS, max_terms=MAX_TERMS,
         max_column_repeats=MAX_COLUMN_REPEATS,
         max_column_multiplier=MAX_COLUMN_MULTIPLIER):
    """Search expressions for the demo or a numeric dataframe with a Result column."""
    if population_size < 1 or num_generations < 0:
        raise ValueError('Population size must be positive and generations nonnegative.')
    if max_terms < 1 or max_column_repeats < 1:
        raise ValueError('Expression length and column-repeat limits must be positive.')
    if not isinstance(max_column_multiplier, int) or max_column_multiplier < 1:
        raise ValueError('The column multiplier limit must be a positive integer.')
    for name, probability in (
        ('CROSSOVER_PROBABILITY', CROSSOVER_PROBABILITY),
        ('MUTATION_PROBABILITY', MUTATION_PROBABILITY),
        ('GENE_MUTATION_PROBABILITY', GENE_MUTATION_PROBABILITY),
        ('IMMIGRANT_FRACTION', IMMIGRANT_FRACTION),
    ):
        if not 0 <= probability <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')
    if not OPERATORS or not TRANSFORMS:
        raise ValueError('Enable at least one binary operator and one column transform.')

    df = create_demo_data(seed) if dataframe is None else dataframe
    if df.empty or 'Result' not in df.columns:
        raise ValueError('The dataset must have rows and a Result column.')

    features = df.drop(columns='Result')
    if features.shape[1] == 0:
        raise ValueError('The dataset must have at least one feature column.')
    data = features.to_numpy(dtype=float)
    target = df['Result'].to_numpy(dtype=float)
    if not np.isfinite(data).all() or not np.isfinite(target).all():
        raise ValueError('The dataset must contain only finite numeric values.')

    column_names = list(features.columns)
    random.seed(seed)

    # Reuse DEAP classes so calling main repeatedly does not redefine them.
    if not hasattr(creator, 'OperatorFitnessMin'):
        creator.create('OperatorFitnessMin', base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'OperatorIndividual'):
        creator.create('OperatorIndividual', list, fitness=creator.OperatorFitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('individual', create_individual, num_columns=len(column_names),
                     max_terms=max_terms, max_column_repeats=max_column_repeats,
                     max_column_multiplier=max_column_multiplier)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate, data=data, target=target)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', mate, num_columns=len(column_names),
                     max_terms=max_terms, max_column_repeats=max_column_repeats,
                     max_column_multiplier=max_column_multiplier)
    toolbox.register('mutate', mutate, indpb=GENE_MUTATION_PROBABILITY,
                     num_columns=len(column_names), max_terms=max_terms,
                     max_column_repeats=max_column_repeats,
                     max_column_multiplier=max_column_multiplier)

    population = toolbox.population(n=population_size)
    hall_of_fame = tools.HallOfFame(
        maxsize=10,
        similar=lambda left, right: expression_key(left) == expression_key(right),
    )

    for individual, fitness in zip(population, toolbox.map(toolbox.evaluate, population)):
        individual.fitness.values = fitness
    hall_of_fame.update(population)

    for generation in range(num_generations):
        if IMMIGRANT_FRACTION == 1.0:
            # Pure random search: each generation is independent of the previous one.
            offspring = toolbox.population(n=population_size)
        else:
            parents = toolbox.select(population, len(population))
            offspring = algorithms.varAnd(parents, toolbox, cxpb=CROSSOVER_PROBABILITY,
                                          mutpb=MUTATION_PROBABILITY)

            # Fresh candidates keep evolutionary generations diverse.
            if IMMIGRANT_FRACTION > 0:
                count = max(1, int(len(offspring) * IMMIGRANT_FRACTION))
                for index in random.sample(range(len(offspring)), count):
                    offspring[index] = toolbox.individual()
        invalid = [individual for individual in offspring if not individual.fitness.valid]
        for individual, fitness in zip(invalid, toolbox.map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness

        hall_of_fame.update(offspring)
        population[:] = offspring
        best_fitness = hall_of_fame[0].fitness.values[0]
        print(f'Generation {generation + 1}: Best Fitness = {best_fitness}')

    best_individual = hall_of_fame[0]
    selected_columns, best_operators = describe_individual(best_individual, column_names)
    best_fitness = best_individual.fitness.values[0]

    print('Hall of Fame:')
    for index, individual in enumerate(hall_of_fame, start=1):
        print(f'Individual {index}: Expression={format_expression(individual, column_names)}, '
              f'Fitness={individual.fitness.values[0]}')

    print('Best individual:')
    print('Selected Columns:', selected_columns)
    print('Selected Operators:', best_operators)
    print('Expression:', format_expression(best_individual, column_names))
    print('Fitness obtained:', best_fitness)
    return best_individual, best_operators, best_fitness, hall_of_fame


if __name__ == '__main__':
    # Load file dataset: uncomment these lines and comment out main() below.
    # df = pd.read_csv('xxxxx.csv')
    # df = df.loc[:, ['Column0', 'Column1', 'Column2', 'Result']]
    # main(dataframe=df)

    main()
