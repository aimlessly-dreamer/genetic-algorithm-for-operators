import operator
import random

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools


# Dataset setting: False creates demo data; True loads example_dataset.csv.
USE_CSV_DATA = True

# Search settings.
NUM_GENERATIONS = 10
POPULATION_SIZE = 10
MAX_TERMS = 4  # Maximum number of column references in an expression.
MAX_COLUMN_REPEATS = 2  # Each column may appear this many times per expression.
MAX_COLUMN_CONSTANT = 5  # Integers from 1 to n for column multipliers, divisors or exponents.
RANDOM_SEED = None  # Different runs; use an integer for reproducible results.

# Random-only test preset: no selection, crossover or mutation; 100% fresh candidates.
# Lower IMMIGRANT_FRACTION and increase mutation/crossover to resume evolution.
CROSSOVER_PROBABILITY = 0.10
MUTATION_PROBABILITY = 0.10
GENE_MUTATION_PROBABILITY = 0.10
IMMIGRANT_FRACTION = 0.70


# Comment out an operator to disable it; uncomment an optional one to enable it.
OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    ## '//': operator.floordiv,  # Floor division.
    ## '**': operator.pow,  # Power.
}

# Operations with an integer constant are independent of operators between columns.
# Comment out an entry to disable it.
COLUMN_CONSTANT_OPERATORS = {
    '*': operator.mul,  # n * Column.
    '/': operator.truediv,  # Column / n.
    '**': operator.pow,  # Column ** n.
}

# Unary transforms apply to raw columns, before constant operations and binary operations.
TRANSFORMS = {
    None: lambda values: values,  # Leave the column unchanged.
    #  'np.log': np.log,  # Natural logarithm; requires positive column values.
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
    """Keep order and repetitions while normalizing unused and identity operations."""
    selected = [
        (index, symbol, transform,
         None if constant == 1 and constant_symbol in ('*', '/', '**') else constant_symbol,
         constant)
        for index, symbol, transform, constant_symbol, constant in individual
    ]
    if selected:
        index, _, transform, constant_symbol, constant = selected[0]
        selected[0] = index, None, transform, constant_symbol, constant
    return tuple(selected)


def describe_individual(individual, column_names):
    """Return column terms with transforms and constant operations, plus operators."""
    selected = expression_key(individual)
    columns = []
    for index, _, transform, constant_symbol, constant in selected:
        column = f'{transform}({column_names[index]})' if transform else column_names[index]
        if constant_symbol == '*':
            column = f'({constant} * {column})'
        elif constant_symbol is not None:
            column = f'({column} {constant_symbol} {constant})'
        columns.append(column)
    symbols = [symbol for _, symbol, _, _, _ in selected[1:]]
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
        for index, symbol, transform, constant_symbol, constant in selected:
            # Log transforms the raw column before multiplication, division or powers.
            values = TRANSFORMS[transform](data[:, index])
            if constant_symbol is not None:
                values = COLUMN_CONSTANT_OPERATORS[constant_symbol](values, constant)
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


def create_gene(num_columns, column_index=None, max_column_constant=MAX_COLUMN_CONSTANT):
    """Choose a column, transform and operations with columns and an integer constant."""
    if column_index is None:
        column_index = random.randrange(num_columns)
    symbol = random.choice(tuple(OPERATORS))
    transforms = tuple(TRANSFORMS)
    transform = random.choice(transforms) if len(transforms) > 1 else transforms[0]
    constant_symbol = random.choice(tuple(COLUMN_CONSTANT_OPERATORS))
    constant = random.randint(1, max_column_constant)
    return column_index, symbol, transform, constant_symbol, constant


def create_individual(num_columns, max_terms, max_column_repeats,
                      max_column_constant=MAX_COLUMN_CONSTANT):
    """Create an expression with arbitrary column order and bounded repetition."""
    length = random.randint(1, min(max_terms, num_columns * max_column_repeats))
    counts = [0] * num_columns
    available = list(range(num_columns))
    genes = []
    for _ in range(length):
        column_index = random.choice(available)
        genes.append(create_gene(num_columns, column_index, max_column_constant))
        counts[column_index] += 1
        if counts[column_index] == max_column_repeats:
            available.remove(column_index)
    return creator.OperatorIndividual(genes)


def repair_individual(individual, num_columns, max_terms, max_column_repeats,
                      max_column_constant=MAX_COLUMN_CONSTANT):
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
                                         max_column_constant=max_column_constant)]
    return individual


def mate(first, second, num_columns, max_terms, max_column_repeats,
         max_column_constant=MAX_COLUMN_CONSTANT):
    """Exchange variable-length expression segments, including single-term parents."""
    first_cut = random.randrange(len(first) + 1)
    second_cut = random.randrange(len(second) + 1)
    first_genes = first[:first_cut] + second[second_cut:]
    second_genes = second[:second_cut] + first[first_cut:]
    first[:] = first_genes
    second[:] = second_genes
    repair_individual(first, num_columns, max_terms, max_column_repeats, max_column_constant)
    repair_individual(second, num_columns, max_terms, max_column_repeats, max_column_constant)
    return first, second


def mutate(individual, indpb, num_columns, max_terms, max_column_repeats,
           max_column_constant=MAX_COLUMN_CONSTANT):
    """Change columns, transforms, constant operations and values; reorder terms."""
    for index, gene in enumerate(individual):
        column_index, symbol, transform, constant_symbol, constant = gene
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
            alternatives = tuple(op for op in COLUMN_CONSTANT_OPERATORS if op != constant_symbol)
            if alternatives:
                constant_symbol = random.choice(alternatives)
        if random.random() < indpb:
            alternatives = tuple(value for value in range(1, max_column_constant + 1)
                                 if value != constant)
            if alternatives:
                constant = random.choice(alternatives)
        individual[index] = column_index, symbol, transform, constant_symbol, constant

    if random.random() < indpb and len(individual) < max_terms:
        individual.insert(random.randrange(len(individual) + 1),
                          create_gene(num_columns, max_column_constant=max_column_constant))
    if random.random() < indpb and len(individual) > 1:
        del individual[random.randrange(len(individual))]
    if random.random() < indpb and len(individual) > 1:
        first, second = random.sample(range(len(individual)), 2)
        individual[first], individual[second] = individual[second], individual[first]
    repair_individual(individual, num_columns, max_terms, max_column_repeats,
                      max_column_constant)
    return individual,


def main(dataframe=None, seed=RANDOM_SEED, population_size=POPULATION_SIZE,
         num_generations=NUM_GENERATIONS, max_terms=MAX_TERMS,
         max_column_repeats=MAX_COLUMN_REPEATS,
         max_column_constant=MAX_COLUMN_CONSTANT):
    """Search expressions for the demo or a numeric dataframe with a Result column."""
    if population_size < 1 or num_generations < 0:
        raise ValueError('Population size must be positive and generations nonnegative.')
    if max_terms < 1 or max_column_repeats < 1:
        raise ValueError('Expression length and column-repeat limits must be positive.')
    if not isinstance(max_column_constant, int) or max_column_constant < 1:
        raise ValueError('The column constant limit must be a positive integer.')
    for name, probability in (
        ('CROSSOVER_PROBABILITY', CROSSOVER_PROBABILITY),
        ('MUTATION_PROBABILITY', MUTATION_PROBABILITY),
        ('GENE_MUTATION_PROBABILITY', GENE_MUTATION_PROBABILITY),
        ('IMMIGRANT_FRACTION', IMMIGRANT_FRACTION),
    ):
        if not 0 <= probability <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')
    if not OPERATORS or not TRANSFORMS or not COLUMN_CONSTANT_OPERATORS:
        raise ValueError('Enable at least one binary operator, column transform and constant operator.')

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
                     max_column_constant=max_column_constant)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate, data=data, target=target)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', mate, num_columns=len(column_names),
                     max_terms=max_terms, max_column_repeats=max_column_repeats,
                     max_column_constant=max_column_constant)
    toolbox.register('mutate', mutate, indpb=GENE_MUTATION_PROBABILITY,
                     num_columns=len(column_names), max_terms=max_terms,
                     max_column_repeats=max_column_repeats,
                     max_column_constant=max_column_constant)

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
    if USE_CSV_DATA:
        df = pd.read_csv('example_dataset.csv')
        df = df.loc[:, ['Column0', 'Column1', 'Column2', 'Result']]
        # Solution: Result = Column2 ** 2 + Column1 / 2 - 3 * Column0
        main(dataframe=df)
    else:
        main()
