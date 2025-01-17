---
title: 'Automatically Designing Optimization Algorithms Using Genetic Programming'
date: 2024-02-07
permalink: /posts/2024/02/Genetic-Programming-ADO/
tags:
  - papers
---

### Automatic Design of Optimization Algorithm Based on Genetic Programming

It is well known that an important research topic in evolutionary computation is the design of new optimization algorithms. This process is usually carried out by human experts. However, can we let computers automatically design optimization algorithms? The answer to this question is affirmative. This article will introduce how to automatically design optimization algorithms based on genetic programming.

**With a tool for automatic algorithm design like this, once we obtain an algorithmic formula, we only need to observe whether there is a corresponding biological behavior in nature to get a new intelligent optimization algorithm.**

For example, this article will attempt to use genetic programming to automatically design the Arctic Fox Algorithm!

![Arctic Fox Algorithm](/assets/Fox2.png)

### Optimization Function

For example, we hope that the automatically designed algorithm can perform well on a spherical function. The spherical function is a classic test function in the field of single-objective optimization, with the following formula:


```python
import operator
import random

from deap import base, creator, tools, gp, algorithms
import numpy as np

np.random.seed(0)
random.seed(0)


def sphere(x, c=[1, 1, 1]):
    """
    Shifted Sphere function.

    Parameters:
    - x: Input vector.
    - c: Shift vector indicating the new optimal location.

    Returns:
    - The value of the shifted Sphere function at x.
    """
    return sum((xi - ci) ** 2 for xi, ci in zip(x, c))
```

### Classic Optimization Algorithms

In the literature, differential evolution can be used to solve this spherical function optimization problem.


```python
# DE
dim = 3
bounds = np.array([[-5, 5]] * dim)


# Define a simple DE algorithm to test the crossover
def differential_evolution(
        crossover_func, bounds, population_size=10, max_generations=50
):
    population = [
        np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        for _ in range(population_size)
    ]
    population = np.array(population)
    best = min(population, key=lambda ind: sphere(ind))
    for gen in range(max_generations):
        for i, x in enumerate(population):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            mutant = np.clip(crossover_func(a, b, c, np.random.randn(dim)), bounds[:, 0], bounds[:, 1])
            if sphere(mutant) < sphere(x):
                population[i] = mutant
                if sphere(mutant) < sphere(best):
                    best = mutant
    return sphere(best)


print("Optimization result obtained by traditional DE algorithm:",
      np.mean([differential_evolution(lambda a, b, c, F: a + F * (b - c), bounds) for _ in range(10)]))
```
The optimization result obtained by the traditional DE algorithm is 4.506377260849465e-05.

The optimization result obtained by the traditional DE algorithm is quite good. However, can we automatically design a better algorithm?

### Automatic Design of Optimization Algorithm Based on Genetic Programming

In fact, the crossover operator of DE is essentially a function that takes three vectors and one random vector as input, and then outputs a vector. Therefore, we can use genetic programming to automatically design this crossover operator.


```python
# GP Operators
pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(np.add, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(np.subtract, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(np.multiply, [np.ndarray, np.ndarray], np.ndarray)
pset.addEphemeralConstant("rand100", lambda: np.random.randn(dim), np.ndarray)

pset.context["array"] = np.array

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# Evaluate function for GP individuals
def evalCrossover(individual):
    # Convert the individual into a function
    func = toolbox.compile(expr=individual)
    return (differential_evolution(func, bounds),)


toolbox.register("evaluate", evalCrossover)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# Evolve crossover operators
population = toolbox.population(n=50)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(population, toolbox, 0.9, 0.1, 30, stats, halloffame=hof)

# Best crossover operator
best_crossover = hof[0]
print(f"Best Crossover Operator:\n{best_crossover}")
print(f"Fitness: {best_crossover.fitness.values}")
```

### Analyzing the New Algorithm

Now, we have obtained a new crossover operator. Let's take a look at the formula of this crossover operator.

$X_{new}=X+(F*X-F)$, where $F$ is a set of random variables.

```python
add = np.add
subtract = np.subtract
multiply = np.multiply
square = np.square
array = np.array

crossover_operator = lambda ARG0, ARG1, ARG2, ARG3: add(ARG0, subtract(multiply(ARG0, ARG3), ARG3))
print("Optimization result obtained by the new algorithm:", np.mean([differential_evolution(crossover_operator, bounds) for _ in range(10)]))
```

The optimization result obtained by the new optimization algorithm is 1.0213225557390857e-19.

The optimization result obtained by the new algorithm is better than that of the traditional DE algorithm. This proves that GP has discovered a better new algorithm.

### Arctic Fox Algorithm

Now, we can name this algorithm the Arctic Fox Algorithm. The fur color of the Arctic fox changes with the seasons. In this formula, X changes according to the random variables F. The form of this formula is somewhat similar to the change in the fur color of the Arctic fox. Therefore, we can name this algorithm the Arctic Fox Algorithm.

![Arctic Fox Algorithm](/assets/Fox.png)

The crossover operator of this algorithm is $X_{new}=X+(F*X-F)$.
