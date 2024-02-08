---
title: 'Genetic Programming Automatically Generated Optimization Algorithm Can Also Have Convergence Analysis'
date: 2024-02-08
permalink: /posts/2023/07/Genetic-Programming-Analysis/
tags:
  - papers
---

### What is the Arctic Fox Algorithm?

In the article "Automated Design of Optimization Algorithms based on Genetic Programming (Automatic Discovery of the Arctic Fox Optimization Algorithm)," we utilized genetic programming algorithms to discover a new optimization algorithm within five minutes, $X_{\text{new}} = X + (F \cdot X - F)$, named the Arctic Fox Algorithm.

As we know, when designing algorithms, we not only aim for good experimental results but also seek theoretical guarantees for the designed algorithms. So, can we prove that the Arctic Fox Algorithm necessarily converges to the global optimum?

### Convergence of the Arctic Fox Algorithm

**Assumptions:**
- $\mathcal{S} \subset \mathbb{R}^n$ is a bounded search space containing the global optimum $X^*$.
- $F$ is a random perturbation vector drawn from some distribution, and the range of $F$ covers the entire search space $\mathcal{S}$.

**Proof:**
Define $A_\epsilon$ as the $\epsilon$-neighborhood of $X^\*$, i.e., $A_\epsilon = \{X \in \mathcal{S} : \|X - X^\*\| < \epsilon\}$. Let $P_\epsilon$ be the probability of $X_{\text{new}} = X + (F \cdot X - F)$ falling into $A_\epsilon$ in one operation. Since $F$ can cover the entire search space, even for sufficiently small $\epsilon$, we still have $P_\epsilon > 0$.

In $N$ iterations, the probability of $X_{\text{new}}$ falling into $A_\epsilon$ at least once is $1 - (1 - P_\epsilon)^N$. Using limits, we can express the behavior of this probability as $N$ approaches infinity:

$$\lim_{N \to \infty} \left[ 1 - (1 - P_\epsilon)^N \right] = 1$$

This limit expresses that as the number of iterations $N$ increases, the probability of $X_{\text{new}}$ falling into the $\epsilon$-neighborhood of $X^*$ approaches 1.

### Conclusion:
Now, we have demonstrated that the Arctic Fox Algorithm $X_{\text{new}} = X + (F \cdot X - F)$, in the case of an infinite number of iterations, can approach the global optimum $X^*$ with probability 1.
