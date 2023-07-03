---
title: "Introduction to Score-matching"
author: "Justin T Chiu"
theme: "metropolis"
fonttheme: "default"
section-titles: false
aspectratio: 169
date: \today
---

## Goals
1. What is an energy-based model and why are they hard to train?
\vspace{2em}
2. What is score-matching, and how can it be used to train an EBM?
\vspace{2em}
3. How does score-matching relate to diffusion models?

# Energy-Based Models (EBM)

## Problem setup: Density estimation
* Observations $x$
\vspace{2em}
* Goal: Learn a model $p(x)$
    * Capture uncertainty / variability over $x$
\vspace{2em}
* Participation: Give examples of an $x$ we model, and how $p(x)$ is parameterized
    * Ex: Language modeling uses Transformers for $p(x) = \prod_t p(x_t | x_{<t})$

## Running example: Image generation
* "Solved": Finite-class density estimation
    * Softmax assigns a score to each $E(x)$ then normalizes
    $$softmax(x) = \frac{\exp(E(x))}{\sum_x \exp(E(x))}$$
\vspace{2em}
* Image generation
    * Can consider every small change in a single pixel as a new clas
    * Size: 1024 x 1024, each pixel has 256 * 3 values
\vspace{2em}
* More efficient paradigms

## Image generation models
* Autoregressive: Break down generation from left-to-right
$$p(x) = \prod_t p(x_t | x_{<t})$$
\vspace{2em}
* Latent variable model: Break down generation more flexibly
$$p(x) = \sum_z p(x|z)p(z)$$
\vspace{2em}
* Energy-based model: Don't enforce breakdown of decision process

## What is an EBM?
* Globally normalized over $x$, eg sentences
\begin{align*}
p(x) &= \frac{\exp(E(x))}{Z}\\
Z &= \int_x \exp(E(x))
\end{align*}
* Computation of the partition function $Z$ is hard
