# Recycling Robot: Tabular vs Function Approximation

This project implements the **Recycling Robot problem** from Sutton & Barto (Example 3.3) and extends it from a simple tabular Markov Decision Process (MDP) to a continuous state-space setting with function approximation.

---

## Overview

The project is divided into three parts:

### Part A – Tabular Value Iteration
- Exact 2-state MDP implementation  
- Bellman optimality updates  
- Convergence analysis  
- Optimal policy extraction  

### Part B – Continuous MDP (Discretized Baseline)
- Continuous battery level \( b \in [0,2] \)  
- Discretization into 1000 bins  
- Tabular value iteration baseline  

### Part C – Function Approximation (Tile Coding)
- 1D tile coding representation  
- Linear value function approximation  
- Batch least squares updates  
- Convergence and error analysis  

---

## Results Summary

- Tabular and FA methods converge successfully  
- Approximation error ≈ 0 (MSE ≈ 10⁻²⁴)  
- Policy agreement = 100%  
- FA trades computational cost for scalability  

---

## Author
Waruni Liyanapathirana 
BSc (Hons) Computer Science (Artificial Intelligence) 
University of Hertfordshire  
Intelligent Adaptive systems
