# GPMST
Genetic Programming with Multi-Source Transfer Learning for Scheduling Applications in Edge and Fog Computing

## Overview
This repository contains the official implementation of the **GPMST** algorithm proposed in the paper *Genetic Programming with Multi-Source Transfer Learning for Scheduling Applications in Edge and Fog Computing*, submitted to Swarm and Evolutionary Computation (SWEVO-D-26-02411R1).

GPMST is a genetic programming hyper-heuristic (GPHH) approach with multi-source transfer learning, designed to automatically evolve scheduling heuristics for dynamic IoT application scheduling in edge and fog computing environments.

## Paper Information
- **Title**: Genetic Programming with Multi-Source Transfer Learning for Scheduling Applications in Edge and Fog Computing
- **Venue**: Swarm and Evolutionary Computation (Under Review, Revision R1)
- **Manuscript ID**: SWEVO-D-26-02411R1

## Core Files & Paper Correspondence
- **`gpiot-stage1.py`**: Implements Stage I of GPMST. It evolves scheduling heuristics on the target problem, with initialization by injecting best heuristics from all source problems, corresponding to **Algorithm 1** in the paper.
- **`gpiot-stage2.py`**: Implements Stage II of GPMST. It optimizes the ensemble weights of source and target heuristics via CMA-ES with softmax normalization, corresponding to **Algorithm 2** and Section 4.3.
- **`Edgenetwork.py`**: Constructs the heterogeneous edge/fog/cloud computing environment with server capacity, bandwidth and latency parameters, consistent with **Section 3** and **Table 3**.
- **`Taskqueue.py`**: Generates dynamic IoT application sequences with Poisson arrival, random data size and workload, following the simulation settings in **Section 5.1**.
