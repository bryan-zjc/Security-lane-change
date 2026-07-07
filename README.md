# Security-lane-change

This is the demo code for our paper **Secure Lane-Change Trajectory Planning for Connected and Autonomous Vehicles under Cyberattacks.**

## **Abstract**
>Connected and autonomous vehicles (CAVs) can improve driving experience by using shared driving-intention information from surrounding vehicles, but they also face crash risks when such information is falsified by cyberattacks. Existing studies on cyberattack-resilient CAV decision-making mainly focus on single-lane scenarios, while secure decision-making in multi-lane environments remains insufficiently studied. This study proposes a secure lane-change trajectory planning model for CAVs under potential inter-vehicle cyberattacks. The proposed model cautiously utilizes the communicated trajectories of surrounding vehicles. Instead of directly imposing safety constraints on the received future trajectories of leading vehicles, the ego CAV constructs protective leader trajectories by considering possible emergency deceleration. To improve computational efficiency, we develop a FAST-SLC (FAST-secure Lane Change) algorithm that integrates candidate selection and rolling-horizon warm start to accelerate computation. The proposed model is compared with a fully-trust model and a no-trust model in both manually generated scenarios and real-world lane-changing trajectories. The proposed model can avoid rear-end collisions under both current-lane leader and target-lane leader attacks, while the fully-trust model leads to collisions. In real-world scenarios, the proposed secure model increases the minimum Time-to-Collision (TTC) and minimum gap by 70.1\% and 41.3\%, respectively, compared with the fully-trust model, and reduces the lane change time by 35.9\% compared with the conservative no-trust model. These results demonstrate that the proposed method enables safety-guaranteed decision-making for CAVs against potential cyberattacks in complex multi-lane scenarios, supporting more robust applications of CAVs.

<img width="4489" height="2623" alt="fig1" src="https://github.com/user-attachments/assets/73c316c4-576e-4ea4-97be-3795bf6b7f2b" />
_Figure 1: Illustration of the lane-changing process under potential cyberattacks._

## **Requirements**

The code is tested with Python 3.11. The main dependencies are:

```bash
numpy
matplotlib
gurobipy
```

Gurobi is required for the optimization-based security and fully-trust models. Please make sure that Gurobi and a valid Gurobi license are correctly installed before running the demo.

## **Scenarios**

This demo contains two manually generated cyberattack scenarios:

- `lc_hard_brake`: the current-lane leading vehicle LC is attacked.
- `lt_hard_brake`: the target-lane leading vehicle LT is attacked.

## **Methods**

Three lane-changing strategies are compared:

- `security`: the proposed secure lane-changing model.
- `fully_trust`: a baseline model that fully trusts the received trajectories.
- `no_trust`: a conservative baseline model that does not use the communicated future trajectory.

## **Run**

The default command runs all three models under the LT hard-brake attack scenario using FAST-SLC:

```bash
python main.py
```

To run a specific attack scenario:

```bash
python main.py --scenario lt_hard_brake
python main.py --scenario lc_hard_brake
```

To run a specific lane-changing model:

```bash
python main.py --method security
python main.py --method fully_trust
python main.py --method no_trust
python main.py --method all
```

To choose the optimization algorithm for the security and fully-trust models:

```bash
python main.py --algorithm FAST-SLC
python main.py --algorithm gurobi_solve
```
