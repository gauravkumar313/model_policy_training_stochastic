# Determining Policy for a System Dynamics Model using Reinforcement  Learning

## Abstract

#### Using model-based reinforcement learning, we offer a method for policy search in stochastic dynamical systems. With Bayesian neural networks (BNNs) that have stochastic input variables, the dynamics of the system are described. Alternative modeling approaches frequently overlook complex statistical patterns in the transition dynamics (such as multi-modality and heteroskedasticity), but these input variables enable us to do so. Our BNNs are then put into an algorithm that does random roll-outs and employs stochastic optimization for policy learning after learning the dynamics. With α = 0.5, the α-divergences are reduced, we train our BNNs, which often yields better results than alternative methods like variational Bayes. We use the control of a wind turbine and industrial standard to demonstrate the effectiveness of our approach by resolving a difficult issue where model-based solutions typically fall short and by achieving encouraging outcomes in real-world circumstances.

## Running the model

### Create Bathces of Data 
##### Run src/components/generate_batches.py. This will generate data inside model/data as X_instruct_data.txt, X_trial_data.txt, Y_instruct_data.txt and Y_trial_data.txt.

### Perform Model Training
##### Run src/components/model_trainer.py [passing parameter 0.5 as α]. This will create pickle file inside model/ as AlphaWrapper_0.5.p.

### Perform Policy Search Training
##### Run src/pipeline/policy_trainer.py [passing parameter 0.5 as α]. This will create pickle file inside src/pipeline/ as AlphaWrapper_0.5.p.

### Finally perform policy evaluation
##### Run src/pipeline/policy_evaluation_after_train.py to evaluate the policy.
