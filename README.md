# Probabilistic Artificial Intelligence
## Gaussian Process Regression
<img align="right" height="140" src="https://github.com/riccardodesanti/probabilistic-AI/blob/main/images/GP_1.png"></img>
Implementation of a Gaussian Process regression from scratch and proposed a way to approximate the hard computation (of the update) of the model posterior. Then applied to an inference problem based on space data.
<br/><br/>
## Bayesian Neural Network
<img align="right" height="110" src="https://github.com/riccardodesanti/probabilistic-AI/blob/main/images/BNN_1.png"></img>
Coding exercise based on the theory shown in [Variational Inference for Neural Networks](https://www.cs.toronto.edu/~graves/nips_2011.pdf), implementing a simple Bayesian NN, and applying it on the [Rotated MNIST](https://github.com/ChaitanyaBaweja/RotNIST) and [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) datasets.
<br/><br/>
## Bayesian Optimization
<img align="right" height="140" src="https://github.com/riccardodesanti/probabilistic-AI/blob/main/images/BO_1.png"></img>
Implementation of a custom [Bayesian optimization algorithm to an hyperparameter tuning problem](https://papers.nips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf). In particular, we wished to determine the value of a model hyperparameter that maximizes the validation accuracy subject to a constraint on the average prediction speed, as shown in [Bayesian Optimization with Unknown Constraints](https://www.cs.princeton.edu/~rpa/pubs/gelbart2014constraints.pdf) for the general case. 
<br/><br/>
## Actor Critic Reinforcement Learning
<img align="right" height="120" src="https://github.com/riccardodesanti/probabilistic-AI/blob/main/images/RL_1.png"></img>
The task was to implement an algorithm that, by practicing on a simulator, learns a control policy for a lunar lander. The method suggested is a variant of policy gradient with two additional features, namely (1) [Rewards-to-go](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#implementing-reward-to-go-policy-gradient), and (2) [Generalized Advantage Estimatation](https://arxiv.org/pdf/1506.02438.pdf), both aiming at decreasing the variance of the policy gradient estimates while keeping them unbiased.
