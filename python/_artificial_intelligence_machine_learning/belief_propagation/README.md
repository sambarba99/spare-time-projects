## Bayesian Network Belief Propagation demo

Based on this graphical model of a coffee machine:

<p align="center">
	<img src="images/coffee_machine_graphical_model.png"/>
</p>

Where e.g. `cw` &rarr; `pl` $= P(\text{pl|cw}) = P(\text{power light switches on | circuitry works})$

1. First, we calculate the distribution of the random variables (both marginal and conditional) given the raw observations:

<p align="center">
	<img src="images/raw_data.png"/>
	<br/>
	<img src="images/random_variables_from_raw_data.png"/>
</p>

2. Next, we generate the factor graph which we'll use for message sending (Belief Propagation to calculate marginals given certain observations)

<p align="center">
	<img src="images/coffee_machine_factor_graph.png"/>
</p>

3. We then calculate the message order: a message can only be sent from node A to node B once A has received all of its messages, except for the message from B.

4. Finally, we use message passing to perform our Belief Propagation to calculate the marginals for some test machine observations:

<p align="center">
	<img src="images/machine_tests.png"/>
</p>
