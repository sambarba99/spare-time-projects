"""
Graphical model (Bayes net) and factor graph plotter using GraphViz

Author: Sam Barba
Created 02/12/2022
"""

from graphviz import Digraph, Graph


FILL_COLOURS = {'D': '#bbbbff', 'F': '#ffbbbb', 'M': '#bbbbbb'}


def plot_graphical_model(rvs):
	# Set up global attributes and key

	g = Digraph(
		name='graphical model',
		graph_attr={
			'overlap': 'compress', 'splines': 'line', 'fontname': 'consolas',
			'labelloc': 't', 'label': 'Graphical Model'
		},
		node_attr={'style': 'filled,setlinewidth(0.5)', 'fontname': 'consolas'},
		engine='neato'
	)

	g.node(
		'key',
		label=f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
				<TR><TD bgcolor="{FILL_COLOURS['D']}">Diagnostic (observable)</TD></TR>
				<TR><TD bgcolor="{FILL_COLOURS['F']}">Failures (observable)</TD></TR>
				<TR><TD bgcolor="{FILL_COLOURS['M']}">Mechanism (unobservable)</TD></TR>
			</TABLE>>''',
		shape='plain'
	)

	# Create nodes and edges

	for rv in rvs:
		prob_name = rv[1]  # E.g. P(ne) or P(he|me,fh)
		kind = rv[3]       # E.g. M (mechanism)
		prob_name = prob_name.split('|')[0].replace('P(', '').replace(')', '')  # E.g. 'P(he|me,fh)' -> 'he'
		g.node(prob_name, label=prob_name, shape='circle', fillcolor=FILL_COLOURS[kind])

	for rv in rvs:
		prob_name = rv[1]
		if '|' not in prob_name:
			# Edges represent conditional dependencies,
			# e.g. P(he|me,fh) -> edge from 'me' to 'he' and edge from 'fh' to 'he'
			continue

		dest, src = prob_name.replace('P(', '').replace(')', '').split('|')
		for s in src.split(','):  # In case of multiple conditions
			g.edge(s, dest)

	# Render graph

	g.render('./images/coffee_machine_graphical_model', view=True, cleanup=True, format='png')


def plot_factor_graph(edges, itn, rvs):
	# Set up global attributes and key

	g = Graph(
		name='factor graph',
		graph_attr={
			'overlap': 'compress', 'sep': '0', 'splines': 'line', 'fontname': 'consolas', 'labelloc': 't',
			'label': 'Factor Graph'
		},
		node_attr={'style': 'filled,setlinewidth(0.5)', 'fontname': 'consolas'},
		engine='neato'
	)

	g.node(
		'key',
		label=f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
				<TR><TD bgcolor="{FILL_COLOURS['D']}">Diagnostic (observable)</TD></TR>
				<TR><TD bgcolor="{FILL_COLOURS['F']}">Failures (observable)</TD></TR>
				<TR><TD bgcolor="{FILL_COLOURS['M']}">Mechanism (unobservable)</TD></TR>
				<TR><TD bgcolor="white">Factor</TD></TR>
			</TABLE>>''',
		shape='plain'
	)

	# Create nodes and edges

	for node in set([e[0] for e in edges] + [e[1] for e in edges]):
		if node < len(itn):  # RV
			rv = None
			for rv in rvs:  # Get rv kind (D/F/M)
				if itn[node] in rv[1].split('|')[0]:
					break
			kind = rv[3]
			g.node(str(node), label=itn[node], shape='circle', fillcolor=FILL_COLOURS[kind])
		else:  # Factor
			g.node(str(node), label=str(node), shape='square', fillcolor='white')

	for src, dest in edges:
		g.edge(str(src), str(dest))

	# Render graph

	g.render('./images/coffee_machine_factor_graph', view=True, cleanup=True, format='png')
