"""
Critical path diagram plotter using GraphViz

Author: Sam Barba
Created 06/11/2022
"""

from graphviz import Digraph


def plot_critical_path_diagram(df, duration):
	# 1. Set up global attributes

	lbl = f'Critical path diagram\n(critical path duration: {duration})'
	g = Digraph(
		name='critical path diagram',
		graph_attr={'rankdir': 'LR', 'fontsize': '20', 'fontname': 'consolas', 'label': lbl, 'labelloc': 't'},
		node_attr={'style': 'filled,setlinewidth(0)', 'fontname': 'consolas', 'label': '', 'shape': 'rect', 'fillcolor': '#80c0ff'}
	)

	# 2. Create nodes

	for idx, (desc, es, ef, ls, lf, slack, dur) in enumerate(zip(df['DESC'], df['ES'], df['EF'], df['LS'], df['LS'], df['SLACK'], df['DURATION'])):
		lbl = f'{desc}\n(duration {dur})\n\nES, EF: {es}, {ef}\nLS, LF: {ls}, {lf}\nSlack: {slack}'
		g.node(str(idx), label=lbl)

	# 3. Create edges

	for idx, successors in enumerate(df['SUCCESSORS']):
		if successors:
			for s in successors:
				is_critical = df['CRITICAL'][s] == df['CRITICAL'][idx] == 'YES'
				g.edge(str(idx), str(s), color='red' if is_critical else 'black')

	# 4. Render graph

	g.render('critical_path_diagram', view=True, cleanup=True, format='png')
