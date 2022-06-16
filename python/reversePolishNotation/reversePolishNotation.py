"""
Infix to RPN converter

Author: Sam Barba
Created 11/09/2021
"""

import tkinter as tk

# Nested dictionary of precedence and left-associativity of elementary operations, together with equivalent lambdas
OPS = {'^': {'prec': 3, 'left-assoc': False, 'calc': lambda op1, op2: op1 ** op2},
	'*': {'prec': 2, 'left-assoc': True, 'calc': lambda op1, op2: op1 * op2},
	'/': {'prec': 2, 'left-assoc': True, 'calc': lambda op1, op2: op1 / op2},
	'+': {'prec': 1, 'left-assoc': True, 'calc': lambda op1, op2: op1 + op2},
	'-': {'prec': 1, 'left-assoc': True, 'calc': lambda op1, op2: op1 - op2}}

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def infix_to_rpn(infix_str):
	op_stack = []
	rpn = []

	for token in infix_str.split():
		if token in OPS.keys():
			while op_stack and op_stack[-1] != '(' \
				and (OPS[op_stack[-1]]['prec'] > OPS[token]['prec']
					or (OPS[op_stack[-1]]['prec'] == OPS[token]['prec'] and OPS[token]['left-assoc'])):

				rpn.append(op_stack.pop())

			op_stack.append(token)

		elif token == '(':
			op_stack.append(token)

		elif token == ')':
			while op_stack[-1] != '(':
				rpn.append(op_stack.pop())
			op_stack.pop()

		else:
			rpn.append(token)

	if op_stack:
		rpn += op_stack[::-1]

	return ' '.join(rpn)

def evaluate_rpn(rpn_str):
	global output_rpn, output_num

	output_rpn.config(text=rpn_str)

	rpn_stack, txt_stack = [], []

	for token in rpn_str.split():
		if token in OPS.keys():
			operand2, operand1 = float(rpn_stack.pop()), float(rpn_stack.pop())
			appnd = OPS[token]['calc'](operand1, operand2)
		else:
			appnd = float(token)

		rpn_stack.append(int(appnd) if appnd % 1 == 0 else appnd)
		txt_stack.append(rpn_stack[:])

	output_num.config(state='normal')
	output_num.delete('1.0', tk.END)
	output_num.insert('1.0', '\n'.join(str(i) for i in txt_stack))
	output_num.tag_add('center', '1.0', tk.END)
	output_num.config(state='disabled')

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Infix to RPN Converter')
	root.config(width=400, height=500, bg='#000045')
	root.eval('tk::PlaceWindow . center')

	enter_exp_lbl = tk.Label(root, text='Enter an infix expression:',
		font='consolas', bg='#000045', fg='white')
	enter_exp_lbl.place(relwidth=0.8, relheight=0.05, relx=0.5, rely=0.07, anchor='center')

	entry_box = tk.Entry(root, font='consolas', justify='center')
	entry_box.place(relwidth=0.8, relheight=0.06, relx=0.5, rely=0.13, anchor='center')

	button = tk.Button(root, text='Convert', font='consolas',
		command=lambda: evaluate_rpn(infix_to_rpn(entry_box.get())))
	button.place(relwidth=0.3, relheight=0.08, relx=0.5, rely=0.24, anchor='center')

	rpn_result_lbl = tk.Label(root, text='In RPN:', font='consolas', bg='#000045', fg='white')
	rpn_result_lbl.place(relwidth=0.9, relheight=0.04, relx=0.5, rely=0.32, anchor='center')

	output_rpn = tk.Label(root, font='consolas', bg='white')
	output_rpn.place(relwidth=0.9, relheight=0.06, relx=0.5, rely=0.39, anchor='center')

	evaluation_lbl = tk.Label(root, text='Evaluation:', font='consolas', bg='#000045', fg='white')
	evaluation_lbl.place(relwidth=0.9, relheight=0.04, relx=0.5, rely=0.46, anchor='center')

	output_num = tk.Text(root, bg='white', font='consolas', state='disabled')
	output_num.tag_configure('center', justify='center')
	output_num.place(relwidth=0.9, relheight=0.45, relx=0.5, rely=0.72, anchor='center')

	root.mainloop()
