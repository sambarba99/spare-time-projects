"""
Infix to RPN converter and evaluator

Author: Sam Barba
Created 11/09/2021
"""

import re
import tkinter as tk


NUMBER_REG = r'-?\d+(\.\d+)?'
OPEN_PAREN_REG = r'\('
CLOSED_PAREN_REG = r'\)'
EXPRESSION_REG = r'^open*num(closed*[+\-*/^]open*num)*closed*$' \
	.replace('num', NUMBER_REG) \
	.replace('open', OPEN_PAREN_REG) \
	.replace('closed', CLOSED_PAREN_REG)

# For removing redundant parentheses later, e.g. (3.14) -> 3.14, (((2))) -> 2
SINGLE_NUM_WITHIN_PAREN = 'open+(num)closed+' \
	.replace('num', NUMBER_REG) \
	.replace('open', OPEN_PAREN_REG) \
	.replace('closed', CLOSED_PAREN_REG)

# For adding spaces between operators/operands/parentheses so expression can be split later
OPERATORS_AND_PAREN_REG = r'[+\-*/^()]'
SEPARATION_REG = '(op)(num|op)' \
	.replace('op', OPERATORS_AND_PAREN_REG) \
	.replace('num', NUMBER_REG)

# Nested dictionary of precedence and left-associativity of elementary operations, together with equivalent lambdas
OPS = {
	'^': {'prec': 3, 'left-assoc': False, 'calc': lambda op1, op2: op1 ** op2},
	'*': {'prec': 2, 'left-assoc': True, 'calc': lambda op1, op2: op1 * op2},
	'/': {'prec': 2, 'left-assoc': True, 'calc': lambda op1, op2: op1 / op2},
	'+': {'prec': 1, 'left-assoc': True, 'calc': lambda op1, op2: op1 + op2},
	'-': {'prec': 1, 'left-assoc': True, 'calc': lambda op1, op2: op1 - op2}
}


def convert_and_solve(*_):
	expression = sv.get().strip()

	if not is_valid(expression):
		output_rpn.config(text='')

		output_num.config(state='normal')
		output_num.delete('1.0', 'end')
		output_num.insert('1.0', 'Bad expression!'
			'\n1. Must be numeric'
			'\n2. Parentheses must balance')
		output_num.tag_add('center', '1.0', 'end')
		output_num.config(state='disabled')
		return

	# Remove redundant parentheses around numbers
	# Group 1 (\g<1>) matches the only group () in the regex (numbers)
	expression = re.sub(SINGLE_NUM_WITHIN_PAREN, r'\g<1>', expression)

	# Add spaces between operators/operands/parentheses
	# Group 1 (\g<1>) matches operators or parentheses, group 2 matches this as well as numbers
	expression = re.sub(SEPARATION_REG, r' \g<1> \g<2> ', expression)
	expression = expression.replace('  ', ' ')  # Remove any double spaces

	rpn = infix_to_rpn(expression)
	output_rpn.config(text=rpn)
	try:
		error_msg = evaluate_rpn(rpn)
	except Exception as e:
		error_msg = e.args[-1]

	if error_msg:
		output_num.config(state='normal')
		output_num.delete('1.0', 'end')
		output_num.insert('1.0', f'Bad expression! {error_msg}')
		output_num.tag_add('center', '1.0', 'end')
		output_num.config(state='disabled')


def is_valid(expression):
	def parentheses_balanced(expression):
		n = 0
		for c in expression:
			if c == '(': n += 1
			elif c == ')': n -= 1

			if n < 0: return False
		return n == 0


	return parentheses_balanced(expression) and re.search(EXPRESSION_REG, expression)


def infix_to_rpn(expression):
	op_stack, rpn = [], []

	for token in expression.split():
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

	return ' '.join(str(i if i in '+-*/^() ' else (int(float(i)) if float(i) % 1 == 0 else float(i))) for i in rpn)


def evaluate_rpn(rpn_expression):
	rpn_stack, txt_stack = [], []

	for token in rpn_expression.split():
		if token in OPS.keys():
			operand2, operand1 = float(rpn_stack.pop()), float(rpn_stack.pop())
			if operand1 == operand2 == 0 and token == '^':
				return 'Results in 0 ^ 0'
			if operand2 == 0 and token == '/':
				return 'Results in division by 0'
			appnd = OPS[token]['calc'](operand1, operand2)
			txt_stack.append(f'Found operator: {token}')
		else:
			appnd = float(token)

		rpn_stack.append(int(appnd) if appnd % 1 == 0 else appnd)
		txt_stack.append(rpn_stack[:])

	txt_stack[-1] = f'Final answer = {txt_stack[-1][0]}'

	output_num.config(state='normal')
	output_num.delete('1.0', 'end')
	output_num.insert('1.0', '\n'.join(map(str, txt_stack)))
	output_num.tag_add('center', '1.0', 'end')
	output_num.config(state='disabled')

	return None


if __name__ == '__main__':
	root = tk.Tk()
	root.title('Infix to RPN Converter')
	root.config(width=500, height=600, background='#101010')
	root.eval('tk::PlaceWindow . center')
	root.resizable(False, False)

	enter_exp_lbl = tk.Label(root, text='Enter an infix expression:',
		font='consolas', background='#101010', foreground='white')

	sv = tk.StringVar(value='1+((23-4.5)*6/7)^0.89')
	sv.trace_add(mode='write', callback=convert_and_solve)
	entry_box = tk.Entry(root, textvariable=sv, font='consolas', justify='center')

	rpn_result_lbl = tk.Label(root, text='In RPN:', font='consolas', background='#101010', foreground='white')
	output_rpn = tk.Label(root, font='consolas', background='white')
	evaluation_lbl = tk.Label(root, text='Evaluation:', font='consolas', background='#101010', foreground='white')
	output_num = tk.Text(root, background='white', font='consolas', state='disabled')
	output_num.tag_configure('center', justify='center')

	enter_exp_lbl.place(width=400, height=24, relx=0.5, y=42, anchor='center')
	entry_box.place(width=450, height=30, relx=0.5, y=72, anchor='center')
	rpn_result_lbl.place(width=450, height=24, relx=0.5, y=114, anchor='center')
	output_rpn.place(width=450, height=30, relx=0.5, y=144, anchor='center')
	evaluation_lbl.place(width=450, height=24, relx=0.5, y=186, anchor='center')
	output_num.place(width=450, height=360, relx=0.5, y=384, anchor='center')

	convert_and_solve()

	root.mainloop()
