# RPN demo
# Author: Sam Barba
# Created 11/09/2021

# Nested dictionary of precedence and left-associativity of elementary operations, together with equivalent lambdas
OPS = {"^": {"prec": 3, "left-assoc": False, "calc": lambda op1, op2: op1 ** op2},
	"*": {"prec": 2, "left-assoc": True, "calc": lambda op1, op2: op1 * op2},
	"/": {"prec": 2, "left-assoc": True, "calc": lambda op1, op2: op1 / op2},
	"+": {"prec": 1, "left-assoc": True, "calc": lambda op1, op2: op1 + op2},
	"-": {"prec": 1, "left-assoc": True, "calc": lambda op1, op2: op1 - op2}}

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def infix_to_rpn(infix_str):
	op_stack = []
	rpn = []

	for token in infix_str.split():
		if token in OPS.keys():
			while (op_stack and op_stack[-1] != "("
				and (OPS[op_stack[-1]]["prec"] > OPS[token]["prec"]
					or (OPS[op_stack[-1]]["prec"] == OPS[token]["prec"]
						and OPS[token]["left-assoc"]))):

				rpn.append(op_stack.pop())

			op_stack.append(token)

		elif token == "(":
			op_stack.append(token)

		elif token == ")":
			while op_stack[-1] != "(":
				rpn.append(op_stack.pop())
			op_stack.pop()

		else:
			rpn.append(token)

	if op_stack:
		rpn += op_stack[::-1]

	return " ".join(rpn)

def evaluate_rpn(rpn_str):
	rpn_stack = []

	for token in rpn_str.split():
		if token in OPS.keys():
			operand2, operand1 = float(rpn_stack.pop()), float(rpn_stack.pop())
			rpn_stack.append(OPS[token]["calc"](operand1, operand2))
		else:
			rpn_stack.append(token)
		print(rpn_stack)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

rpn_str = infix_to_rpn(input("Enter an infix expression: "))

print("\nIn RPN:", rpn_str, "\n")

evaluate_rpn(rpn_str)
