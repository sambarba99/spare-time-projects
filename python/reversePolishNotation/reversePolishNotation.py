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

def infixToRPN(infixStr):
	opStack = []
	rpn = []

	for token in infixStr.split():
		if token in OPS.keys():
			while (opStack and opStack[-1] != "("
				and (OPS[opStack[-1]]["prec"] > OPS[token]["prec"]
					or (OPS[opStack[-1]]["prec"] == OPS[token]["prec"]
						and OPS[token]["left-assoc"]))):

				rpn.append(opStack.pop())

			opStack.append(token)

		elif token == "(":
			opStack.append(token)

		elif token == ")":
			while opStack[-1] != "(":
				rpn.append(opStack.pop())
			opStack.pop()

		else:
			rpn.append(token)

	if opStack:
		rpn += opStack[::-1]

	return " ".join(rpn)

def evaluateRPN(rpnStr):
	rpnStack = []

	for token in rpnStr.split():
		if token in OPS.keys():
			operand2, operand1 = float(rpnStack.pop()), float(rpnStack.pop())
			rpnStack.append(OPS[token]["calc"](operand1, operand2))
		else:
			rpnStack.append(token)
		print(rpnStack)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	rpnStr = infixToRPN(input("Enter an infix expression: "))

	print("\nIn RPN:", rpnStr, "\n")

	evaluateRPN(rpnStr)

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
