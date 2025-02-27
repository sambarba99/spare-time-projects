/*
Infix to RPN converter and evaluator

Author: Sam Barba
Created 11/09/2021
*/

#include <cmath>
#include <iostream>
#include <map>
#include <stack>

using std::string;


long double charToLongDouble(const char c) {
	int value = c - '0';
	return static_cast<long double>(value);
}


bool isOperator(const char c) {
	return c == '+' || c == '-' || c == '*' || c == '/' || c == '^';
}


bool isOperand(const char c) {
	return c >= '0' && c <= '9';
}


long double operation(const long double a, const long double b, const char op) {
	switch (op) {
		case '+': return b + a;
		case '-': return b - a;
		case '*': return b * a;
		case '/': return b / a;
		case '^': return pow(b, a);
		default: return -1;
	}
}


string infixToPostfix(const string expression) {
	std::map<char, int> precedence = {{'^', 3}, {'/', 2}, {'*', 2}, {'+', 1}, {'-', 1}};
	std::stack<char> stk;
    string postfix;
  
    for (const char c : expression) {
		if (isOperand(c)) postfix += c;
        else if (c == '(') stk.push('(');
        else if (c == ')') {
            while (stk.top() != '(') {
                postfix += stk.top();
                stk.pop();
            }
            stk.pop();
        } else if (isOperator(c)) {
            while (!stk.empty() && precedence[c] <= precedence[stk.top()]) {
                postfix += stk.top();
                stk.pop();
            }
            stk.push(c);
        }
    }
  
    while (!stk.empty()) {
        postfix += stk.top();
        stk.pop();
    }
  
	return postfix;
}


long double evaluate(const string postfix) {
	long double a, b;
	std::stack<long double> stk;

	for (const char c : postfix) {
		if (isOperand(c)) {
			stk.push(charToLongDouble(c));
		} else if (isOperator(c)) {
			a = stk.top();
			stk.pop();
			b = stk.top();
			stk.pop();
			stk.push(operation(a, b, c));
		}
	}

	return stk.top();
}


int main() {
	string expression;

	while (true) {
		std::cout << "Input an expression e.g. 1+2/3 (or X to exit)\n>>> ";
		std::cin >> expression;

		if (toupper(expression[0]) == 'X') break;
		else {
			string postfix = infixToPostfix(expression);
			long double result = evaluate(postfix);
			std::cout << "Postfix: " << postfix << '\n';
			std::cout << "Result: " << result << "\n\n";
		}
	}

	return 0;
}
