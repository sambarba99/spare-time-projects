/*
Infix to RPN converter and evaluator

Author: Sam Barba
Created 11/09/2021
*/

#include <cmath>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stack>
#include <unordered_map>

using std::stack;
using std::string;
using std::vector;


std::unordered_map<char, int> OPERATOR_PRECEDENCE = {{'^', 3}, {'/', 2}, {'*', 2}, {'+', 1}, {'-', 1}};


bool is_operator(const string& c) {
	return c == "+" || c == "-" || c == "*" || c == "/" || c == "^";
}


long double operation(const long double a, const long double b, const char op) {
	switch (op) {
		case '+': return b + a;
		case '-': return b - a;
		case '*': return b * a;
		case '/': return b / a;
		case '^': return pow(b, a);
		default: throw std::exception();
	}
}


vector<string> infix_to_postfix(const string& expression) {
	// Split expression on this regex in order to read operators/operands
	std::regex re(R"(\d+(\.\d+)?|[+\-*/^()])");  // Numbers (ints or decimals), +, -, *, /, ^, (, )
	std::sregex_token_iterator it(expression.begin(), expression.end(), re);
	std::sregex_token_iterator end;
	vector<string> tokens;
	while (it != end) {
		tokens.emplace_back(it->str());
		it++;
	}

	stack<string> stk;
	vector<string> postfix;

	for (int i = 0; i < tokens.size(); i++) {
		string t = tokens[i];

		if (t == "(") {
			stk.push(t);
		} else if (t == ")") {
			while (!stk.empty() && stk.top() != "(") {
				postfix.emplace_back(stk.top());
				stk.pop();
			}
			stk.pop();  // Remove '('
		} else if (t == "-") {
			// Check if '-' is a unary minus (negative number)
			if (i == 0 || tokens[i - 1] == "(" || is_operator(tokens[i - 1])) {
				// If negative number, merge with next token
				if (i + 1 < tokens.size() && isdigit(tokens[i + 1][0])) {
					postfix.emplace_back("-" + tokens[i + 1]);
					i++;  // Skip next token (the number) as we've merged it
				}
			} else {
				// Normal subtraction operator
				while (!stk.empty() && OPERATOR_PRECEDENCE[t[0]] <= OPERATOR_PRECEDENCE[stk.top()[0]]) {
					postfix.emplace_back(stk.top());
					stk.pop();
				}
				stk.push(t);
			}
		} else if (is_operator(t)) {
			while (!stk.empty() && OPERATOR_PRECEDENCE[t[0]] <= OPERATOR_PRECEDENCE[stk.top()[0]]) {
				postfix.emplace_back(stk.top());
				stk.pop();
			}
			stk.push(t);
		} else {
			// Operand (number)
			postfix.emplace_back(t);
		}
	}

	while (!stk.empty()) {
		postfix.emplace_back(stk.top());
		stk.pop();
	}

	return postfix;
}


void print_operand_stack(const stack<long double>& stk) {
	stack<long double> stk_copy = stk;
	vector<long double> elements;  // To store elements in bottom-to-top order

	while (!stk_copy.empty()) {
		elements.emplace_back(stk_copy.top());
		stk_copy.pop();
	}

	std::cout << "Operand stack:";
	for (int i = elements.size() - 1; i >= 0; i--)
		std::cout << ' ' << elements[i];
	std::cout << '\n';
}


long double evaluate(const vector<string>& postfix) {
	long double a, b;
	stack<long double> operands;

	for (const string& s : postfix) {
		if (is_operator(s)) {
			std::cout << "Found operator: " << s << '\n';
			a = operands.top();
			operands.pop();
			b = operands.top();
			operands.pop();
			operands.push(operation(a, b, s[0]));
		} else if (s != "(" && s != ")") {
			// Operand
			operands.push(std::stold(s));
		}
		print_operand_stack(operands);
	}

	return operands.top();
}


int main() {
	string expression;

	while (true) {
		std::cout << "Input an expression e.g. 1 + 2/3 (or X to exit)\n>>> ";
		std::getline(std::cin, expression);
		expression.erase(remove(expression.begin(), expression.end(), ' '), expression.end());

		if (toupper(expression[0]) == 'X' || expression == "") {
			break;
		} else {
			try {
				vector<string> postfix = infix_to_postfix(expression);
				std::cout << "Postfix:";
				for (const string& s : postfix)
					std::cout << ' ' << s;
				std::cout << '\n';
				std::ostringstream result_str;
				result_str << std::setprecision(20) << evaluate(postfix);
				std::cout << "Result: " << result_str.str() << "\n\n";
			} catch (...) {
				std::cout << "Error: bad expression\n\n";
			}
		}
	}

	return 0;
}
