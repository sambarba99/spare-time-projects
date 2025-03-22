/*
Infix to RPN converter and evaluator

Author: Sam Barba
Created 11/09/2021
*/

#include <cmath>
#include <iostream>
#include <map>
#include <regex>
#include <stack>

using std::string;
using std::vector;


bool isOperator(const string& c) {
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


vector<string> infixToPostfix(const string& expression) {
	// Split expression on this regex in order to read operators/operands
	std::regex re(R"(\d+|[+\-*/^()])");  // Numbers, +, -, *, /, ^, (, )
	std::regex_token_iterator it(expression.begin(), expression.end(), re);
	std::sregex_token_iterator end;
	vector<string> tokens;
	while (it != end) {
		tokens.push_back(it->str());
		it++;
	}

	std::map<char, int> precedence = {{'^', 3}, {'/', 2}, {'*', 2}, {'+', 1}, {'-', 1}};
	std::stack<string> stk;
	vector<string> postfix;

    for (size_t i = 0; i < tokens.size(); i++) {
        string t = tokens[i];

        if (t == "(") {
            stk.push(t);
        } else if (t == ")") {
            while (!stk.empty() && stk.top() != "(") {
                postfix.push_back(stk.top());
                stk.pop();
            }
            stk.pop();  // Remove '('
        } else if (t == "-") {
            // Check if '-' is a unary minus (negative number)
            if (i == 0 || tokens[i - 1] == "(" || isOperator(tokens[i - 1])) {
                // It's a negative number, merge with next token
                if (i + 1 < tokens.size() && isdigit(tokens[i + 1][0])) {
                    postfix.push_back("-" + tokens[i + 1]);
                    i++;  // Skip next token (the number) since we merged it
                }
            } else {
                // Normal subtraction operator
                while (!stk.empty() && precedence[t[0]] <= precedence[stk.top()[0]]) {
                    postfix.push_back(stk.top());
                    stk.pop();
                }
                stk.push(t);
            }
        } else if (isOperator(t)) {
            while (!stk.empty() && precedence[t[0]] <= precedence[stk.top()[0]]) {
                postfix.push_back(stk.top());
                stk.pop();
            }
            stk.push(t);
        } else {
            // Operand (number)
            postfix.push_back(t);
        }
    }

    while (!stk.empty()) {
        postfix.push_back(stk.top());
        stk.pop();
    }

	return postfix;
}


long double evaluate(const vector<string>& postfix) {
	long double a, b;
	std::stack<long double> operands;

	for (const string& s : postfix)
		if (isOperator(s)) {
			a = operands.top();
			operands.pop();
			b = operands.top();
			operands.pop();
			operands.push(operation(a, b, s[0]));
		} else if (s != "(" && s != ")") {
			// Operand
			operands.push(std::stold(s));
		}

	return operands.top();
}


int main() {
	string expression;

	while (true) {
		std::cout << "Input an expression e.g. 1+2/3 (or X to exit)\n>>> ";
		std::cin >> expression;

		if (toupper(expression[0]) == 'X') {
			break;
		} else {
			vector<string> postfix = infixToPostfix(expression);
			std::cout << "Postfix:";
			for (const string& s : postfix)
				std::cout << ' ' << s;
			std::cout << '\n';
			long double result = evaluate(postfix);
			std::cout << "Result: " << result << "\n\n";
		}
	}

	return 0;
}
