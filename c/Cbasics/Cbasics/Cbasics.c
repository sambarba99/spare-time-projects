//C basics
//Author: Sam Barba
//Created 29/10/2018

//always have these
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#pragma warning(disable: 4996)

void halfAdder(bool A, bool B, bool* S, bool* C) { //* here means that we will assign the result to a given variable when we call this function
	*S = A ^ B; //XOR to get the result, and assign it (*) to 'sum'
	*C = A && B; //AND to get the carry, and assign it to 'carry'
}
void fullAdder(bool A, bool B, bool inC, bool* S, bool* outC) {
	bool S1, C1, S2, C2;

	halfAdder(A, B, &S1, &C1); //taking A and B as inputs, and assigning the sum to sum1, and the carry bit to carry1
	halfAdder(S1, inC, &S2, &C2);

	*S = S2;
	*outC = C1 || C2;
}

void print(char* name, bool b) { //char* is how to handle strings of text
	if (b) printf("%s = 1\n", name);
	else printf("%s = 0\n", name);
}
int main() {
	printf("FULL ADDER\n\n");

	bool A[4] = { false, true, true, false }; //0110
	bool B[4] = { true, false, true, false }; //1010

	bool S[4]; //sum
	bool C = false; //carry bit which serves as first input, starting as 0

	for (int i = 3; i >= 0; i--) {
		//taking A and B as inputs, outputting sum to S, and carry bit to C
		//so C keeps getting updated
		fullAdder(A[i], B[i], C, &S[i], &C);
	}
	//when loop stops, the final C is the carry bit

	print("S[0]", S[0]);
	
	//################################################################################################################################
	
	printf("\n\nLOOPING\n\n");
	
	int num = 0;

	//outputs 5678910
	for (num = 5; num <= 10; num++)
		printf("%d", num);
	printf("\n");

	num = 0;
	//outputs 02468
	while (num < 9) {
		printf("%d", num);
		num += 2;
	}
	printf("\n");

	num = 1;
	//outputs 123
	do {
		printf("%d", num);
		num++;
	} while (num < 4);

	//################################################################################################################################

	printf("\n\n\nCHARS AND CHAR ARRAYS\n\n");

	char textArr[] = { "COMP" }; //initialise string (array of chars) - contains "COMP" and '\0', so size = 5
	char* textStr = &textArr[0]; //set string 'textStr' to the address of the first char in the array - & means address of

	for (int i = 0; i < 4; i++)
		printf("textArr[%d] = %c\n", i, textArr[i]);

	printf("\ntextArr = %s\n", textArr); //printing string in form of array
	printf("textStr = %s", textStr); //printing string in form of string

	//################################################################################################################################
	
	printf("\n\n\nCALCULATOR\n\n");
	
	float previousNumber = 0.0, operand1, operand2;
	char operator;

	while (true) { //runs FOREVER (unless input[0] = 'q')
		char input[256]; //intilialises an array with 256 elements
		scanf("%[^\n]%*c", input); //reads up until the newline character (enter is pressed) and stores it in the arrary input

		if (input[0] == 'p') {
			char temp;
			sscanf(input, "%c %c %f", &temp, &operator, &operand2); //reads in the first character, second char (operator) and the second number
			operand1 = previousNumber;
		}
		else if (input[0] == 'q')
			return 0; //quit
		else
			sscanf(input, "%f %c %f", &operand1, &operator, &operand2);

		switch (operator) {
			case '+': previousNumber = operand1 + operand2; break;
			case '-': previousNumber = operand1 - operand2; break;
			case '*': previousNumber = operand1 * operand2; break;
			case '/': previousNumber = operand1 / operand2; break;
		}
		printf("ANS=%f\n", previousNumber);
	}

	getchar(); //waits for user to enter a char

	return 0; //to exit program
}