//Binary Search and Quicksort
//Autor: Sam Barba
//Created 29/10/2018

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#pragma warning(disable: 4996)

int binarySearch(int arr[], int target, int left, int right) {
	if (left > right) return -1; //not in array

	int mid = (int) floor((left + right) / 2);

	if (target < arr[mid])
		return binarySearch(arr, target, left, mid - 1);

	if (target > arr[mid])
		return binarySearch(arr, target, mid + 1, right);

	return mid;
}

void quicksort(int arr[], int start, int end) {
	if (start < end) {
		int p = partition(arr, start, end);
		quicksort(arr, start, p);
		quicksort(arr, p + 1, end);
	}
}
int partition(int arr[], int start, int end) {
	int pivot = arr[(start + end) / 2];
	int i = start - 1;
	int j = end + 1;
	while (true) {
		while (arr[i] < pivot) {
			i++;
		}
		while (arr[j] > pivot) {
			j--;
		}
		if (i >= j) {
			return j;
		}
		int temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}
}

int main() {
	int arr[] = { 1,2 };
	int target = 1;
	int left = 0;
	int right = sizeof(arr) / sizeof(arr[0]);

	printf("In position %d", binarySearch(arr, target, left, right));

	quicksort(arr, 0, right - 1);

	getchar();

	return 0;
}
