/*
Demo of some different sorting algorithms

Author: Sam Barba
Created 06/09/2021
*/

#include <bits/stdc++.h>
#include <chrono>
#include <iostream>
#include <vector>

using namespace std::chrono;
using std::cout;
using std::setw;
using std::string;
using std::to_string;
using std::vector;

void swap(vector<int> &arr, const int i, const int j) {
	int temp = arr[i];
	arr[i] = arr[j];
	arr[j] = temp;
}

void bubbleSort(vector<int> arr) {
	bool swapped = true;
	int end = arr.size() - 1;

	while (swapped) {
		swapped = false;

		for (int i = 0; i < end; i++) {
			if (arr[i] > arr[i + 1]) {
				swap(arr, i, i + 1);
				swapped = true;
			}
		}

		end--;
	}
}

void cocktailShakerSort(vector<int> arr) {
	bool swapped = true;
	int start = 0, end = arr.size() - 1;

	while (swapped) {
		for (int i = start; i < end; i++) {
			if (arr[i] > arr[i + 1]) {
				swap(arr, i, i + 1);
				swapped = true;
			}
		}

		if (!swapped) break;

		swapped = false;
		end--;

		for (int i = end - 1; i >= start; i--) {
			if (arr[i] > arr[i + 1]) {
				swap(arr, i, i + 1);
				swapped = true;	
			}
		}

		start++;
	}
}

void combSort(vector<int> arr) {
	int gap = arr.size();
	float k = 1.3;
	bool done = false;

	while (!done) {
		gap /= k;
		if (gap <= 1) {
			gap = 1;
			done = true;
		}

		int i = 0;
		while (i + gap < arr.size()) {
			if (arr[i] > arr[i + gap]) {
				swap(arr, i, i + gap);
				done = false;
			}
			i++;
		}
	}
}

void countingSort(vector<int> arr) {
	int maxNum = *max_element(arr.begin(), arr.end());
	vector<int> counts(maxNum + 1, 0);

	for (int i : arr)
		counts[i]++;

	vector<int> output;
	for (int i = 0; i < counts.size(); i++) {
		for (int j = 0; j < counts[i]; j++)
			output.push_back(i);
	}
}

void insertionSort(vector<int> arr) {
	for (int i = 1; i < arr.size(); i++) {
		int key = arr[i];
		int j = i - 1;
		while (j >= 0 && arr[j] > key) {
			arr[j + 1] = arr[j];
			j--;
		}
		arr[j + 1] = key;
	}
}

void mergeSort(vector<int> arr) {
	if (arr.size() < 2) return;

	int mid = arr.size() / 2;
	vector<int> l = {arr.begin(), arr.begin() + mid};
	vector<int> r = {arr.begin() + mid, arr.end()};

	mergeSort(l);  // Sort copy of first half
	mergeSort(r);  // Sort copy of second half

	// Merge sorted halves back into arr
	int i = 0, j = 0;
	while (i + j < arr.size()) {
		if (j == r.size() || (i < l.size() && l[i] < r[j])) {
			arr[i + j] = l[i];
			i++;
		} else {
			arr[i + j] = r[j];
			j++;
		}
	}
}

void quicksort(vector<int> arr) {
	if (arr.size() < 2) return;

	int pivot = arr[arr.size() / 2];
	vector<int> less, equal, greater;
	for (int i : arr) {
		if (i < pivot) less.push_back(i);
		if (i == pivot) equal.push_back(i);
		if (i > pivot) greater.push_back(i);
	}

	quicksort(less);
	quicksort(greater);

	arr.clear();
	arr.insert(arr.end(), less.begin(), less.end());
	arr.insert(arr.end(), equal.begin(), equal.end());
	arr.insert(arr.end(), greater.begin(), greater.end());
}

void radixSort(vector<int> arr) {  // Least Significant Digit
	int max = *max_element(arr.begin(), arr.end());
	int maxDigits = to_string(max).length();

	for (int i = 0; i < maxDigits; i++) {
		vector<vector<int>> buckets(10);

		for (int n : arr) {
			int idx = int(n / (pow(10, i))) % 10;
			buckets[idx].push_back(n);
		}

		arr.clear();
		for (vector<int> b : buckets)
			arr.insert(arr.end(), b.begin(), b.end());
	}
}

void selectionSort(vector<int> arr) {
	for (int i = 0; i < arr.size() - 1; i++) {
		int minIdx = i;
		for (int j = i + 1; j < arr.size(); j++) {
			if (arr[j] < arr[minIdx])
				minIdx = j;
		}
		swap(arr, i, minIdx);
	}
}

void shellSort(vector<int> arr) {
	int gap = arr.size() / 2;

	while (gap) {
		for (int i = gap; i < arr.size(); i++) {
			int key = arr[i];
			while (i >= gap && arr[i - gap] > key) {
				arr[i] = arr[i - gap];
				i -= gap;
			}
			arr[i] = key;
		}
		gap = gap == 2 ? 1 : (gap * 5) / 11;
	}
}

void testFunc(const string funcName, vector<int> arr, void (*func)(vector<int>)) {
	cout << setw(20) << funcName;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	func(arr);
	high_resolution_clock::time_point finish = high_resolution_clock::now();
	auto millis = duration_cast<milliseconds>(finish - start);
	cout << setw(15) << millis.count() << '\n';
}

int main() {
	vector<int> arr(10000);
	generate(arr.begin(), arr.end(), rand);
	cout << setw(20) << "Algorithm" << setw(25) << " Completion time (ms)\n";
	testFunc("bubble sort", arr, bubbleSort);
	testFunc("cocktail shaker sort", arr, cocktailShakerSort);
	testFunc("comb sort", arr, combSort);
	testFunc("counting sort", arr, countingSort);
	testFunc("insertion sort", arr, insertionSort);
	testFunc("merge sort", arr, mergeSort);
	testFunc("quicksort", arr, quicksort);
	testFunc("radix sort", arr, radixSort);
	testFunc("selection sort", arr, selectionSort);
	testFunc("shell sort", arr, shellSort);
}
