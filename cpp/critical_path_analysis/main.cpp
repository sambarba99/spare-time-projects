/*
Critical Path Analysis demo on this task data:
Example: engineering project
+------------------+------+----------+--------------+
| Task description | Code | Duration | Predecessors |
+------------------+------+----------+--------------+
|     Analysis     |  0   |   120    |     None     |
|      Design      |  1   |    60    |      0       |
|      Layout      |  2   |    15    |      0       |
| Request material |  3   |    3     |     1,2      |
|  Request parts   |  4   |    3     |     1,2      |
| Receive material |  5   |    7     |      3       |
|  Receive parts   |  6   |    7     |      4       |
|   Fabrication    |  7   |    25    |     2,5      |
|     Assembly     |  8   |    60    |    2,6,7     |
|     Testing      |  9   |    90    |      8       |
+------------------+------+----------+--------------+

Author: Sam Barba
Created 15/12/2021
*/

#include <bits/stdc++.h>
#include <iostream>
#include <vector>

using std::cin;
using std::cout;
using std::setw;
using std::string;
using std::to_string;
using std::vector;

//                              0    1   2  3  4  5  6   7   8   9
const vector<int> DURATIONS = {120, 60, 15, 3, 3, 7, 7, 25, 60, 90};
const vector<vector<int>> PREDECESSORS = {{-1}, {0}, {0}, {1, 2}, {1, 2}, {3}, {4}, {2, 5}, {2, 6, 7}, {8}};

string vecToString(const vector<int> vec) {
	if (vec.empty()) return "None";
	if (vec[0] == -1) return "None";

	string res = "";
	for (int i : vec)
		res += " " + to_string(i);
	return res;
}

void cpa() {
	int nTasks = DURATIONS.size();

	// 1. Forward pass

	// Early start, early finish
	vector<int> es(nTasks, 0);
	vector<int> ef(nTasks, 0);
	
	for (int i = 0; i < nTasks; i++) {
		if (PREDECESSORS[i][0] != -1) {
			int maxEf = 0;
			for (int taskCode : PREDECESSORS[i])
				if (ef[taskCode] > maxEf) maxEf = ef[taskCode];
			es[i] = maxEf;
		}
		ef[i] = es[i] + DURATIONS[i];
	}

	// 2. Backward pass

	// Late start, late finish
	vector<int> ls(nTasks, 0);
	vector<int> lf(nTasks, 0);
	vector<vector<int>> successors(nTasks);

	for (int i = nTasks - 1; i > -1; i--) {
		if (PREDECESSORS[i][0] != -1) {
			for (int taskCode : PREDECESSORS[i])
				successors[taskCode].push_back(i);
		}
	}

	for (int i = 0; i < nTasks; i++)
		sort(successors[i].begin(), successors[i].end());
	
	for (int i = nTasks - 1; i > -1; i--) {
		if (successors[i].empty())
			lf[i] = *max_element(ef.begin(), ef.end());
		else {
			int minLs = INT_MAX;
			for (int taskCode : successors[i])
				if (ls[taskCode] < minLs) minLs = ls[taskCode];
			lf[i] = minLs;
		}
		ls[i] = lf[i] - DURATIONS[i];
	}

	// 3. Compute slack
	
	vector<int> slack;
	vector<string> isCritical;
	for (int i = 0; i < nTasks; i++) {
		int thisSlack = ls[i] - es[i];
		slack.push_back(thisSlack);
		isCritical.push_back(thisSlack == 0 ? "Yes" : "No");
	}

	// 4. Print task data

	cout << setw(6) << "Code" << setw(10) << "Duration" << setw(14) << "Predecessors";
	cout << setw(5) << "ES" << setw(5) << "EF" << setw(12) << "Successors";
	cout << setw(5) << "LS" << setw(5) << "LF" << setw(7) << "Slack" << setw(10) << "Critical" << '\n';
	for (int taskCode = 0; taskCode < nTasks; taskCode++) {
		cout << setw(6) << taskCode << setw(10) << DURATIONS[taskCode] << setw(14) << vecToString(PREDECESSORS[taskCode]);
		cout << setw(5) << es[taskCode] << setw(5) << ef[taskCode] << setw(12) << vecToString(successors[taskCode]);
		cout << setw(5) << ls[taskCode] << setw(5) << lf[taskCode] << setw(7) << slack[taskCode] << setw(10) << isCritical[taskCode] << '\n';
	}

	// 5. Print critical path and duration

	vector<int> criticalPath;
	int criticalDuration = 0;

	for (int i = 0; i < nTasks; i++) {
		if (slack[i] == 0) {
			criticalPath.push_back(i);  // i = task code
			criticalDuration += DURATIONS[i];
		}
	}

	cout << "\nCritical path (duration " << criticalDuration << "): ";
	for (int i : criticalPath)
		cout << i << " ";
}

int main() {
	cpa();
}
