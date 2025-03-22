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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

using std::cout;
using std::setw;
using std::string;
using std::vector;

//                              0    1   2  3  4  5  6   7   8   9
const vector<int> DURATIONS = {120, 60, 15, 3, 3, 7, 7, 25, 60, 90};
const vector<vector<int>> PREDECESSORS = {{-1}, {0}, {0}, {1, 2}, {1, 2}, {3}, {4}, {2, 5}, {2, 6, 7}, {8}};


string vec_to_string(const vector<int>& vec) {
	if (vec.empty() || vec[0] == -1)
		return "None";

	string res = "";
	for (int i : vec)
		res += " " + std::to_string(i);
	return res;
}


void cpa() {
	int num_tasks = DURATIONS.size();

	// 1. Forward pass

	// Early start, early finish
	vector<int> es(num_tasks, 0);
	vector<int> ef(num_tasks, 0);

	for (int i = 0; i < num_tasks; i++) {
		if (PREDECESSORS[i][0] != -1) {
			int max_ef = INT_MIN;
			for (int task_code : PREDECESSORS[i])
				if (ef[task_code] > max_ef)
					max_ef = ef[task_code];
			es[i] = max_ef;
		}
		ef[i] = es[i] + DURATIONS[i];
	}

	// 2. Backward pass

	// Late start, late finish
	vector<int> ls(num_tasks, 0);
	vector<int> lf(num_tasks, 0);
	vector<vector<int>> successors(num_tasks);

	for (int i = num_tasks - 1; i > -1; i--)
		if (PREDECESSORS[i][0] != -1)
			for (int task_code : PREDECESSORS[i])
				successors[task_code].emplace_back(i);

	for (int i = 0; i < num_tasks; i++)
		sort(successors[i].begin(), successors[i].end());

	for (int i = num_tasks - 1; i > -1; i--) {
		if (successors[i].empty()) {
			lf[i] = *max_element(ef.begin(), ef.end());
		} else {
			int min_ls = INT_MAX;
			for (int task_code : successors[i])
				if (ls[task_code] < min_ls)
					min_ls = ls[task_code];
			lf[i] = min_ls;
		}
		ls[i] = lf[i] - DURATIONS[i];
	}

	// 3. Compute slack

	vector<int> slack(num_tasks);
	vector<string> is_critical(num_tasks);
	for (int i = 0; i < num_tasks; i++) {
		slack[i] = ls[i] - es[i];
		is_critical[i] = slack[i] == 0 ? "Yes" : "No";
	}

	// 4. Print task data

	cout << setw(6) << "Code" << setw(10) << "Duration" << setw(14) << "Predecessors";
	cout << setw(5) << "ES" << setw(5) << "EF" << setw(12) << "Successors";
	cout << setw(5) << "LS" << setw(5) << "LF" << setw(7) << "Slack" << setw(10) << "Critical" << '\n';
	for (int task_code = 0; task_code < num_tasks; task_code++) {
		cout << setw(6) << task_code << setw(10) << DURATIONS[task_code] << setw(14) << vec_to_string(PREDECESSORS[task_code]);
		cout << setw(5) << es[task_code] << setw(5) << ef[task_code] << setw(12) << vec_to_string(successors[task_code]);
		cout << setw(5) << ls[task_code] << setw(5) << lf[task_code] << setw(7) << slack[task_code] << setw(10) << is_critical[task_code] << '\n';
	}

	// 5. Print critical path and duration

	vector<int> critical_path;
	int critical_duration = 0;

	for (int i = 0; i < num_tasks; i++)
		if (slack[i] == 0) {
			critical_path.emplace_back(i);  // i = task code
			critical_duration += DURATIONS[i];
		}

	cout << "\nCritical path (duration " << critical_duration << "): ";
	for (int i : critical_path)
		cout << i << " ";
}


int main() {
	cpa();
	return 0;
}
