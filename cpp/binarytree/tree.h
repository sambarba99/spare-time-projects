#ifndef TREE
#define TREE

#include <algorithm>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

using std::cout;
using std::max;
using std::queue;
using std::string;
using std::vector;

class Tree {
	public:
		string data;
		Tree* leftChild;
		Tree* rightChild;

		Tree(const string dataParam) {
			data = dataParam;
			leftChild = NULL;
			rightChild = NULL;
		}

		void insert(const string newData) {
			if (data == newData) return;  // No duplicates allowed
			else if (newData.compare(data) == -1) {
				if (leftChild)
					leftChild->insert(newData);
				else
					leftChild = new Tree(newData);
			} else {
				if (rightChild)
					rightChild->insert(newData);
				else
					rightChild = new Tree(newData);
			}
		}

		int getHeight() {
			if (!this) return 0;
			return max(leftChild->getHeight(), rightChild->getHeight()) + 1;
		}

		vector<string> inOrderTraversal() {
			if (!this) return vector<string>();

			vector<string> leftTraversal = leftChild->inOrderTraversal();
			vector<string> rightTraversal = rightChild->inOrderTraversal();
			vector<string> result;
			result.insert(result.end(), leftTraversal.begin(), leftTraversal.end());
			result.push_back(data);
			result.insert(result.end(), rightTraversal.begin(), rightTraversal.end());
			return result;
		}

		vector<string> preOrderTraversal() {
			if (!this) return vector<string>();

			vector<string> leftTraversal = leftChild->preOrderTraversal();
			vector<string> rightTraversal = rightChild->preOrderTraversal();
			vector<string> result;
			result.push_back(data);
			result.insert(result.end(), leftTraversal.begin(), leftTraversal.end());
			result.insert(result.end(), rightTraversal.begin(), rightTraversal.end());
			return result;
		}

		vector<string> postOrderTraversal() {
			if (!this) return vector<string>();

			vector<string> leftTraversal = leftChild->postOrderTraversal();
			vector<string> rightTraversal = rightChild->postOrderTraversal();
			vector<string> result;
			result.insert(result.end(), leftTraversal.begin(), leftTraversal.end());
			result.insert(result.end(), rightTraversal.begin(), rightTraversal.end());
			result.push_back(data);
			return result;
		}

		vector<string> breadthFirstTraversal() {
			vector<string> result({data});
			queue<Tree*> q;
			q.push(this);

			while (!q.empty()) {
				Tree* front = q.front(); 
				q.pop();

				if (front->leftChild) {
					q.push(front->leftChild);
					result.push_back(front->leftChild->data);
				}
				if (front->rightChild) {
					q.push(front->rightChild);
					result.push_back(front->rightChild->data);
				}
			}

			return result;
		}

		bool isBST(string prev = "0") {
			if (!this) return true;

			if (!leftChild->isBST(prev))
				return false;
			// Handle equal-valued nodes
			if (data.compare(prev) == -1)
				return false;
			prev = data;
			return rightChild->isBST(prev);
		}

		bool isBalanced() {
			if (!this) return true;

			int leftHeight = leftChild->getHeight();
			int rightHeight = rightChild->getHeight();
			return abs(leftHeight - rightHeight) <= 1
				&& leftChild->isBalanced()
				&& rightChild->isBalanced();
		}

		void display(const bool isLeft = false, const string prefix = "") {
			if (!this) return;

			cout << prefix << (isLeft ? "├─" : "└─") << data << '\n';
			leftChild->display(true, prefix + (isLeft ? "│  " : "   "));
			rightChild->display(false, prefix + (isLeft ? "│  " : "   "));
		}
};

#endif
