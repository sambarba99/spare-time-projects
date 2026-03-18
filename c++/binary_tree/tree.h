#ifndef TREE
#define TREE

#include <queue>

using std::string;
using std::vector;


class Tree {
	public:
		string data;
		Tree* left_child;
		Tree* right_child;

		Tree(const string& data) {
			this->data = data;
			left_child = NULL;
			right_child = NULL;
		}

		void insert(const string& new_data) {
			if (data == new_data) {
				return;  // No duplicates allowed
			} else if (new_data.compare(data) == -1) {
				if (left_child)
					left_child->insert(new_data);
				else
					left_child = new Tree(new_data);
			} else {
				if (right_child)
					right_child->insert(new_data);
				else
					right_child = new Tree(new_data);
			}
		}

		int get_height() {
			if (!this)
				return 0;
			return std::max(left_child->get_height(), right_child->get_height()) + 1;
		}

		vector<string> in_order_traversal() {
			if (!this)
				return vector<string>();

			vector<string> left_traversal = left_child->in_order_traversal();
			vector<string> right_traversal = right_child->in_order_traversal();
			vector<string> result;
			result.insert(result.end(), left_traversal.begin(), left_traversal.end());
			result.emplace_back(data);
			result.insert(result.end(), right_traversal.begin(), right_traversal.end());
			return result;
		}

		vector<string> pre_order_traversal() {
			if (!this)
				return vector<string>();

			vector<string> left_traversal = left_child->pre_order_traversal();
			vector<string> right_traversal = right_child->pre_order_traversal();
			vector<string> result;
			result.emplace_back(data);
			result.insert(result.end(), left_traversal.begin(), left_traversal.end());
			result.insert(result.end(), right_traversal.begin(), right_traversal.end());
			return result;
		}

		vector<string> post_order_traversal() {
			if (!this)
				return vector<string>();

			vector<string> left_traversal = left_child->post_order_traversal();
			vector<string> right_traversal = right_child->post_order_traversal();
			vector<string> result;
			result.insert(result.end(), left_traversal.begin(), left_traversal.end());
			result.insert(result.end(), right_traversal.begin(), right_traversal.end());
			result.emplace_back(data);
			return result;
		}

		vector<string> breadth_first_traversal() {
			vector<string> result({data});
			std::queue<Tree*> q;
			q.push(this);

			while (!q.empty()) {
				Tree* front = q.front();
				q.pop();

				if (front->left_child) {
					q.push(front->left_child);
					result.emplace_back(front->left_child->data);
				}
				if (front->right_child) {
					q.push(front->right_child);
					result.emplace_back(front->right_child->data);
				}
			}

			return result;
		}

		bool is_bst(string prev = "0") {
			if (!this)
				return true;
			if (!left_child->is_bst(prev))
				return false;
			// Handle equal-valued nodes
			if (data.compare(prev) == -1)
				return false;
			prev = data;
			return right_child->is_bst(prev);
		}

		bool is_balanced() {
			if (!this)
				return true;

			int left_height = left_child->get_height();
			int right_height = right_child->get_height();
			return abs(left_height - right_height) <= 1
				&& left_child->is_balanced()
				&& right_child->is_balanced();
		}

		void display(const bool is_left = false, const string& prefix = "") {
			if (!this)
				return;

			std::cout << prefix << (is_left ? "├─" : "└─") << data << '\n';
			left_child->display(true, prefix + (is_left ? "│  " : "   "));
			right_child->display(false, prefix + (is_left ? "│  " : "   "));
		}
};

#endif
