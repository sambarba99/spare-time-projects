/*
Binary tree demo

Author: Sam Barba
Created 08/09/2021
*/

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>

#include "tree.h"

using std::cout;
using std::setw;


std::random_device rd;
std::mt19937 gen(rd());


void display_tree_info(Tree* tree) {
	cout << "Tree (height " << tree->get_height() << "):\n\n";
	tree->display();

	cout << '\n';
	cout << setw(24) << "In-order traversal:";
	for (const string& name : tree->in_order_traversal())
		cout << ' ' << name;

	cout << '\n';
	cout << setw(24) << "Pre-order traversal:";
	for (const string& name : tree->pre_order_traversal())
		cout << ' ' << name;

	cout << '\n';
	cout << setw(24) << "Post-order traversal:";
	for (const string& name : tree->post_order_traversal())
		cout << ' ' << name;

	cout << "\nBreadth-first traversal:";
	for (const string& name : tree->breadth_first_traversal())
		cout << ' ' << name;

	cout << '\n';
	cout << setw(24) << "Is Binary Search Tree:";
	cout << (tree->is_bst() ? " Yes\n" : " No\n");
	cout << setw(24) << "Is balanced:";
	cout << (tree->is_balanced() ? " Yes" : " No");
}


Tree* make_balanced_bst(const vector<string>& data, const int lo = 0, int hi = INT_MIN) {
	// Make a balanced binary search tree

	sort(data.begin(), data.end());

	if (hi == INT_MIN)
		hi = data.size() - 1;
	if (lo > hi)
		return NULL;

	int mid = (lo + hi) / 2;

	Tree* tree = new Tree(data[mid]);
	tree->left_child = make_balanced_bst(data, lo, mid - 1);
	tree->right_child = make_balanced_bst(data, mid + 1, hi);

	return tree;
}


int main() {
	vector<string> data = {"alice", "bob", "charlie", "david", "emily", "francis", "george",
		"harrison", "isaac", "jason", "leo", "maria", "nathan", "olivia", "penelope"};
	std::shuffle(data.begin(), data.end(), gen);

	Tree* tree = new Tree(data[0]);
	for (int i = 1; i < data.size(); i++)
		tree->insert(data[i]);

	display_tree_info(tree);

	tree = make_balanced_bst(data);

	cout << "\n\n-------------------- After balancing --------------------\n\n";

	display_tree_info(tree);

	return 0;
}
