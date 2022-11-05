/*
Binary tree demo

Author: Sam Barba
Created 08/09/2021
*/

#include <bits/stdc++.h>

#include "tree.h"

using std::default_random_engine;
using std::setw;
using std::shuffle;

void displayTreeInfo(Tree* tree) {
	cout << "Tree (height " << tree->getHeight() << "):\n\n";
	tree->display();

	cout << '\n';
	cout << setw(24);
	cout << "In-order traversal:";
	for (string name : tree->inOrderTraversal())
		cout << ' ' << name;
	
	cout << '\n';
	cout << setw(24);
	cout << "Pre-order traversal:";
	for (string name : tree->preOrderTraversal())
		cout << ' ' << name;
	
	cout << '\n';
	cout << setw(24);
	cout << "Post-order traversal:";
	for (string name : tree->postOrderTraversal())
		cout << ' ' << name;
	
	cout << "\nBreadth-first traversal:";
	for (string name : tree->breadthFirstTraversal())
		cout << ' ' << name;
	
	cout << '\n';
	cout << setw(24) << "Is Binary Search Tree:";
	cout << (tree->isBST() ? " Yes\n" : " No\n");
	cout << setw(24);
	cout << "Is balanced:";
	cout << (tree->isBalanced() ? " Yes" : " No");
}

Tree* makeBalancedBST(vector<string> data, int lo = 0, int hi = INT_MIN) {
	sort(data.begin(), data.end());

	if (hi == INT_MIN)
		hi = data.size() - 1;
	if (lo > hi)
		return NULL;

	int mid = (lo + hi) / 2;

	Tree* tree = new Tree(data[mid]);
	tree->leftChild = makeBalancedBST(data, lo, mid - 1);
	tree->rightChild = makeBalancedBST(data, mid + 1, hi);

	return tree;
}

int main() {
	vector<string> data = {"alice", "bob", "charlie", "david", "emily", "francis", "george",
		"harrison", "isaac", "jason", "leo", "maria", "nathan", "olivia", "penelope"};
	auto rnd = default_random_engine {};
	shuffle(data.begin(), data.end(), rnd);

	Tree* tree = new Tree(data[0]);
	for (int i = 1; i < data.size(); i++)
		tree->insert(data[i]);

	displayTreeInfo(tree);

	tree = makeBalancedBST(data);

	cout << "\n\n-------------------- After balancing --------------------\n\n";

	displayTreeInfo(tree);
}
