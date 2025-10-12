#pragma once
#include <vector>
#include <memory>

class Branch
{
public:
	std::vector<int> birds;
	size_t unfillness;

	Branch(void);
	Branch(std::vector<int> birds_configuration);
	bool operator == (const Branch& other) const;

private:
	size_t measure_unfillness(std::vector<int> birds_configuration);
};

class Tree
{
public:
	std::vector<Branch> Branches;
	unsigned long amount_of_empty_branches;
	size_t unperfectness;
	size_t branch_len;
	size_t _hash;

	Tree(void);
	Tree(std::vector<std::vector<int>> TreeState, unsigned long empty_branches = 0);
	bool operator == (const Tree& other) const;
	std::vector<std::vector<int>> get_TreeState(void);

private:
	unsigned long parse_empty_branches(std::vector<std::vector<int>> TreeState);
	std::vector<Branch> parse_non_empty_branches(std::vector<std::vector<int>> TreeState);
	size_t measure_tree_unperfectness(void);
	size_t get_hash(std::vector<std::vector<int>> tree);
};

class Node
{
public:
	Tree nodeTree;          // Дерево узла
	int g;             // Расстояние от начального узла до текущего
	size_t h;           // Эвристическое расстояние от текущего до конечного узла
	size_t f;
	std::shared_ptr<Node> parent;
	std::vector<Branch> parent_diff;
	size_t _hash;

	Node(void);
	Node(Tree Tree, std::shared_ptr<Node> parent, std::vector<Branch> parent_diff, int g = 0);
	bool operator == (const Node& other) const;
	bool operator < (const Node& other) const;
	bool operator > (const Node& other) const;
};

int AStar(std::vector<std::vector<int>> startTreeState);