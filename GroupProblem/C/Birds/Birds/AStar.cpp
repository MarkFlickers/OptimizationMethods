#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <memory>
#include "AStar.h"

namespace std {
	template <> struct hash<Node> {
		size_t operator()(const Node& o) const
		{
			// Some custom logic for calculating hash of Node
			return o._hash;
		}
	};
	template <> struct hash<Tree> {
		size_t operator()(const Tree& o) const
		{
			// Some custom logic for calculating hash of Tree
			return o._hash;
		}
	};
	template <> struct hash<std::vector<std::vector<int>>> {
		size_t operator()(const std::vector<std::vector<int>>& o) const
		{
			// Some custom logic for calculating hash of TreeState
			size_t h = 0;
			for(auto vec : o)
			{
				for(auto num : vec)
				{
					auto hash = std::hash<int>{}(num);
					h ^= hash + 0x9e3779b9 + (h << 6) + (h >> 2);
				}
			}
			return h;
		}
	};
}

Branch::Branch(void)
{
	birds = std::vector<int>{0};
	unfillness = 0;
}

Branch::Branch(std::vector<int> birds_configuration)
{
	birds = birds_configuration;
	unfillness = measure_unfillness(birds);
}

bool Branch::operator == (const Branch& other) const
{
	return (birds == other.birds);
}

size_t Branch::measure_unfillness(std::vector<int> birds_configuration)
{
	if(birds_configuration[0] == 0)
	{
		return 0;
	}
	std::map<int, int> frequencies;

	for(auto bird : birds_configuration)
	{
		frequencies[bird]++;
	}
	using pair_type = decltype(frequencies)::value_type;
	auto max = std::max_element(std::begin(frequencies), std::end(frequencies), [](const pair_type& p1, const pair_type& p2) {return p1.second < p2.second; });
	return birds_configuration.size() - max->second;
}

Tree::Tree(void)
{
	amount_of_empty_branches = 0;
	Branches = {Branch()};
	unperfectness = 0;
	branch_len = 0;
	_hash = get_hash({{0}});
}

Tree::Tree(std::vector<std::vector<int>> TreeState, unsigned long empty_branches)
{
	branch_len = TreeState[0].size();
	amount_of_empty_branches = parse_empty_branches(TreeState) + empty_branches;
	Branches = parse_non_empty_branches(TreeState);
	unperfectness = measure_tree_unperfectness();
	_hash = get_hash(TreeState);
}

bool Tree::operator == (const Tree& other) const
{
	return (Branches == other.Branches);
}

std::vector<std::vector<int>> Tree::get_TreeState(void)
{
	std::vector<std::vector<int>> TreeState = {};
	for(auto branch : Branches)
	{
		TreeState.push_back(branch.birds);
	}
	return TreeState;
}

unsigned long Tree::parse_empty_branches(std::vector<std::vector<int>> TreeState)
{
	long sum = std::count_if(TreeState.begin(), TreeState.end(), [](const std::vector<int>& elem) {return elem[0] == 0 ? true : false; });
	return sum;
}

std::vector<Branch> Tree::parse_non_empty_branches(std::vector<std::vector<int>> TreeState)
{
	std::vector<Branch> Branches = {};
	for(auto branch_state : TreeState)
	{
		Branch branch_candidate(branch_state);
		if(branch_candidate.unfillness == 0)
			continue;
		Branches.push_back(branch_candidate);
	}
	if(amount_of_empty_branches > 0)
	{
		Branch empty_branch(std::vector<int>(branch_len, 0));
		Branches.push_back(empty_branch);
		amount_of_empty_branches--;
	}
	return Branches;
}

size_t Tree::measure_tree_unperfectness(void)
{
	size_t unperfectness = std::accumulate(Branches.begin(), Branches.end(), 0, [](int sum, const Branch &b) {return sum + b.unfillness; });
	return unperfectness;
}

size_t Tree::get_hash(std::vector<std::vector<int>> tree)
{
	size_t h = 0;
	for(auto vec : tree)
	{
		for(auto num : vec)
		{
			auto hash = std::hash<int>{}(num);
			h ^= hash + 0x9e3779b9 + (h << 6) + (h >> 2);
		}
	}
	return h;
}

Node::Node(void)
{
	nodeTree = Tree();
	g = 0;
	h = 0;
	f = 0;
	parent = nullptr;
	parent_diff = {Branch(), Branch()};
	_hash = std::hash<Tree>{}(nodeTree);
}

Node::Node(Tree itsTree, std::shared_ptr<Node> parent, std::vector<Branch> parent_diff, int g)
{
	this->nodeTree = itsTree;
	this->g = g;
	this->h = itsTree.unperfectness;
	this->f = this->g + this->h;
	this->parent = parent;
	this->parent_diff = parent_diff;
	_hash = std::hash<Tree>{}(nodeTree);
}

bool Node::operator == (const Node& other) const
{
	return nodeTree == other.nodeTree;
}
bool Node::operator < (const Node& other) const
{
	return f < other.f;
}
bool Node::operator > (const Node& other) const
{
	return f > other.f;
}

std::vector<std::vector<int>> move_bird(std::vector<std::vector<int>> TreeState, unsigned int src_branch_number, unsigned int dst_branch_number)
{
	std::vector<int> src_branch = TreeState[src_branch_number];
	std::vector<int> dst_branch = TreeState[dst_branch_number];

	if(src_branch[0] == 0 or src_branch_number == dst_branch_number)
		return {{0}};

	auto src_non_zero_elem_p = std::find(src_branch.begin(), src_branch.end(), 0);
	src_non_zero_elem_p--;

	auto dst_zero_elem_p = std::find(dst_branch.begin(), dst_branch.end(), 0);
	if(dst_zero_elem_p == dst_branch.end())
		return {{0}};

	if(dst_zero_elem_p == dst_branch.begin() or *(dst_zero_elem_p - 1) == *src_non_zero_elem_p)
	{
		*dst_zero_elem_p = *src_non_zero_elem_p;
		*src_non_zero_elem_p = 0;
		TreeState[src_branch_number] = src_branch;
		TreeState[dst_branch_number] = dst_branch;
		return TreeState;
	}

	return {{0}};
}

std::vector<Node> get_neighbours(std::shared_ptr<Node> curr_node)
{
	std::vector<Node> neighbours = {};
	auto CurrentTree = curr_node->nodeTree.get_TreeState();

	for(unsigned int src_idx = 0; src_idx < CurrentTree.size(); src_idx++)
	{
		for(unsigned int dst_idx = 0; dst_idx < CurrentTree.size(); dst_idx++)
		{
			auto NewTree = move_bird(CurrentTree, src_idx, dst_idx);
			if(NewTree != std::vector<std::vector<int>>{{0}})
			{
				auto neigbour = Node(Tree(NewTree, curr_node->nodeTree.amount_of_empty_branches), curr_node, {Branch(), Branch()}, curr_node->g + 1);
				neighbours.push_back(neigbour);
			}
		}
	}
	return neighbours;
}

int AStar(std::vector<std::vector<int>> startTreeState)
{
	auto start_node_ptr = std::make_shared<Node>(Tree(startTreeState), nullptr, std::vector<Branch>{Branch(), Branch()});
	Node start_node = *start_node_ptr; // Создаем копию для open_list

	std::vector<Node> open_list = {start_node};
	std::ranges::make_heap(open_list, std::greater{});

	std::unordered_set<Node> closed_set = {};



	int nodes_processed = 0;

	while(not open_list.empty())
	{
		std::ranges::pop_heap(open_list, std::greater{});
		Node current_node = open_list.back();
		open_list.pop_back();

		if(closed_set.find(current_node) != closed_set.end())
		{
			continue;
		}

		if(current_node.h == 0)
		{
			int steps_number = 0;
			std::shared_ptr<Node> current = std::make_shared<Node>(current_node);
			while(current != nullptr)
			{
				steps_number++;
				current = current->parent;
			}
			return steps_number;
		}

		closed_set.insert(current_node);

		auto current_node_ptr = std::make_shared<Node>(current_node);
		auto neighbours = get_neighbours(current_node_ptr);

		for(auto neighbour : neighbours)
		{
			if(closed_set.find(neighbour) != closed_set.end())
			{
				continue;
			}

			auto existing_node = std::find(open_list.begin(), open_list.end(), neighbour);
			if(existing_node != open_list.end())
			{
				if(neighbour.g < existing_node->g)
				{
					existing_node->g = neighbour.g;
					existing_node->f = neighbour.f;
					existing_node->parent = neighbour.parent;
					existing_node->parent_diff = neighbour.parent_diff;
					std::ranges::make_heap(open_list, std::greater{});     // Maybe change this to erase + push_back + push_heap?
				}
			}
			else
			{
				open_list.push_back(neighbour);
				std::ranges::push_heap(open_list, std::greater{});
			}

		}
		nodes_processed++;
	}

	return 0;
}