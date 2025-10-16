#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <unordered_map>
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
			size_t h = o.size();
			for(const auto& vec : o)
			{
				for(auto num : vec)
				{
					h ^= std::hash<int>{}(num)+0x9e3779b9 + (h << 6) + (h >> 2);
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

Branch::Branch(std::vector<int> &birds_configuration)
{
	birds = birds_configuration;
	unfillness = measure_unfillness(birds);
}

bool Branch::operator == (const Branch& other) const
{
	return (birds == other.birds);
}

size_t Branch::measure_unfillness(std::vector<int> &birds_configuration)
{
	if(birds_configuration[0] == 0)
	{
		return 0;
	}

	std::unordered_map<int, int> frequencies;
	frequencies.reserve(birds_configuration.size());

	int max_freq = 0;
	for(auto bird : birds_configuration)
	{
		max_freq = std::max(max_freq, ++frequencies[bird]);
	}

	return birds_configuration.size() - max_freq;
}

Tree::Tree(void)
{
	amount_of_empty_branches = 0;
	Branches = {Branch()};
	unperfectness = 0;
	branch_len = 0;
	std::vector<std::vector<int>> vec = { {0} };
	_hash = get_hash(vec);
}

Tree::Tree(std::vector<std::vector<int>> &TreeState, unsigned long empty_branches)
{
	branch_len = TreeState[0].size();
	Branches.reserve(TreeState.size());
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

unsigned long Tree::parse_empty_branches(std::vector<std::vector<int>> &TreeState)
{
	long sum = std::count_if(TreeState.begin(), TreeState.end(), [](const std::vector<int>& elem) {return elem[0] == 0 ? true : false; });
	return sum;
}

std::vector<Branch> Tree::parse_non_empty_branches(std::vector<std::vector<int>> &TreeState)
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
		auto vec = std::vector<int>(branch_len, 0);
		Branch empty_branch(vec);
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

size_t Tree::get_hash(std::vector<std::vector<int>> &tree)
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
	static const std::vector<std::vector<int>> empty_result = {{0}};

	if(src_branch_number == dst_branch_number || TreeState[src_branch_number][0] == 0)
		return empty_result;

	const auto& src_branch = TreeState[src_branch_number];
	const auto& dst_branch = TreeState[dst_branch_number];

	int src_last_non_zero = -1;
	for(int i = src_branch.size() - 1; i >= 0; --i)
	{
		if(src_branch[i] != 0) {
			src_last_non_zero = i;
			break;
		}
	}

	if(src_last_non_zero == -1) 
		return empty_result;

	int dst_first_zero = -1;
	for(int i = 0; i < dst_branch.size(); ++i)
	{
		if(dst_branch[i] == 0) {
			dst_first_zero = i;
			break;
		}
	}

	if(dst_first_zero == -1)
		return empty_result;

	if(dst_first_zero > 0 && dst_branch[dst_first_zero - 1] != src_branch[src_last_non_zero])
	{
		return empty_result;
	}

	std::vector<std::vector<int>> new_state = TreeState;

	new_state[src_branch_number][src_last_non_zero] = 0;
	new_state[dst_branch_number][dst_first_zero] = src_branch[src_last_non_zero];

	return new_state;
}

std::vector<Node> get_neighbours(std::shared_ptr<Node> curr_node)
{
	std::vector<Node> neighbours;
	const auto& CurrentTree = curr_node->nodeTree.get_TreeState();
	size_t tree_size = CurrentTree.size();

	neighbours.reserve(tree_size * (tree_size - 1));

	for(unsigned int src_idx = 0; src_idx < tree_size; src_idx++)
	{
		if(CurrentTree[src_idx][0] == 0)
			continue;

		for(unsigned int dst_idx = 0; dst_idx < tree_size; dst_idx++)
		{
			if(src_idx == dst_idx)
				continue;

			auto NewTree = move_bird(CurrentTree, src_idx, dst_idx);
			if(NewTree[0][0] != 0)
			{
				neighbours.emplace_back(
					Tree(NewTree, curr_node->nodeTree.amount_of_empty_branches),
					curr_node,
					std::vector<Branch>{},
					curr_node->g + 1
				);
			}
		}
	}
	return neighbours;
}

int AStar(std::vector<std::vector<int>> &startTreeState)
{
	auto start_node_ptr = std::make_shared<Node>(Tree(startTreeState), nullptr, std::vector<Branch>{});
	const Node& start_node = *start_node_ptr;

	auto node_compare = [](const Node* a, const Node* b) { return a->f > b->f; };
	std::priority_queue<Node*, std::vector<Node*>, decltype(node_compare)> open_list(node_compare);

	std::vector<std::unique_ptr<Node>> all_nodes;
	all_nodes.push_back(std::make_unique<Node>(*start_node_ptr));
	open_list.push(all_nodes.back().get());

	std::unordered_set<size_t> closed_set;

	while(!open_list.empty())
	{
		Node* current_node_ptr = open_list.top();
		open_list.pop();

		Node& current_node = *current_node_ptr;

		if(closed_set.count(current_node._hash))
			continue;

		if(current_node.h == 0)
		{
			int steps_number = 0;
			const Node* current = &current_node;
			while(current->parent != nullptr)
			{
				steps_number++;
				current = current->parent.get();
			}
			return steps_number;
		}

		closed_set.insert(current_node._hash);

		auto neighbours = get_neighbours(std::make_shared<Node>(current_node));

		for(auto& neighbour : neighbours)
		{
			if(closed_set.count(neighbour._hash))
				continue;

			bool found = false;
			for(const auto& node : all_nodes) 
			{
				if(node->_hash == neighbour._hash) 
				{
					if(neighbour.g < node->g) 
					{
						node->g = neighbour.g;
						node->f = neighbour.f;
						node->parent = neighbour.parent;
						open_list.push(node.get());
					}
					found = true;
					break;
				}
			}

			if(!found) 
			{
				all_nodes.push_back(std::make_unique<Node>(std::move(neighbour)));
				open_list.push(all_nodes.back().get());
			}
		}
	}
	return 0;
}