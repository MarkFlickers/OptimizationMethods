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
	template<> struct hash<TreeState> {
		size_t operator()(const TreeState& ts) const {
			return ts.getHash();
		}
	};

	template<> struct hash<Tree> {
		size_t operator()(const Tree& t) const {
			return t.getHash();
		}
	};
}

// TreeState implementation
TreeState::TreeState(uint32_t branches, uint32_t branch_len,
	const std::vector<std::vector<char>>& state)
	: branches_count_(branches), branch_len_(branch_len) {
	flatten(state);
	computeHash();
}

TreeState::TreeState(uint32_t branches, uint32_t branch_len,
	const std::vector<char>& flat_data)
	: data_(flat_data), branches_count_(branches), branch_len_(branch_len) {
	computeHash();
}

void TreeState::flatten(const std::vector<std::vector<char>>& state) {
	data_.resize(branches_count_ * branch_len_);
	for(uint32_t i = 0; i < branches_count_; ++i) {
		for(uint32_t j = 0; j < branch_len_; ++j) {
			data_[i * branch_len_ + j] = state[i][j];
		}
	}
}

void TreeState::computeHash(void)
{
	size_t h = 0;
	for(auto num : data_)
	{
		auto hash = std::hash<int>{}(num);
		h ^= hash + 0x9e3779b9 + (h << 6) + (h >> 2);
	}
	hash_ = h;
}

bool TreeState::operator==(const TreeState& other) const {
	return data_ == other.data_;
}

// Tree implementation
Tree::Tree(const TreeState& state) : state_(state) {
	computeUnperfectness();
}

Tree::Tree(const Tree& parent, const Move& move) {
	// Быстрое копирование flat данных
	state_ = TreeState(parent.state_.getBranchesCount(),
		parent.state_.getBranchLen(),
		parent.state_.getData());

	// Применяем ход - всего 2 изменения в массиве
	state_.at(move.src_branch, move.src_pos) = 0;
	state_.at(move.dst_branch, move.dst_pos) = move.bird;

	// Пересчитываем хэш
	state_.computeHash();
	computeUnperfectness();
}

bool Tree::operator==(const Tree& other) const {
	return state_ == other.state_;
}

void Tree::computeUnperfectness() {
	unperfectness_ = 0;
	for(uint32_t i = 0; i < state_.getBranchesCount(); ++i) {
		unperfectness_ += computeBranchUnperfectness(i);
	}
}

size_t Tree::computeBranchUnperfectness(uint32_t branch_index) {
	if(state_.at(branch_index, 0) == 0) return 0;

	std::unordered_map<char, uint32_t> freq;
	uint32_t max_freq = 0;

	for(uint32_t j = 0; j < state_.getBranchLen(); ++j) {
		char bird = state_.at(branch_index, j);
		if(bird != 0) {
			max_freq = std::max(max_freq, ++freq[bird]);
		}
	}

	return state_.getBranchLen() - max_freq;
}

// Node implementation
Node::Node(Tree&& tree, std::shared_ptr<Node> parent, const Move& move, int g)
	: tree_(std::move(tree)), parent_(parent), move_(move), g_(g) {
	f_ = g_ + tree_.getUnperfectness();
}

bool Node::operator>(const Node& other) const {
	return f_ > other.f_;
}

// AStarSolver implementation
AStarSolver::AStarSolver(const TreeState& start_state)
	: start_state_(start_state) {
}

SolvedTree AStarSolver::solve() {
	Tree start_tree(start_state_);
	auto start_node = std::make_shared<Node>(std::move(start_tree), nullptr, Move{}, 0);

	std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_list;
	open_list.push(*start_node);

	std::unordered_set<size_t> closed_set;
	std::vector<std::shared_ptr<Node>> all_nodes;
	all_nodes.push_back(std::move(start_node));

	while(!open_list.empty()) {
		Node current_node = open_list.top();
		open_list.pop();

		if(closed_set.count(current_node.getHash())) continue;

		if(current_node.getTree().getUnperfectness() == 0) {
			SolvedTree answer;
			answer.steps_amount = 0;
			answer.Moves.push_back(current_node.getMove());
			auto current = current_node.getParent();
			while(current) {
				answer.Moves.insert(answer.Moves.begin(), current->getMove());
				answer.steps_amount++;
				current = current->getParent();
			}
			return answer;
		}

		closed_set.insert(current_node.getHash());
		processNeighbors(current_node, open_list, closed_set, all_nodes);
	}
	SolvedTree answer
	{
		.steps_amount = 0
	};
	return answer;
}

std::vector<Move> AStarSolver::findPossibleMoves(const Tree& tree) const {
	std::vector<Move> moves;
	const auto& state = tree.getState();
	uint32_t branches_count = state.getBranchesCount();
	uint32_t branch_len = state.getBranchLen();

	// Предварительное резервирование памяти
	moves.reserve(branches_count * (branches_count - 1));

	for(uint32_t src_branch = 0; src_branch < branches_count; ++src_branch) {
		// Быстрая проверка на пустую ветку
		if(state.at(src_branch, 0) == 0) continue;

		// Находим последнюю птицу в исходной ветке
		int src_pos = -1;
		for(int j = branch_len - 1; j >= 0; --j) {
			if(state.at(src_branch, j) != 0) {
				src_pos = j;
				break;
			}
		}
		if(src_pos == -1) continue;

		char bird = state.at(src_branch, src_pos);

		for(uint32_t dst_branch = 0; dst_branch < branches_count; ++dst_branch) {
			if(src_branch == dst_branch) continue;

			// Находим первую свободную позицию в целевой ветке
			int dst_pos = -1;
			for(uint32_t j = 0; j < branch_len; ++j) {
				if(state.at(dst_branch, j) == 0) {
					dst_pos = j;
					break;
				}
			}
			if(dst_pos == -1) continue;

			// Проверяем совместимость птиц
			if(dst_pos > 0 && state.at(dst_branch, dst_pos - 1) != bird) continue;

			moves.push_back({src_branch, static_cast<uint32_t>(src_pos),
						   dst_branch, static_cast<uint32_t>(dst_pos), bird});
		}
	}

	return moves;
}

Tree AStarSolver::applyMove(const Tree& tree, const Move& move) const {
	return Tree(tree, move);
}

void AStarSolver::processNeighbors(const Node& current_node,
	std::priority_queue<Node, std::vector<Node>, std::greater<Node>>& open_list,
	std::unordered_set<size_t>& closed_set,
	std::vector<std::shared_ptr<Node>>& all_nodes) const {
	auto moves = findPossibleMoves(current_node.getTree());

	for(const auto& move : moves) {
		Tree new_tree = applyMove(current_node.getTree(), move);
		size_t new_hash = new_tree.getHash();

		if(closed_set.count(new_hash)) continue;

		auto new_node_ptr = std::make_shared<Node>(std::move(new_tree),
			std::make_shared<Node>(current_node),
			move, current_node.getG() + 1);

		// TODO: оптимизировать поиск в open_list
		// Временное решение - всегда добавлять
		open_list.push(*new_node_ptr);
		all_nodes.push_back(std::move(new_node_ptr));
	}
}

// Вспомогательная функция для конвертации из старого формата
TreeState convertToFlatTreeState(const std::vector<std::vector<char>>& state) {
	if(state.empty()) return TreeState(0, 0, state);
	return TreeState(state.size(), state[0].size(), state);
}