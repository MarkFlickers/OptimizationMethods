#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include "AStar.h"

extern AStarSolver *GlobalSolver_p;

namespace std {
	template<> struct hash<TreeState> {
		size_t operator()(const TreeState& ts) const
		{
			return ts.getHash();
		}
	};

	template<> struct hash<Tree> {
		size_t operator()(const Tree& t) const
		{
			return t.getHash();
		}
	};
}

// TreeState implementation
TreeState::TreeState(uint16_t branches, uint8_t branch_len,
	const std::vector<std::vector<char>>& state)
	: branches_count_(branches), branch_len_(branch_len)
{
	flatten(state);
	computeHash();
}

TreeState::TreeState(uint16_t branches, uint8_t branch_len,
	const std::vector<char>& flat_data)
	: data_(flat_data), branches_count_(branches), branch_len_(branch_len)
{
	computeHash();
}

void TreeState::flatten(const std::vector<std::vector<char>>& state)
{
	data_.resize(branches_count_ * branch_len_);
	for(uint32_t i = 0; i < branches_count_; ++i)
	{
		for(uint32_t j = 0; j < branch_len_; ++j)
		{
			data_[i * branch_len_ + j] = state[i][j];
		}
	}
}

void TreeState::computeHash(void) const
{
	if(!hash_computed_)
	{
		const size_t prime = 0x100000001b3;
		hash_ = 0xcbf29ce484222325;

		for(char c : data_)
		{
			hash_ ^= static_cast<size_t>(c);
			hash_ *= prime;
		}
		hash_computed_ = true;
	}
}

size_t TreeState::getHash() const
{
	computeHash();
	return hash_;
}

bool TreeState::operator==(const TreeState& other) const
{
	return data_ == other.data_;
}

// Tree implementation
Tree::Tree(const TreeState& state) : state_(state)
{
	computeUnperfectness();
}

Tree::Tree(const Tree& parent, const Move& move)
{
	// Быстрое копирование flat данных
	state_ = TreeState(parent.state_.getBranchesCount(),
		parent.state_.getBranchLen(),
		parent.state_.getData());

	// Применяем ход - всего 2 изменения в массиве
	state_.at(move.src_branch, move.src_pos) = 0;
	state_.at(move.dst_branch, move.dst_pos) = move.bird;

	// Пересчитываем хэш
	state_.computeHash();
	unperfectness_ = computeUnperfectnessIncremental(parent, move);
}

bool Tree::operator==(const Tree& other) const
{
	return state_ == other.state_;
}

size_t Tree::computeUnperfectnessIncremental(const Tree& parent, const Move& move)
{
	// Вычисляем изменения только для затронутых веток
	size_t delta = 0;

	// Вычитаем старые значения для измененных веток
	delta -= ::computeBranchUnperfectnessWithCache(parent.state_, move.src_branch);
	delta -= ::computeBranchUnperfectnessWithCache(parent.state_, move.dst_branch);

	// Прибавляем новые значения для измененных веток
	delta += ::computeBranchUnperfectnessWithCache(state_, move.src_branch);
	delta += ::computeBranchUnperfectnessWithCache(state_, move.dst_branch);

	return parent.unperfectness_ + delta;
}

void Tree::computeUnperfectness()
{
	unperfectness_ = 0;
	for(uint32_t i = 0; i < state_.getBranchesCount(); ++i)
	{
		unperfectness_ += computeBranchUnperfectnessWithCache(state_, i);
	}
}

size_t Tree::computeBranchUnperfectness(uint32_t branch_index)
{
	if(state_.at(branch_index, 0) == 0) return 0;

	const int MAX_BIRD_TYPES = 26;
	uint32_t freq[MAX_BIRD_TYPES] = {0};
	uint32_t max_freq = 0;

	for(uint32_t j = 0; j < state_.getBranchLen(); ++j)
	{
		char bird = state_.at(branch_index, j);
		if(bird != 0)
		{
			freq[bird]++;
		}
	}
	for(uint8_t i = 0; i < MAX_BIRD_TYPES; i++)
	{
		if(freq[i] > max_freq)
			max_freq = freq[i];
	}

	return state_.getBranchLen() - max_freq;
}

// Node implementation
Node::Node(Tree&& tree, std::shared_ptr<Node> parent, const Move& move, int g)
	: tree_(std::move(tree)), parent_(parent), move_(move), g_(g)
{
	f_ = g_ + tree_.getUnperfectness();
}

bool Node::operator>(const Node& other) const
{
	return f_ > other.f_;
}

// AStarSolver implementation
AStarSolver::AStarSolver(const TreeState& start_state)
	: start_state_(start_state)
{}

SolvedTree AStarSolver::solve()
{
	Tree start_tree(start_state_);
	auto start_node = std::make_shared<Node>(std::move(start_tree), nullptr, Move{}, 0);

	auto compare = [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
		return a->getF() > b->getF();
	};

	std::priority_queue<std::shared_ptr<Node>,
		std::vector<std::shared_ptr<Node>>,
		decltype(compare)> open_list(compare);

	open_list.push(start_node);

	std::unordered_set<size_t> closed_set;
	std::vector<std::shared_ptr<Node>> all_nodes;
	all_nodes.push_back(start_node);

	while(!open_list.empty())
	{
		auto current_node = open_list.top();
		open_list.pop();

		if(closed_set.count(current_node->getHash())) continue;

		if(current_node->getTree().getUnperfectness() == 0)
		{
			SolvedTree answer = {0};
			auto current = current_node;
			while(current->getParent() != nullptr)
			{
				answer.Moves.insert(answer.Moves.begin(), current->getMove());
				answer.steps_amount++;
				current = current->getParent();
			}
			return answer;
		}

		closed_set.insert(current_node->getHash());
		//processNeighbors(current_node, open_list, closed_set, all_nodes);
		auto moves = findPossibleMovesWithCache(current_node->getTree());

		for(const auto& move : moves)
		{
			Tree new_tree = applyMove(current_node->getTree(), move);
			size_t new_hash = new_tree.getHash();

			if(closed_set.count(new_hash)) continue;

			auto new_node = std::make_shared<Node>(std::move(new_tree),
				current_node, move,
				current_node->getG() + 1);

			if(registerNode(new_node))
			{
				open_list.push(new_node);
			}
		}
	}

	return SolvedTree{0};
}

bool AStarSolver::registerNode(const std::shared_ptr<Node>& node)
{
	size_t hash = node->getHash();
	auto it = node_registry_.find(hash);

	if(it != node_registry_.end())
	{
		// Узел с таким хэшем уже существует
		if(it->second->getG() <= node->getG())
		{
			return false; // Существующий узел не хуже
		}
		// Новый узел лучше - обновляем регистр
		it->second = node;
		return true;
	}
	else
	{
		// Новый узел - добавляем в регистр
		node_registry_[hash] = node;
		return true;
	}
}

std::vector<Move> AStarSolver::findPossibleMoves(const Tree& tree) const
{
	std::vector<Move> moves;
	const auto& state = tree.getState();
	uint16_t branches_count = state.getBranchesCount();
	uint8_t branch_len = state.getBranchLen();

	// Предварительное резервирование памяти
	moves.reserve(branches_count * (branches_count - 1));

	for(uint16_t src_branch = 0; src_branch < branches_count; ++src_branch)
	{
		// Быстрая проверка на пустую ветку
		if(state.at(src_branch, 0) == 0) continue;

		// Находим последнюю птицу в исходной ветке
		int8_t src_pos = -1;
		for(int8_t j = branch_len - 1; j >= 0; --j)
		{
			if(state.at(src_branch, j) != 0)
			{
				src_pos = j;
				break;
			}
		}
		if(src_pos == -1) continue;

		char bird = state.at(src_branch, src_pos);

		for(uint16_t dst_branch = 0; dst_branch < branches_count; ++dst_branch)
		{
			if(src_branch == dst_branch) continue;

			// Находим первую свободную позицию в целевой ветке
			int8_t dst_pos = -1;
			for(int8_t j = 0; j < branch_len; ++j)
			{
				if(state.at(dst_branch, j) == 0)
				{
					dst_pos = j;
					break;
				}
			}
			if(dst_pos == -1) continue;

			// Проверяем совместимость птиц
			if(dst_pos > 0 && state.at(dst_branch, dst_pos - 1) != bird) continue;

			moves.push_back({src_branch, static_cast<uint8_t>(src_pos),
						   dst_branch, static_cast<uint8_t>(dst_pos), bird});
		}
	}

	return moves;
}

Tree AStarSolver::applyMove(const Tree& tree, const Move& move) const
{
	return Tree(tree, move);
}

void AStarSolver::processNeighbors(const Node& current_node,
	std::priority_queue<Node, std::vector<Node>, std::greater<Node>>& open_list,
	std::unordered_set<size_t>& closed_set,
	std::vector<std::shared_ptr<Node>>& all_nodes) const
{
	auto moves = findPossibleMovesWithCache(current_node.getTree());

	for(const auto& move : moves)
	{
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

std::vector<Move> AStarSolver::findPossibleMovesWithCache(const Tree& tree) const
{
	size_t tree_hash = tree.getHash();
	auto it = moves_cache_.find(tree_hash);
	if(it != moves_cache_.end())
	{
		return it->second;
	}

	std::vector<Move> moves = findPossibleMoves(tree);
	moves_cache_[tree_hash] = moves;
	return moves;
}

void AStarSolver::initializeBranchCache(uint32_t max_branches, uint32_t branch_len)
{
	branch_cache_size_ = max_branches * (1 << (branch_len * 2)); // Эвристика размера
	branch_unperfectness_cache_.resize(branch_cache_size_, 0);
	branch_cache_valid_.resize(branch_cache_size_, false);
}

size_t AStarSolver::computeBranchUnperfectnessWithCache(uint32_t branch_index, const TreeState& state)
{
	const char* branch_data = &state.getData()[branch_index * state.getBranchLen()];
	size_t branch_hash = computeBranchHash(branch_data, state.getBranchLen()) % branch_cache_size_;

	if(branch_cache_valid_[branch_hash])
	{
		return branch_unperfectness_cache_[branch_hash];
	}

	size_t result = computeBranchUnperfectness(state, branch_index);
	branch_unperfectness_cache_[branch_hash] = result;
	branch_cache_valid_[branch_hash] = true;
	return result;
}


// Вспомогательная функция для конвертации из старого формата
TreeState convertToFlatTreeState(const std::vector<std::vector<char>>& state)
{
	if(state.empty()) return TreeState(0, 0, state);
	return TreeState(state.size(), state[0].size(), state);
}

size_t computeBranchUnperfectnessWithCache(const TreeState& state, uint32_t branch_index)
{
	static std::unordered_map<size_t, size_t> branch_unperfectness_cache_ = {};

	const char* branch_data = &state.getData()[branch_index * state.getBranchLen()];
	size_t branch_hash = computeBranchHash(branch_data, state.getBranchLen());

	auto it = branch_unperfectness_cache_.find(branch_hash);
	if(it != branch_unperfectness_cache_.end())
	{
		return it->second;
	}

	size_t result = computeBranchUnperfectness(state, branch_index);
	branch_unperfectness_cache_[branch_hash] = result;
	return result;
}

size_t computeBranchHash(const char* branch_data, uint32_t branch_len)
{
	size_t h = 0;
	for(uint32_t i = 0; i < branch_len; ++i)
	{
		h = h * 31 + static_cast<size_t>(branch_data[i]);
	}
	return h;
}

size_t computeBranchUnperfectness(const TreeState & state, uint32_t branch_index)
{
	if(state.at(branch_index, 0) == 0) return 0;

	const int MAX_BIRD_TYPES = 26;
	uint32_t freq[MAX_BIRD_TYPES] = {0};
	uint32_t max_freq = 0;

	for(uint32_t j = 0; j < state.getBranchLen(); ++j)
	{
		char bird = state.at(branch_index, j);
		if(bird != 0)
		{
			freq[bird]++;
		}
	}
	for(uint8_t i = 0; i < MAX_BIRD_TYPES; i++)
	{
		if(freq[i] > max_freq)
			max_freq = freq[i];
	}

	return state.getBranchLen() - max_freq;
}