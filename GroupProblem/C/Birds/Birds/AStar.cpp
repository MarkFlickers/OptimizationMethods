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

char& TreeState::at(uint16_t branch, uint8_t pos)
{
	hash_computed_ = false;
	return data_[branch * branch_len_ + pos];
}

const char& TreeState::at(uint16_t branch, uint8_t pos) const
{
	return data_[branch * branch_len_ + pos];
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

// Node implementation
Node::Node(Tree&& tree, Node* parent, const Move& move, int g)
	: tree_(std::move(tree)), parent_(parent), move_(move), g_(g),
	hash_(tree_.getHash())
{
	// Эвристика: штраф за недособранные ветки
	//size_t h = (tree_.getMaxL() - tree_.getCurrentL()) * 100 * tree_.getState().getBranchLen();
	size_t h = tree_.getUnperfectness();
	f_ = g_ + h;
}

bool Node::operator>(const Node& other) const
{
	return f_ > other.f_;
}

void NodePool::initialize(size_t capacity)
{
	nodes_.reserve(capacity);
	for(size_t i = 0; i < capacity; ++i)
	{
		nodes_.push_back(std::make_unique<Node>());
	}
}

Node* NodePool::createNode(Tree&& tree, Node* parent, const Move& move, int g)
{
	if(next_index_ >= nodes_.size())
	{
		nodes_.push_back(std::make_unique<Node>(std::move(tree), parent, move, g));
		return nodes_.back().get();
	}

	Node* node = nodes_[next_index_++].get();
	new (node) Node(std::move(tree), parent, move, g);
	return node;
}

void NodePool::reset()
{
	next_index_ = 0;
}

// AStarSolver implementation
AStarSolver::AStarSolver(const TreeState& start_state) : start_state_(start_state)
{
	node_pool_.initialize(1000000); // 1 миллион узлов
}

SolvedTree AStarSolver::solve()
{
	// Очищаем кэши
	moves_cache_.clear();
	apply_move_cache_.clear();
	node_registry_.clear();
	node_pool_.reset();

	Tree start_tree(start_state_);
	Node* start_node = node_pool_.createNode(std::move(start_tree), nullptr, Move{}, 0);
	node_registry_[start_node->getHash()] = start_node;

	auto compare = [](Node* a, Node* b) { return a->getF() > b->getF(); };
	std::priority_queue<Node*, std::vector<Node*>, decltype(compare)> open_list(compare);
	open_list.push(start_node);

	std::unordered_set<size_t> closed_set;
	int nodes_processed = 0;

	while(!open_list.empty())
	{
		Node* current_node = open_list.top();
		open_list.pop();

		// Периодическая очистка кэшей
		if(nodes_processed % 10000 == 0)
		{
			cleanupCaches();
		}

		if(closed_set.count(current_node->getHash()))
		{
			continue;
		}

		if(current_node->getTree().isTargetState())
		{
			SolvedTree solution{0};
			const Node* current = current_node;
			while(current != nullptr && current->getParent() != nullptr)
			{
				solution.Moves.insert(solution.Moves.begin(), current->getMove());
				solution.steps_amount++;
				current = current->getParent();
			}
			return solution;
		}

		if(shouldPrune(*current_node))
		{
			continue;
		}

		closed_set.insert(current_node->getHash());
		nodes_processed++;

		// Используем оптимизированную версию поиска ходов
		auto moves = findPossibleMovesOptimized(current_node->getTree());

		for(const auto& move : moves)
		{
			Tree new_tree = applyMoveWithCache(current_node->getTree(), move);
			size_t new_hash = new_tree.getHash();
			int new_g = current_node->getG() + 1;

			if(closed_set.count(new_hash))
			{
				continue;
			}

			auto it = node_registry_.find(new_hash);
			if(it != node_registry_.end())
			{
				Node* existing_node = it->second;
				if(new_g < existing_node->getG())
				{
					// Обновляем существующий узел
					*existing_node = Node(std::move(new_tree), current_node, move, new_g);
					open_list.push(existing_node);
				}
			}
			else
			{
				Node* new_node = node_pool_.createNode(std::move(new_tree), current_node, move, new_g);
				if(registerNode(new_node))
				{
					open_list.push(new_node);
				}
			}
		}
	}

	return SolvedTree{0};
}

bool AStarSolver::registerNode(Node* node)
{
	size_t hash = node->getHash();
	auto it = node_registry_.find(hash);

	if(it != node_registry_.end())
	{
		if(it->second->getG() <= node->getG())
		{
			return false;
		}
		it->second = node;
	}
	else
	{
		node_registry_[hash] = node;
	}
	return true;
}

Tree AStarSolver::applyMove(const Tree& tree, const Move& move) const
{
	return Tree(tree, move);
}

Tree AStarSolver::applyMoveWithCache(const Tree& tree, const Move& move) {
    size_t tree_hash = tree.getHash();
    size_t move_hash = hashMove(move);
    
    auto tree_it = apply_move_cache_.find(tree_hash);
    if (tree_it != apply_move_cache_.end()) {
        auto move_it = tree_it->second.find(move_hash);
        if (move_it != tree_it->second.end()) {
            return move_it->second;
        }
    }
    
    Tree result = applyMove(tree, move);
    apply_move_cache_[tree_hash][move_hash] = result;
    return result;
}

size_t AStarSolver::hashMove(const Move& move) const {
    return (static_cast<size_t>(move.src_branch) << 24) | 
           (static_cast<size_t>(move.src_pos) << 16) | 
           (static_cast<size_t>(move.dst_branch) << 8) | 
           static_cast<size_t>(move.dst_pos);
}

bool AStarSolver::shouldPrune(const Node& node) const {
    return node.getG() > MAX_DEPTH;
}

void AStarSolver::cleanupCaches() {
    if (moves_cache_.size() > CACHE_CLEANUP_THRESHOLD) {
        moves_cache_.clear();
    }
    if (apply_move_cache_.size() > CACHE_CLEANUP_THRESHOLD) {
        apply_move_cache_.clear();
    }
}

std::vector<Move> AStarSolver::findPossibleMovesOptimized(const Tree& tree) const {
    size_t tree_hash = tree.getHash();
    auto it = moves_cache_.find(tree_hash);
    if (it != moves_cache_.end()) {
        return it->second;
    }

    std::vector<Move> moves;
    const auto& state = tree.getState();
    uint16_t branches_count = state.getBranchesCount();
    uint8_t branch_len = state.getBranchLen();

    // Предварительно вычисляем информацию о ветках
    struct BranchInfo {
        int8_t last_bird_pos = -1;
        char last_bird = 0;
        std::vector<uint8_t> free_positions;
    };
    
    std::vector<BranchInfo> branches_info(branches_count);
    
    // Собираем информацию о ветках за один проход
    for (uint16_t i = 0; i < branches_count; ++i) {
        auto& info = branches_info[i];
        
        // Ищем последнюю птицу
        for (int8_t j = branch_len - 1; j >= 0; --j) {
            if (state.at(i, j) != 0) {
                info.last_bird_pos = j;
                info.last_bird = state.at(i, j);
                break;
            }
        }
        
        // Ищем свободные позиции
        for (uint8_t j = 0; j < branch_len; ++j) {
            if (state.at(i, j) == 0) {
                info.free_positions.push_back(j);
            }
        }
    }

    // Генерируем ходы только для релевантных пар веток
    moves.reserve(branches_count * (branches_count - 1));
    
    for (uint16_t src = 0; src < branches_count; ++src) {
        const auto& src_info = branches_info[src];
        if (src_info.last_bird_pos == -1) continue;
        
        for (uint16_t dst = 0; dst < branches_count; ++dst) {
            if (src == dst) continue;
            if (branches_info[dst].free_positions.empty()) continue;
            
            uint8_t dst_pos = branches_info[dst].free_positions[0];
            
            // Проверяем совместимость птиц
            if (dst_pos > 0 && state.at(dst, dst_pos - 1) != src_info.last_bird) {
                continue;
            }
            
            moves.push_back({
                src, static_cast<uint8_t>(src_info.last_bird_pos),
                dst, dst_pos, src_info.last_bird
            });
        }
    }
    
    moves_cache_[tree_hash] = moves;
    return moves;
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