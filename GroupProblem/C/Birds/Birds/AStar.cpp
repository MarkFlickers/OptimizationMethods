#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include "AStar.h"

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

    data_.resize(branches_count_ * branch_len_);
    for(uint16_t i = 0; i < branches_count_; ++i)
    {
        for(uint8_t j = 0; j < branch_len_; ++j)
        {
            data_[i * branch_len_ + j] = state[i][j];
        }
    }
    computeHash();
}

void TreeState::computeHash() const
{
    if(!hash_computed_)
    {
        // Упрощенная хэш-функция для скорости
        hash_ = 0;
        for(char c : data_)
        {
            hash_ = hash_ * 131 + static_cast<size_t>(c);
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

Tree::Tree(const Tree& parent, const Move& move) : state_(parent.state_)
{
    // Применяем ход
    state_.at(move.src_branch, move.src_pos) = 0;
    state_.at(move.dst_branch, move.dst_pos) = move.bird;
    state_.computeHash();

    unperfectness_ = computeUnperfectnessIncremental(parent, move);
}

bool Tree::operator==(const Tree& other) const
{
    return state_ == other.state_;
}

size_t Tree::computeUnperfectnessIncremental(const Tree& parent, const Move& move)
{
    size_t delta = 0;

    delta -= ::computeBranchUnperfectnessWithCache(parent.state_, move.src_branch);
    delta -= ::computeBranchUnperfectnessWithCache(parent.state_, move.dst_branch);

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

bool Tree::isBranchComplete(uint16_t branch_index) const
{
    char first_bird = state_.at(branch_index, 0);
    if(first_bird == 0) return false;

    for(uint8_t j = 1; j < state_.getBranchLen(); ++j)
    {
        if(state_.at(branch_index, j) != first_bird)
        {
            return false;
        }
    }
    return true;
}

// Node implementation
Node::Node(Tree&& tree, Node* parent, const Move& move, int g)
    : tree_(std::move(tree)), parent_(parent), move_(move), g_(g),
    hash_(tree_.getHash())
{
    // Упрощенная эвристика
    f_ = g_ + tree_.getUnperfectness();
}

bool Node::operator>(const Node& other) const
{
    return f_ > other.f_;
}

// AStarSolver implementation
AStarSolver::AStarSolver(const TreeState& start_state) : start_state_(start_state) {}

SolvedTree AStarSolver::solve()
{
    node_registry_.clear();

    Tree start_tree(start_state_);
    auto start_node = new Node(std::move(start_tree), nullptr, Move{}, 0);
    node_registry_[start_node->getHash()] = start_node;

    auto compare = [](Node* a, Node* b) { return a->getF() > b->getF(); };
    std::priority_queue<Node*, std::vector<Node*>, decltype(compare)> open_list(compare);
    open_list.push(start_node);

    std::unordered_set<size_t> closed_set;

    while(!open_list.empty())
    {
        Node* current_node = open_list.top();
        open_list.pop();

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

            // Освобождаем память
            for(auto& pair : node_registry_)
            {
                delete pair.second;
            }

            return solution;
        }

        if(shouldPrune(*current_node))
        {
            continue;
        }

        closed_set.insert(current_node->getHash());

        // Оптимизированная генерация ходов
        auto moves = findPossibleMovesOptimized(current_node->getTree());

        for(const auto& move : moves)
        {
            Tree new_tree = applyMove(current_node->getTree(), move);
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
                Node* new_node = new Node(std::move(new_tree), current_node, move, new_g);
                if(registerNode(new_node))
                {
                    open_list.push(new_node);
                }
                else
                {
                    delete new_node;
                }
            }
        }
    }

    // Освобождаем память
    for(auto& pair : node_registry_)
    {
        delete pair.second;
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
        delete it->second;  // Удаляем старый узел
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

bool AStarSolver::shouldPrune(const Node& node) const
{
    return node.getG() > MAX_DEPTH;
}

std::vector<Move> AStarSolver::findPossibleMovesOptimized(const Tree& tree) const
{
    std::vector<Move> moves;
    const auto& state = tree.getState();
    uint16_t branches_count = state.getBranchesCount();
    uint8_t branch_len = state.getBranchLen();

    // Предварительное резервирование памяти
    moves.reserve(branches_count * 2); // Эмпирически подобранное значение

    // Кэшируем информацию о ветках в локальных массивах
    std::vector<int8_t> last_bird_pos(branches_count, -1);
    std::vector<char> last_bird(branches_count, 0);
    std::vector<int8_t> first_free_pos(branches_count, -1);

    // Однократный проход для сбора информации
    for(uint16_t i = 0; i < branches_count; ++i)
    {
        // Ищем последнюю птицу
        for(int8_t j = branch_len - 1; j >= 0; --j)
        {
            if(state.at(i, j) != 0)
            {
                last_bird_pos[i] = j;
                last_bird[i] = state.at(i, j);
                break;
            }
        }

        // Ищем первую свободную позицию
        for(uint8_t j = 0; j < branch_len; ++j)
        {
            if(state.at(i, j) == 0)
            {
                first_free_pos[i] = j;
                break;
            }
        }
    }

    // Генерируем ходы только для релевантных пар
    for(uint16_t src = 0; src < branches_count; ++src)
    {
        if(last_bird_pos[src] == -1) continue;

        for(uint16_t dst = 0; dst < branches_count; ++dst)
        {
            if(src == dst) continue;
            if(first_free_pos[dst] == -1) continue;

            // Проверяем совместимость птиц
            if(first_free_pos[dst] > 0 &&
                state.at(dst, first_free_pos[dst] - 1) != last_bird[src])
            {
                continue;
            }

            moves.push_back({
                src, static_cast<uint8_t>(last_bird_pos[src]),
                dst, static_cast<uint8_t>(first_free_pos[dst]),
                last_bird[src]
                });
        }
    }

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