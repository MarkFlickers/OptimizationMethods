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
TreeState::TreeState(const std::vector<std::vector<char>>& state)
{
    if(state.empty())
    {
        total_branches_ = 0;
        branch_len_ = 0;
        return;
    }

    branches_ = state;
    total_branches_ = static_cast<uint16_t>(state.size());
    branch_len_ = static_cast<uint8_t>(state[0].size());

    // Применяем нормализацию
    normalizeBirds();
    sortBranches();
    computeHash();
}

void TreeState::normalizeBirds(void)
{
    std::unordered_map<char, char> type_map;
    char next_type = 1;  // 0 зарезервирован для пустоты

    for(auto& branch : branches_)
    {
        for(auto& bird : branch)
        {
            if(bird != 0)
            {
                if(type_map.find(bird) == type_map.end())
                {
                    type_map[bird] = next_type++;
                }
                bird = type_map[bird];
            }
        }
    }
}

void TreeState::sortBranches()
{
    std::sort(branches_.begin(), branches_.end(),
        [this](const std::vector<char>& a, const std::vector<char>& b) {
        return compareBranches(a, b) < 0;
    });
}

bool TreeState::isBranchEmpty(const std::vector<char>& branch) const
{
    if(branch[0] != 0)
        return false;
    else
        return true;
}

int TreeState::compareBranches(const std::vector<char>& a, const std::vector<char>& b) const
{
    // Пустые ветки идут в конце
    bool a_empty = isBranchEmpty(a);
    bool b_empty = isBranchEmpty(b);

    if(a_empty && !b_empty) return 1;
    if(!a_empty && b_empty) return -1;
    if(a_empty && b_empty) return 0;

    // Лексикографическое сравнение
    for(uint8_t i = 0; i < branch_len_; ++i)
    {
        if(a[i] != b[i])
        {
            return a[i] - b[i];
        }
    }
    return 0;
}

void TreeState::computeHash() const
{
    if(!hash_computed_)
    {
        hash_ = 0;
        for(const auto& branch : branches_)
        {
            for(char c : branch)
            {
                hash_ = hash_ * 131 + static_cast<size_t>(c);
            }
        }
        hash_computed_ = true;
    }
}

TreeState TreeState::applyMove(const Move& move) const
{
    // Создаем копию текущего состояния
    TreeState new_state = *this;

    // Применяем ход
    auto& src_branch = new_state.branches_[move.src_branch];
    auto& dst_branch = new_state.branches_[move.dst_branch];

    char bird_to_move = src_branch[move.src_pos];
    src_branch[move.src_pos] = 0;
    dst_branch[move.dst_pos] = bird_to_move;

    // Перенормализуем и отсортируем
    new_state.normalizeBirds();
    new_state.sortBranches();
    new_state.hash_computed_ = false;
    new_state.computeHash();

    return new_state;
}

size_t TreeState::getHash() const
{
    computeHash();
    return hash_;
}

bool TreeState::operator==(const TreeState& other) const
{
    return branches_ == other.branches_;
}

// Tree implementation
Tree::Tree(const TreeState& state) : state_(state)
{
    computeUnperfectness();
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
    for(uint16_t i = 0; i < state_.getTotalBranches(); ++i)
    {
        unperfectness_ += computeBranchUnperfectnessWithCache(state_, i);
    }
}

// Node implementation
Node::Node(Tree&& tree, Node* parent, const Move& move, int g)
    : tree_(std::move(tree)), parent_(parent), move_(move), g_(g),
    hash_(tree_.getHash())
{
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
    Node* start_node = new Node(std::move(start_tree), nullptr, Move{}, 0);
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
            return solution;
        }

        if(shouldPrune(*current_node))
        {
            continue;
        }

        closed_set.insert(current_node->getHash());

        auto moves = findPossibleMoves(current_node->getTree());

        for(const auto& move : moves)
        {
            TreeState new_state = current_node->getTree().getState().applyMove(move);
            Tree new_tree(new_state);
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
                    *existing_node = Node(std::move(new_tree), current_node, move, new_g);
                    open_list.push(existing_node);
                }
            }
            else
            {
                Node* new_node = new Node(std::move(new_tree), current_node, move, new_g);
                node_registry_[new_hash] = new_node;
                open_list.push(new_node);
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
        delete it->second;  // Удаляем старый узел
        it->second = node;
    }
    else
    {
        node_registry_[hash] = node;
    }
    return true;
}

bool AStarSolver::shouldPrune(const Node& node) const
{
    return node.getG() > MAX_DEPTH;
}

std::vector<Move> AStarSolver::findPossibleMoves(const Tree& tree) const
{
    std::vector<Move> moves;
    const auto& state = tree.getState();
    const auto& branches = state.getBranches();
    uint16_t branches_count = state.getTotalBranches();
    uint8_t branch_len = state.getBranchLen();

    // Кэшируем информацию о ветках
    std::vector<int8_t> last_bird_pos(branches_count, -1);
    std::vector<int8_t> first_free_pos(branches_count, -1);

    // Однократный проход для сбора информации
    for(uint16_t i = 0; i < branches_count; ++i)
    {
        const auto& branch = branches[i];

        // Ищем последнюю птицу
        for(int8_t j = branch_len - 1; j >= 0; --j)
        {
            if(branch[j] != 0)
            {
                last_bird_pos[i] = j;
                break;
            }
        }

        // Ищем первую свободную позицию
        for(uint8_t j = 0; j < branch_len; ++j)
        {
            if(branch[j] == 0)
            {
                first_free_pos[i] = j;
                break;
            }
        }
    }

    // Генерируем ходы для всех пар веток
    for(uint16_t src = 0; src < branches_count; ++src)
    {
        if(last_bird_pos[src] == -1) continue;

        for(uint16_t dst = 0; dst < branches_count; ++dst)
        {
            if(src == dst) continue;
            if(first_free_pos[dst] == -1) continue;

            const auto& src_branch = branches[src];
            const auto& dst_branch = branches[dst];
            char bird_to_move = src_branch[last_bird_pos[src]];

            // Проверяем совместимость птиц
            if(first_free_pos[dst] > 0 &&
                dst_branch[first_free_pos[dst] - 1] != bird_to_move)
            {
                continue;
            }

            moves.push_back({
                src, static_cast<uint8_t>(last_bird_pos[src]),
                dst, static_cast<uint8_t>(first_free_pos[dst]),
                bird_to_move
                });
        }
    }

    return moves;
}

size_t computeBranchUnperfectnessWithCache(const TreeState& state, uint32_t branch_index)
{
    static std::unordered_map<size_t, size_t> branch_unperfectness_cache_ = {};

    const std::vector<char> & branch_data = state.getBranches().at(branch_index);
    size_t branch_hash = computeBranchHash(branch_data, state.getBranchLen());

    auto it = branch_unperfectness_cache_.find(branch_hash);
    if(it != branch_unperfectness_cache_.end())
    {
        return it->second;
    }

    size_t result = computeBranchUnperfectness(branch_data, state.getBranchLen());
    branch_unperfectness_cache_[branch_hash] = result;
    return result;
}

size_t computeBranchHash(const std::vector<char> &branch_data, uint8_t branch_len)
{
    size_t h = 0;
    for(uint32_t i = 0; i < branch_len; ++i)
    {
        h = h * 31 + static_cast<size_t>(branch_data[i]);
    }
    return h;
}

size_t computeBranchUnperfectness(const std::vector<char> &branch_data, uint8_t branch_len)
{
    if(branch_data[0] == 0) return 0;

    const int MAX_BIRD_TYPES = 26;
    uint16_t freq[MAX_BIRD_TYPES] = {0};
    uint16_t max_freq = 0;

    for(uint8_t j = 0; j < branch_len; ++j)
    {
        char bird = branch_data[j];
        if(bird != 0)
        {
            max_freq = std::max(max_freq, ++freq[bird]);
        }
    }

    return branch_len - max_freq;
}