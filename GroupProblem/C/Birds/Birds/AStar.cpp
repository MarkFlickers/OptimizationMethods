#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <array>
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

    normalizeBirds();
    sortBranches();
    computeHash();
}

void TreeState::normalizeBirds(void)
{
    std::array<char, 256> type_map;
    type_map.fill(0);
    char next_type = 1;

    for(auto& branch : branches_)
    {
        for(auto& bird : branch)
        {
            if(bird != 0)
            {
                unsigned char ub = static_cast<unsigned char>(bird);
                if(type_map[ub] == 0)
                {
                    type_map[ub] = next_type++;
                }
                bird = type_map[ub];
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
	// Empty branches in the end
    bool a_empty = isBranchEmpty(a);
    bool b_empty = isBranchEmpty(b);

    if(a_empty && !b_empty) return 1;
    if(!a_empty && b_empty) return -1;
    if(a_empty && b_empty) return 0;

	// Lexicographical comparison
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
    TreeState new_state = *this;

    auto& src_branch = new_state.branches_[move.src_branch];
    auto& dst_branch = new_state.branches_[move.dst_branch];

    char bird_to_move = src_branch[move.src_pos];
    src_branch[move.src_pos] = 0;
    dst_branch[move.dst_pos] = bird_to_move;

    new_state.normalizeBirds();
    new_state.sortBranches();
    new_state.hash_computed_ = false;

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

void Tree::computeUnperfectness()
{
    unperfectness_ = 0;
    for(const auto& branch : state_.getBranches())
    {
        if(branch[0] == 0) continue;

        char first_bird = branch[0];
        size_t same_count = 1;

        for(size_t j = 1; j < branch.size(); ++j)
        {
            if(branch[j] == first_bird)
            {
                same_count++;
            }
        }

        unperfectness_ += branch.size() - same_count;
    }
}

// Node implementation
Node::Node(Tree&& tree, Node* parent, const Move& move, int g)
    : tree_(std::move(tree)), parent_(parent), move_(move), g_(g),
    hash_(tree_.getHash())
{
    f_ = g_ + tree_.getUnperfectness();
}

void Node::update(Node* parent, const Move& move, int g)
{
    parent_ = parent;
    move_ = move;
    g_ = g;
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
    node_registry_.reserve(16384);

    Tree start_tree(start_state_);
    Node* start_node = new Node(std::move(start_tree), nullptr, Move{}, 0);
    node_registry_[start_node->getHash()] = start_node;

    auto compare = [](Node* a, Node* b) { return a->getF() > b->getF(); };
    std::vector<Node*> open_container;
    open_container.reserve(16384);
    std::priority_queue<Node*, std::vector<Node*>, decltype(compare)> open_list(compare, std::move(open_container));
    open_list.push(start_node);

    std::unordered_set<size_t> closed_set;
    closed_set.reserve(16384);

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
			// Creater new state by applying the move
            TreeState new_state = current_node->getTree().getState().applyMove(move);
            size_t new_hash = new_state.getHash();
            int new_g = current_node->getG() + 1;

            if(closed_set.count(new_hash))
            {
                continue;
            }

            auto it = node_registry_.find(new_hash);
            if(it != node_registry_.end())
            {
                Node* existing_node = it->second;
				// If the states are equal, check if we found a better path
                if(existing_node->getTree().getState() == new_state)
                {
                    if(new_g < existing_node->getG())
                    {
						// Update existing node with better path
                        existing_node->update(current_node, move, new_g);
                        open_list.push(existing_node);
                    }
                }
                else
                {
					// Hash collision detected, create a new node
                    Tree new_tree(new_state);
                    *existing_node = Node(std::move(new_tree), current_node, move, new_g);
                    open_list.push(existing_node);
                }
            }
            else
            {
                Tree new_tree(new_state);
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
        delete it->second;
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

	moves.reserve(branches_count * 8); // Some arbitrary number to reduce reallocations

    for(uint16_t src = 0; src < branches_count; ++src)
    {
        const auto& src_branch = branches[src];

		// Skip empty branches
        if(src_branch[0] == 0) continue;

		// Find the topmost bird in the source branch
        int8_t src_pos = -1;
        for(int8_t j = branch_len - 1; j >= 0; --j)
        {
            if(src_branch[j] != 0)
            {
                src_pos = j;
                break;
            }
        }
        if(src_pos == -1) continue;

        char bird_to_move = src_branch[src_pos];

        for(uint16_t dst = 0; dst < branches_count; ++dst)
        {
            if(src == dst) continue;

            const auto& dst_branch = branches[dst];

			// Find the first empty position in the destination branch
            int8_t dst_pos = -1;
            for(uint8_t j = 0; j < branch_len; ++j)
            {
                if(dst_branch[j] == 0)
                {
                    dst_pos = j;
                    break;
                }
            }
            if(dst_pos == -1) continue;

			// Check the movement rules
            if(dst_pos > 0 && dst_branch[dst_pos - 1] != bird_to_move)
            {
                continue;
            }

            moves.push_back({
                src, static_cast<uint8_t>(src_pos),
                dst, static_cast<uint8_t>(dst_pos),
                bird_to_move
                });
        }
    }

    return moves;
}