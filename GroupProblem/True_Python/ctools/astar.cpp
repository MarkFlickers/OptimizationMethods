#include <iostream>
#include <vector>
#include <tuple>
#include <queue>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;

// ---------------------------
// Move
// ---------------------------
struct Move {
    int src_branch, src_pos, dst_branch, dst_pos, bird;
};

// ---------------------------
// State
// ---------------------------
struct State {
    vector<vector<int>> branches;
    vector<int> tops;
    vector<int> first_birds;
    int unperfectness = 0;
    size_t hash_cache = 0;

    State() = default;

    State(const vector<vector<int>>& br) : branches(br) {
        int bcount = branches.size();
        int blen = br.empty() ? 0 : br[0].size();
        tops.resize(bcount, -1);
        first_birds.resize(bcount, 0);
        unperfectness = 0;

        size_t h = 0;
        for (int i = 0; i < bcount; ++i) {
            auto& row = branches[i];
            int top = -1;
            for (int j = blen - 1; j >= 0; --j)
                if (row[j] != 0) { top = j; break; }
            tops[i] = top;
            first_birds[i] = row.empty() ? 0 : row[0];

            if (top == -1) continue;

            int empty_positions = 0;
            for (int j = blen - 1; j >= 0 && row[j] == 0; --j) empty_positions++;

            int ordered_birds = 0;
            int limit = blen - empty_positions;
            int first_bird = row[0];
            for (int j = 0; j < limit; ++j)
                if (row[j] != first_bird) break;
                else ordered_birds++;

            int unordered_birds = limit - ordered_birds;
            int weight_factor = 25;
            unperfectness += unordered_birds * weight_factor + empty_positions;

            // хэш
            for (int val : row)
                h = h * 31 + static_cast<size_t>(val);
        }
        hash_cache = h;
    }

    bool operator==(const State& other) const {
        return branches == other.branches;
    }
};

// ---------------------------
// Hash
// ---------------------------
namespace std {
    template<>
    struct hash<State> {
        size_t operator()(const State& s) const {
            return s.hash_cache;
        }
    };
}

// ---------------------------
// Recalculate unperfectness
// ---------------------------
int compute_unperfectness(const vector<vector<int>>& branches) {
    int unperfectness = 0;
    int bcount = branches.size();
    int blen = branches.empty() ? 0 : branches[0].size();

    for (int i = 0; i < bcount; ++i) {
        int top = -1;
        for (int j = blen - 1; j >= 0; --j)
            if (branches[i][j] != 0) { top = j; break; }
        if (top == -1) continue;

        int empty_positions = 0;
        for (int j = blen - 1; j >= 0 && branches[i][j] == 0; --j) empty_positions++;

        int ordered_birds = 0;
        int limit = blen - empty_positions;
        int first_bird = branches[i][0];
        for (int j = 0; j < limit; ++j)
            if (branches[i][j] != first_bird) break;
            else ordered_birds++;

        int unordered_birds = limit - ordered_birds;
        int weight_factor = 25;
        unperfectness += unordered_birds * weight_factor + empty_positions;
    }
    return unperfectness;
}

// ---------------------------
// Apply move
// ---------------------------
State apply_move(const State& s, const Move& m) {
    vector<vector<int>> bcopy = s.branches;
    int bird = bcopy[m.src_branch][m.src_pos];
    bcopy[m.src_branch][m.src_pos] = 0;
    bcopy[m.dst_branch][m.dst_pos] = bird;

    State ns(bcopy); // пересчёт tops, first_birds, unperfectness, hash
    return ns;
}

// ---------------------------
// Find possible moves
// ---------------------------
vector<Move> find_possible_moves(const State& s) {
    vector<Move> moves;
    int bcount = s.branches.size();
    if (bcount == 0) return moves;
    int blen = s.branches[0].size();

    for (int src = 0; src < bcount; ++src) {
        int src_top = s.tops[src];
        if (src_top == -1) continue;
        int bird = s.branches[src][src_top];
        for (int dst = 0; dst < bcount; ++dst) {
            if (dst == src) continue;
            int dst_pos = -1;
            for (int j = 0; j < blen; ++j)
                if (s.branches[dst][j] == 0) { dst_pos = j; break; }
            if (dst_pos == -1) continue;
            if (dst_pos > 0 && s.branches[dst][dst_pos - 1] != bird) continue;
            moves.push_back({src, src_top, dst, dst_pos, bird});
        }
    }
    return moves;
}

// ---------------------------
// Node
// ---------------------------
struct Node {
    State state;
    int g;
    int f;
    Node(State s, int g_, int f_) : state(s), g(g_), f(f_) {}
};

struct CompareNode {
    bool operator()(const Node& a, const Node& b) { return a.f > b.f; }
};

// ---------------------------
// A* Solver
// ---------------------------
class AStarSolver {
public:
    State start;
    int MAX_DEPTH = 2000;

    AStarSolver(const State& s) : start(s) {}

    tuple<int, vector<Move>, State> solve() {
        priority_queue<Node, vector<Node>, CompareNode> open;
        unordered_map<State, int> g_score;
        unordered_map<State, pair<State, Move>> came_from;

        open.push(Node(start, 0, start.unperfectness));
        g_score[start] = 0;

        while (!open.empty()) {
            Node cur = open.top(); open.pop();
            State cs = cur.state;

            if (cs.unperfectness == 0) {
                vector<Move> path;
                State s = cs;
                while (came_from.find(s) != came_from.end() && came_from[s].first.branches != s.branches) {
                    auto [parent, mv] = came_from[s];
                    path.push_back(mv);
                    s = parent;
                }
                reverse(path.begin(), path.end());
                return { (int)path.size(), path, cs };
            }
            if (cur.g > MAX_DEPTH) continue;

            for (auto& mv : find_possible_moves(cs)) {
                State ns = apply_move(cs, mv);
                int tentative_g = cur.g + 1;

                auto it = g_score.find(ns);
                int cur_g = (it != g_score.end()) ? it->second : INT32_MAX;
                if (tentative_g < cur_g) {
                    g_score[ns] = tentative_g;
                    came_from[ns] = {cs, mv};
                    open.push(Node(ns, tentative_g, tentative_g + ns.unperfectness));
                }
            }
        }
        return {0, {}, start};
    }
};

// ---------------------------
// Main
// ---------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: astar.exe input.json\n";
        return 1;
    }

    ifstream fin(argv[1]);
    json j; fin >> j;
    auto br = j["branches"];
    vector<vector<int>> branches = br.get<vector<vector<int>>>();

    State start(branches);
    AStarSolver solver(start);
    auto [steps_count, sol, final_state] = solver.solve();

    json jout;
    jout["steps"] = steps_count;
    jout["moves"] = json::array();
    for (auto& m : sol) {
        jout["moves"].push_back({
            {"src_branch", m.src_branch},
            {"src_pos", m.src_pos},
            {"dst_branch", m.dst_branch},
            {"dst_pos", m.dst_pos},
            {"bird", m.bird}
        });
    }
    jout["result_tree"] = final_state.branches;
    cout << jout.dump() << endl;

    return 0;
}
