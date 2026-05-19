#pragma once
#include "stdafx.h"

// Coordinate conversion helpers between (row, col) and a flattened node id
static inline NodeId rc_to_id(int r, int c, int cols) {
    return static_cast<NodeId>(r * cols + c);
}
static inline std::pair<int,int> id_to_rc(NodeId id, int cols) {
    return { id / cols, id % cols };
}

// Check whether a coordinate lies inside the grid bounds
static inline bool in_bounds(int r, int c, int rows, int cols) {
    return (0 <= r && r < rows && 0 <= c && c < cols);
}

/**
 * Build a 2-D regular grid graph.
 * @param rows Number of rows (>=1)
 * @param cols Number of columns (>=1)
 * @param undirected Add edges in both directions (default true)
 * @param with_diagonals Include diagonal neighbors as well (default false)
 * @param wrap_rows Enable toroidal wrap in the row direction (default false)
 * @param wrap_cols Enable toroidal wrap in the column direction (default false)
 *
 * All edges are unweighted and node ids follow row-major order in [0, rows*cols).
 */
static inline Graph make_grid_2d(int rows, int cols,
                                 [[maybe_unused]] bool undirected = true,
                                 bool with_diagonals = false,
                                 bool wrap_rows = false,
                                 bool wrap_cols = false)
{
    Graph G(rows * cols);
    if (rows <= 0 || cols <= 0) return G;

    auto wrap = [](int x, int n, bool do_wrap) -> int {
        if (!do_wrap) return x;
        // Use mathematical modulo so negative indices wrap correctly
        int y = x % n;
        if (y < 0) y += n;
        return y;
    };

    // 4-neighborhood offsets: up, down, left, right
    const int dr4[4] = {-1, +1,  0,  0};
    const int dc4[4] = { 0,  0, -1, +1};

    // 4 diagonals (optional)
    const int dr8[4] = {-1, -1, +1, +1};
    const int dc8[4] = {-1, +1, -1, +1};

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            NodeId u = rc_to_id(r, c, cols);

            // Process the 4-neighborhood
            for (int k = 0; k < 4; ++k) {
                int nr = r + dr4[k];
                int nc = c + dc4[k];

                if (!in_bounds(nr, nc, rows, cols)) {
                    // Attempt wrapping on each dimension
                    nr = wrap(nr, rows, wrap_rows);
                    nc = wrap(nc, cols, wrap_cols);
                    // Skip if wrapping is disabled and the coordinate is still out of range
                    if (!in_bounds(nr, nc, rows, cols)) continue;
                }
                NodeId v = rc_to_id(nr, nc, cols);
                G.add_edge(u, v, /*undirected=*/false); // mirror direction will be added when visiting (nr,nc)
            }

            // Optionally include diagonal neighbors
            if (with_diagonals) {
                for (int k = 0; k < 4; ++k) {
                    int nr = r + dr8[k];
                    int nc = c + dc8[k];

                    if (!in_bounds(nr, nc, rows, cols)) {
                        nr = wrap(nr, rows, wrap_rows);
                        nc = wrap(nc, cols, wrap_cols);
                        if (!in_bounds(nr, nc, rows, cols)) continue;
                    }
                    NodeId v = rc_to_id(nr, nc, cols);
                    G.add_edge(u, v, /*undirected=*/false);
                }
            }
        }
    }
    return G;
}

/**
 * Convenience helper: treat every grid node as a candidate.
 */
static inline std::vector<NodeId> make_all_nodes_as_candidates(int rows, int cols) {
    const int n = rows * cols;
    std::vector<NodeId> cand(n);
    std::iota(cand.begin(), cand.end(), 0);
    return cand;
}

struct GridBuildSpec {
    int rows = 0;
    int cols = 0;
    bool diag = false;
    bool torus = false;
};

bool parse_grid_spec(const std::string& spec,
                     GridBuildSpec& out,
                     std::string& error_out);

Graph build_grid_graph(const GridBuildSpec& spec);

bool write_grid_edge_list(const GridBuildSpec& spec,
                          const std::string& path,
                          std::string& error_out);

// ===== Obstacle utilities =====
// Compact representation of blocked/walkable grid cells for obstacle handling.
struct ObstacleMask {
    std::vector<char> blocked;       // size = rows*cols, 1=blocked
    std::vector<NodeId> blocked_nodes;
    std::vector<NodeId> walkable;    // Unblocked nodes
    int rows = 0, cols = 0;
};

// Build an obstacle mask from a list of (row, col) coordinates; out-of-bounds entries are ignored.
static inline ObstacleMask make_obstacles_from_coords(
    int rows, int cols, const std::vector<std::pair<int,int>>& coords)
{
    const int n = rows * cols;
    ObstacleMask ob; ob.rows=rows; ob.cols=cols;
    ob.blocked.assign(n, 0);
    for (auto [r,c] : coords) {
        if (!in_bounds(r,c,rows,cols)) continue;
        NodeId id = rc_to_id(r,c,cols);
        if (!ob.blocked[id]) {
            ob.blocked[id] = 1;
            ob.blocked_nodes.push_back(id);
        }
    }
    ob.walkable.reserve(n - (int)ob.blocked_nodes.size());
    for (int id=0; id<n; ++id) if (!ob.blocked[id]) ob.walkable.push_back(id);
    return ob;
}

// Randomly sample obstacle locations; entries listed in `exclude` are never selected.
static inline ObstacleMask make_random_obstacles(
    int rows, int cols, int num, uint64_t seed,
    const std::vector<NodeId>* exclude = nullptr)
{
    const int n = rows * cols;
    num = std::max(0, std::min(num, n));
    std::vector<char> forbid(n, 0);
    if (exclude) for (auto x : *exclude) if (0<=x && x<n) forbid[x]=1;

    std::vector<NodeId> pool; pool.reserve(n);
    for (int id=0; id<n; ++id) if (!forbid[id]) pool.push_back(id);
    if (num > (int)pool.size()) num = (int)pool.size();

    std::mt19937_64 rng(seed);
    std::shuffle(pool.begin(), pool.end(), rng);

    std::vector<std::pair<int,int>> coords; coords.reserve(num);
    for (int i=0;i<num;++i) {
        int id = pool[i];
        auto rc = id_to_rc(id, cols);
        coords.push_back(rc);
    }
    return make_obstacles_from_coords(rows, cols, coords);
}

// Apply obstacles to the graph by removing edges incident to blocked nodes
static inline void apply_obstacles_to_graph(Graph& G, const ObstacleMask& ob)
{
    const int n = G.n;
    if ((int)ob.blocked.size()!=n) return; // Size mismatch, nothing to do

    // Remove adjacency lists for blocked nodes entirely
    for (auto id : ob.blocked_nodes) {
        if (0<=id && id<n) G.adj[id].clear();
    }
    // Remove any edge that points to a blocked node
    for (int u=0; u<n; ++u) {
        if (u<0 || u>=n) continue;
        if (!G.adj[u].empty()) {
            auto& nbrs = G.adj[u];
            nbrs.erase(std::remove_if(nbrs.begin(), nbrs.end(),
                        [&](NodeId v){ return (0<=v && v<n && ob.blocked[v]); }),
                       nbrs.end());
        }
    }
}

// Return a copy of the walkable nodes to use as candidates
static inline std::vector<NodeId> make_walkable_candidates(const ObstacleMask& ob) {
    return ob.walkable; // return a copy of walkable ids
}
