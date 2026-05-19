#pragma once
#include "stdafx.h"
using namespace std;

using NodeId = int32_t;
using AgentId = int32_t;
using Path = vector<NodeId>;

// Lightweight adjacency-list graph used throughout the project (unweighted).
struct Graph{
    int n;
    vector<vector<NodeId>> adj;
    Graph(int n_=0): n(n_), adj(n_) {}
    void add_edge(NodeId u, NodeId v, bool undirected=true) {
        if ((int)adj.size() < n) adj.resize(n);
        adj[u].push_back(v);
        if (undirected) adj[v].push_back(u);
    }
};

// Weighted neighborhoods for candidate locations
// Each candidate u stores its affected nodes v along with weights w_{v,u}
struct WeightedNeighborhood {
    vector<pair<NodeId,double>> vw;
};
// Stores candidate ids and their weighted neighborhoods (hard/soft radii).
struct WeightsStore {
    vector<NodeId> candId;                 // candIdx -> NodeId
    vector<WeightedNeighborhood> W;        // candIdx -> neighborhood
    vector<int> invCand;                   // NodeId -> candIdx, -1 if not a candidate
};

// Hard cutoff neighbourhood: include nodes within hop-distance <= D (unit weight).
static inline void build_weights_hard_cutoff(
    const Graph& G, const vector<NodeId>& candidates, int D, WeightsStore& out)
{
    out.candId = candidates;
    out.W.assign(candidates.size(), {});
    out.invCand.assign(G.n, -1);
    for (size_t i=0;i<candidates.size();++i) {
        NodeId u = candidates[i];
        if (u<0 || u>=G.n) continue;        // Skip invalid node ids defensively
        out.invCand[u] = (int)i;
    }

    vector<int> dist(G.n, -1);
    vector<int> vis(G.n, 0);  // visitation stamp array
    int stamp = 1;
    deque<NodeId> dq;

    for (size_t ci=0; ci<candidates.size(); ++ci) {
        NodeId src = candidates[ci];
        if (src<0 || src>=G.n) continue;
        auto& neigh = out.W[ci].vw;
        dq.clear();

        // Initialize BFS at the source
        dist[src] = 0; vis[src] = stamp; dq.push_back(src);
        while (!dq.empty()) {
            NodeId x = dq.front(); dq.pop_front();
            if (dist[x] > D) break;
            neigh.emplace_back(x, 1.0);            // assign unit weight within radius D
            if (dist[x] == D) continue;
            for (auto y: G.adj[x]) {
                if (y<0 || y>=G.n) continue;
                if (vis[y] != stamp) {             // equivalent to checking dist[y] == -1
                    vis[y] = stamp;
                    dist[y] = dist[x] + 1;
                    dq.push_back(y);
                }
            }
        }
        // Increment the stamp so the next iteration treats untouched nodes as unseen
        ++stamp;
        if (stamp == INT_MAX){           // Avoid stamp overflow by resetting arrays
            fill(vis.begin(), vis.end(), 0);
            fill(dist.begin(), dist.end(), -1);
            stamp = 1;
        }
    }
}

// Soft decay neighbourhood: weights decay exponentially with distance up to radius R.
static inline void build_weights_soft_decay(
    const Graph& G, const vector<NodeId>& candidates,
    double alpha, int R, double eps, WeightsStore& out,
    bool include_self = true)
{
    out.candId = candidates;
    out.W.assign(candidates.size(), {});
    out.invCand.assign(G.n, -1);
    for (size_t i=0;i<candidates.size();++i) {
        NodeId u = candidates[i];
        if (u<0 || u>=G.n) continue;
        out.invCand[u] = (int)i;
    }

    vector<int> dist(G.n, -1);
    vector<int> vis(G.n, 0);
    int stamp = 1;
    deque<NodeId> dq;

    for (size_t ci=0; ci<candidates.size(); ++ci) {
        NodeId src = candidates[ci];
        if (src<0 || src>=G.n) continue;
        auto& neigh = out.W[ci].vw;
        dq.clear();

        dist[src] = 0; vis[src] = stamp; dq.push_back(src);
        while (!dq.empty()) {
            NodeId x = dq.front(); dq.pop_front();
            if (dist[x] > R) break;
            if (include_self || dist[x] > 0) {
                double w = std::exp(-alpha * dist[x]);
                if (w >= eps) neigh.emplace_back(x, w);
            }
            if (dist[x] == R) continue;
            for (auto y: G.adj[x]) {
                if (y<0 || y>=G.n) continue;
                if (vis[y] != stamp) {
                    vis[y] = stamp;
                    dist[y] = dist[x] + 1;
                    dq.push_back(y);
                }
            }
        }
        ++stamp; if (stamp == INT_MAX) {
            fill(vis.begin(), vis.end(), 0);
            fill(dist.begin(), dist.end(), -1);
            stamp = 1;
        }
    }
}
