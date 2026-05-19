#include "stdafx.h"
#include "alg.h"

using std::make_pair;
using std::max;
using std::min;
using std::mt19937_64;
using std::numeric_limits;
using std::pair;
using std::priority_queue;
using std::queue;
using std::size_t;
using std::string;
using std::tuple;
using std::unordered_map;
using std::unordered_set;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::vector;

static bool logged_psi_assumption = false;

static string cap_mode_to_string(CapMode mode);

static void bfs_on_adj(const vector<vector<NodeId>>& adj,
                       NodeId src,
                       vector<int>& dist) {
    dist.assign(static_cast<int>(adj.size()), -1);
    if (src < 0 || src >= static_cast<NodeId>(adj.size())) return;
    std::deque<NodeId> dq;
    dist[src] = 0;
    dq.push_back(src);
    while (!dq.empty()) {
        NodeId u = dq.front();
        dq.pop_front();
        for (NodeId v : adj[u]) {
            if (v < 0 || v >= static_cast<NodeId>(adj.size())) continue;
            if (dist[v] != -1) continue;
            dist[v] = dist[u] + 1;
            dq.push_back(v);
        }
    }
}

static void build_reverse_adjacency(const Graph& G,
                                    vector<vector<NodeId>>& rev_adj) {
    rev_adj.assign(G.n, {});
    for (NodeId u = 0; u < G.n; ++u) {
        for (NodeId v : G.adj[u]) {
            if (v < 0 || v >= G.n) continue;
            rev_adj[v].push_back(u);
        }
    }
}

static bool compute_centerline_and_corridor(const Graph& G,
                                            NodeId s, NodeId t,
                                            const vector<vector<NodeId>>& rev_adj,
                                            int corridor_band_r,
                                            vector<char>& corridor_mask,
                                            vector<NodeId>& corridor_nodes) {
    if (s < 0 || s >= G.n || t < 0 || t >= G.n) return false;
    vector<int> dist_s(G.n, -1);
    vector<int> dist_t(G.n, -1);
    bfs_on_adj(G.adj, s, dist_s);
    if (dist_s[t] == -1) return false;
    bfs_on_adj(rev_adj, t, dist_t);

    vector<char> centerline(G.n, 0);
    for (NodeId v = 0; v < G.n; ++v) {
        if (dist_s[v] >= 0 && dist_t[v] >= 0 && dist_s[v] + dist_t[v] == dist_s[t]) {
            centerline[v] = 1;
        }
    }
    if (!centerline[s] || !centerline[t]) return false;

    corridor_mask.assign(G.n, 0);
    corridor_nodes.clear();
    vector<int> dist(G.n, -1);
    std::deque<NodeId> dq;

    for (NodeId v = 0; v < G.n; ++v) {
        if (!centerline[v]) continue;
        corridor_mask[v] = 1;
        dist[v] = 0;
        dq.push_back(v);
        corridor_nodes.push_back(v);
    }

    if (corridor_band_r > 0) {
        while (!dq.empty()) {
            NodeId u = dq.front();
            dq.pop_front();
            if (dist[u] == corridor_band_r) continue;
            for (NodeId v : G.adj[u]) {
                if (v < 0 || v >= G.n) continue;
                if (dist[v] != -1) continue;
                int nd = dist[u] + 1;
                if (nd > corridor_band_r) continue;
                dist[v] = nd;
                corridor_mask[v] = 1;
                corridor_nodes.push_back(v);
                dq.push_back(v);
            }
        }
    }

    if (!corridor_mask[s] || !corridor_mask[t]) return false;

    std::sort(corridor_nodes.begin(), corridor_nodes.end());
    corridor_nodes.erase(std::unique(corridor_nodes.begin(), corridor_nodes.end()), corridor_nodes.end());
    return true;
}

static inline int path_length_edges(const Path& P){
    if (P.empty()) return 0;
    return static_cast<int>(P.size()) > 0 ? std::max(0, static_cast<int>(P.size()) - 1) : 0;
}

struct AgentUnionData {
    vector<vector<NodeId>> nodes_per_agent;
    vector<int> min_path_len;
    vector<int> max_path_len;
    int max_paths_per_agent = 0;
};

static AgentUnionData build_agent_union_data(const vector<vector<Path>>& paths_per_agent,
                                             int n,
                                             const vector<vector<NodeId>>* corridor_nodes_per_agent){
    AgentUnionData data;
    size_t num_agents = paths_per_agent.size();
    data.nodes_per_agent.assign(num_agents, {});
    data.min_path_len.assign(num_agents, 0);
    data.max_path_len.assign(num_agents, 0);
    data.max_paths_per_agent = 0;

    vector<int> last_seen(n, -1);
    bool use_corridor_nodes = corridor_nodes_per_agent && corridor_nodes_per_agent->size() == num_agents;
    for (size_t i = 0; i < num_agents; ++i){
        const auto& candidate_paths = paths_per_agent[i];
        data.max_paths_per_agent = std::max<int>(data.max_paths_per_agent,
                                                 static_cast<int>(candidate_paths.size()));
        int min_len = std::numeric_limits<int>::max();
        int max_len = 0;
        auto& nodes = data.nodes_per_agent[i];
        nodes.clear();
        for (const Path& P : candidate_paths){
            int len = path_length_edges(P);
            min_len = std::min(min_len, len);
            max_len = std::max(max_len, len);
            if (!use_corridor_nodes){
                for (NodeId v : P){
                    if (v < 0 || v >= n) continue;
                    if (last_seen[v] != static_cast<int>(i)){
                        last_seen[v] = static_cast<int>(i);
                        nodes.push_back(v);
                    }
                }
            }
        }
        if (min_len == std::numeric_limits<int>::max()) min_len = 0;
        data.min_path_len[i] = min_len;
        data.max_path_len[i] = max_len;
        if (use_corridor_nodes){
            const auto& corridor_nodes = (*corridor_nodes_per_agent)[i];
            for (NodeId v : corridor_nodes){
                if (v < 0 || v >= n) continue;
                nodes.push_back(v);
            }
            std::sort(nodes.begin(), nodes.end());
            nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        }
    }
    return data;
}

static bool psi_is_bounded(const Psi& psi){
    return psi.type == Psi::Type::ClipMinC || psi.type == Psi::Type::ExpSat;
}

static double psi_upper_bound(const Psi& psi){
    switch (psi.type){
        case Psi::Type::ClipMinC: return psi.C;
        case Psi::Type::ExpSat:   return 1.0;
        default: return std::numeric_limits<double>::infinity();
    }
}

static bool psi_is_monotone_nonnegative(const Psi&){
    // All supported psi types are nonnegative and nondecreasing on [0, +inf).
    return true;
}

CapResolution resolve_cap_strategy(const SandwichConfig& cfg){
    CapResolution res;
    res.effective_mode = cfg.cap_mode;
    res.psi_bounded = psi_is_bounded(cfg.psi);
    if (cfg.cap_mode == CapMode::Auto){
        res.auto_selected = true;
        res.effective_mode = res.psi_bounded ? CapMode::PathlenOnly : CapMode::TopkLocal;
    }
    if (res.effective_mode == CapMode::PathlenOnly && !res.psi_bounded){
        res.pathlen_fallback = true;
        res.effective_mode = CapMode::TopkLocal;
    }
    if (res.effective_mode == CapMode::TopkLocal){
        int topk = cfg.cap_topk_k > 0 ? cfg.cap_topk_k : cfg.k;
        if (topk <= 0 && cfg.k > 0) topk = cfg.k;
        if (topk < 0) topk = 0;
        res.topk_k = topk;
    } else {
        res.topk_k = 0;
    }
    return res;
}

static uint64_t hash_weights_store(const WeightsStore& W){
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < W.candId.size(); ++i){
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(W.candId[i]));
        h *= 1099511628211ull;
        for (const auto& vw : W.W[i].vw){
            h ^= static_cast<uint64_t>(static_cast<uint32_t>(vw.first));
            h *= 1099511628211ull;
            uint64_t bits;
            static_assert(sizeof(double) == sizeof(uint64_t), "Unexpected double size");
            std::memcpy(&bits, &vw.second, sizeof(double));
            h ^= bits;
            h *= 1099511628211ull;
        }
    }
    return h;
}

static std::string to_hex_string(uint64_t value){
    std::ostringstream oss;
    oss << std::hex << value;
    return oss.str();
}

struct CapComputationResult {
    vector<double> caps;
    CapMode mode_used = CapMode::PathlenOnly;
    int topk_k_used = 0;
    double cap_min = 0.0;
    double cap_median = 0.0;
    double cap_max = 0.0;
    std::string cache_key_hash;
};

static CapComputationResult compute_caps(const SandwichConfig& cfg,
                                         const Psi& psi,
                                         const vector<vector<Path>>& paths,
                                         const AgentUnionData& union_data,
                                         const WeightsStore& W,
                                         int n,
                                         const vector<NodeId>& candidates);

// Inverse standard normal CDF (Acklam approximation)
static double inverse_normal_cdf(double p){
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return  std::numeric_limits<double>::infinity();

    static const double a1 = -3.969683028665376e+01;
    static const double a2 =  2.209460984245205e+02;
    static const double a3 = -2.759285104469687e+02;
    static const double a4 =  1.383577518672690e+02;
    static const double a5 = -3.066479806614716e+01;
    static const double a6 =  2.506628277459239e+00;

    static const double b1 = -5.447609879822406e+01;
    static const double b2 =  1.615858368580409e+02;
    static const double b3 = -1.556989798598866e+02;
    static const double b4 =  6.680131188771972e+01;
    static const double b5 = -1.328068155288572e+01;

    static const double c1 = -7.784894002430293e-03;
    static const double c2 = -3.223964580411365e-01;
    static const double c3 = -2.400758277161838e+00;
    static const double c4 = -2.549732539343734e+00;
    static const double c5 =  4.374664141464968e+00;
    static const double c6 =  2.938163982698783e+00;

    static const double d1 =  7.784695709041462e-03;
    static const double d2 =  3.224671290700398e-01;
    static const double d3 =  2.445134137142996e+00;
    static const double d4 =  3.754408661907416e+00;

    const double plow  = 0.02425;
    const double phigh = 1.0 - plow;
    double q, r;

    if (p < plow){
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
               ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    }
    if (p > phigh){
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
                ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
    }

    q = p - 0.5;
    r = q * q;
    return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
           (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0);
}

static EvalStats summarize_samples(const vector<double>& samples, double alpha){
    EvalStats stats;
    stats.M = static_cast<int>(samples.size());
    if (samples.empty()) return stats;

    double mean = 0.0;
    double m2 = 0.0;
    int n = 0;
    for (double x : samples){
        ++n;
        double delta = x - mean;
        mean += delta / n;
        double delta2 = x - mean;
        m2 += delta * delta2;
    }
    double variance = (n > 1) ? m2 / (n - 1) : 0.0;
    double standard_error = (n > 0) ? std::sqrt(variance / n) : 0.0;
    double clamped_alpha = (alpha <= 0.0 || alpha >= 1.0) ? 0.05 : alpha;
    double z = inverse_normal_cdf(1.0 - clamped_alpha * 0.5);

    stats.mean = mean;
    stats.var = variance;
    stats.se = standard_error;
    stats.ci_low = mean - z * standard_error;
    stats.ci_high = mean + z * standard_error;
    return stats;
}

// ========================= 1) Psi =========================
Psi Psi::Clip(double C_) { Psi p; p.type = Type::ClipMinC; p.C = C_; return p; }
Psi Psi::Log1p()        { Psi p; p.type = Type::Log1p; return p; }
Psi Psi::ExpSat(double beta_) { Psi p; p.type = Type::ExpSat; p.beta = beta_; return p; }
Psi Psi::Power(double alpha_) { Psi p; p.type = Type::Power; p.alpha = alpha_; return p; }

double Psi::apply(double t) const {
    if (t <= 0.0) return 0.0;
    switch (type) {
        case Type::ClipMinC: return t < C ? t : C; 
        case Type::Log1p:    return std::log1p(t);
        case Type::ExpSat:   return 1.0 - std::exp(-beta * t);
        case Type::Power:    return std::pow(t, std::max(0.0, std::min(1.0, alpha)));
    }
    return t; // fallback
}

// ========================= 2) AgentSampler =========================
void AgentSampler::init(const AgentSamplerConfig& cfg,
                        const vector<NodeId>& walkable,
                        const vector<char>* blocked){
    cfg_ = cfg; blocked_ = blocked; walkable_ = walkable;
    if (cfg_.seed == 0){
        std::random_device rd;
        cfg_.seed = static_cast<uint64_t>(rd());
    }
    rng_.seed(cfg_.seed);
    rebuild_hotspot_cache();
}

void AgentSampler::reseed(uint64_t seed){
    cfg_.seed = seed;
    if (cfg_.seed == 0){
        std::random_device rd;
        cfg_.seed = static_cast<uint64_t>(rd());
    }
    rng_.seed(cfg_.seed);
    rebuild_hotspot_cache();
}

void AgentSampler::rebuild_hotspot_cache(){
    hotspot_nodes_.clear();
    hotspot_prefix_.clear();
    hotspot_total_ = 0.0;
    if (cfg_.hotspots.empty()) return;

    for (const auto& hw : cfg_.hotspots){
        NodeId node = hw.first;
        double weight = hw.second;
        if (weight <= 0.0) continue;
        if (node < 0) continue;
        if (blocked_ && static_cast<size_t>(node) < blocked_->size() && (*blocked_)[node]) continue;
        hotspot_nodes_.push_back(node);
        hotspot_total_ += weight;
        hotspot_prefix_.push_back(hotspot_total_);
    }

    if (hotspot_total_ <= 0.0){
        hotspot_nodes_.clear();
        hotspot_prefix_.clear();
        hotspot_total_ = 0.0;
    }
}

bool AgentSampler::sample_hotspot(NodeId& out) const{
    if (hotspot_nodes_.empty() || hotspot_total_ <= 0.0) return false;
    std::uniform_real_distribution<double> dist(0.0, hotspot_total_);
    double r = dist(rng_);
    auto it = std::lower_bound(hotspot_prefix_.begin(), hotspot_prefix_.end(), r);
    if (it == hotspot_prefix_.end()){
        out = hotspot_nodes_.back();
        if (blocked_ && static_cast<size_t>(out) < blocked_->size() && (*blocked_)[out]) return false;
        return true;
    }
    size_t idx = static_cast<size_t>(std::distance(hotspot_prefix_.begin(), it));
    out = hotspot_nodes_[idx];
    if (blocked_ && static_cast<size_t>(out) < blocked_->size() && (*blocked_)[out]) return false;
    return true;
}

static inline int manhattan_dist_rc([[maybe_unused]] int rows, [[maybe_unused]] int cols, NodeId a, NodeId b){
    auto ra = id_to_rc(a, cols); auto rb = id_to_rc(b, cols);
    return std::abs(ra.first - rb.first) + std::abs(ra.second - rb.second);
}

static bool bfs_reachable(const Graph& G, NodeId s, NodeId t){
    if (s == t) return true;

    vector<char> vis(G.n, 0);
    std::deque<NodeId> dq;
    vis[s] = 1;
    dq.push_back(s);

    while (!dq.empty()){
        NodeId u = dq.front();
        dq.pop_front();
        for (NodeId v : G.adj[u]){
            if (vis[v]) continue;
            vis[v] = 1;
            if (v == t) return true;
            dq.push_back(v);
        }
    }
    return false;
}

bool AgentSampler::sample_od(const Graph& G, NodeId& s, NodeId& t, int max_tries) const {
    if (walkable_.empty()) return false;
    std::uniform_int_distribution<int> U(0, (int)walkable_.size()-1);
    int tries = 0; 
    const bool band = (cfg_.dist==ODDistribution::UniformWithDistBand);
    const bool gauss = (cfg_.dist==ODDistribution::GaussianAroundStart);
    const bool hot   = (cfg_.dist==ODDistribution::HotspotMixture);
    const bool grav  = (cfg_.dist==ODDistribution::Gravity);
    int dmin = cfg_.dist_band.first, dmax = cfg_.dist_band.second;

    while (tries++ < max_tries){
        NodeId a = -1;
        NodeId b = -1;
        if (hot){
            // Hotspot mixture: draw s and t from weighted hotspots, fall back to uniform if invalid
            if (!sample_hotspot(a)) a = walkable_[U(rng_)];
            if (!sample_hotspot(b)) b = walkable_[U(rng_)];
        } else {
            a = walkable_[U(rng_)];
            if (gauss){
                // GaussianAroundStart: sample (dr,dc) ~ N(0, sigma^2) and round to a lattice move
                std::normal_distribution<double> N(0.0, cfg_.sigma);
                auto rc = id_to_rc(a, cfg_.cols);
                int tries2 = 0;
                do {
                    int rr = rc.first  + (int)std::llround(N(rng_));
                    int cc = rc.second + (int)std::llround(N(rng_));
                    if (!in_bounds(rr, cc, cfg_.rows, cfg_.cols)) continue;
                    NodeId cand = rc_to_id(rr, cc, cfg_.cols);
                    if (blocked_ && (*blocked_)[cand]) continue;
                    b = cand;
                    break;
                } while (++tries2 < 50);
                if (b == -1) b = walkable_[U(rng_)];
            } else if (grav){
                // Gravity: sample b uniformly and accept with probability exp(-beta * d)
                int tries2 = 0;
                do {
                    b = walkable_[U(rng_)];
                    if (a == b) continue;
                    int d = manhattan_dist_rc(cfg_.rows, cfg_.cols, a, b);
                    double accp = std::exp(-cfg_.decay_beta * (double)d); // bounded by 1
                    std::uniform_real_distribution<double> U01(0.0,1.0);
                    if (U01(rng_) <= accp) break;
                } while (++tries2 < 50);
            } else {
                // Uniform or UniformWithDistBand
                b = walkable_[U(rng_)];
            }
        }
        if (a == b) continue;
        if (band){
            int d = manhattan_dist_rc(cfg_.rows, cfg_.cols, a, b);
            if (d < dmin || d > dmax) continue;
        }
        if (!bfs_reachable(G, a, b)) continue;
        s = a;
        t = b;
        return true;
    }
    return false;
}

// ========================= 3) Shortest-path DAG & sampling =========================
bool build_shortestpath_dag(const Graph& G, NodeId s, NodeId t,
                            vector<vector<NodeId>>& dag_adj,
                            vector<int>& dist,
                            const vector<char>* allowed_mask){
    dist.assign(G.n, -1);
    if (allowed_mask) {
        if (s < 0 || s >= G.n || t < 0 || t >= G.n) return false;
        if (!(*allowed_mask)[s] || !(*allowed_mask)[t]) return false;
    }
    std::deque<NodeId> dq;
    dq.push_back(s);
    dist[s] = 0;

    while (!dq.empty()){
        NodeId u = dq.front();
        dq.pop_front();
        for (NodeId v : G.adj[u]){
            if (allowed_mask && (v < 0 || v >= G.n || !(*allowed_mask)[v])) continue;
            if (dist[v] != -1) continue;
            dist[v] = dist[u] + 1;
            dq.push_back(v);
        }
    }
    if (t < 0 || t >= G.n || dist[t] == -1) return false;
    dag_adj.assign(G.n, {});
    for (NodeId u = 0; u < G.n; ++u){
        if (dist[u] == -1) continue;
        for (NodeId v : G.adj[u]){
            if (allowed_mask && (v < 0 || v >= G.n || !(*allowed_mask)[v])) continue;
            if (dist[v] == dist[u] + 1) dag_adj[u].push_back(v);
        }
    }
    return true;
}

// Layered bottom-up DP over dist to count shortest paths to t in the DAG
static bool dp_count_paths_to_t(const vector<vector<NodeId>>& dag_adj,
                                const vector<int>& dist,
                                NodeId t,
                                vector<double>& cnt){
    const int n = static_cast<int>(dag_adj.size());
    cnt.assign(n, 0.0);
    if (t < 0 || t >= n || dist[t] < 0) return false;
    cnt[t] = 1.0;

    int dmax = 0;
    for (int i = 0; i < n; ++i){
        if (dist[i] >= 0) dmax = std::max(dmax, dist[i]);
    }

    for (int d = dmax - 1; d >= 0; --d){
        for (int u = 0; u < n; ++u){
            if (dist[u] != d) continue;
            double s = 0.0;
            for (auto v : dag_adj[u]) s += cnt[v];
            cnt[u] = s;
        }
    }
    return true;
}

static void sample_one_path_from_dag(const vector<vector<NodeId>>& dag_adj,
                                     NodeId s, NodeId t,
                                     const vector<double>& cnt,
                                     mt19937_64& rng,
                                     Path& out){
    out.clear();
    NodeId cur = s;
    out.push_back(cur);
    while (cur != t){
        const auto& nxts = dag_adj[cur];
        if (nxts.empty()){
            out.clear();
            return;
        }
        double total = 0.0;
        for (auto v : nxts) total += cnt[v];
        std::uniform_real_distribution<double> U(0.0, total);
        double r = U(rng);
        double acc = 0.0;
        NodeId pick = nxts.back();
        for (auto v : nxts){
            acc += cnt[v];
            if (r <= acc){
                pick = v;
                break;
            }
        }
        out.push_back(pick);
        cur = pick;
    }
}

static uint64_t hash_path(const Path& p){
    uint64_t h = 1469598103934665603ull; // FNV-1a
    for (NodeId x: p){ h ^= (uint64_t)(uint32_t)x; h *= 1099511628211ull; }
    return h;
}

bool sample_k_shortest_paths_uniform_from_dag(const vector<vector<NodeId>>& dag_adj,
                                              NodeId s, NodeId t, int K, uint64_t seed,
                                              vector<Path>& out_paths){
    out_paths.clear();
    if (K <= 0) return true;

    const int n = static_cast<int>(dag_adj.size());
    vector<int> dist(n, -1);
    dist[s] = 0;
    std::deque<NodeId> dq;
    dq.push_back(s);
    while (!dq.empty()){
        NodeId u = dq.front();
        dq.pop_front();
        for (auto v : dag_adj[u]){
            if (dist[v] != -1) continue;
            dist[v] = dist[u] + 1;
            dq.push_back(v);
        }
    }

    vector<double> cnt;
    if (!dp_count_paths_to_t(dag_adj, dist, t, cnt)) return false;

    std::mt19937_64 rng(seed);
    unordered_set<uint64_t> seen;
    const int MAX_TRY = std::max(10 * K, 50);
    for (int tries = 0; static_cast<int>(out_paths.size()) < K && tries < MAX_TRY; ++tries){
        Path p;
        sample_one_path_from_dag(dag_adj, s, t, cnt, rng, p);
        if (p.empty()) continue;
        auto h = hash_path(p);
        if (seen.insert(h).second) out_paths.push_back(std::move(p));
    }
    return !out_paths.empty();
}

bool sample_k_shortest_paths_uniform(const Graph& G, NodeId s, NodeId t,
                                     int K, uint64_t seed, vector<Path>& out_paths,
                                     const vector<char>* allowed_mask){
    vector<vector<NodeId>> dag;
    vector<int> dist;
    if (!build_shortestpath_dag(G, s, t, dag, dist, allowed_mask)) return false;

    vector<double> cnt;
    if (!dp_count_paths_to_t(dag, dist, t, cnt)) return false;

    out_paths.clear();
    if (K <= 0) return true;

    std::mt19937_64 rng(seed);
    unordered_set<uint64_t> seen;
    const int MAX_TRY = std::max(10 * K, 50);
    for (int tries = 0; static_cast<int>(out_paths.size()) < K && tries < MAX_TRY; ++tries){
        Path p;
        sample_one_path_from_dag(dag, s, t, cnt, rng, p);
        if (p.empty()) continue;
        auto h = hash_path(p);
        if (seen.insert(h).second) out_paths.push_back(std::move(p));
    }
    return !out_paths.empty();
}

// Grid combinatorial sampler
static inline double logC(int n, int k){
    if (k<0 || k>n) return -1e300; // -inf
    if (k>n-k) k=n-k;
    double s=0.0; for (int i=1;i<=k;++i){ s += std::log((double)(n-k+i)) - std::log((double)i); }
    return s;
}

bool sample_k_l1_shortest_paths_grid([[maybe_unused]] int rows, int cols, NodeId s, NodeId t,
                                     int K, uint64_t seed, vector<Path>& out_paths){
    out_paths.clear();
    if (K <= 0) return true;

    auto rs = id_to_rc(s, cols);
    auto rt = id_to_rc(t, cols);
    int dr = rt.first - rs.first;
    int dc = rt.second - rs.second;
    int vr = (dr > 0) ? +1 : -1;
    int vc = (dc > 0) ? +1 : -1;
    int R = std::abs(dr);
    int C = std::abs(dc);
    int steps = R + C;
    if (steps == 0) return false;

    std::mt19937_64 rng(seed);
    unordered_set<uint64_t> seen;
    const int MAX_TRY = std::max(10 * K, 50);
    for (int tries = 0; static_cast<int>(out_paths.size()) < K && tries < MAX_TRY; ++tries){
        int r = rs.first;
        int c = rs.second;
        Path p;
        p.reserve(steps + 1);
        p.push_back(s);
        int remR = R;
        int remC = C;
        int rem = steps;
        while (rem > 0){
            double log_tot = logC(rem, remR); // choose positions of vertical steps
            double probV = 0.0;
            if (remR > 0){
                double log_v = logC(rem - 1, remR - 1);
                probV = std::exp(log_v - log_tot);
            }
            std::uniform_real_distribution<double> U(0.0, 1.0);
            bool takeV = (U(rng) < probV);
            if (takeV){
                r += vr;
                --remR;
            } else {
                c += vc;
                --remC;
            }
            p.push_back(rc_to_id(r, c, cols));
            --rem;
        }
        auto h = hash_path(p);
        if (seen.insert(h).second) out_paths.push_back(std::move(p));
    }
    return !out_paths.empty();
}

// ========================= 4) CorridorAccumulator =========================
void CorridorAccumulator::init(const CorridorAccumulatorConfig& cfg){
    n_ = cfg.n; r_ = std::max(1, cfg.r);
    b_r_.assign(n_, 0); in_union_.assign(n_, 0); c_total_.assign(n_, 0);
    Lsum_r_ = 0; Lmin1_ = 0; Lsum_all_ = 0;
}

void CorridorAccumulator::add_agent_paths(const vector<Path>& paths_for_agent){
    if (n_ <= 0) return;
    if (paths_for_agent.empty()) return;

    vector<int> cnt(n_, 0);
    vector<char> seen(n_, 0);
    int K = (int)paths_for_agent.size();
    int Lmin = INT_MAX;
    for (const auto& P : paths_for_agent){
        int Lp = std::max(0, (int)P.size() - 1);
        Lmin = std::min(Lmin, Lp);
        for (NodeId v : P){
            if (!seen[v]){
                seen[v] = 1;
                in_union_[v] += 1;
            }
            cnt[v] += 1;
            c_total_[v] += 1;
        }
        Lsum_all_ += Lp;
    }
    if (Lmin == INT_MAX) Lmin = 0;
    Lmin1_ += Lmin;
    for (int v = 0; v < n_; ++v){
        if (cnt[v] > 0){
            b_r_[v] += std::min(r_, cnt[v]);
        }
    }
    if (K > 0) Lsum_r_ += (long long)std::min(r_, K) * Lmin;
}

// ========================= 5) UB / LB Objectives =========================
void LinearWeightsObjective::init(const WeightsStore& W, int n,
                                  const Psi& psi, double lambda,
                                  const vector<int>* weight_multiplier,
                                  double const_term,
                                  const vector<NodeId>& candidate_nodes){
    W_ = &W;
    n_ = n;
    psi_ = psi;
    lambda_ = lambda;
    weight_multiplier_ = weight_multiplier;
    const_term_ = const_term;
    candidates_ = candidate_nodes;

    m_v_.assign(n_, 0.0);
    psi_v_.assign(n_, psi_.apply(0.0));
    S_.clear();

    inv_.assign(n_, -1);
    for (size_t i = 0; i < W.candId.size(); ++i){
        inv_[W.candId[i]] = static_cast<int>(i);
    }
    chosen_.assign(W.candId.size(), 0);
}

double LinearWeightsObjective::value() const{
    const vector<int>* w = weight_multiplier_;
    double sum = 0.0;
    if (w){
        for (int v = 0; v < n_; ++v){
            if ((*w)[v] > 0){
                sum += static_cast<double>((*w)[v]) * psi_v_[v];
            }
        }
    }
    return sum - lambda_*const_term_;
}

double LinearWeightsObjective::marginal(NodeId u) const{
    int ci = (u >= 0 && u < n_) ? inv_[u] : -1;
    if (ci < 0 || chosen_[ci]) return -1e100;
    const auto& neigh = W_->W[ci].vw;
    const vector<int>* w = weight_multiplier_;
    double gain = 0.0;
    for (auto [v, wvu] : neigh){
        if (!w || (*w)[v] == 0) continue;
        double nw = psi_.apply(m_v_[v] + wvu);
        gain += static_cast<double>((*w)[v]) * (nw - psi_v_[v]);
    }
    return gain; // const term independent of S
}

void LinearWeightsObjective::add(NodeId u){
    int ci = (u >= 0 && u < n_) ? inv_[u] : -1;
    if (ci < 0 || chosen_[ci]) return;
    chosen_[ci] = 1;
    S_.push_back(u);
    const auto& neigh = W_->W[ci].vw;
    for (auto [v, wvu] : neigh){
        m_v_[v] += wvu;
        psi_v_[v] = psi_.apply(m_v_[v]);
    }
}

void LinearWeightsObjective::reset_solution(){
    std::fill(m_v_.begin(), m_v_.end(), 0.0);
    for (int v = 0; v < n_; ++v){
        psi_v_[v] = psi_.apply(0.0);
    }
    std::fill(chosen_.begin(), chosen_.end(), 0);
    S_.clear();
}

void CappedUnionObjective::init(const WeightsStore& W, int n,
                                const Psi& psi, double lambda,
                                const vector<vector<NodeId>>& agent_nodes,
                                const vector<double>& caps,
                                const vector<int>& Lmin,
                                const vector<NodeId>& candidate_nodes,
                                double* out_sum_Lmin){
    W_ = &W;
    n_ = n;
    psi_ = psi;
    lambda_ = lambda;
    candidates_ = candidate_nodes;

    m_v_.assign(n_, 0.0);
    psi_v_.assign(n_, psi_.apply(0.0));
    inv_.assign(n_, -1);
    for (size_t i = 0; i < W.candId.size(); ++i){
        inv_[W.candId[i]] = static_cast<int>(i);
    }
    chosen_.assign(W.candId.size(), 0);
    S_.clear();

    caps_ = caps;
    agent_sum_.assign(caps_.size(), 0.0);
    agent_delta_.assign(caps_.size(), 0.0);
    agent_mark_.assign(caps_.size(), 0);
    touched_agents_.clear();

    vector<int> deg(n_, 0);
    for (size_t i = 0; i < agent_nodes.size(); ++i){
        for (NodeId v : agent_nodes[i]){
            if (v < 0 || v >= n_) continue;
            ++deg[v];
        }
    }
    node_offsets_.assign(n_ + 1, 0);
    for (int v = 0; v < n_; ++v){
        node_offsets_[v + 1] = node_offsets_[v] + deg[v];
    }
    node_agents_.assign(node_offsets_.back(), 0);
    vector<int> cursor = node_offsets_;
    for (size_t i = 0; i < agent_nodes.size(); ++i){
        for (NodeId v : agent_nodes[i]){
            if (v < 0 || v >= n_) continue;
            node_agents_[cursor[v]++] = static_cast<int>(i);
        }
    }

    long long sum_Lmin = 0;
    for (int L : Lmin) sum_Lmin += static_cast<long long>(L);
    lambda_const_term_ = lambda_ * static_cast<double>(sum_Lmin);
    if (out_sum_Lmin) *out_sum_Lmin = static_cast<double>(sum_Lmin);
}

void CappedUnionObjective::reset_solution(){
    std::fill(m_v_.begin(), m_v_.end(), 0.0);
    for (int v = 0; v < n_; ++v){
        psi_v_[v] = psi_.apply(0.0);
    }
    std::fill(chosen_.begin(), chosen_.end(), 0);
    S_.clear();
    std::fill(agent_sum_.begin(), agent_sum_.end(), 0.0);
}

double CappedUnionObjective::total_cap_sum() const{
    return std::accumulate(caps_.begin(), caps_.end(), 0.0);
}

double CappedUnionObjective::average_agents_per_node() const{
    if (n_ == 0) return 0.0;
    return node_agents_.empty() ? 0.0 : static_cast<double>(node_agents_.size()) / static_cast<double>(n_);
}

double CappedUnionObjective::value() const{
    double sum = std::accumulate(agent_sum_.begin(), agent_sum_.end(), 0.0);
    return sum - lambda_const_term_;
}

double CappedUnionObjective::marginal(NodeId u){
    int ci = (u >= 0 && u < n_) ? inv_[u] : -1;
    if (ci < 0 || chosen_[ci]) return -1e100;

    touched_agents_.clear();
    double gain = 0.0;

    const auto& neigh = W_->W[ci].vw;
    for (const auto& vw : neigh){
        NodeId v = vw.first;
        if (v < 0 || v >= n_) continue;
        double new_m = m_v_[v] + vw.second;
        double new_psi = psi_.apply(new_m);
        double delta = new_psi - psi_v_[v];
        if (delta <= 0.0) continue;
        for (int idx = node_offsets_[v]; idx < node_offsets_[v + 1]; ++idx){
            int agent = node_agents_[idx];
            if (!agent_mark_[agent]){
                agent_mark_[agent] = 1;
                touched_agents_.push_back(agent);
            }
            agent_delta_[agent] += delta;
        }
    }

    for (int agent : touched_agents_){
        double slack = caps_[agent] - agent_sum_[agent];
        if (slack < 0.0) slack = 0.0;
        double inc = std::min(slack, agent_delta_[agent]);
        gain += inc;
        agent_delta_[agent] = 0.0;
        agent_mark_[agent] = 0;
    }

    return gain;
}

void CappedUnionObjective::add(NodeId u){
    int ci = (u >= 0 && u < n_) ? inv_[u] : -1;
    if (ci < 0 || chosen_[ci]) return;

    chosen_[ci] = 1;
    S_.push_back(u);
    touched_agents_.clear();

    const auto& neigh = W_->W[ci].vw;
    for (const auto& vw : neigh){
        NodeId v = vw.first;
        if (v < 0 || v >= n_) continue;
        double new_m = m_v_[v] + vw.second;
        double new_psi = psi_.apply(new_m);
        double delta = new_psi - psi_v_[v];
        if (delta <= 0.0){
            m_v_[v] = new_m;
            psi_v_[v] = new_psi;
            continue;
        }
        for (int idx = node_offsets_[v]; idx < node_offsets_[v + 1]; ++idx){
            int agent = node_agents_[idx];
            if (!agent_mark_[agent]){
                agent_mark_[agent] = 1;
                touched_agents_.push_back(agent);
            }
            agent_delta_[agent] += delta;
        }
        m_v_[v] = new_m;
        psi_v_[v] = new_psi;
    }

    for (int agent : touched_agents_){
        double slack = caps_[agent] - agent_sum_[agent];
        if (slack < 0.0) slack = 0.0;
        double inc = std::min(slack, agent_delta_[agent]);
        agent_sum_[agent] += inc;
        agent_delta_[agent] = 0.0;
        agent_mark_[agent] = 0;
    }
}

// ---- LBObjective ----
void LBObjective::init(const WeightsStore& W, int n,
                       const Psi& psi, double lambda,
                       const LBFixedData& data,
                       const vector<NodeId>& candidate_nodes){
    W_ = &W;
    n_ = n;
    psi_ = psi;
    lambda_ = lambda;
    data_ = &data;
    candidates_ = candidate_nodes;

    m_v_.assign(n_, 0.0);
    psi_v_.assign(n_, psi_.apply(0.0));
    S_.clear();

    inv_.assign(n_, -1);
    for (size_t i = 0; i < W.candId.size(); ++i){
        inv_[W.candId[i]] = static_cast<int>(i);
    }
    chosen_.assign(W.candId.size(), 0);

    // Pre-compute coefficients and the constant term.
    coeff_.assign(n_, 0.0);
    const_term_ = 0.0;
    if (data_){
        for (size_t i = 0; i < data_->P_i.size(); ++i){
            const auto& paths_i = data_->P_i[i];
            if (paths_i.empty()) continue;

            if (data_->alpha_iP.empty() || data_->alpha_iP[i].empty()){
                const Path& P = paths_i[0];
                for (NodeId v : P){
                    coeff_[v] += 1.0;
                }
                const_term_ += static_cast<double>(std::max(0, (int)P.size() - 1));
            } else {
                for (size_t p = 0; p < paths_i.size(); ++p){
                    double a = data_->alpha_iP[i][p];
                    if (a <= 0) continue;
                    const Path& P = paths_i[p];
                    for (NodeId v : P){
                        coeff_[v] += a;
                    }
                    const_term_ += a * static_cast<double>(std::max(0, (int)P.size() - 1));
                }
            }
        }
    }
}

double LBObjective::value() const{
    double sum = 0.0;
    for (int v = 0; v < n_; ++v){
        if (coeff_[v] != 0.0){
            sum += coeff_[v] * psi_v_[v];
        }
    }
    return sum - lambda_ * const_term_;
}

double LBObjective::marginal(NodeId u) const{
    int ci = (u >= 0 && u < n_) ? inv_[u] : -1;
    if (ci < 0 || chosen_[ci]) return -1e100;
    double gain = 0.0;
    for (auto [v, wvu] : W_->W[ci].vw){
        if (coeff_[v] == 0.0) continue;
        double nw = psi_.apply(m_v_[v] + wvu);
        gain += coeff_[v] * (nw - psi_v_[v]);
    }
    return gain;
}

void LBObjective::add(NodeId u){
    int ci = (u >= 0 && u < n_) ? inv_[u] : -1;
    if (ci < 0 || chosen_[ci]) return;
    chosen_[ci] = 1;
    S_.push_back(u);
    for (auto [v, wvu] : W_->W[ci].vw){
        m_v_[v] += wvu;
        psi_v_[v] = psi_.apply(m_v_[v]);
    }
}

void LBObjective::reset_solution(){
    std::fill(m_v_.begin(), m_v_.end(), 0.0);
    for (int v = 0; v < n_; ++v){
        psi_v_[v] = psi_.apply(0.0);
    }
    std::fill(chosen_.begin(), chosen_.end(), 0);
    S_.clear();
}

// ========================= 6) AgentPool & Greedy (original objective) =========================

bool AgentPool::init(const Graph& G, const WeightsStore& W,
                     const AgentPoolConfig& cfg,
                     const vector<pair<NodeId,NodeId>>& agents,
                     const vector<vector<Path>>& paths_per_agent){
    G_ = &G;
    W_ = &W;
    cfg_ = cfg;
    n_ = G.n;
    agents_ = agents;
    paths_ = paths_per_agent;

    // Build inverse index: map each node to the (agent, path) pairs that include it.
    affect_.assign(n_, {});
    for (int i = 0; i < (int)paths_.size(); ++i){
        for (int p = 0; p < (int)paths_[i].size(); ++p){
            const Path& P = paths_[i][p];
            for (NodeId v : P){
                affect_[v].push_back({i, p});
            }
        }
    }

    // Initialize node-level accumulators.
    m_v_.assign(n_, 0.0);
    psi_v_.assign(n_, 0.0);

    // Initialize per-path scores and each agent's current best path.
    path_score_.assign(paths_.size(), {});
    best_path_idx_.assign(paths_.size(), -1);
    best_val_.assign(paths_.size(), 0.0);
    for (int i = 0; i < (int)paths_.size(); ++i){
        path_score_[i].assign(paths_[i].size(), 0.0);
        double best = -1e100;
        int arg = -1;
        for (int p = 0; p < (int)paths_[i].size(); ++p){
            const Path& P = paths_[i][p];
            double s = 0.0;
            for (NodeId v : P){
                s += psi_v_[v];
            }
            s -= cfg_.lambda * (double)std::max(0, (int)P.size() - 1);
            path_score_[i][p] = s;
            if (s > best){
                best = s;
                arg = p;
            }
        }
        best_val_[i] = best;
        best_path_idx_[i] = arg;
    }

    chosen_.assign(W.candId.size(), 0);
    S_.clear();

    // Cache node -> candidate index mapping.
    inv_.assign(n_, -1);
    for (size_t ci = 0; ci < W.candId.size(); ++ci){
        inv_[W.candId[ci]] = static_cast<int>(ci);
    }
    return true;
}

double AgentPool::total_value() const{
    double sum = 0.0;
    for (double x : best_val_){
        sum += x;
    }
    return sum;
}

static inline long long key_ip(int i, int p){ return ( ( (long long)i) << 32 ) ^ (unsigned long long)(uint32_t)p; }

// Evaluate the marginal gain of adding u without mutating internal state
double AgentPool::marginal_if_add(NodeId u){
    int ci = (u >= 0 && u < G_->n) ? inv_[u] : -1;
    if (ci < 0 || chosen_[ci]) return -1e100;

    const auto& neigh = W_->W[ci].vw;
    if (neigh.empty()) return 0.0;

    unordered_map<long long, double> delta_path;
    delta_path.reserve(neigh.size() * 4);
    vector<char> agent_touched(paths_.size(), 0);

    for (auto [v, wvu] : neigh){
        double dpsi = cfg_.psi.apply(m_v_[v] + wvu) - psi_v_[v];
        if (dpsi == 0.0) continue;
        for (auto [i, p] : affect_[v]){
            delta_path[key_ip(i, p)] += dpsi;
            agent_touched[i] = 1;
        }
    }

    double gain = 0.0;
    for (int i = 0; i < (int)paths_.size(); ++i){
        if (!agent_touched[i]) continue;
        double best = best_val_[i];
        double best_new = best;
        for (int p = 0; p < (int)paths_[i].size(); ++p){
            auto it = delta_path.find(key_ip(i, p));
            double inc = (it == delta_path.end() ? 0.0 : it->second);
            double val = path_score_[i][p] + inc;
            if (val > best_new) best_new = val;
        }
        gain += (best_new - best);
    }
    return gain;
}

void AgentPool::add(NodeId u){
    int ci = (u >= 0 && u < G_->n) ? inv_[u] : -1;
    if (ci < 0 || chosen_[ci]) return;
    chosen_[ci] = 1;
    S_.push_back(u);

    const auto& neigh = W_->W[ci].vw;
    if (neigh.empty()) return;

    unordered_map<long long, double> delta_path;
    delta_path.reserve(neigh.size() * 4);
    vector<char> agent_touched(paths_.size(), 0);

    for (auto [v, wvu] : neigh){
        double old = psi_v_[v];
        m_v_[v] += wvu;
        psi_v_[v] = cfg_.psi.apply(m_v_[v]);
        double dpsi = psi_v_[v] - old;
        if (dpsi == 0.0) continue;
        for (auto [i, p] : affect_[v]){
            delta_path[key_ip(i, p)] += dpsi;
            agent_touched[i] = 1;
        }
    }

    for (int i = 0; i < (int)paths_.size(); ++i){
        if (!agent_touched[i]) continue;
        double best = -1e100;
        int arg = -1;
        for (int p = 0; p < (int)paths_[i].size(); ++p){
            auto it = delta_path.find(key_ip(i, p));
            double inc = (it == delta_path.end() ? 0.0 : it->second);
            double val = path_score_[i][p] + inc;
            path_score_[i][p] = val;
            if (val > best){
                best = val;
                arg = p;
            }
        }
        best_val_[i] = best;
        best_path_idx_[i] = arg;
    }
}

template <typename Objective>
static vector<NodeId> greedy_select(Objective& obj,
                                    const vector<NodeId>& candidates,
                                    int k,
                                    vector<double>* iter_times = nullptr){
    vector<NodeId> S;
    S.reserve(k);
    Timer iter_timer{"greedy_select_iter", false};
    Timer* iter_timer_ptr = nullptr;
    if (iter_times){
        iter_timer.start();
        iter_timer_ptr = &iter_timer;
    }
    for (int it = 0; it < k; ++it){
        double best = -1e100;
        NodeId bestu = -1;
        for (NodeId u : candidates){
            double g = obj.marginal(u);
            if (g > best){
                best = g;
                bestu = u;
            }
        }
        if (best <= 0 || bestu < 0) break;
        obj.add(bestu);
        S.push_back(bestu);
        if (iter_timer_ptr){
            iter_times->push_back(iter_timer_ptr->lap().count());
        }
    }
    return S;
}

// Greedy full recomputation for the original objective
vector<NodeId> greedy_original_full_recompute(AgentPool& pool,
                                              const vector<NodeId>& candidates,
                                              int k){
    vector<NodeId> S; S.reserve(k);
    for (int it=0; it<k; ++it){
        double best=-1e100; NodeId bestu=-1;
        for (NodeId u: candidates){ double g = pool.marginal_if_add(u); if (g>best){ best=g; bestu=u; } }
        if (best <= 0 || bestu < 0) {
            break;
        }
        pool.add(bestu);
        S.push_back(bestu);
    }
    return S;
}

// Greedy incremental with CELF-like lazy (heuristic)
struct CandGain {
    double gain;
    NodeId u;
    int stamp;
    bool operator<(const CandGain& o) const { return gain < o.gain; }
};

vector<NodeId> greedy_original_incremental(AgentPool& pool,
                                           const vector<NodeId>& candidates,
                                           int k,
                                           bool use_celf_like_heuristic){
    vector<NodeId> S; S.reserve(k);
    if (!use_celf_like_heuristic){ return greedy_original_full_recompute(pool, candidates, k); }
    // initial gains
    vector<double> gains(candidates.size(), 0.0);
    for (size_t i=0;i<candidates.size();++i) gains[i] = pool.marginal_if_add(candidates[i]);
    priority_queue<CandGain> pq; for (size_t i=0;i<candidates.size();++i) pq.push({gains[i], candidates[i], 0});
    int curStamp = 0;
    for (int it=0; it<k && !pq.empty(); ++it){
        while(true){
            auto top = pq.top(); pq.pop();
            double fresh = pool.marginal_if_add(top.u);
            if (top.stamp == curStamp){ // up-to-date
                if (fresh < 1e-12) { // no more gain
                    return S;
                }
                pool.add(top.u); S.push_back(top.u); ++curStamp; break;
            } else {
                pq.push({fresh, top.u, curStamp});
            }
            if (pq.empty()) break;
        }
    }
    return S;
}

// ========================= 7) LB3 softmax weights and sandwich integration =========================

// Compute g_{i,P}(S0) for a reference solution and derive softmax mixture weights alpha_iP (temperature = eta)
static void compute_softmax_mixture_weights(const Graph& G,
                                            const WeightsStore& W,
                                            const Psi& psi, double lambda,
                                            const vector<vector<Path>>& paths,
                                            const vector<NodeId>& S0,
                                            double eta,
                                            LBFixedData& out_data){
    const int n = G.n;
    // 1) Evaluate m_v and psi_v under S0
    vector<double> m(n,0.0), ps(n,0.0);
    // Pre-compute node->candidate lookups
    vector<int> inv(n,-1); for (size_t ci=0; ci<W.candId.size(); ++ci) inv[W.candId[ci]]=(int)ci;
    for (NodeId u : S0){
        int ci = (u >= 0 && u < n) ? inv[u] : -1;
        if (ci < 0) continue;
        for (auto [v, wvu] : W.W[ci].vw){
            m[v] += wvu;
        }
    }
    for (int v=0; v<n; ++v) ps[v]=psi.apply(m[v]);
    // 2) Score each path and apply a numerically stable softmax
    out_data.P_i = paths; out_data.alpha_iP.assign(paths.size(), {});
    for (size_t i = 0; i < paths.size(); ++i){
        const auto& Pi = paths[i];
        vector<double> g;
        g.reserve(Pi.size());
        double maxg = -1e100;
        for (const auto& P : Pi){
            double s = 0.0;
            for (NodeId v : P){
                s += ps[v];
            }
            s -= lambda * static_cast<double>(std::max(0, (int)P.size() - 1));
            g.push_back(s);
            maxg = std::max(maxg, s);
        }
        // Numerically stable softmax
        double Z = 0.0;
        for (double x : g){
            Z += std::exp(eta * (x - maxg));
        }
        out_data.alpha_iP[i].resize(Pi.size(), 0.0);
        if (Z == 0.0){
            // Degenerate case: every path has negligible weight; fall back to uniform mixture
            double a = 1.0 / std::max<size_t>(1, Pi.size());
            for (size_t p = 0; p < Pi.size(); ++p){
                out_data.alpha_iP[i][p] = a;
            }
        } else {
            for (size_t p = 0; p < Pi.size(); ++p){
                out_data.alpha_iP[i][p] = std::exp(eta * (g[p] - maxg)) / Z;
            }
        }
    }
}

static double evaluate_once_with_pool(const Graph& G,
                                      const WeightsStore& W,
                                      const Psi& psi, double lambda,
                                      const vector<pair<NodeId,NodeId>>& agents,
                                      const vector<vector<Path>>& paths,
                                      const vector<NodeId>& S);

bool run_sandwich(const Graph& G,
                  const vector<NodeId>& candidates,
                  const vector<char>* blocked,
                  const WeightsStore& W,
                  const SandwichConfig& cfg,
                  SandwichResult& out){
    if (!logged_psi_assumption){
        if (cfg.psi.type == Psi::Type::Power && (cfg.psi.alpha <= 0.0 || cfg.psi.alpha > 1.0)){
            std::cerr << "[warn] psi.Power(alpha) expects 0 < alpha <= 1; behaviour outside this range may break monotonicity assumptions.\n";
        }
        if (!psi_is_monotone_nonnegative(cfg.psi)){
            std::cerr << "[warn] psi might not be monotone nondecreasing; certified bounds assume monotonicity.\n";
        } else {
            std::cerr << "[info] psi treated as concave, nonnegative, monotone for certified bounds.\n";
        }
        logged_psi_assumption = true;
    }

    vector<NodeId> walkable = candidates;
    AgentSampler sampler;
    AgentSamplerConfig od = cfg.od;
    od.rows = cfg.od.rows ? cfg.od.rows : od.rows;
    od.cols = cfg.od.cols ? cfg.od.cols : od.cols;
    sampler.init(od, walkable, blocked);
    vector<pair<NodeId,NodeId>> agents;
    vector<vector<Path>> paths;
    vector<vector<NodeId>> corridor_nodes;
    vector<vector<NodeId>> rev_adj;
    if (cfg.corridor_band_r >= 0) {
        build_reverse_adjacency(G, rev_adj);
    }
    int dropped_corridor = 0;
    int resample_corridor = 0;
    if (!generate_agents_and_paths(G, sampler,
                                   cfg.M_agents,
                                   cfg.K_paths_per_agent,
                                   cfg.seed,
                                   agents,
                                   paths,
                                   (cfg.od.rows > 0 && cfg.od.cols > 0),
                                   cfg.od.rows,
                                   cfg.od.cols,
                                   cfg.corridor_band_r,
                                   cfg.corridor_band_r >= 0 ? &rev_adj : nullptr,
                                   cfg.corridor_band_r >= 0 ? &corridor_nodes : nullptr,
                                   &dropped_corridor,
                                   &resample_corridor)) {
        std::cerr << "[error] Failed to sample agent paths; corridor may be too restrictive" << '\n';
        return false;
    }

    out.corridor_band_r_used = cfg.corridor_band_r;
    out.agents_dropped_due_corridor = dropped_corridor;
    out.corridor_resample_attempts = resample_corridor;

    AgentUnionData union_data = build_agent_union_data(paths, G.n,
                                                       cfg.corridor_band_r >= 0 ? &corridor_nodes : nullptr);
    vector<int> union_counts(G.n, 0);
    for (const auto& nodes : union_data.nodes_per_agent){
        for (NodeId v : nodes){
            if (v >= 0 && v < G.n) union_counts[v] += 1;
        }
    }
    long long sum_Lmin = 0;
    for (int L : union_data.min_path_len) sum_Lmin += static_cast<long long>(L);
    long long union_total = 0;
    for (int c : union_counts) union_total += c;
    out.avg_agents_per_node = (G.n > 0) ? static_cast<double>(union_total) / static_cast<double>(G.n) : 0.0;
    Timer selection_timer{"selection", false};
    bool selection_timer_running = false;
    out.selection_iter_times.clear();
    vector<double>* iter_times_ptr = cfg.use_LB3_softmax ? nullptr : &out.selection_iter_times;
    if (!cfg.use_LB3_softmax){
        selection_timer.start();
        selection_timer_running = true;
    }
    out.lb3_run_mode = "none";
    out.lb3_iters_done = 0;
   out.lb3_improved = false;
   out.lb3_best_value = 0.0;
   out.lb3_base_value = 0.0;
   out.cap_cache_key_hash.clear();

    CapResolution cap_info = resolve_cap_strategy(cfg);
    out.cap_mode_used = cap_info.effective_mode;
    out.cap_k_used = cap_info.topk_k;

    UBMode selection_mode = cfg.ub_mode;
    bool routed_r_equals_one = false;
    bool sum_deprecated_used = false;
    int proxy_selection_r = std::max(1, cfg.corridor_r);
    static bool sum_deprecated_warning_emitted = false;
    if (selection_mode == UBMode::SumDeprecated){
        sum_deprecated_used = true;
        proxy_selection_r = std::max(1, union_data.max_paths_per_agent);
        selection_mode = UBMode::ProxyCorridor;
        if (!sum_deprecated_warning_emitted){
            std::cerr << "[warn] Deprecated: mapped to proxy_corridor with r = "
                      << proxy_selection_r << " (greedy order preserved; constants differ).\n";
            sum_deprecated_warning_emitted = true;
        }
    }
    if (selection_mode == UBMode::ProxyCorridor && proxy_selection_r <= 1){
        selection_mode = UBMode::CertifiedUnion;
        routed_r_equals_one = true;
        proxy_selection_r = 1;
    }

    bool need_proxy_corridor = (selection_mode == UBMode::ProxyCorridor) || cfg.enable_proxy_baseline || sum_deprecated_used;
    CorridorAccumulator corridor_proxy;
    if (need_proxy_corridor){
        corridor_proxy.init({G.n, proxy_selection_r});
        for (const auto& vp : paths) corridor_proxy.add_agent_paths(vp);
    }

    bool need_caps = (selection_mode == UBMode::CappedUnion) || (cfg.cap_mode != CapMode::Auto);
    CapComputationResult cap_result;
    bool caps_ready = false;
    if (need_caps){
        cap_result = compute_caps(cfg, cfg.psi, paths, union_data, W, G.n, candidates);
        caps_ready = !cap_result.caps.empty() && cap_result.caps.size() == union_data.nodes_per_agent.size();
        out.cap_mode_used = cap_result.mode_used;
        out.cap_k_used = cap_result.topk_k_used;
        out.cap_min = cap_result.cap_min;
        out.cap_median = cap_result.cap_median;
        out.cap_max = cap_result.cap_max;
        out.cap_cache_key_hash = cap_result.cache_key_hash;
    } else {
        out.cap_mode_used = cap_info.effective_mode;
        out.cap_k_used = cap_info.topk_k;
        out.cap_min = out.cap_median = out.cap_max = 0.0;
        out.cap_cache_key_hash.clear();
    }

    vector<NodeId> selection_set;
    double selection_cert_value = 0.0;
    double selection_capped_value = 0.0;
    double selection_proxy_value = 0.0;
    bool selection_used_proxy = false;
    bool selection_used_capped = false;

    if (selection_mode == UBMode::CappedUnion && !caps_ready){
        std::cerr << "[warn] CappedUnion selection requested but caps unavailable; using CertifiedUnion instead.\n";
        selection_mode = UBMode::CertifiedUnion;
    }

    if (selection_mode == UBMode::CappedUnion){
        CappedUnionObjective capped_obj;
        capped_obj.init(W, G.n, cfg.psi, cfg.lambda,
                        union_data.nodes_per_agent,
                        cap_result.caps,
                        union_data.min_path_len,
                        candidates,
                        nullptr);
        selection_set = greedy_select(capped_obj, candidates, cfg.k, iter_times_ptr);
        selection_capped_value = capped_obj.value();
        selection_used_capped = true;
    } else if (selection_mode == UBMode::CertifiedUnion){
        CertifiedUnionObjective cert_obj;
        cert_obj.init(W, G.n, cfg.psi, cfg.lambda,
                      &union_counts,
                      static_cast<double>(sum_Lmin),
                      candidates);
        selection_set = greedy_select(cert_obj, candidates, cfg.k, iter_times_ptr);
        selection_cert_value = cert_obj.value();
    } else { // ProxyCorridor
        const vector<int>* weights = need_proxy_corridor ? &corridor_proxy.b_r() : nullptr;
        double const_term = need_proxy_corridor ? static_cast<double>(corridor_proxy.Lsum_r()) : 0.0;
        ProxyCorridorObjective proxy_obj;
        proxy_obj.init(W, G.n, cfg.psi, cfg.lambda, weights, const_term, candidates);
        selection_set = greedy_select(proxy_obj, candidates, cfg.k, iter_times_ptr);
        selection_proxy_value = proxy_obj.value();
        selection_used_proxy = true;
    }
    out.S_UB = selection_set;
    out.selection_mode_effective = selection_mode;
    out.selection_objective_type = (selection_mode == UBMode::ProxyCorridor) ? "proxy" : "certified";
    out.selection_proxy_r = (selection_mode == UBMode::ProxyCorridor) ? proxy_selection_r : 0;

    if (!cfg.use_LB3_softmax && selection_timer_running){
        out.selection_time_sec = selection_timer.stop().count();
        selection_timer_running = false;
    }

    if (cfg.use_LB3_softmax){
        LBFixedData lb3_data;
        compute_softmax_mixture_weights(G, W, cfg.psi, cfg.lambda,
                                        paths, out.S_UB,
                                        cfg.lb3_softmax_eta, lb3_data);
        LBObjective lb3_obj;
        lb3_obj.init(W, G.n, cfg.psi, cfg.lambda, lb3_data, candidates);
        out.selection_iter_times.clear();
        selection_timer.start();
        selection_timer_running = true;
        out.S_UB = greedy_select(lb3_obj, candidates, cfg.k, &out.selection_iter_times);
        selection_set = out.S_UB;

        out.selection_time_sec = selection_timer.stop().count();
        selection_timer_running = false;

        if (selection_mode == UBMode::CappedUnion){
            CappedUnionObjective capped_obj;
            capped_obj.init(W, G.n, cfg.psi, cfg.lambda,
                            union_data.nodes_per_agent,
                            cap_result.caps,
                            union_data.min_path_len,
                            candidates,
                            nullptr);
            for (NodeId u : out.S_UB) capped_obj.add(u);
            selection_capped_value = capped_obj.value();
        } else if (selection_mode == UBMode::CertifiedUnion){
            CertifiedUnionObjective cert_obj;
            cert_obj.init(W, G.n, cfg.psi, cfg.lambda,
                          &union_counts,
                          static_cast<double>(sum_Lmin),
                          candidates);
            for (NodeId u : out.S_UB) cert_obj.add(u);
            selection_cert_value = cert_obj.value();
        } else {
            const vector<int>* weights = need_proxy_corridor ? &corridor_proxy.b_r() : nullptr;
            double const_term = need_proxy_corridor ? static_cast<double>(corridor_proxy.Lsum_r()) : 0.0;
            ProxyCorridorObjective proxy_obj;
            proxy_obj.init(W, G.n, cfg.psi, cfg.lambda, weights, const_term, candidates);
            for (NodeId u : out.S_UB) proxy_obj.add(u);
            selection_proxy_value = proxy_obj.value();
        }
    }

    auto recompute_selection_metrics = [&](){
        selection_cert_value = 0.0;
        selection_capped_value = 0.0;
        selection_proxy_value = 0.0;
        selection_used_proxy = false;
        selection_used_capped = false;

        if (out.selection_mode_effective == UBMode::CappedUnion && caps_ready){
            CappedUnionObjective capped_obj;
            capped_obj.init(W, G.n, cfg.psi, cfg.lambda,
                            union_data.nodes_per_agent,
                            cap_result.caps,
                            union_data.min_path_len,
                            candidates,
                            nullptr);
            for (NodeId u : out.S_UB) capped_obj.add(u);
            selection_capped_value = capped_obj.value();
            selection_used_capped = true;
        } else if (out.selection_mode_effective == UBMode::CertifiedUnion){
            CertifiedUnionObjective cert_obj;
            cert_obj.init(W, G.n, cfg.psi, cfg.lambda,
                          &union_counts,
                          static_cast<double>(sum_Lmin),
                          candidates);
            for (NodeId u : out.S_UB) cert_obj.add(u);
            selection_cert_value = cert_obj.value();
        } else if (out.selection_mode_effective == UBMode::ProxyCorridor){
            const vector<int>* weights = need_proxy_corridor ? &corridor_proxy.b_r() : nullptr;
            double const_term = need_proxy_corridor ? static_cast<double>(corridor_proxy.Lsum_r()) : 0.0;
            ProxyCorridorObjective proxy_obj;
            proxy_obj.init(W, G.n, cfg.psi, cfg.lambda, weights, const_term, candidates);
            for (NodeId u : out.S_UB) proxy_obj.add(u);
            selection_proxy_value = proxy_obj.value();
            selection_used_proxy = true;
        }
    };

    bool run_lb3_iter = cfg.use_LB3_iter;
    bool run_lb3_single = cfg.use_LB3_single && !run_lb3_iter;
    if (run_lb3_iter || run_lb3_single){
        LB3IterConfig lb3cfg;
        lb3cfg.k = cfg.k;
        lb3cfg.eta = cfg.lb3_eta;
        lb3cfg.max_iters = run_lb3_iter ? cfg.lb3_max_iters : 1;
        if (lb3cfg.max_iters <= 0) lb3cfg.max_iters = 1;
        lb3cfg.tol = cfg.lb3_tol;
        lb3cfg.start_from_UB = cfg.lb3_start_from_UB;
        lb3cfg.keep_best = cfg.lb3_keep_best;
        lb3cfg.exact_iters = false;
        lb3cfg.seed = cfg.seed + 424242ull;

        vector<NodeId> S_lb3;
        int lb3_iters_done = 0;
        bool lb3_ok = run_softmax_lb3_solver(G, candidates, blocked, W,
                                             cfg.psi, cfg.lambda,
                                             std::max(1, cfg.corridor_r),
                                             cfg.od,
                                             cfg.M_agents, cfg.K_paths_per_agent,
                                             cfg.corridor_band_r,
                                             lb3cfg,
                                             S_lb3,
                                             &lb3_iters_done);

        out.lb3_run_mode = run_lb3_iter ? "iter" : "single";
        out.lb3_iters_done = lb3_iters_done;

        double base_value = evaluate_once_with_pool(G, W, cfg.psi, cfg.lambda,
                                                    agents, paths, out.S_UB);
        out.lb3_base_value = base_value;

        if (lb3_ok && !S_lb3.empty()){
            double lb3_value = evaluate_once_with_pool(G, W, cfg.psi, cfg.lambda,
                                                       agents, paths, S_lb3);
            out.lb3_best_value = lb3_value;
            if (lb3_value > base_value + 1e-9){
                out.lb3_improved = true;
                out.S_UB = S_lb3;
                selection_set = out.S_UB;
            }
        } else {
            out.lb3_best_value = base_value;
        }
    }

    recompute_selection_metrics();

    // Proxy baseline (heuristic only)
    out.proxy_solution.clear();
    out.proxy_score = 0.0;
    out.proxy_corridor_r_effective = 0;
    if (cfg.enable_proxy_baseline){
        int baseline_r = std::max(1, cfg.proxy_corridor_r);
        CorridorAccumulator baseline_acc;
        baseline_acc.init({G.n, baseline_r});
        for (const auto& vp : paths) baseline_acc.add_agent_paths(vp);
        Timer proxy_timer{"proxy_baseline"};
        ProxyCorridorObjective baseline_obj;
        baseline_obj.init(W, G.n, cfg.psi, cfg.lambda,
                          &baseline_acc.b_r(),
                          static_cast<double>(baseline_acc.Lsum_r()),
                          candidates);
        out.proxy_solution = greedy_select(baseline_obj, candidates, cfg.k);
        out.proxy_score = baseline_obj.value();
        out.proxy_corridor_r_effective = baseline_r;
        out.proxy_baseline_time_sec = proxy_timer.stop().count();
    } else if (selection_used_proxy){
        out.proxy_solution = selection_set;
        out.proxy_score = selection_proxy_value;
        out.proxy_corridor_r_effective = proxy_selection_r;
        out.proxy_baseline_time_sec = 0.0;
    } else {
        out.proxy_baseline_time_sec = 0.0;
    }

    out.routed_r_equals_one = routed_r_equals_one;
    out.sum_deprecated_used = sum_deprecated_used;

    // 3) Lower bound greedy
    LBFixedData lbdata;
    lbdata.P_i = paths;
    if (cfg.use_LB2_fixed_mixture){
        lbdata.alpha_iP.assign(paths.size(), {});
        for (size_t i = 0; i < paths.size(); ++i){
            lbdata.alpha_iP[i].assign(paths[i].size(),
                                      1.0 / std::max<size_t>(1, paths[i].size()));
        }
    }
    LBObjective lb;
    lb.init(W, G.n, cfg.psi, cfg.lambda, lbdata, candidates);
    out.S_LB = greedy_select(lb, candidates, cfg.k);

    AgentPoolConfig pcfg{cfg.psi, cfg.lambda};
    EvalStats statsUB, statsLB, statsPick;
    double fUB = 0.0, fLB = 0.0;
    auto stats_total = [](const EvalStats& stats)->double{
        return (stats.M > 0) ? stats.mean * static_cast<double>(stats.M) : 0.0;
    };

    if (cfg.M_eval > 0){
        if (cfg.reuse_eval_samples){
            AgentPool poolUB, poolLB;
            poolUB.init(G, W, pcfg, agents, paths);
            for (NodeId u : out.S_UB) poolUB.add(u);
            statsUB = summarize_samples(poolUB.agent_values(), cfg.ci_alpha);
            statsUB.M = static_cast<int>(poolUB.agent_values().size());
            fUB = stats_total(statsUB);

            poolLB.init(G, W, pcfg, agents, paths);
            for (NodeId u : out.S_LB) poolLB.add(u);
            statsLB = summarize_samples(poolLB.agent_values(), cfg.ci_alpha);
            statsLB.M = static_cast<int>(poolLB.agent_values().size());
            fLB = stats_total(statsLB);

            if (fUB >= fLB){
                out.S_pick = out.S_UB;
                statsPick = statsUB;
            } else {
                out.S_pick = out.S_LB;
                statsPick = statsLB;
            }
            out.fhat_pick = stats_total(statsPick);
        } else {
            AgentSampler sampler_eval;
            sampler_eval.init(od, walkable, blocked);

            statsUB = evaluate_original_objective_mc_stats(
                G, W, cfg.psi, cfg.lambda,
                out.S_UB,
                sampler_eval,
                cfg.M_eval,
                cfg.K_paths_per_agent,
                cfg.seed + 999,
                cfg.ci_alpha,
                (cfg.od.rows > 0 && cfg.od.cols > 0),
                cfg.od.rows,
                cfg.od.cols,
                cfg.corridor_band_r);
            fUB = stats_total(statsUB);

            statsLB = evaluate_original_objective_mc_stats(
                G, W, cfg.psi, cfg.lambda,
                out.S_LB,
                sampler_eval,
                cfg.M_eval,
                cfg.K_paths_per_agent,
                cfg.seed + 1999,
                cfg.ci_alpha,
                (cfg.od.rows > 0 && cfg.od.cols > 0),
                cfg.od.rows,
                cfg.od.cols,
                cfg.corridor_band_r);
            fLB = stats_total(statsLB);

            if (fUB >= fLB){
                out.S_pick = out.S_UB;
                statsPick = statsUB;
            } else {
                out.S_pick = out.S_LB;
                statsPick = statsLB;
            }
            out.fhat_pick = stats_total(statsPick);
        }
    } else {
        AgentPool poolUB, poolLB;
        poolUB.init(G, W, pcfg, agents, paths);
        for (NodeId u : out.S_UB) poolUB.add(u);
        fUB = poolUB.total_value();

        poolLB.init(G, W, pcfg, agents, paths);
        for (NodeId u : out.S_LB) poolLB.add(u);
        fLB = poolLB.total_value();

        if (fUB >= fLB){
            out.S_pick = out.S_UB;
            out.fhat_pick = fUB;
        } else {
            out.S_pick = out.S_LB;
            out.fhat_pick = fLB;
        }
        out.stats_UB = EvalStats{};
        out.stats_LB = EvalStats{};
        out.stats_pick = EvalStats{};
    }

    out.fhat_UB = fUB;
    out.fhat_LB = fLB;
    if (cfg.M_eval > 0){
        out.stats_UB = statsUB;
        out.stats_LB = statsLB;
        out.stats_pick = statsPick;
        out.fhat_pick = stats_total(statsPick);
    }

    CertifiedUnionObjective certificate_union;
    certificate_union.init(W, G.n, cfg.psi, cfg.lambda,
                           &union_counts,
                           static_cast<double>(sum_Lmin),
                           candidates);
    for (NodeId u : out.S_pick) certificate_union.add(u);
    out.certified_union_value = certificate_union.value();

    if (caps_ready){
        CappedUnionObjective certificate_capped;
        certificate_capped.init(W, G.n, cfg.psi, cfg.lambda,
                                 union_data.nodes_per_agent,
                                 cap_result.caps,
                                 union_data.min_path_len,
                                 candidates,
                                 nullptr);
        for (NodeId u : out.S_pick) certificate_capped.add(u);
        out.capped_union_value = certificate_capped.value();
    } else {
        out.capped_union_value = out.certified_union_value;
    }

    if (selection_mode == UBMode::CappedUnion && caps_ready){
        out.certified_ub_value = out.capped_union_value;
    } else {
        out.certified_ub_value = out.certified_union_value;
    }

    double ub_total_S_UB = selection_used_capped ? selection_capped_value : selection_cert_value;
    out.ub_objective_total_S_UB = ub_total_S_UB;
    double agent_count_double = static_cast<double>(union_data.nodes_per_agent.size());
    out.ub_objective_mean_S_UB = (agent_count_double > 0.0)
                                 ? ub_total_S_UB / agent_count_double
                                 : 0.0;
    double ub_total_pick = out.certified_ub_value;
    out.ub_objective_mean_pick = (agent_count_double > 0.0)
                                 ? ub_total_pick / agent_count_double
                                 : 0.0;

    if (!cfg.enable_proxy_baseline && selection_used_proxy){
        out.proxy_score = selection_proxy_value;
        out.proxy_corridor_r_effective = proxy_selection_r;
    }

    return true;
}
// ========================= 8) Baselines =========================

vector<NodeId> baseline_random_k(const vector<NodeId>& candidates, int k, uint64_t seed){
    vector<NodeId> idx = candidates;
    std::mt19937_64 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    if (static_cast<int>(idx.size()) > k) idx.resize(k);
    return idx;
}

vector<NodeId> baseline_regular_spacing_grid(int rows, int cols, const vector<char>* blocked, int k){
    vector<NodeId> S;
    if (k <= 0) return S;
    int n = rows * cols;
    if (n <= 0) return S;
    double area = (double)n;
    double step = std::sqrt(area / std::max(1, k));
    int sr = std::max(1, (int)std::round(step));
    int sc = sr;
    for (int r = sr / 2; r < rows && static_cast<int>(S.size()) < k; r += sr){
        for (int c = sc / 2; c < cols && static_cast<int>(S.size()) < k; c += sc){
            NodeId id = rc_to_id(r, c, cols);
            if (blocked && static_cast<int>(blocked->size()) == n && (*blocked)[id]) continue;
            S.push_back(id);
        }
    }
    // fall back to a dense sweep if spacing undershoots
    if ((int)S.size()<k){
        // Sweep row-major to fill any remaining budget slots
        for (int r=0; r<rows && (int)S.size()<k; ++r){
            for (int c=0; c<cols && (int)S.size()<k; ++c){
                NodeId id = rc_to_id(r,c,cols);
                if (blocked && (*blocked)[id]) continue;
                bool in=false; for (auto x:S) if (x==id) {in=true; break;}
                if (!in) S.push_back(id);
            }
        }
    }
    return S;
}

// Static coverage greedy baseline (demand=nullptr implies unit weights)
vector<NodeId> baseline_static_com_greedy(const WeightsStore& W,
                                          int n,
                                          const Psi& psi,
                                          int k,
                                          const vector<NodeId>& candidates,
                                          const vector<double>* demand){
    vector<double> m(n, 0.0);
    vector<double> ps(n, 0.0);
    vector<char> chosen(W.candId.size(), 0);
    vector<NodeId> S;
    S.reserve(k);

    auto get_weight = [&](int v) -> double {
        if (!demand) return 1.0;
        double w = (*demand)[v];
        return (w < 0) ? 0.0 : w;
    };

    vector<int> inv(n, -1);
    for (size_t i = 0; i < W.candId.size(); ++i) inv[W.candId[i]] = (int)i;

    for (int it = 0; it < k; ++it){
        double best = -1e100;
        NodeId bestu = -1;
        for (NodeId u : candidates){
            int ci = (u >= 0 && u < n) ? inv[u] : -1;
            if (ci < 0 || chosen[ci]) continue;
            double g = 0.0;
            for (auto [v, wvu] : W.W[ci].vw){
                double nw = psi.apply(m[v] + wvu);
                g += get_weight(v) * (nw - ps[v]);
            }
            if (g > best){
                best = g;
                bestu = u;
            }
        }
        if (best <= 0 || bestu < 0) break;
        int ci = inv[bestu];
        chosen[ci] = 1;
        S.push_back(bestu);
        for (auto [v, wvu] : W.W[ci].vw){
            m[v] += wvu;
            ps[v] = psi.apply(m[v]);
        }
    }
    return S;
}

vector<NodeId> baseline_greedy_original_full(AgentPool& pool,
                                             const vector<NodeId>& candidates,
                                             int k){
    return greedy_original_full_recompute(pool, candidates, k);
}

// Exhaustive search (small instances) evaluated under a linear-weight template
static double eval_ub_set(LinearWeightsObjective& obj, const vector<NodeId>& S){
    obj.reset_solution();
    for (NodeId u : S){
        obj.add(u);
    }
    return obj.value();
}

static bool next_comb(vector<int>& idx, int n, int k){
    if (k < 0 || k > n) return false;
    if (static_cast<int>(idx.size()) != k) return false;

    int i = k - 1;
    while (i >= 0 && idx[i] == n - k + i){
        --i;
    }
    if (i < 0) return false;

    ++idx[i];
    for (int j = i + 1; j < k; ++j){
        idx[j] = idx[j - 1] + 1;
    }
    return true;
}

vector<NodeId> baseline_bruteforce_ub(const LinearWeightsObjective& objective_template,
                                      const vector<NodeId>& candidates,
                                      int k,
                                      size_t max_combinations_guard){
    vector<NodeId> bestS;
    double bestVal = -1e100;
    if (k <= 0 || candidates.empty()) return bestS;
    size_t n = candidates.size();
    if (static_cast<size_t>(k) > n) return bestS;
    vector<int> idx(k);
    for (int i = 0; i < k; ++i) idx[i] = i;
    size_t cnt = 0;
    do {
        LinearWeightsObjective obj = objective_template; // copy per combination
        vector<NodeId> S;
        S.reserve(k);
        for (int i = 0; i < k; ++i) S.push_back(candidates[idx[i]]);
        double v = eval_ub_set(obj, S);
        if (v > bestVal){
            bestVal = v;
            bestS = S;
        }
        ++cnt;
        if (max_combinations_guard > 0 && cnt >= max_combinations_guard) break;
    } while (next_comb(idx, (int)n, k));
    return bestS;
}

// ========================= 9) Utilities: generation & MC eval =========================

bool generate_agents_and_paths(const Graph& G,
                               const AgentSampler& sampler_in,
                               int M, int K, uint64_t seed,
                               vector<pair<NodeId,NodeId>>& agents,
                               vector<vector<Path>>& paths_per_agent,
                               bool use_grid_accel,
                               int rows, int cols,
                               int corridor_band_r,
                               const vector<vector<NodeId>>* rev_adj_in,
                               vector<vector<NodeId>>* corridor_nodes_per_agent,
                               int* agents_dropped_due_corridor,
                               int* corridor_resample_attempts){
    agents.clear();
    paths_per_agent.clear();
    if (corridor_nodes_per_agent) corridor_nodes_per_agent->clear();

    AgentSampler sampler = sampler_in;
    sampler.reseed(seed);

    vector<vector<NodeId>> local_rev_adj;
    const vector<vector<NodeId>>* rev_adj_ptr = rev_adj_in;
    if (corridor_band_r >= 0 && !rev_adj_ptr) {
        build_reverse_adjacency(G, local_rev_adj);
        rev_adj_ptr = &local_rev_adj;
    }

    vector<char> corridor_mask;
    vector<NodeId> corridor_nodes;

    int dropped_corridor = 0;
    int resample_corridor = 0;

    const int max_attempts = std::max(M * 20, 200);
    int attempts = 0;
    uint64_t path_seed_base = seed ? seed : 1;

    while (static_cast<int>(agents.size()) < M && attempts < max_attempts) {
        ++attempts;
        NodeId s = -1;
        NodeId t = -1;
        if (!sampler.sample_od(G, s, t)) {
            continue;
        }

        bool corridor_ok = true;
        if (corridor_band_r >= 0) {
            if (!rev_adj_ptr) {
                corridor_ok = false;
            } else {
                corridor_ok = compute_centerline_and_corridor(G, s, t, *rev_adj_ptr,
                                                             corridor_band_r,
                                                             corridor_mask,
                                                             corridor_nodes);
            }
            if (!corridor_ok) {
                ++dropped_corridor;
                ++resample_corridor;
                continue;
            }
        }

        vector<Path> pp;
        bool ok = false;
        if (corridor_band_r >= 0) {
            ok = sample_k_shortest_paths_uniform(G, s, t, K,
                                                 path_seed_base + 1337ull * attempts,
                                                 pp,
                                                 &corridor_mask);
        } else if (use_grid_accel && rows > 0 && cols > 0) {
            ok = sample_k_l1_shortest_paths_grid(rows, cols, s, t, K,
                                                 path_seed_base + 1337ull * attempts,
                                                 pp);
        } else {
            ok = sample_k_shortest_paths_uniform(G, s, t, K,
                                                 path_seed_base + 1337ull * attempts,
                                                 pp,
                                                 nullptr);
        }

        if (!ok || pp.empty()) {
            if (corridor_band_r >= 0) {
                ++dropped_corridor;
                ++resample_corridor;
            }
            continue;
        }

        agents.emplace_back(s, t);
        paths_per_agent.push_back(std::move(pp));
        if (corridor_nodes_per_agent) {
            if (corridor_band_r >= 0) {
                corridor_nodes_per_agent->push_back(corridor_nodes);
            } else {
                corridor_nodes_per_agent->push_back({});
            }
        }
    }

    if (agents_dropped_due_corridor) {
        *agents_dropped_due_corridor += dropped_corridor;
    }
    if (corridor_resample_attempts) {
        *corridor_resample_attempts += resample_corridor;
    }
    return !agents.empty();
}

double evaluate_original_objective_mc(const Graph& G,
                                      const WeightsStore& W,
                                      const Psi& psi, double lambda,
                                      const vector<NodeId>& S,
                                      const AgentSampler& sampler_in,
                                      int M_eval, int K_paths,
                                      uint64_t seed,
                                      bool use_grid_accel,
                                      int rows, int cols,
                                      int corridor_band_r){
    vector<pair<NodeId,NodeId>> agents; vector<vector<Path>> paths; 
    AgentSampler sampler = sampler_in; sampler.reseed(seed);
    vector<vector<NodeId>> rev_adj;
    if (corridor_band_r >= 0) {
        build_reverse_adjacency(G, rev_adj);
    }
    if (!generate_agents_and_paths(G, sampler, M_eval, K_paths, seed, agents, paths,
                                   use_grid_accel, rows, cols, corridor_band_r,
                                   corridor_band_r >= 0 ? &rev_adj : nullptr)) {
        return 0.0;
    }
    AgentPool pool; AgentPoolConfig cfg{psi, lambda};
    pool.init(G, W, cfg, agents, paths);
    for (NodeId u : S) pool.add(u);
    return pool.total_value();
}

// Evaluate the original objective once using a pre-initialized agent/path set.
static double evaluate_once_with_pool(const Graph& G,
                                      const WeightsStore& W,
                                      const Psi& psi, double lambda,
                                      const vector<pair<NodeId,NodeId>>& agents,
                                      const vector<vector<Path>>& paths,
                                      const vector<NodeId>& S){
    AgentPool pool;
    AgentPoolConfig cfg{psi, lambda};
    pool.init(G, W, cfg, agents, paths);
    for (NodeId u : S){
        pool.add(u);
    }
    return pool.total_value();
}

EvalStats evaluate_original_objective_mc_stats(const Graph& G,
                                               const WeightsStore& W,
                                               const Psi& psi, double lambda,
                                               const vector<NodeId>& S,
                                               const AgentSampler& sampler_in,
                                               int M_eval, int K_paths,
                                               uint64_t seed,
                                               double alpha,
                                               bool use_grid_accel,
                                               int rows, int cols,
                                               int corridor_band_r){
    EvalStats stats;
    if (M_eval <= 0) return stats;

    AgentSampler sampler = sampler_in;
    sampler.reseed(seed);

    vector<pair<NodeId,NodeId>> agents;
    vector<vector<Path>> paths;
    vector<vector<NodeId>> rev_adj;
    if (corridor_band_r >= 0) {
        build_reverse_adjacency(G, rev_adj);
    }
    if (!generate_agents_and_paths(G, sampler, M_eval, K_paths, seed, agents, paths,
                                   use_grid_accel, rows, cols, corridor_band_r,
                                   corridor_band_r >= 0 ? &rev_adj : nullptr)){
        stats.M = 0;
        return stats;
    }

    AgentPool pool; AgentPoolConfig cfg{psi, lambda};
    pool.init(G, W, cfg, agents, paths);
    for (NodeId u : S) pool.add(u);

    stats = summarize_samples(pool.agent_values(), alpha);
    stats.M = static_cast<int>(pool.agent_values().size());
    return stats;
}

EvalStats evaluate_diff_mc_stats(const Graph& G,
                                 const WeightsStore& W,
                                 const Psi& psi, double lambda,
                                 const vector<NodeId>& S_A,
                                 const vector<NodeId>& S_B,
                                 const AgentSampler& sampler_in,
                                 int M_eval, int K_paths,
                                 uint64_t seed,
                                 double alpha,
                                 bool use_grid_accel,
                                 int rows, int cols,
                                 int corridor_band_r){
    EvalStats stats;
    if (M_eval <= 0) return stats;

    AgentSampler sampler = sampler_in;
    sampler.reseed(seed);

    vector<pair<NodeId,NodeId>> agents;
    vector<vector<Path>> paths;
    vector<vector<NodeId>> rev_adj;
    if (corridor_band_r >= 0) {
        build_reverse_adjacency(G, rev_adj);
    }
    if (!generate_agents_and_paths(G, sampler, M_eval, K_paths, seed, agents, paths,
                                   use_grid_accel, rows, cols, corridor_band_r,
                                   corridor_band_r >= 0 ? &rev_adj : nullptr)){
        stats.M = 0;
        return stats;
    }

    AgentPoolConfig cfg{psi, lambda};
    AgentPool poolA; poolA.init(G, W, cfg, agents, paths); for (NodeId u : S_A) poolA.add(u);
    AgentPool poolB; poolB.init(G, W, cfg, agents, paths); for (NodeId u : S_B) poolB.add(u);

    const auto& valuesA = poolA.agent_values();
    const auto& valuesB = poolB.agent_values();
    size_t n = std::min(valuesA.size(), valuesB.size());
    vector<double> diffs;
    diffs.reserve(n);
    for (size_t i=0; i<n; ++i) diffs.push_back(valuesA[i] - valuesB[i]);

    stats = summarize_samples(diffs, alpha);
    stats.M = static_cast<int>(diffs.size());
    return stats;
}

bool run_softmax_lb3_solver(const Graph& G,
                            const vector<NodeId>& candidates,
                            const vector<char>* blocked,
                            const WeightsStore& W,
                            const AgentSamplerConfig& od,
                            int M_agents, int K_paths,
                            int corridor_band_r,
                            const LB3IterConfig& lb3cfg,
                            vector<NodeId>& S_out,
                            int* iters_done){
    if (iters_done) *iters_done = 0;
    // Backward-compatible wrapper: assume psi=Clip(1.0) and lambda=0.
    // Use k (budget) as a fallback corridor radius for UB warm start.
    Psi default_psi = Psi::Clip(1.0);
    double default_lambda = 0.0;
    int corridor_r = std::max(1, lb3cfg.k);
    return run_softmax_lb3_solver(G, candidates, blocked, W,
                                  default_psi, default_lambda, corridor_r,
                                  od, M_agents, K_paths, corridor_band_r,
                                  lb3cfg, S_out, iters_done);
}

// LB3 iterative solver using softmax linearization with optional UB warm start.
bool run_softmax_lb3_solver(const Graph& G,
                            const vector<NodeId>& candidates,
                            const vector<char>* blocked,
                            const WeightsStore& W,
                            const Psi& psi, double lambda,
                            int corridor_r_for_ub,
                            const AgentSamplerConfig& od,
                            int M_agents, int K_paths,
                            int corridor_band_r,
                            const LB3IterConfig& lb3cfg,
                            vector<NodeId>& S_out,
                            int* iters_done){
    S_out.clear();
    if (iters_done) *iters_done = 0;
    const int k = lb3cfg.k;
    if (k <= 0) return false;

    int max_iters = lb3cfg.max_iters;
    if (max_iters <= 0) max_iters = 1;

    AgentSampler sampler;
    sampler.init(od, candidates, blocked);
    sampler.reseed(lb3cfg.seed);

    vector<pair<NodeId,NodeId>> agents;
    vector<vector<Path>> paths;
    vector<vector<NodeId>> rev_adj;
    if (corridor_band_r >= 0) {
        build_reverse_adjacency(G, rev_adj);
    }
    if (!generate_agents_and_paths(G, sampler, M_agents, K_paths, lb3cfg.seed,
                                   agents, paths,
                                   (od.rows > 0 && od.cols > 0), od.rows, od.cols,
                                   corridor_band_r,
                                   corridor_band_r >= 0 ? &rev_adj : nullptr)){
        return false;
    }

    vector<NodeId> S_cur;
    if (lb3cfg.start_from_UB){
        int corr_r = std::max(1, corridor_r_for_ub);
        CorridorAccumulator acc;
        acc.init({G.n, corr_r});
        for (const auto& vp : paths){
            acc.add_agent_paths(vp);
        }
        ProxyCorridorObjective ub_obj;
        ub_obj.init(W, G.n, psi, lambda,
                    &acc.b_r(),
                    static_cast<double>(acc.Lsum_r()),
                    candidates);
        S_cur = greedy_select(ub_obj, candidates, k);
        if (static_cast<int>(S_cur.size()) > k) S_cur.resize(k);
    }

    double f_cur = evaluate_once_with_pool(G, W, psi, lambda, agents, paths, S_cur);
    vector<NodeId> bestS = S_cur;
    double bestF = f_cur;

    const bool use_tol = (!lb3cfg.exact_iters && lb3cfg.tol > 0.0);

    int iterations_done = 0;
    for (int it = 0; it < max_iters; ++it){
        ++iterations_done;
        LBFixedData lbdata;
        compute_softmax_mixture_weights(G, W, psi, lambda, paths, S_cur, lb3cfg.eta, lbdata);

        LBObjective lb;
        lb.init(W, G.n, psi, lambda, lbdata, candidates);
        vector<NodeId> S_next = greedy_select(lb, candidates, k);

        double f_next = evaluate_once_with_pool(G, W, psi, lambda, agents, paths, S_next);

        if (lb3cfg.keep_best && (it == 0 || f_next > bestF)){
            bestF = f_next;
            bestS = S_next;
        }

        bool converged = use_tol && std::fabs(f_next - f_cur) < lb3cfg.tol;
        S_cur = std::move(S_next);
        f_cur = f_next;

        if (converged) break;
    }

    S_out = lb3cfg.keep_best ? bestS : S_cur;
    if (iters_done) *iters_done = iterations_done;
    return !S_out.empty();
}

// ========================= 10) Reporting helpers =========================

void fill_report_stats(const EvalStats& src, ReportStats& dst){
    dst.mean    = src.mean;
    dst.var     = src.var;
    dst.se      = src.se;
    dst.ci_low  = src.ci_low;
    dst.ci_high = src.ci_high;
    dst.M       = src.M;
}

static string psi_type_to_string(const Psi& psi){
    switch (psi.type){
        case Psi::Type::ClipMinC: return "ClipMinC";
        case Psi::Type::Log1p:    return "Log1p";
        case Psi::Type::ExpSat:   return "ExpSat";
        case Psi::Type::Power:    return "Power";
    }
    return "Unknown";
}

static string cap_mode_to_string(CapMode mode){
    switch (mode){
        case CapMode::Auto:         return "Auto";
        case CapMode::PathlenOnly:  return "PathlenOnly";
        case CapMode::TopkLocal:    return "TopkLocal";
        case CapMode::GlobalAll:    return "GlobalAll";
    }
    return "Unknown";
}

static string ub_mode_to_string(UBMode mode){
    switch (mode){
        case UBMode::CertifiedUnion: return "CertifiedUnion";
        case UBMode::CappedUnion:    return "CappedUnion";
        case UBMode::ProxyCorridor:  return "ProxyCorridor";
        case UBMode::SumDeprecated:  return "SumDeprecated";
    }
    return "Unknown";
}

static string od_distribution_to_string(ODDistribution dist){
    switch (dist){
        case ODDistribution::Uniform:              return "Uniform";
        case ODDistribution::UniformWithDistBand:  return "UniformWithDistBand";
        case ODDistribution::GaussianAroundStart:  return "GaussianAroundStart";
        case ODDistribution::HotspotMixture:       return "HotspotMixture";
        case ODDistribution::Gravity:              return "Gravity";
    }
    return "Unknown";
}

static string bool_to_string(bool v){
    return v ? "true" : "false";
}

static string to_string_compact(double v){
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6) << v;
    return oss.str();
}

void build_sandwich_metadata(const SandwichConfig& cfg,
                             const vector<NodeId>& candidates,
                             vector<pair<string,string>>& meta_out,
                             const vector<pair<string,string>>* extra_meta){
    auto push = [&](const string& key, const string& value){
        meta_out.emplace_back(key, value);
    };

    push("k", std::to_string(cfg.k));
    push("corridor_r", std::to_string(cfg.corridor_r));
    push("ub_mode", ub_mode_to_string(cfg.ub_mode));
    push("cap_mode", cap_mode_to_string(cfg.cap_mode));
    push("cap_topk_k", std::to_string(cfg.cap_topk_k));
    push("cap_cache", bool_to_string(cfg.cap_cache));
    push("lambda", to_string_compact(cfg.lambda));
    push("psi_type", psi_type_to_string(cfg.psi));
    if (cfg.psi.type == Psi::Type::ClipMinC){
        push("psi_C", to_string_compact(cfg.psi.C));
    } else if (cfg.psi.type == Psi::Type::ExpSat){
        push("psi_beta", to_string_compact(cfg.psi.beta));
    } else if (cfg.psi.type == Psi::Type::Power){
        push("psi_alpha", to_string_compact(cfg.psi.alpha));
    }
    push("corridor_band_r", std::to_string(cfg.corridor_band_r));

    push("M_agents", std::to_string(cfg.M_agents));
    push("K_paths_per_agent", std::to_string(cfg.K_paths_per_agent));
    push("M_eval", std::to_string(cfg.M_eval));
    push("reuse_eval_samples", bool_to_string(cfg.reuse_eval_samples));
    push("ci_alpha", to_string_compact(cfg.ci_alpha));
    push("proxy_baseline_enabled", bool_to_string(cfg.enable_proxy_baseline));
    push("proxy_corridor_r", std::to_string(cfg.proxy_corridor_r));
    push("use_LB2_fixed_mixture", bool_to_string(cfg.use_LB2_fixed_mixture));
    push("use_LB3_softmax", bool_to_string(cfg.use_LB3_softmax));
    push("lb3_softmax_eta", to_string_compact(cfg.lb3_softmax_eta));
    push("use_LB3_single", bool_to_string(cfg.use_LB3_single));
    push("use_LB3_iter", bool_to_string(cfg.use_LB3_iter));
    push("lb3_start_from_UB", bool_to_string(cfg.lb3_start_from_UB));
    push("lb3_max_iters", std::to_string(cfg.lb3_max_iters));
    push("lb3_eta", to_string_compact(cfg.lb3_eta));
    push("lb3_tol", to_string_compact(cfg.lb3_tol));
    push("lb3_keep_best", bool_to_string(cfg.lb3_keep_best));
    push("seed", std::to_string(cfg.seed));

    push("od_distribution", od_distribution_to_string(cfg.od.dist));
    push("od_rows", std::to_string(cfg.od.rows));
    push("od_cols", std::to_string(cfg.od.cols));
    push("od_dist_band", std::to_string(cfg.od.dist_band.first) + "," + std::to_string(cfg.od.dist_band.second));
    push("od_sigma", to_string_compact(cfg.od.sigma));
    push("od_decay_beta", to_string_compact(cfg.od.decay_beta));
    push("od_hotspot_count", std::to_string(cfg.od.hotspots.size()));

    push("candidate_count", std::to_string(candidates.size()));

    if (extra_meta){
        meta_out.insert(meta_out.end(), extra_meta->begin(), extra_meta->end());
    }
}
static vector<double> make_node_topk_weights(const WeightsStore& W,
                                             int n,
                                             int k,
                                             bool use_cache,
                                             uint64_t hash_key,
                                             const Psi& psi){
    static std::unordered_map<std::string, vector<double>> cache;
    std::string cache_key;
    if (use_cache){
        cache_key = std::to_string(hash_key) + ":topk:" + std::to_string(k) + ":" + std::to_string(static_cast<int>(psi.type));
        if (psi.type == Psi::Type::ClipMinC) cache_key += ":" + std::to_string(psi.C);
        if (psi.type == Psi::Type::ExpSat) cache_key += ":" + std::to_string(psi.beta);
        if (psi.type == Psi::Type::Power) cache_key += ":" + std::to_string(psi.alpha);
        auto it = cache.find(cache_key);
        if (it != cache.end()) return it->second;
    }

    vector<vector<double>> node_weights(n);
    for (size_t ci = 0; ci < W.candId.size(); ++ci){
        for (const auto& vw : W.W[ci].vw){
            NodeId v = vw.first;
            if (v < 0 || v >= n) continue;
            node_weights[v].push_back(vw.second);
        }
    }

    vector<double> result(n, 0.0);
    for (int v = 0; v < n; ++v){
        auto& arr = node_weights[v];
        if (arr.empty()) continue;
        if (static_cast<int>(arr.size()) > k){
            std::nth_element(arr.begin(), arr.begin() + k, arr.end(), std::greater<double>());
            double sum = 0.0;
            for (int i = 0; i < k; ++i) sum += arr[i];
            result[v] = sum;
        } else {
            double sum = std::accumulate(arr.begin(), arr.end(), 0.0);
            result[v] = sum;
        }
    }

    if (use_cache){
        cache[cache_key] = result;
    }
    return result;
}

static vector<double> make_node_global_weights(const WeightsStore& W,
                                               int n,
                                               bool use_cache,
                                               uint64_t hash_key){
    static std::unordered_map<uint64_t, vector<double>> cache;
    if (use_cache){
        auto it = cache.find(hash_key);
        if (it != cache.end()) return it->second;
    }
    vector<double> result(n, 0.0);
    for (size_t ci = 0; ci < W.candId.size(); ++ci){
        for (const auto& vw : W.W[ci].vw){
            NodeId v = vw.first;
            if (v < 0 || v >= n) continue;
            result[v] += vw.second;
        }
    }
    if (use_cache){
        cache.emplace(hash_key, result);
    }
    return result;
}

static CapComputationResult compute_caps(const SandwichConfig& cfg,
                                         const Psi& psi,
                                         const vector<vector<Path>>& paths,
                                         const AgentUnionData& union_data,
                                         const WeightsStore& W,
                                         int n,
                                         [[maybe_unused]] const vector<NodeId>& candidates){
    CapComputationResult result;
    CapMode mode = cfg.cap_mode;
    if (mode == CapMode::Auto){
        mode = psi_is_bounded(psi) ? CapMode::PathlenOnly : CapMode::TopkLocal;
        std::cerr << "[info] cap_mode auto -> "
                  << (mode == CapMode::PathlenOnly ? "PathlenOnly" : (mode == CapMode::TopkLocal ? "TopkLocal" : "GlobalAll"))
                  << " based on psi type\n";
    }

    if (mode == CapMode::PathlenOnly && !psi_is_bounded(psi)){
        std::cerr << "[warn] PathlenOnly cap requested but psi is unbounded; falling back to TopkLocal\n";
        mode = CapMode::TopkLocal;
    }

    int budget_k = cfg.cap_topk_k > 0 ? cfg.cap_topk_k : cfg.k;
    if (budget_k <= 0) budget_k = cfg.k;
    result.topk_k_used = (mode == CapMode::TopkLocal) ? budget_k : 0;

    size_t num_agents = union_data.nodes_per_agent.size();
    result.caps.assign(num_agents, 0.0);

    uint64_t weight_hash = hash_weights_store(W);
    result.cache_key_hash = to_hex_string(weight_hash);

    if (mode == CapMode::PathlenOnly){
        double psi_at_k = psi.apply(static_cast<double>(std::max(0, cfg.k)));
        for (size_t i = 0; i < num_agents; ++i){
            result.caps[i] = psi_at_k * static_cast<double>(union_data.max_path_len[i] + 1);
        }
        result.mode_used = CapMode::PathlenOnly;
        result.cache_key_hash += ":pathlen";
    } else {
        vector<double> node_weights;
        if (mode == CapMode::TopkLocal){
            node_weights = make_node_topk_weights(W, n, budget_k, cfg.cap_cache, weight_hash, psi);
            result.mode_used = CapMode::TopkLocal;
            result.cache_key_hash += ":topk:" + std::to_string(budget_k) + ":psi" + std::to_string(static_cast<int>(psi.type));
            if (psi.type == Psi::Type::ClipMinC) result.cache_key_hash += ":" + to_string_compact(psi.C);
            if (psi.type == Psi::Type::ExpSat)   result.cache_key_hash += ":" + to_string_compact(psi.beta);
            if (psi.type == Psi::Type::Power)    result.cache_key_hash += ":" + to_string_compact(psi.alpha);
        } else {
            node_weights = make_node_global_weights(W, n, cfg.cap_cache, weight_hash);
            result.mode_used = CapMode::GlobalAll;
            result.cache_key_hash += ":global";
        }
        vector<double> node_psi(node_weights.size(), 0.0);
        for (size_t v = 0; v < node_weights.size(); ++v){
            node_psi[v] = psi.apply(node_weights[v]);
        }
        for (size_t i = 0; i < num_agents; ++i){
            double max_sum = 0.0;
            const auto& candidate_paths = paths[i];
            for (const Path& P : candidate_paths){
                double sum = 0.0;
                for (NodeId v : P){
                    if (v < 0 || v >= n) continue;
                    sum += node_psi[v];
                }
                if (sum > max_sum) max_sum = sum;
            }
            result.caps[i] = max_sum;
        }
    }

    if (!result.caps.empty()){
        vector<double> sorted_caps = result.caps;
        std::sort(sorted_caps.begin(), sorted_caps.end());
        result.cap_min = sorted_caps.front();
        result.cap_max = sorted_caps.back();
        size_t mid = sorted_caps.size() / 2;
        if (sorted_caps.size() % 2 == 0 && sorted_caps.size() > 1){
            result.cap_median = 0.5 * (sorted_caps[mid - 1] + sorted_caps[mid]);
        } else {
            result.cap_median = sorted_caps[mid];
        }
    }

    return result;
}
