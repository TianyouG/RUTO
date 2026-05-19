#pragma once
#include "stdafx.h"

// ============ 1) Psi compression family (declarations) ============
// Represents the concave compression applied to aggregated shadow weights.
struct Psi {
    enum class Type { ClipMinC, Log1p, ExpSat, Power };
    Type type = Type::ClipMinC;
    double C = 1.0;       // for ClipMinC
    double beta = 1.0;    // for ExpSat
    double alpha = 1.0;   // for Power (0 < alpha <= 1)

    Psi() = default;
    static Psi Clip(double C_);
    static Psi Log1p();
    static Psi ExpSat(double beta_);
    static Psi Power(double alpha_);

    double apply(double t) const;
};

// ============ 2) Agent origin-destination sampler (declarations) ============
// Configuration for generating synthetic agent OD pairs under the supported
// distributions. Parameters are interpreted as follows:
//  - dist:                ODDistribution selector (default Uniform).
//  - rows/cols:           Optional grid dimensions (inherit from --make_grid when absent).
//  - dist_band:           Inclusive [d_min, d_max] hop range for UniformWithDistBand.
//  - sigma:               Standard deviation (in hops) for GaussianAroundStart.
//  - hotspots:            Weighted node list for HotspotMixture (positive weights).
//  - decay_beta:          Gravity acceptance parameter β (prob ∝ exp(-β·distance)).
//  - seed:                RNG seed; 0 triggers std::random_device.
enum class ODDistribution {
    Uniform,
    UniformWithDistBand,
    GaussianAroundStart,
    HotspotMixture,
    Gravity
};

struct AgentSamplerConfig {
    ODDistribution dist = ODDistribution::Uniform;
    int rows = 0, cols = 0;                      // optional grid dimensions for coordinate helpers
    std::pair<int,int> dist_band = {0, INT_MAX}; // distance band [d_min, d_max]
    double sigma = 2.0;                          // step size for GaussianAroundStart
    std::vector<std::pair<NodeId,double>> hotspots; // per-node weights for the hotspot mixture
    double decay_beta = 0.2;                     // Gravity: acceptance exp(-beta * d); base sampling is uniform
    uint64_t seed = 1;                           // seed==0 means pull from random_device
};

// RNG-backed sampler supporting the configured OD distribution, walkable mask,
// and optional hotspot cache for faster weighted draws.
struct AgentSampler {
    void init(const AgentSamplerConfig& cfg,
              const std::vector<NodeId>& walkable,
              const std::vector<char>* blocked = nullptr);
    void reseed(uint64_t seed);
    bool sample_od(const Graph& G, NodeId& s, NodeId& t, int max_tries = 50) const;
    const std::vector<NodeId>& walkable() const { return walkable_; }
private:
    AgentSamplerConfig cfg_;
    std::vector<NodeId> walkable_;
    const std::vector<char>* blocked_ = nullptr;
    mutable std::mt19937_64 rng_;
    std::vector<NodeId> hotspot_nodes_;
    std::vector<double> hotspot_prefix_;
    double hotspot_total_ = 0.0;

    void rebuild_hotspot_cache();
    bool sample_hotspot(NodeId& out) const;
};

// ============ 3) Shortest-path sampling (declarations) ============
// Utilities to build shortest-path DAGs and sample shortest paths uniformly.
bool build_shortestpath_dag(const Graph& G, NodeId s, NodeId t,
                            std::vector<std::vector<NodeId>>& dag_adj,
                            std::vector<int>& dist,
                            const std::vector<char>* allowed_mask = nullptr);

bool sample_k_shortest_paths_uniform_from_dag(const std::vector<std::vector<NodeId>>& dag_adj,
                                              NodeId s, NodeId t, int K, uint64_t seed,
                                              std::vector<Path>& out_paths);

bool sample_k_shortest_paths_uniform(const Graph& G, NodeId s, NodeId t,
                                     int K, uint64_t seed, std::vector<Path>& out_paths,
                                     const std::vector<char>* allowed_mask = nullptr);

bool sample_k_l1_shortest_paths_grid(int rows, int cols, NodeId s, NodeId t,
                                     int K, uint64_t seed, std::vector<Path>& out_paths);

// ============ 4) Corridor statistics accumulator ============
// Aggregates per-node capped counts, union indicators, and path length totals.
struct CorridorAccumulatorConfig { int n = 0; int r = 1; };
struct CorridorAccumulator {
    void init(const CorridorAccumulatorConfig& cfg);
    void add_agent_paths(const std::vector<Path>& paths_for_agent);
    const std::vector<int>& b_r() const { return b_r_; }
    long long Lsum_r() const { return Lsum_r_; }
    const std::vector<int>& in_union() const { return in_union_; }
    const std::vector<int>& c_total() const { return c_total_; }
    long long Lmin1() const { return Lmin1_; }
    long long Lsum_all() const { return Lsum_all_; }
private:
    int n_ = 0, r_ = 1;
    std::vector<int> b_r_, in_union_, c_total_;
    long long Lsum_r_ = 0, Lmin1_ = 0, Lsum_all_ = 0;
};

// ============ 5) Certified submodular objectives (UB/LB wrappers) ============
// Submodular surrogates derived from corridor stats (UB) or fixed path mixtures (LB).
enum class UBMode { CertifiedUnion, CappedUnion, ProxyCorridor, SumDeprecated };
enum class CapMode { Auto, PathlenOnly, TopkLocal, GlobalAll };

struct LinearWeightsObjective {
    void init(const WeightsStore& W, int n,
              const Psi& psi, double lambda,
              const std::vector<int>* weight_multiplier,
              double const_term,
              const std::vector<NodeId>& candidate_nodes);

    void reset_solution();
    double value() const;
    double marginal(NodeId u) const;
    void add(NodeId u);
    const std::vector<NodeId>& solution() const { return S_; }

protected:
    const WeightsStore* W_ = nullptr; int n_ = 0;
    Psi psi_;
    double lambda_ = 0.0;
    const std::vector<int>* weight_multiplier_ = nullptr;
    double const_term_ = 0.0;
    std::vector<NodeId> candidates_;
    std::vector<double> m_v_, psi_v_;
    std::vector<char> chosen_;
    std::vector<NodeId> S_;
    std::vector<int> inv_;
};

struct ProxyCorridorObjective : LinearWeightsObjective {};
struct CertifiedUnionObjective : LinearWeightsObjective {};

struct CappedUnionObjective {
    void init(const WeightsStore& W, int n,
              const Psi& psi, double lambda,
              const std::vector<std::vector<NodeId>>& agent_nodes,
              const std::vector<double>& caps,
              const std::vector<int>& Lmin,
              const std::vector<NodeId>& candidates,
              double* out_sum_Lmin = nullptr);

    void reset_solution();
    double value() const;
    double marginal(NodeId u);
    void add(NodeId u);
    const std::vector<NodeId>& solution() const { return S_; }

    double total_cap_sum() const;
    double average_agents_per_node() const;

private:
    const WeightsStore* W_ = nullptr; int n_ = 0;
    Psi psi_;
    double lambda_ = 0.0;
    double lambda_const_term_ = 0.0;
    std::vector<NodeId> candidates_;
    std::vector<double> m_v_, psi_v_;
    std::vector<char> chosen_;
    std::vector<NodeId> S_;
    std::vector<int> inv_;

    std::vector<double> caps_;
    std::vector<double> agent_sum_;

    std::vector<int> node_offsets_;
    std::vector<int> node_agents_;

    std::vector<double> agent_delta_;
    std::vector<char> agent_mark_;
    std::vector<int> touched_agents_;
};

// Holds per-agent path sets and optional mixture weights for LB objectives.
struct LBFixedData {
    std::vector<std::vector<Path>> P_i;                 // fixed candidate paths per agent
    std::vector<std::vector<double>> alpha_iP;          // optional: LB2/LB3 mixture weights
};

struct LBObjective {
    void init(const WeightsStore& W, int n,
              const Psi& psi, double lambda,
              const LBFixedData& data,
              const std::vector<NodeId>& candidate_nodes);

    void reset_solution();
    double value() const;
    double marginal(NodeId u) const;
    void add(NodeId u);
    const std::vector<NodeId>& solution() const { return S_; }

private:
    const WeightsStore* W_ = nullptr; int n_ = 0;
    Psi psi_; double lambda_ = 0.0; const LBFixedData* data_ = nullptr;
    std::vector<NodeId> candidates_;
    std::vector<double> m_v_, psi_v_;
    std::vector<char> chosen_;
    std::vector<NodeId> S_;
    std::vector<int> inv_;
    std::vector<double> coeff_; double const_term_ = 0.0; // pre-computed coefficients
};

// ============ 6) AgentPool for the original objective (non-submodular heuristic) ============
// Maintains per-agent best-path values under the true objective for greedy heuristics.
struct AgentPoolConfig { Psi psi; double lambda = 0.0; };
struct AgentPool {
    bool init(const Graph& G, const WeightsStore& W,
              const AgentPoolConfig& cfg,
              const std::vector<std::pair<NodeId,NodeId>>& agents,
              const std::vector<std::vector<Path>>& paths_per_agent);
    double total_value() const;
    double marginal_if_add(NodeId u);
    void add(NodeId u);
    const std::vector<NodeId>& solution() const { return S_; }
    const std::vector<double>& agent_values() const { return best_val_; } // contribution per agent
private:
    const Graph* G_ = nullptr; const WeightsStore* W_ = nullptr; AgentPoolConfig cfg_; int n_ = 0;
    std::vector<std::pair<NodeId,NodeId>> agents_;
    std::vector<std::vector<Path>> paths_;
    std::vector<std::vector<std::pair<int,int>>> affect_;
    std::vector<double> m_v_, psi_v_;
    std::vector<std::vector<double>> path_score_;
    std::vector<int> best_path_idx_; std::vector<double> best_val_;
    std::vector<char> chosen_;
    std::vector<NodeId> S_;
    std::vector<int> inv_;
};

std::vector<NodeId> greedy_original_full_recompute(AgentPool& pool,
                                                   const std::vector<NodeId>& candidates,
                                                   int k);
std::vector<NodeId> greedy_original_incremental(AgentPool& pool,
                                                const std::vector<NodeId>& candidates,
                                                int k,
                                                bool use_celf_like_heuristic = false);

// Forward declaration for report helpers (ReportStats lives in io.hpp).
struct ReportStats;

// ============ 7) Sandwich approximation framework ============
// Configuration and result bundles for the overall sandwich procedure.
struct SandwichConfig {
    int M_agents = 10000;
    int K_paths_per_agent = 10;
    int corridor_r = 1;
    int corridor_band_r = -1;
    int k = 50;
    double lambda = 0.0;
    Psi psi = Psi::Clip(1.0);
    AgentSamplerConfig od;
    uint64_t seed = 1; // seed==0 triggers random_device
    UBMode ub_mode = UBMode::CertifiedUnion;
    CapMode cap_mode = CapMode::Auto;
    int cap_topk_k = 0;             // 0 => auto (defaults to budget)
    bool cap_cache = true;
    bool enable_proxy_baseline = true;
    int proxy_corridor_r = 2;
    bool use_LB2_fixed_mixture = false;
    // Evaluation settings
    int M_eval = 0;                 // 0 => reuse training samples
    bool reuse_eval_samples = true; // if false and M_eval>0, resample for evaluation
    double ci_alpha = 0.05;         // confidence interval significance level
    // LB3 (softmax-based fixed mixture) configuration
    bool   use_LB3_softmax = false; // perform one softmax linearization pass
    double lb3_softmax_eta = 1.0;   // temperature used for the softmax weights
    bool use_LB3_single = false;    // run a single tangent at S0 (optionally seeded by UB)
    bool use_LB3_iter = false;      // run iterative MM refinements
    bool lb3_start_from_UB = true;  // start LB3 iterations from the UB solution when true
    int lb3_max_iters = 0;          // 0 => treat as single iteration
    double lb3_eta = 1.0;
    double lb3_tol = 1e-6;
    bool lb3_keep_best = true;      // keep the best original-objective solution across iterations
};

struct CapResolution {
    CapMode effective_mode = CapMode::Auto;
    int topk_k = 0;
    bool auto_selected = false;
    bool psi_bounded = false;
    bool pathlen_fallback = false;
};

CapResolution resolve_cap_strategy(const SandwichConfig& cfg);

struct EvalStats { double mean=0, var=0, se=0, ci_low=0, ci_high=0; int M=0; };

struct SandwichResult {
    std::vector<NodeId> S_UB, S_LB, S_pick;
    double fhat_UB = 0.0, fhat_LB = 0.0, fhat_pick = 0.0;
    double certified_union_value = 0.0;
    double capped_union_value = 0.0;
    double certified_ub_value = 0.0;
    double ub_objective_total_S_UB = 0.0;
    double ub_objective_mean_S_UB = 0.0;
    double ub_objective_mean_pick = 0.0;
    double proxy_score = 0.0;
    int corridor_band_r_used = -1;
    int agents_dropped_due_corridor = 0;
    int corridor_resample_attempts = 0;
    int proxy_corridor_r_effective = 0;
    int selection_proxy_r = 0;
    std::vector<NodeId> proxy_solution;
    bool routed_r_equals_one = false;
    bool sum_deprecated_used = false;
    UBMode selection_mode_effective = UBMode::CertifiedUnion;
    std::string selection_objective_type;
    CapMode cap_mode_used = CapMode::Auto;
    int cap_k_used = 0;
    double cap_min = 0.0;
    double cap_median = 0.0;
    double cap_max = 0.0;
    double avg_agents_per_node = 0.0;
    double selection_time_sec = 0.0;
    double proxy_baseline_time_sec = 0.0;
    std::vector<double> selection_iter_times;
    std::string lb3_run_mode;
    int lb3_iters_done = 0;
    bool lb3_improved = false;
    double lb3_best_value = 0.0;
    double lb3_base_value = 0.0;
    std::string cap_cache_key_hash;
    // Evaluation statistics for each candidate solution
    EvalStats stats_UB, stats_LB, stats_pick;
};

bool run_sandwich(const Graph& G,
                  const std::vector<NodeId>& candidates,
                  const std::vector<char>* blocked,
                  const WeightsStore& W,
                  const SandwichConfig& cfg,
                  SandwichResult& out);

// Helper utilities for experiment orchestration ---------------------------------
// Copy an EvalStats bundle into a ReportStats container (for save_eval_report).
void fill_report_stats(const EvalStats& src, ReportStats& dst);

// Populate metadata entries summarizing the SandwichConfig (appends to meta_out).
// `extra_meta` can supply additional key/value pairs (e.g., dataset or cache info).
void build_sandwich_metadata(const SandwichConfig& cfg,
                             const std::vector<NodeId>& candidates,
                             std::vector<std::pair<std::string,std::string>>& meta_out,
                             const std::vector<std::pair<std::string,std::string>>* extra_meta = nullptr);

// Standalone LB3 iterative solver configuration (used for ablation experiments)
// Parameter set for the LB3 iterative (softmax linearization) solver.
struct LB3IterConfig {
    int k = 0;
    double eta = 1.0;
    int max_iters = 0;
    double tol = 1e-6;
    bool start_from_UB = true;
    bool keep_best = true;
    bool exact_iters = false;
    uint64_t seed = 1;
};

bool run_softmax_lb3_solver(const Graph& G,
                            const std::vector<NodeId>& candidates,
                            const std::vector<char>* blocked,
                            const WeightsStore& W,
                            const AgentSamplerConfig& od,
                            int M_agents, int K_paths,
                            int corridor_band_r,
                            const LB3IterConfig& lb3cfg,
                            std::vector<NodeId>& S_out,
                            int* iters_done = nullptr);

bool run_softmax_lb3_solver(const Graph& G,
                            const std::vector<NodeId>& candidates,
                            const std::vector<char>* blocked,
                            const WeightsStore& W,
                            const Psi& psi, double lambda,
                            int corridor_r_for_ub,
                            const AgentSamplerConfig& od,
                            int M_agents, int K_paths,
                            int corridor_band_r,
                            const LB3IterConfig& lb3cfg,
                            std::vector<NodeId>& S_out,
                            int* iters_done = nullptr);

// ============ 8) Baseline algorithms ============
std::vector<NodeId> baseline_random_k(const std::vector<NodeId>& candidates,
                                      int k, uint64_t seed);
std::vector<NodeId> baseline_regular_spacing_grid(int rows, int cols,
                                                  const std::vector<char>* blocked,
                                                  int k);
std::vector<NodeId> baseline_static_com_greedy(const WeightsStore& W,
                                               int n,
                                               const Psi& psi,
                                               int k,
                                               const std::vector<NodeId>& candidates,
                                               const std::vector<double>* demand_or_weight = nullptr);
std::vector<NodeId> baseline_greedy_original_full(AgentPool& pool,
                                                  const std::vector<NodeId>& candidates,
                                                  int k);
std::vector<NodeId> baseline_bruteforce_ub(const LinearWeightsObjective& objective_template,
                                           const std::vector<NodeId>& candidates,
                                           int k,
                                           size_t max_combinations_guard = 0);

// ============ 9) Agent/path generation and Monte Carlo evaluation ============
bool generate_agents_and_paths(const Graph& G,
                               const AgentSampler& sampler,
                               int M, int K, uint64_t seed,
                               std::vector<std::pair<NodeId,NodeId>>& agents,
                               std::vector<std::vector<Path>>& paths_per_agent,
                               bool use_grid_accel = false,
                               int rows = 0, int cols = 0,
                               int corridor_band_r = -1,
                               const std::vector<std::vector<NodeId>>* rev_adj = nullptr,
                               std::vector<std::vector<NodeId>>* corridor_nodes_per_agent = nullptr,
                               int* agents_dropped_due_corridor = nullptr,
                               int* corridor_resample_attempts = nullptr);

double evaluate_original_objective_mc(const Graph& G,
                                      const WeightsStore& W,
                                      const Psi& psi, double lambda,
                                      const std::vector<NodeId>& S,
                                      const AgentSampler& sampler,
                                      int M_eval, int K_paths,
                                      uint64_t seed,
                                      bool use_grid_accel = false,
                                      int rows = 0, int cols = 0,
                                      int corridor_band_r = -1);

EvalStats evaluate_original_objective_mc_stats(const Graph& G,
                                               const WeightsStore& W,
                                               const Psi& psi, double lambda,
                                               const std::vector<NodeId>& S,
                                               const AgentSampler& sampler,
                                               int M_eval, int K_paths,
                                               uint64_t seed,
                                               double alpha = 0.05,
                                               bool use_grid_accel = false,
                                               int rows = 0, int cols = 0,
                                               int corridor_band_r = -1);

EvalStats evaluate_diff_mc_stats(const Graph& G,
                                 const WeightsStore& W,
                                 const Psi& psi, double lambda,
                                 const std::vector<NodeId>& S_A,
                                 const std::vector<NodeId>& S_B,
                                 const AgentSampler& sampler,
                                 int M_eval, int K_paths,
                                 uint64_t seed,
                                 double alpha = 0.05,
                                 bool use_grid_accel = false,
                                 int rows = 0, int cols = 0,
                                 int corridor_band_r = -1);
