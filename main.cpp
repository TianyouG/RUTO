/*
 * treeopt: command-line driver for the TreePlacement sandwich solver.
 *
 * Responsibilities
 * -----------------
 *  1. Parse experiment configuration from CLI flags (see --help).
 *  2. Load a real-world graph (edge list), candidate list, and optional blocked nodes.
 *  3. Build (or load) neighbourhood weights using hard-cutoff or soft-decay rules.
 *  4. Invoke `run_sandwich` (alg.cpp) to optimise certified/heuristic objectives.
 *  5. Monte-Carlo evaluate the picked solution and emit `report.tsv` + `stdout.txt` artefacts.
 *
 * Supported objective families
 * ----------------------------
 *  UBMode: CertifiedUnion (default), CappedUnion, ProxyCorridor (heuristic), SumDeprecated (alias).
 *  CapMode: Auto, PathlenOnly, TopkLocal, GlobalAll.
 *  Psi: Clip(C), ExpSat(beta), Log1p, Power(alpha) with 0 < alpha <= 1.
 *
 * Report overview (report.tsv)
 * ----------------------------
 *  - Header metadata: solver configuration, cap strategy, proxy flags, timings.
 *  - Sections pick/UB/LB with value statistics (mean/var/se/CI/M) and selected node ids.
 *  - Fields align with test.cpp outputs (proxy_score vs certified_ub_value, routed_r_equals_one, etc.).
 *
 * Error codes
 * -----------
 *  EXIT_SUCCESS (0): run completed (or help/dry-run output only).
 *  EXIT_FAILURE (1): configuration/IO errors (missing files, malformed arguments, solver failure).
 *
 * Variable ↔ paper notation
 * -------------------------
 *  k                      ↔ budget k (Eq. (1) in paper).
 *  lambda                 ↔ λ, shortest-path penalty weight (Sec. 3.2).
 *  proxy_r                ↔ corridor width r for ProxyCorridor baseline (Alg. ProxyCorridor).
 *  corridor_r             ↔ r for certified objectives (defaults to proxy_r if unset).
 *  cap_topk_k             ↔ per-node Top-k aggregation k in CappedUnion (Sec. 4.2).
 *  M_agents, K_paths_per_agent ↔ Monte-Carlo agent count / paths per agent (Sec. 2 sampling setup).
 *  M_eval                 ↔ evaluation sample count (0 => reuse training set).
 *
 * Build recipe (annotated)
 * ------------------------
 *  g++ -std=c++17 -O2 -Wall -Wextra main.cpp alg.cpp grid.cpp graphbin.cpp -o treeopt
 *    -std=c++17   : match project headers that rely on C++17 features (filesystem, optional helpers).
 *    -O2          : balanced optimisation level for production runs (use -Og/-O1 when debugging, -O3 to chase peak speed).
 *    -Wall -Wextra: surface unused / suspicious constructs early; append -Werror in CI if desired.
 *    -g           : (optional) include symbols for gdb/perf; pair with -fno-omit-frame-pointer for better profiling stacks.
 *    -I.          : (optional) add project include path if headers are not colocated with sources.
 */

#include "stdafx.h"

namespace {

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

// ----------------------- General constants -----------------------
constexpr const char* kDefaultOutDir = "out";          // default output root directory
constexpr const char* kDefaultTag = "run";             // default experiment tag / subdirectory
constexpr int kDefaultK = 50;                           // budget k (Eq. (1)) default
constexpr int kDefaultMAgents = 10000;                  // M (training agents) default (Sec. 2)
constexpr int kDefaultKPaths = 10;                      // K (paths per agent) default (Sec. 2)
constexpr int kDefaultProxyR = 2;                       // default corridor width r for proxy UB
constexpr double kDefaultLambda = 0.0;                  // λ default (Sec. 3.2) -> no penalty
constexpr double kDefaultSoftRadius = 5.0;              // fallback R when deriving soft radius (~3σ)
constexpr double kDefaultSoftEps = 1e-4;                // smallest soft weight retained during BFS
constexpr double kEpsilonCompare = 1e-6;                // tolerance when comparing iteration timing sums
constexpr char kDefaultCommentChar = '#';               // comment prefix for plain-text inputs

/// Enumeration controlling diagnostic verbosity.
enum class LogLevel { Info, Debug };

enum class PathsMode { Smart, Legacy };

/// Simple POD capturing the soft decay radius parameters (if selected).
struct SoftDecayParams {
    double alpha = 1.0;   // α: exponential decay rate exp(-α·d)
    int radius = 1;       // R: maximum hop distance considered
    double eps = 1e-4;    // ε: minimum retained weight (drop smaller contributions)
};

/// Bundle of all command-line options in canonical types.
struct ProgramOptions {
    // Graph + candidate inputs -------------------------------------------------
    std::string edges_path;       // Input edge list path (u v per line)
    int n = -1;                   // Number of nodes n (paper notation); -1 => infer from edges
    bool n_provided = false;      // Track if user supplied --n explicitly
    bool one_based = false;       // True if input ids are 1-based (subtract 1 on read)
    bool undirected = true;       // True => add mirrored edges while loading
    std::string candidates_path;  // Optional candidate set path
    bool has_candidates = false;  // Whether candidate file was provided
    std::string blocked_path;     // Optional blocked node list path
    bool has_blocked = false;     // Whether blocked file was provided
    std::string graph_bin_in;     // Optional binary graph input
    std::string graph_bin_out;    // Optional binary graph output destination
    std::string pack_candidates_path; // Candidate list to embed when packing (optional)
    std::string pack_blocked_path;    // Blocked list to embed when packing (optional)
    std::string emit_edges_path;      // Target edge list path when generating grids
    bool chain_run = false;           // Continue into optimisation after generation (grid mode)
    bool make_grid = false;           // True when --make_grid rows=...,cols=... provided
    GridBuildSpec grid_spec;          // Parsed grid specification
    bool grid_spec_provided = false;  // Tracks successful grid spec parse

    // Weight construction / caching -------------------------------------------
    bool use_hard_cutoff = false;     // True when using hard cutoff neighbourhoods (Eq. hard UB)
    int hard_cutoff = -1;             // Radius D for hard cutoff; >0 when flag provided
    bool use_soft_decay = false;      // True when using soft decay neighbourhoods
    SoftDecayParams soft;             // Parameters (α,R,ε) for soft decay BFS
    std::string soft_named_profile;   // Copy of original profile string (for metadata echo)
    bool load_weights = false;        // True => attempt to load weights.bin cache
    bool save_weights = false;        // True => persist computed weights.bin cache
    std::string weights_path;         // Explicit cache path (defaults to <out>/<tag>/weights.bin)

    // Sandwich configuration ---------------------------------------------------
    Psi psi = Psi::Clip(1.0);          // ψ compression (Sec. 3): default Clip with C=1
    double lambda = kDefaultLambda;    // λ penalty weight (>=0), Sec. 3.2
    int k = kDefaultK;                 // Budget k (Eq. (1)); must satisfy 1 ≤ k ≤ |candidates|
    int corridor_r = kDefaultProxyR;   // r for certified selection; 0 => inherit proxy_r
    int corridor_band_r = -1;          // Geometric (paper) corridor radius; -1 => off
    int proxy_r = kDefaultProxyR;      // r for proxy baseline / UB factory routing
    UBMode ub_mode = UBMode::CertifiedUnion; // Requested upper-bound mode
    CapMode cap_mode = CapMode::Auto;  // Requested cap strategy (Auto applies psi-dependent default)
    int cap_topk_k = 0;                // Optional Top-k parameter (0 => use budget k)
    bool cap_cache = true;             // Reuse cap cache results across runs when true
    bool enable_proxy_baseline = true; // Emit proxy baseline solution/time when true
    bool reuse_eval_samples = true;    // Reuse training samples for evaluation when M_eval>0
    bool use_LB3_softmax = false;      // Enable softmax linearisation pass
    bool use_LB3_single = false;       // Perform single LB3 MM iteration
    bool use_LB3_iter = false;         // Perform iterative LB3 (overrides single when true)
    bool lb3_keep_best = true;         // Keep best objective value observed during LB3
    bool lb3_start_from_UB = true;     // Seed LB3 with UB solution (Sec. LB3 configuration)
    int M_agents = kDefaultMAgents;    // Training agent count M (Sec. 2)
    int K_paths_per_agent = kDefaultKPaths; // Paths per agent K
    int M_eval = 0;                    // Evaluation sample size (0 => reuse training data)
    int lb3_max_iters = 1;             // Maximum LB3 iterations when use_LB3_iter
    double lb3_eta = 1.0;              // LB3 temperature parameter η
    double lb3_tol = 1e-6;             // LB3 convergence tolerance
    double lb3_softmax_eta = 1.0;      // Softmax linearisation η when enabled

    // OD sampling -------------------------------------------------------------
    std::string od_dist_token = "uniform";             // Raw CLI token for OD distribution
    ODDistribution od_dist = ODDistribution::Uniform;   // Parsed OD distribution enum
    int od_band_min = 0;                                // Distance band lower bound (inclusive)
    int od_band_max = std::numeric_limits<int>::max();  // Distance band upper bound (inclusive)
    bool od_band_specified = false;                     // Tracks explicit --od_band usage
    double od_sigma = 2.0;                              // Gaussian σ ( > 0 ) when applicable
    bool od_sigma_specified = false;                    // Tracks explicit --od_sigma usage
    double od_gravity_beta = 0.2;                       // Gravity decay β (≥0)
    bool od_gravity_specified = false;                  // Tracks explicit --od_gravity_beta usage
    std::vector<std::pair<NodeId,double>> od_hotspots;  // Hotspot mixture weights (node, weight)
    bool od_hotspots_inline = false;                    // True when provided via --od_hotspots
    bool od_hotspots_from_file = false;                 // True when parsed from --od_hotspots_file
    std::string od_hotspots_file;                       // Original hotspots file path (if any)
    std::string od_hotspots_hash;                       // Hash of hotspot file contents (hex)
    int od_rows = 0;                                    // Optional grid rows for OD helper
    int od_cols = 0;                                    // Optional grid cols for OD helper
    bool od_rows_provided = false;                      // Tracks explicit --od_rows usage
    bool od_cols_provided = false;                      // Tracks explicit --od_cols usage
    bool od_rows_cols_inherited = false;                // True when rows/cols inherited from --make_grid

    // Output + miscellany ------------------------------------------------------
    std::string out_dir = kDefaultOutDir; // Output root directory
    std::string tag = kDefaultTag;        // Sub-directory + report tag
    LogLevel log_level = LogLevel::Info;  // Logging verbosity
    uint64_t seed = 1;                    // Global RNG seed used by solver components
    bool dry_run = false;                 // True => parse inputs, echo config, exit without solving
    bool write_json_meta = false;         // True => write run_config.json
    std::string json_meta_path;           // Optional explicit JSON metadata path
    std::string method_name = "sandwich"; // Algorithm method selected via --method (default sandwich)
    std::string method_args_raw;          // Raw --margs string (for reproducibility reporting)
    bool list_methods = false;            // Whether to list available methods and exit
    std::string raw_cmdline;              // Full command line (for metadata / graph bins)

    // Derived reporting helpers -----------------------------------------------
    std::string edges_display;            // Effective dataset descriptor for config echo
    std::string candidates_display;       // Candidate source label for config echo
    std::string blocked_display;          // Blocked source label for config echo
    CapMode cap_mode_effective = CapMode::Auto; // Resolved runtime cap mode
    int cap_topk_effective = 0;            // Resolved Top-k parameter (0 => unused)
    bool cap_mode_auto_resolved = false;   // True when Auto resolved the runtime mode
    bool cap_mode_psi_bounded = false;     // True when psi bounded prompted PathlenOnly
    bool cap_mode_pathlen_fallback = false;// True when PathlenOnly forced to TopkLocal
    std::string cap_mode_log_line;         // Preformatted log message (config echo)

    // Path management helpers -------------------------------------------------
    PathsMode paths_mode = PathsMode::Smart;
    std::string dataset_key;               // Canonical dataset identifier
    std::string dataset_root;              // data/datasets/<dataset_key>
    std::string experiments_root;          // data/experiments/<dataset_key>
    std::string out_dir_effective;         // Effective <out_dir>/<tag> path after rewrites

    // Blocked set bookkeeping -------------------------------------------------
    int blocked_count = 0;
    std::string blocked_source_label;      // "empty_file", path, or "<none>"
    bool blocked_seed_has_value = false;
    long long blocked_seed_value = 0;
    bool blocked_fraction_has_value = false;
    double blocked_fraction_value = 0.0;
};

struct MethodContext {
    const Graph& G;
    const std::vector<NodeId>& candidates;
    const std::vector<char>* blocked;
    WeightsStore& weights;
    const ProgramOptions& opts;
    SandwichConfig base_cfg;
};

struct MethodRunResult {
    SandwichResult result;                               // Filled by the method runner
    SandwichConfig applied_cfg;                          // Actual configuration after overrides
    ReportStats stats_pick{};
    ReportStats stats_UB{};
    ReportStats stats_LB{};
    std::vector<std::pair<std::string, std::string>> extra_meta; // Method-specific metadata
    std::vector<std::string> warnings;                   // Non-fatal parsing or runtime warnings
    std::string note;                                    // Optional note (e.g., bruteforce skip)
    bool success = true;                                 // False indicates hard failure
};

struct MethodUnionData {
    std::vector<std::vector<NodeId>> nodes_per_agent; // nodes covered per agent
    std::vector<int> min_path_len;                    // min path length per agent
    std::vector<int> max_path_len;                    // max path length per agent
    std::vector<int> union_counts;                    // total coverage per node
    int max_paths_per_agent = 0;
    long long sum_Lmin = 0;
    double avg_agents_per_node = 0.0;
};

using MethodRunner = std::function<bool(const MethodContext&,
                                        const std::unordered_map<std::string, std::string>&,
                                        MethodRunResult&)>;

struct MethodSpec {
    std::string name;                    // method identifier (lowercase)
    std::string description;             // short description shown in --list-methods
    std::vector<std::string> key_hints;  // commonly recognised keys for --margs
    MethodRunner runner;                 // adapter that executes the method
};

// Helper prototypes consumed by method runners (definitions provided later in the file).
static void apply_common_overrides(SandwichConfig& cfg,
                                   const std::unordered_map<std::string, std::string>& margs,
                                   std::vector<std::string>& warnings);
static const std::unordered_set<std::string>& common_marg_keys();
static std::string to_string_compact(double value);
static std::string bool_to_string(bool v);
static bool get_marg_int(const std::unordered_map<std::string, std::string>& kv,
                         const std::string& key,
                         int& value_out,
                         std::vector<std::string>& warnings);
static bool get_marg_double(const std::unordered_map<std::string, std::string>& kv,
                            const std::string& key,
                            double& value_out,
                            std::vector<std::string>& warnings);
static bool get_marg_bool(const std::unordered_map<std::string, std::string>& kv,
                          const std::string& key,
                          bool& value_out,
                          std::vector<std::string>& warnings);
static MethodUnionData compute_union_data(const std::vector<std::vector<Path>>& paths,
                                          int n,
                                          const std::vector<std::vector<NodeId>>* corridor_nodes_per_agent = nullptr);
static void populate_selection_metrics(const MethodContext& ctx,
                                       const SandwichConfig& cfg,
                                       const MethodUnionData& union_data,
                                       const std::vector<NodeId>& selection,
                                       const std::vector<std::pair<NodeId, NodeId>>& agents,
                                       const std::vector<std::vector<Path>>& paths,
                                       const std::vector<std::vector<NodeId>>* corridor_nodes_per_agent,
                                       int agents_dropped_corridor,
                                       int corridor_resample_attempts,
                                       MethodRunResult& out);
static bool generate_training_samples(const MethodContext& ctx,
                                      const SandwichConfig& cfg,
                                      std::vector<std::pair<NodeId, NodeId>>& agents,
                                      std::vector<std::vector<Path>>& paths,
                                      std::vector<std::vector<NodeId>>* corridor_nodes = nullptr,
                                      int* agents_dropped_corridor = nullptr,
                                      int* corridor_resample_attempts = nullptr);
static int ensure_budget_within_candidates(const MethodContext& ctx,
                                           SandwichConfig& cfg,
                                           MethodRunResult& out);

// Forward declarations for method runners (implemented later in this file).
static bool run_method_sandwich(const MethodContext& ctx,
                                const std::unordered_map<std::string, std::string>& margs,
                                MethodRunResult& out);
static bool run_method_random_k(const MethodContext& ctx,
                                const std::unordered_map<std::string, std::string>& margs,
                                MethodRunResult& out) {
    SandwichConfig cfg = ctx.base_cfg;
    apply_common_overrides(cfg, margs, out.warnings);
    std::unordered_set<std::string> known_keys = common_marg_keys();
    ensure_budget_within_candidates(ctx, cfg, out);

    Timer timer{"random_k"};
    std::vector<NodeId> selection = baseline_random_k(ctx.candidates, cfg.k, cfg.seed);
    double select_time = timer.stop().count();

    std::vector<std::pair<NodeId, NodeId>> agents;
    std::vector<std::vector<Path>> paths;
    std::vector<std::vector<NodeId>> corridor_nodes;
    int dropped_corridor = 0;
    int resample_corridor = 0;
    bool generated = generate_training_samples(ctx, cfg, agents, paths,
                                               cfg.corridor_band_r >= 0 ? &corridor_nodes : nullptr,
                                               &dropped_corridor,
                                               &resample_corridor);
    if (!generated) {
        out.warnings.push_back("generate_agents_and_paths failed; certificates may be zero");
    }
    MethodUnionData union_data = generated
        ? compute_union_data(paths, ctx.G.n,
                             (cfg.corridor_band_r >= 0) ? &corridor_nodes : nullptr)
        : MethodUnionData{};
    if (!generated) {
        union_data.union_counts.assign(ctx.G.n, 0);
        union_data.max_paths_per_agent = 0;
        union_data.sum_Lmin = 0;
        union_data.avg_agents_per_node = 0.0;
    }

    out.result = SandwichResult{};
    out.result.selection_mode_effective = UBMode::ProxyCorridor;
    out.result.selection_objective_type = "heuristic";
    out.result.selection_proxy_r = 0;
    out.result.routed_r_equals_one = false;
    out.result.sum_deprecated_used = false;
    out.result.selection_time_sec = select_time;
    out.result.selection_iter_times.clear();
    out.result.proxy_score = 0.0;
    out.result.proxy_baseline_time_sec = 0.0;
    out.result.proxy_solution.clear();
    out.result.lb3_run_mode = "none";
    out.result.lb3_iters_done = 0;
    out.result.lb3_improved = false;
    out.result.lb3_best_value = 0.0;
    out.result.lb3_base_value = 0.0;

    populate_selection_metrics(ctx, cfg, union_data, selection, agents, paths,
                               (cfg.corridor_band_r >= 0 && generated) ? &corridor_nodes : nullptr,
                               dropped_corridor,
                               resample_corridor,
                               out);

    fill_report_stats(out.result.stats_pick, out.stats_pick);
    fill_report_stats(out.result.stats_UB, out.stats_UB);
    fill_report_stats(out.result.stats_LB, out.stats_LB);

    out.applied_cfg = cfg;
    out.extra_meta.emplace_back("time_method_sec", to_string_compact(select_time));

    for (const auto& kv : margs) {
        if (!known_keys.count(kv.first)) {
            out.warnings.push_back("margs key '" + kv.first + "' is not recognised for method random_k");
        }
    }

    return true;
}
static bool run_method_regular_spacing(const MethodContext& ctx,
                                       const std::unordered_map<std::string, std::string>& margs,
                                       MethodRunResult& out) {
    SandwichConfig cfg = ctx.base_cfg;
    apply_common_overrides(cfg, margs, out.warnings);
    std::unordered_set<std::string> known_keys = common_marg_keys();
    known_keys.insert({"rows", "cols"});
    ensure_budget_within_candidates(ctx, cfg, out);

    int rows = 0, cols = 0;
    bool have_rows = get_marg_int(margs, "rows", rows, out.warnings);
    bool have_cols = get_marg_int(margs, "cols", cols, out.warnings);
    if (!have_rows && ctx.opts.make_grid && ctx.opts.grid_spec_provided) {
        rows = ctx.opts.grid_spec.rows;
        have_rows = rows > 0;
    }
    if (!have_cols && ctx.opts.make_grid && ctx.opts.grid_spec_provided) {
        cols = ctx.opts.grid_spec.cols;
        have_cols = cols > 0;
    }
    if (!have_rows || rows <= 0) {
        out.warnings.push_back("margs.rows must be provided (>0) for regular_spacing_grid");
        out.success = false;
        return false;
    }
    if (!have_cols || cols <= 0) {
        out.warnings.push_back("margs.cols must be provided (>0) for regular_spacing_grid");
        out.success = false;
        return false;
    }
    if (rows * cols != ctx.G.n) {
        out.warnings.push_back("rows*cols does not match graph node count; got " +
                               std::to_string(rows * cols) + " vs n=" + std::to_string(ctx.G.n));
        out.success = false;
        return false;
    }

    Timer timer{"regular_spacing_grid"};
    std::vector<NodeId> selection = baseline_regular_spacing_grid(rows, cols, ctx.blocked, cfg.k);
    double select_time = timer.stop().count();

    std::vector<std::pair<NodeId, NodeId>> agents;
    std::vector<std::vector<Path>> paths;
    std::vector<std::vector<NodeId>> corridor_nodes;
    int dropped_corridor = 0;
    int resample_corridor = 0;
    bool generated = generate_training_samples(ctx, cfg, agents, paths,
                                               cfg.corridor_band_r >= 0 ? &corridor_nodes : nullptr,
                                               &dropped_corridor,
                                               &resample_corridor);
    if (!generated) {
        out.warnings.push_back("generate_agents_and_paths failed; certificates may be zero");
    }
    MethodUnionData union_data = generated
        ? compute_union_data(paths, ctx.G.n,
                             (cfg.corridor_band_r >= 0) ? &corridor_nodes : nullptr)
        : MethodUnionData{};
    if (!generated) {
        union_data.union_counts.assign(ctx.G.n, 0);
        union_data.max_paths_per_agent = 0;
        union_data.sum_Lmin = 0;
        union_data.avg_agents_per_node = 0.0;
    }

    out.result = SandwichResult{};
    out.result.selection_mode_effective = UBMode::ProxyCorridor;
    out.result.selection_objective_type = "heuristic";
    out.result.selection_proxy_r = 0;
    out.result.routed_r_equals_one = false;
    out.result.sum_deprecated_used = false;
    out.result.selection_time_sec = select_time;
    out.result.selection_iter_times.clear();
    out.result.proxy_score = 0.0;
    out.result.proxy_baseline_time_sec = 0.0;
    out.result.proxy_solution.clear();
    out.result.lb3_run_mode = "none";
    out.result.lb3_iters_done = 0;
    out.result.lb3_improved = false;
    out.result.lb3_best_value = 0.0;
    out.result.lb3_base_value = 0.0;

    populate_selection_metrics(ctx, cfg, union_data, selection, agents, paths,
                               (cfg.corridor_band_r >= 0 && generated) ? &corridor_nodes : nullptr,
                               dropped_corridor,
                               resample_corridor,
                               out);
    fill_report_stats(out.result.stats_pick, out.stats_pick);
    fill_report_stats(out.result.stats_UB, out.stats_UB);
    fill_report_stats(out.result.stats_LB, out.stats_LB);

    out.applied_cfg = cfg;
    out.extra_meta.emplace_back("time_method_sec", to_string_compact(select_time));
    out.extra_meta.emplace_back("grid_rows", std::to_string(rows));
    out.extra_meta.emplace_back("grid_cols", std::to_string(cols));

    for (const auto& kv : margs) {
        if (!known_keys.count(kv.first)) {
            out.warnings.push_back("margs key '" + kv.first + "' is not recognised for method regular_spacing_grid");
        }
    }

    return true;
}
static bool run_method_static_com_greedy(const MethodContext& ctx,
                                         const std::unordered_map<std::string, std::string>& margs,
                                         MethodRunResult& out) {
    SandwichConfig cfg = ctx.base_cfg;
    apply_common_overrides(cfg, margs, out.warnings);
    std::unordered_set<std::string> known_keys = common_marg_keys();
    ensure_budget_within_candidates(ctx, cfg, out);

    Timer timer{"static_com_greedy"};
    std::vector<NodeId> selection = baseline_static_com_greedy(ctx.weights, ctx.G.n, cfg.psi, cfg.k, ctx.candidates, nullptr);
    double select_time = timer.stop().count();

    std::vector<std::pair<NodeId, NodeId>> agents;
    std::vector<std::vector<Path>> paths;
    std::vector<std::vector<NodeId>> corridor_nodes;
    int dropped_corridor = 0;
    int resample_corridor = 0;
    bool generated = generate_training_samples(ctx, cfg, agents, paths,
                                               cfg.corridor_band_r >= 0 ? &corridor_nodes : nullptr,
                                               &dropped_corridor,
                                               &resample_corridor);
    if (!generated) {
        out.warnings.push_back("generate_agents_and_paths failed; certificates may be zero");
    }
    MethodUnionData union_data = generated
        ? compute_union_data(paths, ctx.G.n,
                             (cfg.corridor_band_r >= 0) ? &corridor_nodes : nullptr)
        : MethodUnionData{};
    if (!generated) {
        union_data.union_counts.assign(ctx.G.n, 0);
        union_data.max_paths_per_agent = 0;
        union_data.sum_Lmin = 0;
        union_data.avg_agents_per_node = 0.0;
    }

    out.result = SandwichResult{};
    out.result.selection_mode_effective = UBMode::ProxyCorridor;
    out.result.selection_objective_type = "heuristic";
    out.result.selection_proxy_r = 0;
    out.result.routed_r_equals_one = false;
    out.result.sum_deprecated_used = false;
    out.result.selection_time_sec = select_time;
    out.result.selection_iter_times.clear();
    out.result.proxy_score = 0.0;
    out.result.proxy_baseline_time_sec = 0.0;
    out.result.proxy_solution.clear();
    out.result.lb3_run_mode = "none";
    out.result.lb3_iters_done = 0;
    out.result.lb3_improved = false;
    out.result.lb3_best_value = 0.0;
    out.result.lb3_base_value = 0.0;

    populate_selection_metrics(ctx, cfg, union_data, selection, agents, paths,
                               (cfg.corridor_band_r >= 0 && generated) ? &corridor_nodes : nullptr,
                               dropped_corridor,
                               resample_corridor,
                               out);
    fill_report_stats(out.result.stats_pick, out.stats_pick);
    fill_report_stats(out.result.stats_UB, out.stats_UB);
    fill_report_stats(out.result.stats_LB, out.stats_LB);

    out.applied_cfg = cfg;
    out.extra_meta.emplace_back("time_method_sec", to_string_compact(select_time));

    for (const auto& kv : margs) {
        if (!known_keys.count(kv.first)) {
            out.warnings.push_back("margs key '" + kv.first + "' is not recognised for method static_com_greedy");
        }
    }

    return true;
}
static bool run_method_greedy_original_full(const MethodContext& ctx,
                                            const std::unordered_map<std::string, std::string>& margs,
                                            MethodRunResult& out) {
    SandwichConfig cfg = ctx.base_cfg;
    apply_common_overrides(cfg, margs, out.warnings);
    std::unordered_set<std::string> known_keys = common_marg_keys();
    ensure_budget_within_candidates(ctx, cfg, out);

    std::vector<std::pair<NodeId, NodeId>> agents;
    std::vector<std::vector<Path>> paths;
    std::vector<std::vector<NodeId>> corridor_nodes;
    int dropped_corridor = 0;
    int resample_corridor = 0;
    bool generated = generate_training_samples(ctx, cfg, agents, paths,
                                               cfg.corridor_band_r >= 0 ? &corridor_nodes : nullptr,
                                               &dropped_corridor,
                                               &resample_corridor);
    if (!generated) {
        out.warnings.push_back("generate_agents_and_paths failed; cannot run greedy_original_full");
        out.success = false;
        return false;
    }

    AgentPoolConfig pcfg{cfg.psi, cfg.lambda};
    AgentPool pool;
    if (!pool.init(ctx.G, ctx.weights, pcfg, agents, paths)) {
        out.warnings.push_back("AgentPool initialisation failed");
        out.success = false;
        return false;
    }

    Timer timer{"greedy_original_full"};
    std::vector<NodeId> selection = baseline_greedy_original_full(pool, ctx.candidates, cfg.k);
    double select_time = timer.stop().count();

    MethodUnionData union_data = compute_union_data(paths, ctx.G.n,
                                                    (cfg.corridor_band_r >= 0) ? &corridor_nodes : nullptr);

    out.result = SandwichResult{};
    out.result.selection_mode_effective = UBMode::ProxyCorridor;
    out.result.selection_objective_type = "heuristic";
    out.result.selection_proxy_r = 0;
    out.result.routed_r_equals_one = false;
    out.result.sum_deprecated_used = false;
    out.result.selection_time_sec = select_time;
    out.result.selection_iter_times.clear();
    out.result.proxy_score = 0.0;
    out.result.proxy_baseline_time_sec = 0.0;
    out.result.proxy_solution.clear();
    out.result.lb3_run_mode = "none";
    out.result.lb3_iters_done = 0;
    out.result.lb3_improved = false;
    out.result.lb3_best_value = 0.0;
    out.result.lb3_base_value = 0.0;

    populate_selection_metrics(ctx, cfg, union_data, selection, agents, paths,
                               (cfg.corridor_band_r >= 0) ? &corridor_nodes : nullptr,
                               dropped_corridor,
                               resample_corridor,
                               out);
    fill_report_stats(out.result.stats_pick, out.stats_pick);
    fill_report_stats(out.result.stats_UB, out.stats_UB);
    fill_report_stats(out.result.stats_LB, out.stats_LB);

    out.applied_cfg = cfg;
    out.extra_meta.emplace_back("time_method_sec", to_string_compact(select_time));

    for (const auto& kv : margs) {
        if (!known_keys.count(kv.first)) {
            out.warnings.push_back("margs key '" + kv.first + "' is not recognised for method greedy_original_full");
        }
    }

    return true;
}
static bool run_method_bruteforce_ub(const MethodContext& ctx,
                                     const std::unordered_map<std::string, std::string>& margs,
                                     MethodRunResult& out) {
    SandwichConfig cfg = ctx.base_cfg;
    apply_common_overrides(cfg, margs, out.warnings);
    std::unordered_set<std::string> known_keys = common_marg_keys();
    known_keys.insert({"nmax", "guard"});
    ensure_budget_within_candidates(ctx, cfg, out);

    size_t nmax = 16;
    int nmax_int = 0;
    if (get_marg_int(margs, "nmax", nmax_int, out.warnings) && nmax_int > 0) {
        nmax = static_cast<size_t>(nmax_int);
    }
    size_t max_guard = 0;
    if (get_marg_int(margs, "guard", nmax_int, out.warnings) && nmax_int > 0) {
        max_guard = static_cast<size_t>(nmax_int);
    }

    if (ctx.candidates.size() > nmax) {
        out.note = "skipped: n>nmax";
        out.applied_cfg = cfg;
        out.result = SandwichResult{};
        out.result.selection_mode_effective = UBMode::ProxyCorridor;
        out.result.selection_objective_type = "skipped";
        out.stats_pick = ReportStats{};
        out.stats_UB = ReportStats{};
        out.stats_LB = ReportStats{};
        out.extra_meta.emplace_back("time_method_sec", "0.0");
        return true;
    }

    std::vector<std::pair<NodeId, NodeId>> agents;
    std::vector<std::vector<Path>> paths;
    std::vector<std::vector<NodeId>> corridor_nodes;
    int dropped_corridor = 0;
    int resample_corridor = 0;
    bool generated = generate_training_samples(ctx, cfg, agents, paths,
                                               cfg.corridor_band_r >= 0 ? &corridor_nodes : nullptr,
                                               &dropped_corridor,
                                               &resample_corridor);
    if (!generated) {
        out.warnings.push_back("generate_agents_and_paths failed; cannot run bruteforce_ub");
        out.success = false;
        return false;
    }

    MethodUnionData union_data = compute_union_data(paths, ctx.G.n,
                                                    (cfg.corridor_band_r >= 0) ? &corridor_nodes : nullptr);

    CertifiedUnionObjective objective_template;
    objective_template.init(ctx.weights, ctx.G.n, cfg.psi, cfg.lambda,
                            &union_data.union_counts,
                            static_cast<double>(union_data.sum_Lmin),
                            ctx.candidates);

    Timer timer{"bruteforce_ub"};
    std::vector<NodeId> selection = baseline_bruteforce_ub(objective_template, ctx.candidates, cfg.k, max_guard);
    double select_time = timer.stop().count();

    out.result = SandwichResult{};
    out.result.selection_mode_effective = UBMode::CertifiedUnion;
    out.result.selection_objective_type = "certified";
    out.result.selection_proxy_r = 0;
    out.result.routed_r_equals_one = false;
    out.result.sum_deprecated_used = false;
    out.result.selection_time_sec = select_time;
    out.result.selection_iter_times.clear();
    out.result.proxy_score = 0.0;
    out.result.proxy_baseline_time_sec = 0.0;
    out.result.proxy_solution.clear();
    out.result.lb3_run_mode = "none";
    out.result.lb3_iters_done = 0;
    out.result.lb3_improved = false;
    out.result.lb3_best_value = 0.0;
    out.result.lb3_base_value = 0.0;

    populate_selection_metrics(ctx, cfg, union_data, selection, agents, paths,
                               (cfg.corridor_band_r >= 0) ? &corridor_nodes : nullptr,
                               dropped_corridor,
                               resample_corridor,
                               out);
    fill_report_stats(out.result.stats_pick, out.stats_pick);
    fill_report_stats(out.result.stats_UB, out.stats_UB);
    fill_report_stats(out.result.stats_LB, out.stats_LB);

    out.applied_cfg = cfg;
    out.extra_meta.emplace_back("time_method_sec", to_string_compact(select_time));

    for (const auto& kv : margs) {
        if (!known_keys.count(kv.first)) {
            out.warnings.push_back("margs key '" + kv.first + "' is not recognised for method bruteforce_ub");
        }
    }

    return true;
}

// ----------------------- Utility helpers -----------------------

/// Formats floating-point values to match report precision (fixed, 6 decimals).
/**
 * @param value Numeric value to print.
 * @return Fixed-precision string used in report.tsv/meta outputs.
 */
static std::string to_string_compact(double value) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6) << value;
    return oss.str();
}

/// Converts a boolean into "true"/"false" textual form.
/**
 * @param v Boolean flag.
 * @return Literal "true" or "false".
 */
static std::string bool_to_string(bool v) {
    return v ? "true" : "false";
}

/// Maps a UBMode enum to its CLI/report label.
/**
 * @param mode Upper-bound mode enum.
 * @return Textual name shown in logs and metadata.
 */
static std::string ub_mode_name(UBMode mode) {
    switch (mode) {
        case UBMode::CertifiedUnion: return "CertifiedUnion";
        case UBMode::CappedUnion:    return "CappedUnion";
        case UBMode::ProxyCorridor:  return "ProxyCorridor";
        case UBMode::SumDeprecated:  return "SumDeprecated";
    }
    return "Unknown";
}

/// Maps a CapMode enum to its CLI/report label.
/**
 * @param mode Cap strategy enum.
 * @return Descriptive label for metadata/logs.
 */
static std::string cap_mode_name(CapMode mode) {
    switch (mode) {
        case CapMode::Auto:        return "Auto";
        case CapMode::PathlenOnly: return "PathlenOnly";
        case CapMode::TopkLocal:   return "TopkLocal";
        case CapMode::GlobalAll:   return "GlobalAll";
    }
    return "Unknown";
}

/// Maps Psi type to the label used in reports.
/**
 * @param psi Psi configuration.
 * @return Textual identifier (ClipMinC, Log1p, ExpSat, Power).
 */
static std::string psi_type_name(const Psi& psi) {
    switch (psi.type) {
        case Psi::Type::ClipMinC: return "ClipMinC";
        case Psi::Type::Log1p:    return "Log1p";
        case Psi::Type::ExpSat:   return "ExpSat";
        case Psi::Type::Power:    return "Power";
    }
    return "Unknown";
}

/// Parses a signed integer with range checking and descriptive diagnostics.
/**
 * @param text Raw CLI token.
 * @param value_out Parsed integer result.
 * @param flag Flag name for helpful error messages.
 * @param error Detailed error text if parsing fails.
 * @return True on success, false with @p error populated otherwise.
 */
static bool parse_int_value(const std::string& text,
                            int& value_out,
                            const char* flag,
                            std::string& error) {
    try {
        size_t idx = 0;
        long long parsed = std::stoll(text, &idx, 10);
        if (idx != text.size()) {
            error = std::string(flag) + " expects an integer, got '" + text + "'";
            return false;
        }
        if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
            error = std::string(flag) + " value out of int range";
            return false;
        }
        value_out = static_cast<int>(parsed);
        return true;
    } catch (const std::exception&) {
        error = std::string(flag) + " expects an integer, got '" + text + "'";
        return false;
    }
}

/// Parses a floating-point CLI value with validation and guidance.
/**
 * @param text Raw CLI token.
 * @param value_out Parsed double result.
 * @param flag Flag name for contextual error output.
 * @param error Human-readable error string on failure.
 * @return True when parsing succeeds.
 */
static bool parse_double_value(const std::string& text,
                               double& value_out,
                               const char* flag,
                               std::string& error) {
    try {
        size_t idx = 0;
        double parsed = std::stod(text, &idx);
        if (idx != text.size()) {
            error = std::string(flag) + " expects a floating-point number, got '" + text + "'";
            return false;
        }
        value_out = parsed;
        return true;
    } catch (const std::exception&) {
        error = std::string(flag) + " expects a floating-point number, got '" + text + "'";
        return false;
    }
}

/// Parses an unsigned 64-bit integer from a CLI token.
/**
 * @param text Raw CLI token.
 * @param value_out Parsed unsigned integer.
 * @param flag Flag name (e.g., "--seed").
 * @param error Detailed error string on failure.
 * @return True on success.
 */
static bool parse_uint64_value(const std::string& text,
                               uint64_t& value_out,
                               const char* flag,
                               std::string& error) {
    try {
        size_t idx = 0;
        unsigned long long parsed = std::stoull(text, &idx, 10);
        if (idx != text.size()) {
            error = std::string(flag) + " expects an unsigned integer, got '" + text + "'";
            return false;
        }
        value_out = static_cast<uint64_t>(parsed);
        return true;
    } catch (const std::exception&) {
        error = std::string(flag) + " expects an unsigned integer, got '" + text + "'";
        return false;
    }
}

/// Convenience overload that appends string metadata.
static void append_meta(std::vector<std::pair<std::string, std::string>>& meta,
                        const std::string& key,
                        const std::string& value) {
    meta.emplace_back(key, value);
}

/// Convenience overload that appends integer metadata.
static void append_meta(std::vector<std::pair<std::string, std::string>>& meta,
                        const std::string& key,
                        int value) {
    meta.emplace_back(key, std::to_string(value));
}

/// Convenience overload that appends double metadata after formatting.
static void append_meta(std::vector<std::pair<std::string, std::string>>& meta,
                        const std::string& key,
                        double value) {
    meta.emplace_back(key, to_string_compact(value));
}

/// Convenience overload that appends boolean metadata.
static void append_meta(std::vector<std::pair<std::string, std::string>>& meta,
                        const std::string& key,
                        bool value) {
    meta.emplace_back(key, bool_to_string(value));
}

/// Parses text into a UBMode enum.
/**
 * @param token User-specified string.
 * @param out_mode Destination UBMode.
 * @return True when token matches a known mode.
 */
static bool parse_ub_mode(const std::string& token, UBMode& out_mode) {
    if (token == "CertifiedUnion") { out_mode = UBMode::CertifiedUnion; return true; }
    if (token == "CappedUnion")    { out_mode = UBMode::CappedUnion;    return true; }
    if (token == "ProxyCorridor")  { out_mode = UBMode::ProxyCorridor;  return true; }
    if (token == "SumDeprecated")  { out_mode = UBMode::SumDeprecated;  return true; }
    return false;
}

/// Parses text into a CapMode enum.
/**
 * @param token User-specified string.
 * @param out_mode Destination CapMode.
 * @return True when recognised.
 */
static bool parse_cap_mode(const std::string& token, CapMode& out_mode) {
    if (token == "Auto")        { out_mode = CapMode::Auto;        return true; }
    if (token == "PathlenOnly") { out_mode = CapMode::PathlenOnly; return true; }
    if (token == "TopkLocal")   { out_mode = CapMode::TopkLocal;   return true; }
    if (token == "GlobalAll")   { out_mode = CapMode::GlobalAll;   return true; }
    return false;
}

static std::string lower_copy(std::string s);

static bool parse_paths_mode(const std::string& token, PathsMode& out_mode) {
    const std::string lower = lower_copy(token);
    if (lower == "smart")  { out_mode = PathsMode::Smart;  return true; }
    if (lower == "legacy") { out_mode = PathsMode::Legacy; return true; }
    return false;
}

/// Parses "0"/"1" (or true/false) tokens into bool.
static bool parse_zero_one(const std::string& token, bool& value_out) {
    if (token == "0" || token == "false" || token == "False") { value_out = false; return true; }
    if (token == "1" || token == "true"  || token == "True")  { value_out = true;  return true; }
    return false;
}

/// Case-fold copy helper used by parsers.
static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

/// Trims whitespace from both ends of a string.
static std::string trim_copy(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

static std::string sanitize_dataset_key(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char ch : raw) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        } else if (ch == '_' || ch == '-' || ch == ' ') {
            if (!out.empty() && out.back() != '_') out.push_back('_');
        }
    }
    if (out.empty()) out = "DATASET";
    return out;
}

static std::string basename_without_ext(const std::string& path) {
    if (path.empty()) return {};
    std::error_code ec;
    fs::path p(path);
    std::string stem = p.stem().string();
    return stem;
}

static std::string strip_edges_suffix(const std::string& name) {
    const std::string suffix = "_edges";
    if (name.size() > suffix.size()) {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if (lower.rfind(suffix) == lower.size() - suffix.size()) {
            return name.substr(0, name.size() - suffix.size());
        }
    }
    return name;
}

struct DatasetKeyInferenceResult {
    std::string key;
    std::string rule;
    bool warn_generic = false;
    std::string warn_detail;
};

static bool is_generic_dataset_filename(const std::string& stem_lower) {
    static const std::array<const char*, 4> kGenericNames = {
        "edges",
        "graph",
        "graph_edges",
        "grid_edges"
    };
    for (const char* name : kGenericNames) {
        if (stem_lower == name) return true;
    }
    return false;
}

static DatasetKeyInferenceResult ensure_non_generic_key(DatasetKeyInferenceResult res) {
    if (res.rule != "--dataset_key") {
        std::string lowered = lower_copy(res.key);
        if (is_generic_dataset_filename(lowered)) {
            res.warn_generic = true;
            res.key = sanitize_dataset_key("dataset");
            if (res.warn_detail.empty()) res.warn_detail = res.key;
            if (!res.rule.empty()) res.rule += " -> fallback";
            else res.rule = "fallback";
            return res;
        }
    }
    if (res.warn_generic && res.warn_detail.empty()) {
        res.warn_detail = res.key;
    }
    if (res.rule.empty()) res.rule = "fallback";
    return res;
}

static DatasetKeyInferenceResult infer_dataset_key_from_opts(const ProgramOptions& opts) {
    DatasetKeyInferenceResult res{};

    auto assign_key = [&](const std::string& raw, const std::string& rule) -> bool {
        if (raw.empty()) return false;
        res.key = sanitize_dataset_key(raw);
        res.rule = rule;
        return true;
    };

    if (!opts.dataset_key.empty() && assign_key(opts.dataset_key, "--dataset_key")) {
        return ensure_non_generic_key(res);
    }

    auto handle_generic_path = [&](const std::string& path, const char* label) -> bool {
        if (path.empty()) return false;
        fs::path p(path);
        std::string stem_lower = lower_copy(p.stem().string());
        if (!is_generic_dataset_filename(stem_lower)) return false;
        std::string parent = p.parent_path().filename().string();
        if (parent.empty()) parent = "dataset";
        assign_key(parent, std::string("parent(") + label + ")");
        res.warn_generic = true;
        res.warn_detail = res.key;
        return true;
    };

    if (handle_generic_path(opts.emit_edges_path, "--emit_edges")) {
        return ensure_non_generic_key(res);
    }
    if (handle_generic_path(opts.edges_path, "--edges")) {
        return ensure_non_generic_key(res);
    }

    std::string bin_out = basename_without_ext(opts.graph_bin_out);
    if (!bin_out.empty() && assign_key(bin_out, "--graph_bin_out")) {
        return ensure_non_generic_key(res);
    }

    std::string bin_in = basename_without_ext(opts.graph_bin_in);
    if (!bin_in.empty() && assign_key(bin_in, "--graph_bin_in")) {
        return ensure_non_generic_key(res);
    }

    if (!opts.edges_path.empty()) {
        fs::path p(opts.edges_path);
        std::string stem = p.stem().string();
        if (!stem.empty() && !is_generic_dataset_filename(lower_copy(stem)) && assign_key(stem, "edges filename")) {
            return ensure_non_generic_key(res);
        }
    }

    if (!opts.emit_edges_path.empty()) {
        fs::path p(opts.emit_edges_path);
        std::string stem = p.stem().string();
        if (!stem.empty() && !is_generic_dataset_filename(lower_copy(stem)) && assign_key(stem, "emit_edges filename")) {
            return ensure_non_generic_key(res);
        }
    }

    assign_key("dataset", "fallback");
    res.warn_generic = true;
    res.warn_detail = res.key;
    return ensure_non_generic_key(res);
}

static void copy_if_exists(const fs::path& src, const fs::path& dst) {
    if (src == dst) return;
    std::error_code ec;
    if (!fs::exists(src, ec)) return;
    ensure_parent_dir(dst.string());
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
}

static void rewrite_dataset_path(std::string& path,
                                 const fs::path& dataset_root,
                                 const std::string& default_name,
                                 const std::string& label,
                                 bool copy_existing,
                                 std::ostringstream& log) {
    fs::path dest = dataset_root / default_name;
    if (path.empty()) {
        ensure_parent_dir(dest.string());
        path = dest.string();
        log << "[info] " << label << " path set to " << path << " (paths_mode=smart)\n";
        return;
    }
    fs::path original(path);
    if (original == dest) return;
    if (copy_existing) {
        std::error_code exists_ec;
        if (!fs::exists(original, exists_ec)) {
            return; // keep original path so the downstream load emits the usual error
        }
        copy_if_exists(original, dest);
    } else {
        ensure_parent_dir(dest.string());
    }
    path = dest.string();
    log << "[info] " << label << " path rewritten to " << path << " (paths_mode=smart)\n";
}

static void apply_paths_strategy(ProgramOptions& opts, std::ostringstream& log) {
    if (opts.paths_mode == PathsMode::Legacy) {
        if (!opts.dataset_key.empty()) opts.dataset_key = sanitize_dataset_key(opts.dataset_key);
        opts.out_dir_effective = (fs::path(opts.out_dir) / opts.tag).string();
        return;
    }

    DatasetKeyInferenceResult key_info = infer_dataset_key_from_opts(opts);
    opts.dataset_key = key_info.key;
    if (key_info.warn_generic) {
        log << "[warn] generic filename detected; using parent folder name '" << key_info.warn_detail << "'\n";
    }
    log << "[info] dataset_key=" << key_info.key << " (inferred from " << key_info.rule << ")\n";

    fs::path dataset_root = fs::path("data") / "datasets" / key_info.key;
    fs::path experiments_root = fs::path("data") / "experiments" / key_info.key;
    std::error_code ec;
    fs::create_directories(dataset_root, ec);
    fs::create_directories(experiments_root, ec);
    opts.dataset_root = dataset_root.string();
    opts.experiments_root = experiments_root.string();

    if (opts.make_grid && !opts.emit_edges_path.empty()) {
        rewrite_dataset_path(opts.emit_edges_path, dataset_root, "edges.txt", "emit_edges", false, log);
    }

    if (!opts.edges_path.empty()) {
        rewrite_dataset_path(opts.edges_path, dataset_root, "edges.txt", "edges", true, log);
    }

    if (!opts.pack_candidates_path.empty()) {
        rewrite_dataset_path(opts.pack_candidates_path, dataset_root, "cands.txt", "pack_candidates", true, log);
    }

    if (!opts.pack_blocked_path.empty()) {
        rewrite_dataset_path(opts.pack_blocked_path, dataset_root, "blocked.txt", "pack_blocked", true, log);
    }

    if (!opts.graph_bin_out.empty()) {
        fs::path dest = dataset_root / (key_info.key + ".bin");
        if (opts.graph_bin_out != dest.string()) {
            ensure_parent_dir(dest.string());
            log << "[info] graph_bin_out rewritten to " << dest.string() << " (paths_mode=smart)\n";
            opts.graph_bin_out = dest.string();
        }
    }

    if (!opts.graph_bin_in.empty()) {
        fs::path candidate = dataset_root / (key_info.key + ".bin");
        if (fs::exists(candidate)) {
            if (opts.graph_bin_in != candidate.string()) {
                log << "[info] graph_bin_in rewritten to " << candidate.string() << " (paths_mode=smart)\n";
                opts.graph_bin_in = candidate.string();
            }
        }
    }

    fs::path out_dir_path(opts.out_dir);
    const fs::path dataset_root_path = dataset_root;
    bool rewrite_out_dir = false;
    if (out_dir_path == fs::path("data") || out_dir_path == dataset_root_path ||
        out_dir_path == fs::path("data") / key_info.key ||
        out_dir_path == fs::path("data") / "datasets" / key_info.key) {
        rewrite_out_dir = true;
    }
    if (rewrite_out_dir) {
        opts.out_dir = experiments_root.string();
    }

    opts.dataset_root = dataset_root.string();
    opts.experiments_root = experiments_root.string();
    opts.out_dir_effective = (fs::path(opts.out_dir) / opts.tag).string();

    if (rewrite_out_dir) {
        log << "[info] out_dir rewritten to " << opts.out_dir_effective << " (paths_mode=smart)\n";
    }

    log << "[info] dataset_root=" << dataset_root.string() << '\n';
    log << "[info] out_dir_effective=" << opts.out_dir_effective << '\n';
}
static std::string od_distribution_cli_name(ODDistribution dist) {
    switch (dist) {
        case ODDistribution::Uniform:             return "uniform";
        case ODDistribution::UniformWithDistBand: return "band";
        case ODDistribution::GaussianAroundStart: return "gaussian";
        case ODDistribution::HotspotMixture:      return "hotspot";
        case ODDistribution::Gravity:             return "gravity";
    }
    return "unknown";
}

static bool parse_od_dist_token(const std::string& raw,
                                ODDistribution& dist_out,
                                std::string& error) {
    const std::string token = lower_copy(trim_copy(raw));
    if (token.empty() || token == "uniform") {
        dist_out = ODDistribution::Uniform;
        return true;
    }
    if (token == "band" || token == "uniformwithdistband") {
        dist_out = ODDistribution::UniformWithDistBand;
        return true;
    }
    if (token == "gaussian" || token == "gaussianaroundstart") {
        dist_out = ODDistribution::GaussianAroundStart;
        return true;
    }
    if (token == "hotspot" || token == "hotspots" || token == "hotspotmixture") {
        dist_out = ODDistribution::HotspotMixture;
        return true;
    }
    if (token == "gravity") {
        dist_out = ODDistribution::Gravity;
        return true;
    }
    error = "unknown --od_dist value '" + raw + "' (expected uniform|band|gaussian|hotspot|gravity)";
    return false;
}

static bool parse_hotspots_inline(const std::string& spec,
                                  std::vector<std::pair<NodeId,double>>& out,
                                  std::string& error) {
    out.clear();
    std::string trimmed = trim_copy(spec);
    if (trimmed.empty()) {
        error = "--od_hotspots requires a non-empty list (e.g. \"0:2,7:1\")";
        return false;
    }
    std::stringstream ss(trimmed);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) continue;
        size_t colon = token.find(':');
        if (colon == std::string::npos) {
            error = "invalid hotspot entry '" + token + "' (expected node:weight)";
            out.clear();
            return false;
        }
        std::string node_str = trim_copy(token.substr(0, colon));
        std::string weight_str = trim_copy(token.substr(colon + 1));
        try {
            long long node_ll = std::stoll(node_str, nullptr, 10);
            if (node_ll < 0 || node_ll > std::numeric_limits<NodeId>::max()) {
                error = "hotspot node id out of range: " + node_str;
                out.clear();
                return false;
            }
            double weight = std::stod(weight_str);
            if (!(weight > 0.0)) {
                error = "hotspot weight must be positive, got " + weight_str;
                out.clear();
                return false;
            }
            out.emplace_back(static_cast<NodeId>(node_ll), weight);
        } catch (const std::exception&) {
            error = "invalid hotspot entry '" + token + "'";
            out.clear();
            return false;
        }
    }
    if (out.empty()) {
        error = "--od_hotspots produced no valid entries";
        return false;
    }
    return true;
}

static std::string fnv1a_64_hex(const std::string& data) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : data) {
        h ^= static_cast<uint64_t>(ch);
        h *= 1099511628211ull;
    }
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << h;
    return oss.str();
}

static bool load_hotspots_file(const std::string& path,
                               std::vector<std::pair<NodeId,double>>& out,
                               std::string& hash_out,
                               std::string& error) {
    out.clear();
    hash_out.clear();
    std::ifstream fin(path);
    if (!fin) {
        error = "cannot open --od_hotspots_file '" + path + "'";
        return false;
    }
    std::string raw;
    std::string line;
    int line_no = 0;
    while (std::getline(fin, line)) {
        ++line_no;
        std::string trimmed = trim_copy(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        raw.append(trimmed);
        raw.push_back('\n');
        std::istringstream iss(trimmed);
        long long node_ll;
        double weight;
        if (!(iss >> node_ll >> weight)) {
            error = "invalid hotspot entry on line " + std::to_string(line_no) + " of '" + path + "'";
            out.clear();
            return false;
        }
        if (node_ll < 0 || node_ll > std::numeric_limits<NodeId>::max()) {
            error = "hotspot node id out of range on line " + std::to_string(line_no);
            out.clear();
            return false;
        }
        if (!(weight > 0.0)) {
            error = "hotspot weight must be positive on line " + std::to_string(line_no);
            out.clear();
            return false;
        }
        out.emplace_back(static_cast<NodeId>(node_ll), weight);
    }
    if (out.empty()) {
        error = "--od_hotspots_file '" + path + "' produced no usable entries";
        return false;
    }
    hash_out = fnv1a_64_hex(raw);
    return true;
}

static std::string build_od_params_json(const AgentSamplerConfig& od) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);
    oss << '{';
    bool first = true;
    switch (od.dist) {
        case ODDistribution::Uniform:
            break;
        case ODDistribution::UniformWithDistBand:
            oss << "\"band\":[" << od.dist_band.first << ',' << od.dist_band.second << ']';
            first = false;
            break;
        case ODDistribution::GaussianAroundStart:
            oss << "\"sigma\":" << od.sigma;
            first = false;
            break;
        case ODDistribution::HotspotMixture:
            oss << "\"hotspots\":[";
            for (size_t i = 0; i < od.hotspots.size(); ++i) {
                if (i) oss << ',';
                oss << '[' << od.hotspots[i].first << ',' << od.hotspots[i].second << ']';
            }
            oss << ']';
            first = false;
            break;
        case ODDistribution::Gravity:
            oss << "\"beta\":" << od.decay_beta;
            first = false;
            break;
    }
    if (first) {
        oss << '}';
    } else {
        oss << '}';
    }
    return oss.str();
}

/// Parses a comma-separated key=value list provided via --margs.
static std::unordered_map<std::string, std::string>
parse_method_args(const std::string& raw,
                  std::vector<std::string>& warnings) {
    std::unordered_map<std::string, std::string> kv;
    if (raw.empty()) return kv;
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t next = raw.find(',', pos);
        std::string token = (next == std::string::npos) ? raw.substr(pos) : raw.substr(pos, next - pos);
        token = trim_copy(token);
        if (!token.empty()) {
            size_t eq = token.find('=');
            if (eq == std::string::npos) {
                warnings.push_back("margs entry without '=' ignored: " + token);
            } else {
                std::string key = lower_copy(trim_copy(token.substr(0, eq)));
                std::string value = trim_copy(token.substr(eq + 1));
                if (key.empty()) {
                    warnings.push_back("margs entry with empty key ignored: " + token);
                } else {
                    kv[key] = value;
                }
            }
        }
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return kv;
}

static bool get_marg_int(const std::unordered_map<std::string, std::string>& kv,
                         const std::string& key,
                         int& value_out,
                         std::vector<std::string>& warnings) {
    auto it = kv.find(key);
    if (it == kv.end()) return false;
    std::string err;
    std::string flag = "margs." + key;
    if (!parse_int_value(it->second, value_out, flag.c_str(), err)) {
        warnings.push_back(err);
        return false;
    }
    return true;
}

static const std::vector<MethodSpec>& method_registry() {
    static const std::vector<MethodSpec> registry = {
        {"sandwich", "Certified sandwich optimisation (default)",
         {"k", "seed", "cap_mode", "corridor_r", "cap_topk_k", "cap_cache", "enable_proxy_baseline"},
         run_method_sandwich},
        {"random_k", "Uniform random selection of k candidates",
         {"k", "seed"},
         run_method_random_k},
        {"regular_spacing_grid", "Regular spacing baseline on grid graphs",
         {"k", "rows", "cols"},
         run_method_regular_spacing},
        {"static_com_greedy", "Static coverage greedy baseline",
         {"k"},
         run_method_static_com_greedy},
        {"greedy_original_full", "Original objective greedy with recomputation",
         {"k", "seed"},
         run_method_greedy_original_full},
        {"bruteforce_ub", "Exact brute-force enumeration (small graphs)",
         {"k", "nmax", "seed"},
         run_method_bruteforce_ub}
    };
    return registry;
}

static const MethodSpec* find_method_spec(const std::string& name) {
    const auto& registry = method_registry();
    for (const auto& spec : registry) {
        if (spec.name == name) return &spec;
    }
    return nullptr;
}

static bool get_marg_double(const std::unordered_map<std::string, std::string>& kv,
                            const std::string& key,
                            double& value_out,
                            std::vector<std::string>& warnings) {
    auto it = kv.find(key);
    if (it == kv.end()) return false;
    std::string err;
    std::string flag = "margs." + key;
    if (!parse_double_value(it->second, value_out, flag.c_str(), err)) {
        warnings.push_back(err);
        return false;
    }
    return true;
}

static bool get_marg_bool(const std::unordered_map<std::string, std::string>& kv,
                          const std::string& key,
                          bool& value_out,
                          std::vector<std::string>& warnings) {
    auto it = kv.find(key);
    if (it == kv.end()) return false;
    if (!parse_zero_one(lower_copy(it->second), value_out)) {
        warnings.push_back("margs." + key + " expects 0/1 or true/false");
        return false;
    }
    return true;
}


static void apply_common_overrides(SandwichConfig& cfg,
                                   const std::unordered_map<std::string, std::string>& margs,
                                   std::vector<std::string>& warnings) {
    int int_value;
    double dbl_value;
    bool bool_value;
    if (get_marg_int(margs, "k", int_value, warnings) && int_value > 0) cfg.k = int_value;
    if (get_marg_int(margs, "m_agents", int_value, warnings) && int_value > 0) cfg.M_agents = int_value;
    if (get_marg_int(margs, "k_paths_per_agent", int_value, warnings) && int_value > 0) cfg.K_paths_per_agent = int_value;
    if (get_marg_int(margs, "m_eval", int_value, warnings) && int_value >= 0) cfg.M_eval = int_value;
    if (get_marg_int(margs, "corridor_r", int_value, warnings) && int_value > 0) cfg.corridor_r = int_value;
    if (get_marg_int(margs, "corridor_band_r", int_value, warnings)) {
        if (int_value < -1) {
            warnings.push_back("margs.corridor_band_r must be >= -1");
        } else {
            cfg.corridor_band_r = int_value;
        }
    }
    if (get_marg_int(margs, "proxy_r", int_value, warnings) && int_value > 0) cfg.proxy_corridor_r = int_value;
    if (get_marg_int(margs, "cap_topk_k", int_value, warnings) && int_value >= 0) cfg.cap_topk_k = int_value;
    if (get_marg_int(margs, "lb3_max_iters", int_value, warnings) && int_value > 0) cfg.lb3_max_iters = int_value;
    if (get_marg_double(margs, "lambda", dbl_value, warnings)) cfg.lambda = dbl_value;
    if (get_marg_double(margs, "lb3_eta", dbl_value, warnings)) cfg.lb3_eta = dbl_value;
    if (get_marg_double(margs, "lb3_tol", dbl_value, warnings)) cfg.lb3_tol = dbl_value;
    if (get_marg_double(margs, "lb3_softmax_eta", dbl_value, warnings)) cfg.lb3_softmax_eta = dbl_value;
    if (get_marg_bool(margs, "cap_cache", bool_value, warnings)) cfg.cap_cache = bool_value;
    if (get_marg_bool(margs, "enable_proxy_baseline", bool_value, warnings)) cfg.enable_proxy_baseline = bool_value;
    if (get_marg_bool(margs, "reuse_eval_samples", bool_value, warnings)) cfg.reuse_eval_samples = bool_value;
    if (get_marg_bool(margs, "use_lb3_softmax", bool_value, warnings)) cfg.use_LB3_softmax = bool_value;
    if (get_marg_bool(margs, "use_lb3_single", bool_value, warnings)) cfg.use_LB3_single = bool_value;
    if (get_marg_bool(margs, "use_lb3_iter", bool_value, warnings)) cfg.use_LB3_iter = bool_value;
    if (get_marg_bool(margs, "lb3_keep_best", bool_value, warnings)) cfg.lb3_keep_best = bool_value;
    if (get_marg_bool(margs, "lb3_start_from_ub", bool_value, warnings)) cfg.lb3_start_from_UB = bool_value;

    auto it_seed = margs.find("seed");
    if (it_seed != margs.end()) {
        std::string err;
        uint64_t seed_value = cfg.seed;
        if (parse_uint64_value(it_seed->second, seed_value, "margs.seed", err)) {
            cfg.seed = seed_value;
        } else {
            warnings.push_back(err);
        }
    }

    auto it_cap_mode = margs.find("cap_mode");
    if (it_cap_mode != margs.end()) {
        CapMode parsed;
        if (parse_cap_mode(trim_copy(it_cap_mode->second), parsed)) {
            cfg.cap_mode = parsed;
        } else {
            warnings.push_back("margs.cap_mode unknown value: " + it_cap_mode->second);
        }
    }

    auto it_ub_mode = margs.find("ub_mode");
    if (it_ub_mode != margs.end()) {
        UBMode parsed;
        if (parse_ub_mode(trim_copy(it_ub_mode->second), parsed)) {
            cfg.ub_mode = parsed;
        } else {
            warnings.push_back("margs.ub_mode unknown value: " + it_ub_mode->second);
        }
    }
}

static const std::unordered_set<std::string>& common_marg_keys() {
    static const std::unordered_set<std::string> keys = {
        "k", "seed", "cap_mode", "corridor_r", "corridor_band_r", "cap_topk_k", "cap_cache",
        "enable_proxy_baseline", "m_agents", "k_paths_per_agent", "m_eval",
        "reuse_eval_samples", "lambda", "lb3_eta", "lb3_tol", "lb3_softmax_eta",
        "lb3_max_iters", "use_lb3_softmax", "use_lb3_single", "use_lb3_iter",
        "lb3_keep_best", "lb3_start_from_ub", "proxy_r", "ub_mode"
    };
    return keys;
}

static MethodUnionData compute_union_data(const std::vector<std::vector<Path>>& paths,
                                          int n,
                                          const std::vector<std::vector<NodeId>>* corridor_nodes_per_agent) {
    MethodUnionData data;
    size_t num_agents = paths.size();
    data.nodes_per_agent.assign(num_agents, {});
    data.min_path_len.assign(num_agents, 0);
    data.max_path_len.assign(num_agents, 0);
    data.union_counts.assign(n, 0);
    data.max_paths_per_agent = 0;

    std::vector<int> last_seen(n, -1);
    bool use_corridor_nodes = corridor_nodes_per_agent && corridor_nodes_per_agent->size() == num_agents;
    for (size_t i = 0; i < num_agents; ++i) {
        const auto& candidate_paths = paths[i];
        data.max_paths_per_agent = std::max<int>(data.max_paths_per_agent,
                                                 static_cast<int>(candidate_paths.size()));
        int min_len = std::numeric_limits<int>::max();
        int max_len = 0;
        auto& nodes = data.nodes_per_agent[i];
        nodes.clear();
        for (const Path& P : candidate_paths) {
            int len = static_cast<int>(P.size()) > 0 ? std::max(0, static_cast<int>(P.size()) - 1) : 0;
            min_len = std::min(min_len, len);
            max_len = std::max(max_len, len);
            for (NodeId v : P) {
                if (v < 0 || v >= n) continue;
                if (last_seen[v] != static_cast<int>(i)) {
                    last_seen[v] = static_cast<int>(i);
                    nodes.push_back(v);
                }
            }
        }
        if (min_len == std::numeric_limits<int>::max()) min_len = 0;
        data.min_path_len[i] = min_len;
        data.max_path_len[i] = max_len;
        data.sum_Lmin += min_len;

        if (use_corridor_nodes) {
            const auto& corridor_nodes = (*corridor_nodes_per_agent)[i];
            for (NodeId v : corridor_nodes) {
                if (v < 0 || v >= n) continue;
                if (last_seen[v] != static_cast<int>(i)) {
                    last_seen[v] = static_cast<int>(i);
                    nodes.push_back(v);
                }
            }
        }

        for (NodeId v : nodes) {
            if (v < 0 || v >= n) continue;
            data.union_counts[v] += 1;
        }
    }

    long long union_total = 0;
    for (int c : data.union_counts) union_total += c;
    data.avg_agents_per_node = (n > 0) ? static_cast<double>(union_total) / static_cast<double>(n) : 0.0;
    return data;
}

/// Prints the CLI help text summarising all supported flags and examples.
static void print_help(std::ostream& os) {
    os << "treeopt — sandwich solver driver\n\n"
       << "Required/primary inputs:\n"
       << "  --edges <file>              Edge list with 'u v' per line\n"
       << "  --n <int>                   Optional node count (infers from edges if missing)\n"
       << "  --one_based <0|1>           Whether edge/candidate ids are 1-based (default 0)\n"
       << "  --undirected <0|1>          Treat graph as undirected when loading (default 1)\n"
       << "  --cands <file>              Optional candidate list (defaults to all nodes)\n"
       << "  --blocked <file>            Optional blocked-node list (excluded from selection)\n"
       << "  --graph_bin_in <file>       Load graph/candidates/blocked from a CSR binary bundle\n\n"
       << "Weights (choose one construction path unless loading cache):\n"
       << "  --hard_cutoff <int>         BFS radius for unit-weight neighbourhoods\n"
       << "  --soft_decay <profile>      Soft decay profile; forms supported:\n"
       << "      exp:<alpha>:<R>:<eps>   direct exponential parameters\n"
       << "      gauss:<sigma>           convenience shorthand (alpha=1/sigma^2, R~=3*sigma)\n"
       << "  --weights_bin <file>        Neighbourhood cache path (default <out>/<tag>/weights.bin)\n"
       << "  --load_weights <0|1>        Load weights from cache instead of rebuilding\n"
       << "  --save_weights <0|1>        Persist freshly built weights to cache\n\n"
       << "Synthetic grids:\n"
       << "  --make_grid rows=<R>,cols=<C>[,diag=0|1][,torus=0|1]  Generate an R×C lattice\n"
       << "  --emit_edges <file>          Write generated grid edges (use with --make_grid)\n"
       << "  --chain_run <0|1>           Continue optimisation after grid generation (default 0)\n\n"
       << "Objective & hyper-parameters:\n"
       << "  --psi clip:<C> | expsat:<b> | log1p | power:<alpha> (0 < alpha <= 1)\n"
       << "  --lambda <float>            Flow penalty multiplier (default 0)\n"
       << "  --k <int>                   Budget (default 50)\n"
       << "  --M_agents <int>            Training agent budget (default 10000)\n"
       << "  --K_paths_per_agent <int>   Paths sampled per agent (default 10)\n"
       << "  --M_eval <int>              MC samples for evaluation (0 => reuse training set)\n"
       << "  --reuse_eval_samples <0|1>  Reuse training samples when M_eval>0 (default 1)\n"
       << "  --ub_mode <mode>            CertifiedUnion | CappedUnion | ProxyCorridor | SumDeprecated\n"
       << "  --corridor_r <int>          Corridor r used by UB objectives (default proxy_r)\n"
       << "  --corridor_band_r <int>     Geometric corridor width r from the paper (default -1 off)\n"
       << "  --proxy_r <int>             Corridor r for proxy baseline (default 2)\n"
       << "  --cap_mode <mode>           Auto | PathlenOnly | TopkLocal | GlobalAll (default Auto)\n"
       << "  --cap_topk_k <int>          Explicit k for TopkLocal strategy (default 0 auto)\n"
       << "  --cap_cache <0|1>           Enable cap cache reuse (default 1)\n"
       << "  --enable_proxy_baseline <0|1>  Run proxy baseline (default 1)\n"
       << "  --use_LB3_softmax <0|1>     Enable single softmax linearisation\n"
       << "  --use_LB3_single <0|1>      Run one LB3 refinement pass\n"
       << "  --use_LB3_iter <0|1>        Run iterative LB3 refinements\n"
       << "  --lb3_max_iters <int>       Max LB3 iterations (default 1)\n"
       << "  --lb3_eta <float>           LB3 temperature (default 1.0)\n"
       << "  --lb3_tol <float>           LB3 convergence tolerance (default 1e-6)\n"
       << "  --lb3_keep_best <0|1>       Keep best LB3 solution encountered (default 1)\n"
       << "  --lb3_start_from_UB <0|1>   Seed LB3 iterations with UB solution (default 1)\n"
       << "  --lb3_softmax_eta <float>   Softmax eta when use_LB3_softmax\n\n"
       << "OD distributions:\n"
       << "  --od_dist uniform|band|gaussian|hotspot|gravity   Select OD sampling distribution (default uniform)\n"
       << "  --od_band <dmin>:<dmax>           Distance band for --od_dist band\n"
       << "  --od_sigma <float>                Gaussian sigma for --od_dist gaussian\n"
       << "  --od_hotspots \"node:wt,...\"       Hotspot weights for --od_dist hotspot\n"
       << "  --od_hotspots_file <file>         Hotspot weights from file (node weight per line)\n"
       << "  --od_gravity_beta <float>         Gravity beta for --od_dist gravity\n"
       << "  --od_rows <int> --od_cols <int>   Optional grid dimensions (inherit from --make_grid)\n\n"
       << "Evaluation & output:\n"
       << "  --out_dir <dir>             Root output directory (default 'out')\n"
       << "  --tag <name>                Sub-directory and report tag (default 'run')\n"
       << "  --dataset_key <name>        Dataset identifier used for smart path rewrites\n"
       << "  --paths_mode smart|legacy   Control automatic path rewrites (default smart)\n"
       << "  --log_level info|debug      Console verbosity (default info)\n"
       << "  --seed <uint64>             Global RNG seed (default 1)\n"
       << "  --dry_run <0|1>             Validate inputs, echo config, then exit (default 0)\n"
       << "  --json_meta <file>          Optional path to dump metadata JSON\n"
       << "  --graph_bin_out <file>      Pack graph/candidates/blocked into <file> (+ .json)\n"
       << "  --pack_candidates <file>    Candidate list to embed when writing graph bin\n"
       << "  --pack_blocked <file>       Blocked list to embed when writing graph bin\n"
        << "  --method <name>             Algorithm to run (default sandwich)\n"
        << "  --margs \"k=50,...\"        Comma-separated key=value overrides for the selected method\n"
        << "  --list-methods              Print available methods and exit\n"
        << "  --help                      Show this message and exit\n\n"
       << "Examples:\n"
       << "  # 1) Minimal smoke test (hard cutoff, small budgets)\n"
       << "  treeopt --edges E.txt --hard_cutoff 1 --k 2 --psi clip:1.0 \\\n"
       << "         --M_agents 50 --K_paths_per_agent 5 --M_eval 10 --reuse_eval_samples 1 \\\n"
       << "         --out_dir out --tag smoke_cli --seed 42\n\n"
       << "  # 2) ProxyCorridor guidance (explicit corridor_r required for certified upgrade)\n"
       << "  treeopt --edges E.txt --hard_cutoff 1 --k 4 --ub_mode ProxyCorridor --proxy_r 1\n"
       << "  treeopt --edges E.txt --hard_cutoff 1 --k 4 --ub_mode ProxyCorridor --proxy_r 1 --corridor_r 1\n\n"
        << "  # 3) SumDeprecated alias mapping with one-time warning\n"
        << "  treeopt --edges E.txt --hard_cutoff 2 --k 10 --ub_mode SumDeprecated\n\n"
        << "  # 4) Auto cap -> PathlenOnly (bounded psi)\n"
        << "  treeopt --edges E.txt --hard_cutoff 2 --k 20 --psi expsat:1.0 --cap_mode Auto\n\n"
       << "  # 5) Explicit TopkLocal cap (unbounded psi)\n"
       << "  treeopt --edges E.txt --hard_cutoff 2 --k 20 --psi log1p --cap_mode TopkLocal --cap_topk_k 20\n\n"
       << "  # 6) Soft-decay weights with cache save/load\n"
       << "  treeopt --edges E.txt --soft_decay gauss:1.5 --k 20 --psi clip:1.0 --weights_bin W.bin --save_weights 1\n"
        << "  treeopt --edges E.txt --soft_decay gauss:1.5 --k 20 --psi clip:1.0 --weights_bin W.bin --load_weights 1\n\n"
       << "  # 7) OD band-limited sampling\n"
       << "  treeopt --make_grid rows=32,cols=32 --chain_run 1 --hard_cutoff 2 --psi clip:1.0 --k 32 \\\n"
       << "         --od_dist band --od_band 4:12 --out_dir out --tag od_band\n\n"
       << "  # 8) OD gaussian sampling\n"
       << "  treeopt --make_grid rows=32,cols=32 --chain_run 1 --hard_cutoff 2 --psi clip:1.0 --k 32 \\\n"
       << "         --od_dist gaussian --od_sigma 1.5 --out_dir out --tag od_gauss\n\n"
       << "  # 9) OD hotspot sampling (inline weights)\n"
       << "  treeopt --make_grid rows=32,cols=32 --chain_run 1 --hard_cutoff 2 --psi clip:1.0 --k 32 \\\n"
       << "         --od_dist hotspot --od_hotspots \"0:3,7:1,15:2\" --out_dir out --tag od_hot_inline\n\n"
       << "  # 10) OD gravity sampling\n"
       << "  treeopt --make_grid rows=32,cols=32 --chain_run 1 --hard_cutoff 2 --psi clip:1.0 --k 32 \\\n"
       << "         --od_dist gravity --od_gravity_beta 0.3 --out_dir out --tag od_grav\n\n"
       << "  # 11) Pack and reload graph via CSR binary\n"
       << "  treeopt --edges E.txt --graph_bin_out E.bin --pack_candidates C.txt --pack_blocked B.txt\n"
       << "  treeopt --graph_bin_in E.bin --hard_cutoff 1 --k 2 --psi clip:1.0\n\n"
       << "  # 12) Synthetic grid with chain-run baseline\n"
       << "  treeopt --make_grid rows=16,cols=16 --chain_run 1 --method regular_spacing_grid --k 16\n"
       << "         --hard_cutoff 1 --psi clip:1.0 --out_dir out --tag grid_chain\n\n"
       << "  # 13) LB3 refinement (single vs iterative)\n"
       << "  treeopt --edges E.txt --hard_cutoff 2 --k 50 --psi clip:1.0 --use_LB3_single 1 --lb3_keep_best 1\n"
       << "  treeopt --edges E.txt --hard_cutoff 2 --k 50 --psi clip:1.0 --use_LB3_iter 1 --lb3_max_iters 5 --lb3_eta 1.0 --lb3_tol 1e-6 --lb3_keep_best 1\n\n"
       << "  # 14) Dry-run configuration audit (no optimisation)\n"
       << "  treeopt --edges E.txt --hard_cutoff 1 --k 5 --psi clip:1.0 --dry_run 1\n\n"
        << "  # List supported methods and tunable keys\n"
        << "  treeopt --list-methods\n\n"
       << "All flags are long-form for clarity; omit irrelevant groups.\n";
}

/// Parses the `--psi` specification into a Psi object.
/**
 * Accepted forms: clip:C, expsat:beta, log1p, power:alpha (0 < alpha <= 1).
 * @param token Raw CLI token.
 * @param psi_out Destination Psi configuration.
 * @param error Descriptive error message on failure.
 * @return True when parsing succeeds.
 */
static bool parse_psi_spec(const std::string& token, Psi& psi_out, std::string& error) {
    const std::string lower = lower_copy(token);
    if (lower.rfind("clip:", 0) == 0) {
        try {
            double C = std::stod(lower.substr(5));
            if (C <= 0.0) { error = "clip:<C> expects C > 0"; return false; }
            psi_out = Psi::Clip(C);
            return true;
        } catch (const std::exception&) {
            error = "clip:<C> expects a positive number";
            return false;
        }
    }
    if (lower.rfind("expsat:", 0) == 0) {
        try {
            double beta = std::stod(lower.substr(7));
            if (beta <= 0.0) { error = "expsat:<beta> expects beta > 0"; return false; }
            psi_out = Psi::ExpSat(beta);
            return true;
        } catch (const std::exception&) {
            error = "expsat:<beta> expects a positive number";
            return false;
        }
    }
    if (lower == "log1p") {
        psi_out = Psi::Log1p();
        return true;
    }
    if (lower.rfind("power:", 0) == 0) {
        try {
            double alpha = std::stod(lower.substr(6));
            if (alpha <= 0.0 || alpha > 1.0) {
                error = "power:<alpha> requires 0 < alpha <= 1";
                return false;
            }
            psi_out = Psi::Power(alpha);
            return true;
        } catch (const std::exception&) {
            error = "power:<alpha> expects a number";
            return false;
        }
    }
    error = "unknown psi specification";
    return false;
}

/// Parses the soft decay profile string into concrete parameters.
/// Parses the `--soft_decay` profile specification.
/**
 * Supported forms: exp:<alpha>:<R>:<eps> and gauss:<sigma> (maps σ to α,R,ε defaults).
 * @param token Raw CLI token.
 * @param params_out Output decay parameters.
 * @param profile_name Records original string for metadata logging.
 * @param error Populated on error.
 */
static bool parse_soft_decay_profile(const std::string& token,
                                     SoftDecayParams& params_out,
                                     std::string& profile_name,
                                     std::string& error) {
    const std::string lower = lower_copy(token);
    profile_name = token;
    if (lower.rfind("exp:", 0) == 0) {
        // Format: exp:<alpha>:<R>:<eps>
        std::vector<std::string> pieces;
        std::string current;
        for (char c : token.substr(4)) {
            if (c == ':') {
                pieces.push_back(current);
                current.clear();
            } else {
                current.push_back(c);
            }
        }
        if (!current.empty()) pieces.push_back(current);
        if (pieces.size() != 3) {
            error = "exp profile requires exp:<alpha>:<R>:<eps>";
            return false;
        }
        try {
            params_out.alpha = std::stod(pieces[0]);
            params_out.radius = std::max(1, static_cast<int>(std::stoll(pieces[1])));
            params_out.eps = std::stod(pieces[2]);
        } catch (const std::exception&) {
            error = "exp profile parameters must be numeric";
            return false;
        }
        if (params_out.alpha <= 0.0 || params_out.eps <= 0.0) {
            error = "exp profile needs alpha>0 and eps>0";
            return false;
        }
        return true;
    }
    if (lower.rfind("gauss:", 0) == 0) {
        double sigma;
        try {
            sigma = std::stod(lower.substr(6));
        } catch (const std::exception&) {
            error = "gauss:<sigma> expects a numeric sigma";
            return false;
        }
        if (sigma <= 0.0) {
            error = "gauss:<sigma> requires sigma>0";
            return false;
        }
        params_out.alpha = 1.0 / (sigma * sigma);
        params_out.radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
        params_out.eps = kDefaultSoftEps;
        return true;
    }
    error = "unsupported soft_decay profile";
    return false;
}

/// Parses the `--log_level` value.
/**
 * @param token Should be "info" or "debug" (case insensitive).
 * @param level_out Destination log level.
 * @return True when recognised.
 */
static bool parse_log_level(const std::string& token, LogLevel& level_out) {
    const std::string lower = lower_copy(token);
    if (lower == "info")  { level_out = LogLevel::Info;  return true; }
    if (lower == "debug") { level_out = LogLevel::Debug; return true; }
    return false;
}

/// Parses optional boolean toggles (0/1 or true/false).
/**
 * @param token Raw CLI token.
 * @param out Destination boolean.
 * @param error Error text when parsing fails.
 */
[[maybe_unused]] static bool parse_optional_bool(const std::string& token, bool& out, std::string& error) {
    if (!parse_zero_one(token, out)) {
        error = "expected 0 or 1";
        return false;
    }
    return true;
}

/// Infers node count by scanning the edge list (max id + 1).
/**
 * @param edge_path Path to edge list.
 * @param one_based Whether ids are 1-based (converted before max).
 * @param inferred Output node count.
 * @param error Error message on failure.
 */
static bool infer_node_count(const std::string& edge_path,
                             bool one_based,
                             int& inferred,
                             std::string& error) {
    std::ifstream fin(edge_path);
    if (!fin) {
        error = "cannot open edge list " + edge_path + "; verify the path and file permissions";
        return false;
    }
    long long max_id = -1;
    long long lines = 0;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (line[0] == kDefaultCommentChar) continue;
        std::istringstream iss(line);
        long long u, v;
        if (!(iss >> u >> v)) continue;
        if (one_based) { --u; --v; }
        max_id = std::max({max_id, u, v});
        ++lines;
    }
    if (max_id < 0) {
        error = "edge list contains no usable edges; ensure the file lists integer node pairs";
        return false;
    }
    inferred = static_cast<int>(max_id + 1);
    return true;
}

struct BlockedMetaFileData {
    bool found = false;
    bool has_source = false;
    std::string source;
    bool has_seed = false;
    long long seed = 0;
    bool has_fraction = false;
    double fraction = 0.0;
};

static BlockedMetaFileData load_blocked_meta_file(const std::string& blocked_path) {
    BlockedMetaFileData info;
    if (blocked_path.empty()) return info;
    fs::path meta_path = fs::path(blocked_path).parent_path() / "blocked_meta.json";
    std::ifstream fin(meta_path.string());
    if (!fin) return info;

    info.found = true;
    std::string line;
    while (std::getline(fin, line)) {
        const std::string trimmed = trim_copy(line);
        size_t colon = trimmed.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim_copy(trimmed.substr(0, colon));
        std::string value = trim_copy(trimmed.substr(colon + 1));
        if (!value.empty() && value.back() == ',') value.pop_back();

        if (key.find("\"blocked_source\"") != std::string::npos) {
            if (!value.empty() && value.front() == '"' && value.back() == '"') {
                info.source = value.substr(1, value.size() - 2);
                info.has_source = true;
            }
        } else if (key.find("\"blocked_seed\"") != std::string::npos) {
            char* end = nullptr;
            long long parsed = std::strtoll(value.c_str(), &end, 10);
            if (end && *end == '\0') {
                info.seed = parsed;
                info.has_seed = true;
            }
        } else if (key.find("\"blocked_fraction\"") != std::string::npos) {
            char* end = nullptr;
            double parsed = std::strtod(value.c_str(), &end);
            if (end && *end == '\0') {
                info.fraction = parsed;
                info.has_fraction = true;
            }
        }
    }
    return info;
}

/// Loads blocked nodes (one id per line) into a mask vector.
/**
 * @param path File containing node ids.
 * @param n Graph size (for bounds checking).
 * @param one_based Whether ids are 1-based.
 * @param mask_out Output mask (1 => blocked).
 * @param error Populated when the file is missing or invalid.
 */
static bool load_blocked_nodes(const std::string& path,
                               int n,
                               bool one_based,
                               std::vector<char>& mask_out,
                               std::string& error,
                               bool allow_empty = false,
                               bool* out_was_empty = nullptr) {
    mask_out.assign(n, 0);
    std::ifstream fin(path);
    if (!fin) {
        error = "cannot open blocked file " + path + "; verify the path or disable --blocked";
        return false;
    }
    long long ok = 0, oob = 0;
    std::string line;
    while (std::getline(fin, line)) {
        std::string trimmed = trim_copy(line);
        if (trimmed.empty()) continue;
        if (trimmed[0] == kDefaultCommentChar) continue;
        std::istringstream iss(trimmed);
        int x;
        if (!(iss >> x)) continue;
        if (one_based) --x;
        if (x < 0 || x >= n) { ++oob; continue; }
        mask_out[x] = 1;
        ++ok;
    }
    if (ok == 0) {
        if (allow_empty) {
            if (out_was_empty) *out_was_empty = true;
            std::cerr << "[warn] pack_blocked: file is empty → treating as |blocked|=0 (continue)\n";
        } else {
            error = "blocked file contains no valid ids; confirm the ids and --one_based setting";
            return false;
        }
    }
    if (oob > 0) {
        std::cerr << "[load_blocked] ignored " << oob << " out-of-bound ids\n";
    }
    return true;
}

/// Writes the accumulated log to stdout and mirrors it into stdout.txt.
/**
 * @param log Complete log contents.
 * @param path Destination file (ensures parent dir exists).
 * @return True on success.
 */
static bool flush_log(const std::string& log, const fs::path& path) {
    std::cout << log;
    if (!ensure_parent_dir(path.string())) return false;
    std::ofstream fout(path);
    if (!fout) return false;
    fout << log;
    return static_cast<bool>(fout);
}

/// Serialises the effective configuration into JSON (optional artefact).
static bool write_run_config_json(const ProgramOptions& opts,
                                  const fs::path& path) {
    if (path.empty()) return true;
    if (!ensure_parent_dir(path.string())) return false;
    std::ofstream fout(path);
    if (!fout) return false;
    fout << "{\n"
         << "  \"edges\": \"" << opts.edges_path << "\",\n"
         << "  \"n\": " << opts.n << ",\n"
         << "  \"one_based\": " << (opts.one_based ? "true" : "false") << ",\n"
         << "  \"undirected\": " << (opts.undirected ? "true" : "false") << ",\n"
         << "  \"candidates\": \"" << (opts.has_candidates ? opts.candidates_path : "<all>") << "\",\n"
         << "  \"blocked\": \"" << (opts.has_blocked ? opts.blocked_path : "<none>") << "\",\n"
         << "  \"weights_bin\": \"" << opts.weights_path << "\",\n"
         << "  \"weight_mode\": \"" << (opts.use_hard_cutoff ? "hard_cutoff" : (opts.use_soft_decay ? ("soft:" + opts.soft_named_profile) : "<cache>")) << "\",\n"
         << "  \"hard_cutoff\": " << opts.hard_cutoff << ",\n"
         << "  \"soft_alpha\": " << opts.soft.alpha << ",\n"
         << "  \"soft_radius\": " << opts.soft.radius << ",\n"
         << "  \"soft_eps\": " << opts.soft.eps << ",\n"
         << "  \"psi\": \"" << psi_type_name(opts.psi) << "\",\n"
         << "  \"lambda\": " << opts.lambda << ",\n"
         << "  \"k\": " << opts.k << ",\n"
         << "  \"corridor_r\": " << opts.corridor_r << ",\n"
         << "  \"corridor_band_r\": " << opts.corridor_band_r << ",\n"
         << "  \"proxy_r\": " << opts.proxy_r << ",\n"
         << "  \"ub_mode\": \"" << ub_mode_name(opts.ub_mode) << "\",\n"
         << "  \"cap_mode\": \"" << cap_mode_name(opts.cap_mode) << "\",\n"
         << "  \"cap_topk_k\": " << opts.cap_topk_k << ",\n"
         << "  \"cap_cache\": " << (opts.cap_cache ? "true" : "false") << ",\n"
         << "  \"M_agents\": " << opts.M_agents << ",\n"
         << "  \"K_paths_per_agent\": " << opts.K_paths_per_agent << ",\n"
         << "  \"M_eval\": " << opts.M_eval << ",\n"
         << "  \"od_dist\": \"" << od_distribution_cli_name(opts.od_dist) << "\",\n"
         << "  \"od_band_min\": " << opts.od_band_min << ",\n"
         << "  \"od_band_max\": " << opts.od_band_max << ",\n"
         << "  \"od_sigma\": " << opts.od_sigma << ",\n"
         << "  \"od_gravity_beta\": " << opts.od_gravity_beta << ",\n"
         << "  \"od_rows\": " << opts.od_rows << ",\n"
         << "  \"od_cols\": " << opts.od_cols << ",\n"
         << "  \"od_rows_cols_inherited\": " << (opts.od_rows_cols_inherited ? "true" : "false") << ",\n"
         << "  \"od_hotspot_count\": " << opts.od_hotspots.size() << ",\n"
         << "  \"od_hotspots_file\": \"" << (opts.od_hotspots_from_file ? opts.od_hotspots_file : "") << "\",\n"
         << "  \"od_hotspots_hash\": \"" << (opts.od_hotspots_from_file ? opts.od_hotspots_hash : "") << "\",\n"
         << "  \"seed\": " << opts.seed << "\n"
         << "}\n";
    return static_cast<bool>(fout);
}

/// Translates ProgramOptions into the SandwichConfig consumed by run_sandwich.
static SandwichConfig to_sandwich_config(const ProgramOptions& opts) {
    SandwichConfig cfg;
    cfg.k = opts.k;
    cfg.M_agents = opts.M_agents;
    cfg.K_paths_per_agent = opts.K_paths_per_agent;
    cfg.corridor_r = opts.corridor_r > 0 ? opts.corridor_r : opts.proxy_r; // Sentinel 0 => inherit proxy_r per spec
    cfg.corridor_band_r = opts.corridor_band_r;
    cfg.lambda = opts.lambda;
    cfg.psi = opts.psi;
    cfg.seed = opts.seed;
    cfg.ub_mode = opts.ub_mode;
    cfg.cap_mode = opts.cap_mode;
    cfg.cap_topk_k = opts.cap_topk_k;
    cfg.cap_cache = opts.cap_cache;
    cfg.enable_proxy_baseline = opts.enable_proxy_baseline;
    cfg.proxy_corridor_r = opts.proxy_r;
    cfg.M_eval = opts.M_eval;
    cfg.reuse_eval_samples = opts.reuse_eval_samples;
    cfg.use_LB3_softmax = opts.use_LB3_softmax;
    cfg.lb3_softmax_eta = opts.lb3_softmax_eta;
    cfg.use_LB3_single = opts.use_LB3_single;
    cfg.use_LB3_iter = opts.use_LB3_iter;
    cfg.lb3_max_iters = std::max(1, opts.lb3_max_iters);
    cfg.lb3_eta = opts.lb3_eta;
    cfg.lb3_tol = opts.lb3_tol;
    cfg.lb3_keep_best = opts.lb3_keep_best;
    cfg.lb3_start_from_UB = opts.lb3_start_from_UB;

    cfg.od.dist = opts.od_dist;
    cfg.od.dist_band = {opts.od_band_min, opts.od_band_max};
    cfg.od.sigma = opts.od_sigma;
    cfg.od.decay_beta = opts.od_gravity_beta;
    cfg.od.hotspots = opts.od_hotspots;
    cfg.od.rows = opts.od_rows;
    cfg.od.cols = opts.od_cols;
    return cfg;
}

/// Appends a human-readable configuration summary to the log buffer.
static void log_config_echo(const ProgramOptions& opts, std::ostringstream& log) {
    log << "Configuration summary:\n"
        << "  edges           : " << (opts.edges_display.empty() ? opts.edges_path : opts.edges_display) << '\n'
        << "  n               : " << opts.n << " (provided: " << (opts.n_provided ? "yes" : "no") << ")\n"
        << "  candidates      : " << (opts.candidates_display.empty() ? (opts.has_candidates ? opts.candidates_path : "<all nodes>") : opts.candidates_display) << '\n'
        << "  blocked         : " << (opts.blocked_display.empty() ? (opts.has_blocked ? opts.blocked_path : "<none>") : opts.blocked_display) << '\n'
        << "  weight mode     : "
        << (opts.use_hard_cutoff ? ("hard_cutoff D=" + std::to_string(opts.hard_cutoff))
            : (opts.use_soft_decay ? ("soft_decay " + opts.soft_named_profile)
               : (opts.load_weights ? "load cache" : "<unspecified>"))) << '\n'
        << "  weights cache   : " << opts.weights_path << " (load=" << bool_to_string(opts.load_weights)
        << ", save=" << bool_to_string(opts.save_weights) << ")\n"
        << "  psi             : " << psi_type_name(opts.psi) << '\n'
        << "  lambda          : " << opts.lambda << '\n'
        << "  k               : " << opts.k << '\n'
        << "  corridor_r      : " << opts.corridor_r << '\n'
        << "  corridor_band_r : " << opts.corridor_band_r << '\n'
        << "  proxy_r         : " << opts.proxy_r << '\n'
        << "  ub_mode         : " << ub_mode_name(opts.ub_mode) << '\n'
        << "  cap_mode        : " << cap_mode_name(opts.cap_mode)
        << " (effective=" << cap_mode_name(opts.cap_mode_effective)
        << ", topk_runtime=" << opts.cap_topk_effective
        << ", cache=" << bool_to_string(opts.cap_cache) << ")\n"
        << "  M_agents        : " << opts.M_agents << ", K_paths=" << opts.K_paths_per_agent << '\n'
        << "  M_eval          : " << opts.M_eval << ", reuse_eval=" << bool_to_string(opts.reuse_eval_samples) << '\n'
        << "  LB3 settings    : softmax=" << bool_to_string(opts.use_LB3_softmax)
        << " eta=" << opts.lb3_softmax_eta
        << ", single=" << bool_to_string(opts.use_LB3_single)
        << ", iter=" << bool_to_string(opts.use_LB3_iter)
        << ", max_iters=" << opts.lb3_max_iters
        << ", tol=" << opts.lb3_tol
        << ", keep_best=" << bool_to_string(opts.lb3_keep_best)
        << ", start_from_UB=" << bool_to_string(opts.lb3_start_from_UB) << '\n'
        << "  output dir      : " << opts.out_dir << '/' << opts.tag << '\n'
        << "  seed            : " << opts.seed << '\n';

    AgentSamplerConfig od_preview;
    od_preview.dist = opts.od_dist;
    od_preview.dist_band = {opts.od_band_min, opts.od_band_max};
    od_preview.sigma = opts.od_sigma;
    od_preview.decay_beta = opts.od_gravity_beta;
    od_preview.hotspots = opts.od_hotspots;
    od_preview.rows = opts.od_rows;
    od_preview.cols = opts.od_cols;
    std::string od_params = build_od_params_json(od_preview);

    log << "  od_dist        : " << od_distribution_cli_name(opts.od_dist)
        << " params=" << od_params << '\n';
    log << "  od_rows/cols   : " << opts.od_rows << '/' << opts.od_cols;
    if (opts.od_rows_cols_inherited) {
        log << " [inherited]";
    }
    log << '\n';
    if (opts.od_hotspots_from_file) {
        log << "  od_hotspots    : file=" << opts.od_hotspots_file
            << " (count=" << opts.od_hotspots.size() << ")\n";
    } else if (opts.od_hotspots_inline) {
        log << "  od_hotspots    : inline (count=" << opts.od_hotspots.size() << ")\n";
    } else {
        log << "  od_hotspots    : <none>\n";
    }

    if (!opts.graph_bin_in.empty()) {
        log << "  graph_bin_in    : " << opts.graph_bin_in << '\n';
    }
    if (!opts.graph_bin_out.empty()) {
        log << "  graph_bin_out   : " << opts.graph_bin_out << '\n';
    }
    if (opts.make_grid) {
        log << "  grid_spec       : rows=" << opts.grid_spec.rows
            << ", cols=" << opts.grid_spec.cols
            << ", diag=" << (opts.grid_spec.diag ? 1 : 0)
            << ", torus=" << (opts.grid_spec.torus ? 1 : 0) << '\n';
    }
}

/// Parses all CLI arguments into ProgramOptions.
/// Parses CLI arguments into ProgramOptions.
/**
 * Prints help and returns false without error when --help or no args given.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param opts Output options struct.
 * @param error Detailed error string on failure.
 * @return True on success.
 */
static bool parse_arguments(int argc, char** argv, ProgramOptions& opts, std::string& error) {
    {
        std::ostringstream cmd;
        for (int i = 0; i < argc; ++i) {
            if (i) cmd << ' ';
            cmd << argv[i];
        }
        opts.raw_cmdline = cmd.str();
    }
    if (argc <= 1) {
        print_help(std::cout);
        return false;
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--method=", 0) == 0) {
            opts.method_name = lower_copy(arg.substr(9));
            continue;
        } else if (arg.rfind("--margs=", 0) == 0) {
            opts.method_args_raw = arg.substr(8);
            continue;
        } else if (arg == "--help") {
            print_help(std::cout);
            return false;
        } else if (arg == "--list-methods") {
            opts.list_methods = true;
            continue;
        }
        auto require_value = [&](const char* flag_name) -> std::string {
            if (i + 1 >= argc) {
                error = std::string(flag_name) + " requires an argument";
                return {};
            }
            ++i;
            return argv[i];
        };

        if (arg == "--edges") {
            opts.edges_path = require_value("--edges");
            if (opts.edges_path.empty()) return false;
        } else if (arg == "--n") {
            std::string v = require_value("--n");
            if (!parse_int_value(v, opts.n, "--n", error)) return false;
            opts.n_provided = true;
        } else if (arg == "--one_based") {
            std::string v = require_value("--one_based");
            bool b;
            if (!parse_zero_one(v, b)) { error = "--one_based expects 0 or 1"; return false; }
            opts.one_based = b;
        } else if (arg == "--undirected") {
            std::string v = require_value("--undirected");
            bool b;
            if (!parse_zero_one(v, b)) { error = "--undirected expects 0 or 1"; return false; }
            opts.undirected = b;
        } else if (arg == "--cands") {
            opts.candidates_path = require_value("--cands");
            if (opts.candidates_path.empty()) return false;
            opts.has_candidates = true;
        } else if (arg == "--blocked") {
            opts.blocked_path = require_value("--blocked");
            if (opts.blocked_path.empty()) return false;
            opts.has_blocked = true;
        } else if (arg == "--graph_bin_in") {
            opts.graph_bin_in = require_value("--graph_bin_in");
        } else if (arg == "--graph_bin_out") {
            opts.graph_bin_out = require_value("--graph_bin_out");
        } else if (arg == "--pack_candidates") {
            opts.pack_candidates_path = require_value("--pack_candidates");
        } else if (arg == "--pack_blocked") {
            opts.pack_blocked_path = require_value("--pack_blocked");
        } else if (arg == "--emit_edges") {
            opts.emit_edges_path = require_value("--emit_edges");
        } else if (arg == "--chain_run") {
            std::string v = require_value("--chain_run");
            if (!parse_zero_one(v, opts.chain_run)) {
                error = "--chain_run expects 0 or 1";
                return false;
            }
        } else if (arg == "--make_grid") {
            std::string spec = require_value("--make_grid");
            std::string grid_err;
            if (!parse_grid_spec(spec, opts.grid_spec, grid_err)) {
                error = "--make_grid " + grid_err;
                return false;
            }
            opts.make_grid = true;
            opts.grid_spec_provided = true;
        } else if (arg == "--hard_cutoff") {
            std::string v = require_value("--hard_cutoff");
            if (!parse_int_value(v, opts.hard_cutoff, "--hard_cutoff", error)) return false;
            opts.use_hard_cutoff = true;
        } else if (arg == "--soft_decay") {
            std::string v = require_value("--soft_decay");
            std::string profile, err;
            if (!parse_soft_decay_profile(v, opts.soft, profile, err)) {
                error = "--soft_decay " + err;
                return false;
            }
            opts.use_soft_decay = true;
            opts.soft_named_profile = profile;
        } else if (arg == "--weights_bin") {
            opts.weights_path = require_value("--weights_bin");
        } else if (arg == "--load_weights") {
            std::string v = require_value("--load_weights");
            bool b;
            if (!parse_zero_one(v, b)) { error = "--load_weights expects 0 or 1"; return false; }
            opts.load_weights = b;
        } else if (arg == "--save_weights") {
            std::string v = require_value("--save_weights");
            bool b;
            if (!parse_zero_one(v, b)) { error = "--save_weights expects 0 or 1"; return false; }
            opts.save_weights = b;
        } else if (arg == "--psi") {
            std::string v = require_value("--psi");
            std::string err;
            if (!parse_psi_spec(v, opts.psi, err)) {
                error = "--psi " + err;
                return false;
            }
        } else if (arg == "--lambda") {
            std::string v = require_value("--lambda");
            if (!parse_double_value(v, opts.lambda, "--lambda", error)) return false;
        } else if (arg == "--k") {
            std::string v = require_value("--k");
            if (!parse_int_value(v, opts.k, "--k", error)) return false;
            opts.k = std::max(1, opts.k);
        } else if (arg == "--M_agents") {
            std::string v = require_value("--M_agents");
            if (!parse_int_value(v, opts.M_agents, "--M_agents", error)) return false;
            opts.M_agents = std::max(1, opts.M_agents);
        } else if (arg == "--K_paths_per_agent") {
            std::string v = require_value("--K_paths_per_agent");
            if (!parse_int_value(v, opts.K_paths_per_agent, "--K_paths_per_agent", error)) return false;
            opts.K_paths_per_agent = std::max(1, opts.K_paths_per_agent);
        } else if (arg == "--M_eval") {
            std::string v = require_value("--M_eval");
            if (!parse_int_value(v, opts.M_eval, "--M_eval", error)) return false;
            opts.M_eval = std::max(0, opts.M_eval);
        } else if (arg == "--od_dist") {
            opts.od_dist_token = lower_copy(require_value("--od_dist"));
        } else if (arg == "--od_band") {
            std::string v = require_value("--od_band");
            size_t colon = v.find(':');
            if (colon == std::string::npos) {
                error = "--od_band expects dmin:dmax";
                return false;
            }
            std::string dmin_str = trim_copy(v.substr(0, colon));
            std::string dmax_str = trim_copy(v.substr(colon + 1));
            std::string err;
            if (!parse_int_value(dmin_str, opts.od_band_min, "--od_band", err)) {
                error = err;
                return false;
            }
            if (!parse_int_value(dmax_str, opts.od_band_max, "--od_band", err)) {
                error = err;
                return false;
            }
            if (opts.od_band_min < 0 || opts.od_band_max < 0 || opts.od_band_min > opts.od_band_max) {
                error = "--od_band requires integers with 0 ≤ dmin ≤ dmax";
                return false;
            }
            opts.od_band_specified = true;
        } else if (arg == "--od_sigma") {
            std::string v = require_value("--od_sigma");
            std::string err;
            if (!parse_double_value(v, opts.od_sigma, "--od_sigma", err)) {
                error = err;
                return false;
            }
            opts.od_sigma_specified = true;
        } else if (arg == "--od_gravity_beta") {
            std::string v = require_value("--od_gravity_beta");
            std::string err;
            if (!parse_double_value(v, opts.od_gravity_beta, "--od_gravity_beta", err)) {
                error = err;
                return false;
            }
            opts.od_gravity_specified = true;
        } else if (arg == "--od_hotspots") {
            if (opts.od_hotspots_from_file) {
                error = "--od_hotspots conflicts with --od_hotspots_file";
                return false;
            }
            if (opts.od_hotspots_inline) {
                error = "--od_hotspots specified multiple times";
                return false;
            }
            std::string spec = require_value("--od_hotspots");
            std::vector<std::pair<NodeId,double>> parsed;
            std::string err;
            if (!parse_hotspots_inline(spec, parsed, err)) {
                error = err;
                return false;
            }
            opts.od_hotspots = std::move(parsed);
            opts.od_hotspots_inline = true;
        } else if (arg == "--od_hotspots_file") {
            if (opts.od_hotspots_inline) {
                error = "--od_hotspots_file conflicts with --od_hotspots";
                return false;
            }
            if (opts.od_hotspots_from_file) {
                error = "--od_hotspots_file specified multiple times";
                return false;
            }
            opts.od_hotspots_file = require_value("--od_hotspots_file");
            std::vector<std::pair<NodeId,double>> parsed;
            std::string hash;
            std::string err;
            if (!load_hotspots_file(opts.od_hotspots_file, parsed, hash, err)) {
                error = err;
                return false;
            }
            opts.od_hotspots = std::move(parsed);
            opts.od_hotspots_hash = hash;
            opts.od_hotspots_from_file = true;
        } else if (arg == "--od_rows") {
            std::string v = require_value("--od_rows");
            if (!parse_int_value(v, opts.od_rows, "--od_rows", error)) return false;
            if (opts.od_rows <= 0) {
                error = "--od_rows must be > 0";
                return false;
            }
            opts.od_rows_provided = true;
        } else if (arg == "--od_cols") {
            std::string v = require_value("--od_cols");
            if (!parse_int_value(v, opts.od_cols, "--od_cols", error)) return false;
            if (opts.od_cols <= 0) {
                error = "--od_cols must be > 0";
                return false;
            }
            opts.od_cols_provided = true;
        } else if (arg == "--reuse_eval_samples") {
            std::string v = require_value("--reuse_eval_samples");
            if (!parse_zero_one(v, opts.reuse_eval_samples)) {
                error = "--reuse_eval_samples expects 0 or 1";
                return false;
            }
        } else if (arg == "--ub_mode") {
            std::string v = require_value("--ub_mode");
            if (!parse_ub_mode(v, opts.ub_mode)) {
                error = "unknown ub_mode: " + v;
                return false;
            }
        } else if (arg == "--corridor_r") {
            std::string v = require_value("--corridor_r");
            if (!parse_int_value(v, opts.corridor_r, "--corridor_r", error)) return false;
            opts.corridor_r = std::max(1, opts.corridor_r);
        } else if (arg == "--corridor_band_r") {
            std::string v = require_value("--corridor_band_r");
            if (!parse_int_value(v, opts.corridor_band_r, "--corridor_band_r", error)) return false;
            if (opts.corridor_band_r < -1) {
                error = "--corridor_band_r must be >= -1";
                return false;
            }
        } else if (arg == "--proxy_r") {
            std::string v = require_value("--proxy_r");
            if (!parse_int_value(v, opts.proxy_r, "--proxy_r", error)) return false;
            opts.proxy_r = std::max(1, opts.proxy_r);
        } else if (arg == "--cap_mode") {
            std::string v = require_value("--cap_mode");
            if (!parse_cap_mode(v, opts.cap_mode)) {
                error = "unknown cap_mode: " + v;
                return false;
            }
        } else if (arg == "--cap_topk_k") {
            std::string v = require_value("--cap_topk_k");
            if (!parse_int_value(v, opts.cap_topk_k, "--cap_topk_k", error)) return false;
            opts.cap_topk_k = std::max(0, opts.cap_topk_k);
        } else if (arg == "--cap_cache") {
            std::string v = require_value("--cap_cache");
            if (!parse_zero_one(v, opts.cap_cache)) {
                error = "--cap_cache expects 0 or 1";
                return false;
            }
        } else if (arg == "--enable_proxy_baseline") {
            std::string v = require_value("--enable_proxy_baseline");
            if (!parse_zero_one(v, opts.enable_proxy_baseline)) {
                error = "--enable_proxy_baseline expects 0 or 1";
                return false;
            }
        } else if (arg == "--use_LB3_softmax") {
            std::string v = require_value("--use_LB3_softmax");
            if (!parse_zero_one(v, opts.use_LB3_softmax)) {
                error = "--use_LB3_softmax expects 0 or 1";
                return false;
            }
        } else if (arg == "--use_LB3_single") {
            std::string v = require_value("--use_LB3_single");
            if (!parse_zero_one(v, opts.use_LB3_single)) {
                error = "--use_LB3_single expects 0 or 1";
                return false;
            }
        } else if (arg == "--use_LB3_iter") {
            std::string v = require_value("--use_LB3_iter");
            if (!parse_zero_one(v, opts.use_LB3_iter)) {
                error = "--use_LB3_iter expects 0 or 1";
                return false;
            }
        } else if (arg == "--lb3_max_iters") {
            std::string v = require_value("--lb3_max_iters");
            if (!parse_int_value(v, opts.lb3_max_iters, "--lb3_max_iters", error)) return false;
            opts.lb3_max_iters = std::max(1, opts.lb3_max_iters);
        } else if (arg == "--lb3_eta") {
            std::string v = require_value("--lb3_eta");
            if (!parse_double_value(v, opts.lb3_eta, "--lb3_eta", error)) return false;
        } else if (arg == "--lb3_tol") {
            std::string v = require_value("--lb3_tol");
            if (!parse_double_value(v, opts.lb3_tol, "--lb3_tol", error)) return false;
        } else if (arg == "--lb3_keep_best") {
            std::string v = require_value("--lb3_keep_best");
            if (!parse_zero_one(v, opts.lb3_keep_best)) {
                error = "--lb3_keep_best expects 0 or 1";
                return false;
            }
        } else if (arg == "--lb3_start_from_UB") {
            std::string v = require_value("--lb3_start_from_UB");
            if (!parse_zero_one(v, opts.lb3_start_from_UB)) {
                error = "--lb3_start_from_UB expects 0 or 1";
                return false;
            }
        } else if (arg == "--lb3_softmax_eta") {
            std::string v = require_value("--lb3_softmax_eta");
            if (!parse_double_value(v, opts.lb3_softmax_eta, "--lb3_softmax_eta", error)) return false;
        } else if (arg == "--out_dir") {
            opts.out_dir = require_value("--out_dir");
        } else if (arg == "--tag") {
            opts.tag = require_value("--tag");
        } else if (arg == "--dataset_key") {
            opts.dataset_key = require_value("--dataset_key");
            if (opts.dataset_key.empty()) return false;
        } else if (arg == "--paths_mode") {
            std::string v = require_value("--paths_mode");
            if (!parse_paths_mode(v, opts.paths_mode)) {
                error = "unknown paths_mode: " + v;
                return false;
            }
        } else if (arg == "--log_level") {
            std::string v = require_value("--log_level");
            if (!parse_log_level(v, opts.log_level)) {
                error = "--log_level expects info or debug";
                return false;
            }
        } else if (arg == "--seed") {
            std::string v = require_value("--seed");
            if (!parse_uint64_value(v, opts.seed, "--seed", error)) return false;
        } else if (arg == "--dry_run") {
            std::string v = require_value("--dry_run");
            if (!parse_zero_one(v, opts.dry_run)) {
                error = "--dry_run expects 0 or 1";
                return false;
            }
        } else if (arg == "--json_meta") {
            opts.json_meta_path = require_value("--json_meta");
            opts.write_json_meta = !opts.json_meta_path.empty();
        } else if (arg == "--method") {
            opts.method_name = lower_copy(require_value("--method"));
        } else if (arg == "--margs") {
            opts.method_args_raw = require_value("--margs");
        } else {
            error = "unknown flag: " + arg;
            return false;
        }
    }

    if (!parse_od_dist_token(opts.od_dist_token, opts.od_dist, error)) {
        return false;
    }

    switch (opts.od_dist) {
        case ODDistribution::Uniform:
            break;
        case ODDistribution::UniformWithDistBand:
            if (!opts.od_band_specified) {
                error = "--od_dist band requires --od_band dmin:dmax";
                return false;
            }
            break;
        case ODDistribution::GaussianAroundStart:
            if (!opts.od_sigma_specified) {
                error = "--od_dist gaussian requires --od_sigma <sigma>";
                return false;
            }
            if (!(opts.od_sigma > 0.0)) {
                error = "--od_sigma must be > 0";
                return false;
            }
            break;
        case ODDistribution::HotspotMixture:
            if (!(opts.od_hotspots_inline || opts.od_hotspots_from_file)) {
                error = "--od_dist hotspot requires --od_hotspots or --od_hotspots_file";
                return false;
            }
            break;
        case ODDistribution::Gravity:
            if (!opts.od_gravity_specified) {
                error = "--od_dist gravity requires --od_gravity_beta <beta>";
                return false;
            }
            if (opts.od_gravity_beta < 0.0) {
                error = "--od_gravity_beta must be ≥ 0";
                return false;
            }
            break;
    }

    if (opts.od_rows_provided != opts.od_cols_provided) {
        error = "provide both --od_rows and --od_cols (or neither)";
        return false;
    }

    if (!opts.make_grid && !opts.emit_edges_path.empty()) {
        error = "--emit_edges requires --make_grid";
        return false;
    }
    if (!opts.make_grid && opts.chain_run) {
        error = "--chain_run is only valid together with --make_grid";
        return false;
    }
    if (opts.make_grid && !opts.grid_spec_provided) {
        error = "--make_grid rows=...,cols=... specification is required";
        return false;
    }

    const bool has_edges_source = !opts.edges_path.empty();
    const bool has_graph_bin = !opts.graph_bin_in.empty();
    const bool has_grid_source = opts.make_grid;
    if (!has_edges_source && !has_graph_bin && !has_grid_source) {
        error = "provide --edges, --graph_bin_in, or --make_grid";
        return false;
    }

    const bool grid_emit_only = opts.make_grid && !opts.chain_run;
    if (grid_emit_only && opts.emit_edges_path.empty()) {
        error = "--emit_edges <file> is required when --make_grid is used without --chain_run";
        return false;
    }

    if (!grid_emit_only) {
        if (opts.use_hard_cutoff && opts.use_soft_decay) {
            error = "choose either --hard_cutoff or --soft_decay, not both";
            return false;
        }
        if (!opts.use_hard_cutoff && !opts.use_soft_decay && !opts.load_weights) {
            error = "specify --hard_cutoff or --soft_decay or enable --load_weights";
            return false;
        }
        if (opts.use_hard_cutoff && opts.hard_cutoff <= 0) {
            error = "--hard_cutoff must be > 0";
            return false;
        }
        if (opts.use_soft_decay && opts.soft.radius <= 0) {
            error = "invalid soft_decay radius";
            return false;
        }
    }

    return true;
}

static void print_method_list() {
    std::cout << "Available methods:\n";
    for (const auto& spec : method_registry()) {
        std::cout << "  " << spec.name << " : " << spec.description;
        if (!spec.key_hints.empty()) {
            std::cout << " (keys: ";
            for (size_t i = 0; i < spec.key_hints.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << spec.key_hints[i];
            }
            std::cout << ')';
        }
        std::cout << '\n';
    }
}

static bool run_method_sandwich(const MethodContext& ctx,
                                const std::unordered_map<std::string, std::string>& margs,
                                MethodRunResult& out) {
    SandwichConfig cfg = ctx.base_cfg;
    apply_common_overrides(cfg, margs, out.warnings);

    std::unordered_set<std::string> known_keys = common_marg_keys();

    Timer method_timer{"run_sandwich"};
    SandwichResult result;
    bool ok = run_sandwich(ctx.G, ctx.candidates, ctx.blocked, ctx.weights, cfg, result);
    double run_time = method_timer.stop().count();
    if (!ok) {
        out.success = false;
        return false;
    }

    out.result = result;
    out.applied_cfg = cfg;
    fill_report_stats(result.stats_pick, out.stats_pick);
    fill_report_stats(result.stats_UB, out.stats_UB);
    fill_report_stats(result.stats_LB, out.stats_LB);
    out.extra_meta.emplace_back("time_run_sandwich_sec", to_string_compact(run_time));

    for (const auto& kv : margs) {
        if (!known_keys.count(kv.first)) {
            out.warnings.push_back("margs key '" + kv.first + "' is not recognised for method sandwich");
        }
    }

    return true;
}

static int ensure_budget_within_candidates(const MethodContext& ctx,
                                           SandwichConfig& cfg,
                                           MethodRunResult& out) {
    int max_k = static_cast<int>(ctx.candidates.size());
    if (cfg.k > max_k) {
        out.warnings.push_back("budget k=" + std::to_string(cfg.k) +
                               " exceeds candidate count; clamped to " + std::to_string(max_k));
        cfg.k = max_k;
    }
    if (cfg.k < 0) cfg.k = 0;
    return cfg.k;
}

static void populate_selection_metrics(const MethodContext& ctx,
                                       const SandwichConfig& cfg,
                                       const MethodUnionData& union_data,
                                       const std::vector<NodeId>& selection,
                                       const std::vector<std::pair<NodeId, NodeId>>& agents,
                                       const std::vector<std::vector<Path>>& paths,
                                       const std::vector<std::vector<NodeId>>* corridor_nodes_per_agent,
                                       int agents_dropped_corridor,
                                       int corridor_resample_attempts,
                                       MethodRunResult& out) {
    (void)corridor_nodes_per_agent;
    SandwichResult& result = out.result;
    result.S_pick = selection;
    result.S_UB = selection;
    result.S_LB = selection;
    result.corridor_band_r_used = cfg.corridor_band_r;
    result.agents_dropped_due_corridor = agents_dropped_corridor;
    result.corridor_resample_attempts = corridor_resample_attempts;

    // Certified union certificate
    CertifiedUnionObjective cert_obj;
    int n = ctx.G.n;
    cert_obj.init(ctx.weights, n, cfg.psi, cfg.lambda,
                  &union_data.union_counts,
                  static_cast<double>(union_data.sum_Lmin),
                  ctx.candidates);
    for (NodeId u : selection) cert_obj.add(u);
    double certified_value = cert_obj.value();
    result.certified_union_value = certified_value;
    result.certified_ub_value = certified_value;

    // For baselines we default capped union to the same certified value.
    result.capped_union_value = certified_value;
    CapResolution cap_summary = resolve_cap_strategy(cfg);
    result.cap_mode_used = cap_summary.effective_mode;
    result.cap_k_used = cap_summary.topk_k;
    result.cap_min = result.cap_median = result.cap_max = 0.0;
    result.cap_cache_key_hash.clear();

    result.avg_agents_per_node = union_data.avg_agents_per_node;

    // Monte-Carlo evaluation or agent-pool reuse depending on cfg.M_eval.
    if (cfg.M_eval > 0) {
        AgentSampler sampler_eval;
        AgentSamplerConfig od_eval = cfg.od;
        sampler_eval.init(od_eval, ctx.candidates, ctx.blocked);
        EvalStats stats = evaluate_original_objective_mc_stats(
            ctx.G, ctx.weights, cfg.psi, cfg.lambda,
            selection,
            sampler_eval,
            cfg.M_eval,
            cfg.K_paths_per_agent,
            cfg.seed + 999,
            cfg.ci_alpha,
            (cfg.od.rows > 0 && cfg.od.cols > 0),
            cfg.od.rows,
            cfg.od.cols,
            cfg.corridor_band_r);

        result.stats_pick = stats;
        result.stats_UB = stats;
        result.stats_LB = stats;
        double total = (stats.M > 0) ? stats.mean * static_cast<double>(stats.M) : 0.0;
        result.fhat_pick = total;
        result.fhat_UB = total;
        result.fhat_LB = total;
    } else {
        AgentPoolConfig pcfg{cfg.psi, cfg.lambda};
        AgentPool pool;
        if (pool.init(ctx.G, ctx.weights, pcfg, agents, paths)) {
            for (NodeId u : selection) pool.add(u);
            double total = pool.total_value();
            result.fhat_pick = total;
            result.fhat_UB = total;
            result.fhat_LB = total;
        } else {
            result.fhat_pick = result.fhat_UB = result.fhat_LB = 0.0;
        }
        result.stats_pick = EvalStats{};
        result.stats_UB = EvalStats{};
        result.stats_LB = EvalStats{};
    }

    out.extra_meta.emplace_back("corridor_band_r_used", std::to_string(result.corridor_band_r_used));
    out.extra_meta.emplace_back("agents_dropped_due_corridor", std::to_string(result.agents_dropped_due_corridor));
    out.extra_meta.emplace_back("corridor_resample_attempts", std::to_string(result.corridor_resample_attempts));
}

static bool generate_training_samples(const MethodContext& ctx,
                                      const SandwichConfig& cfg,
                                      std::vector<std::pair<NodeId, NodeId>>& agents,
                                      std::vector<std::vector<Path>>& paths,
                                      std::vector<std::vector<NodeId>>* corridor_nodes,
                                      int* agents_dropped_corridor,
                                      int* corridor_resample_attempts) {
    if (corridor_nodes) corridor_nodes->clear();
    if (agents_dropped_corridor) *agents_dropped_corridor = 0;
    if (corridor_resample_attempts) *corridor_resample_attempts = 0;
    AgentSampler sampler;
    AgentSamplerConfig od = cfg.od;
    sampler.init(od, ctx.candidates, ctx.blocked);
    return generate_agents_and_paths(ctx.G, sampler,
                                     cfg.M_agents,
                                     cfg.K_paths_per_agent,
                                     cfg.seed,
                                     agents,
                                     paths,
                                     (cfg.od.rows > 0 && cfg.od.cols > 0),
                                     cfg.od.rows,
                                     cfg.od.cols,
                                     cfg.corridor_band_r,
                                     nullptr,
                                     corridor_nodes,
                                     agents_dropped_corridor,
                                     corridor_resample_attempts);
}

} // namespace

// ----------------------- Program entry point -----------------------

int main(int argc, char** argv) {
    ProgramOptions opts;
    std::string parse_error;
    if (!parse_arguments(argc, argv, opts, parse_error)) {
        if (!parse_error.empty()) {
            std::cerr << "[error] " << parse_error << '\n';
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS; // help already printed
    }

    std::ostringstream paths_log;
    apply_paths_strategy(opts, paths_log);

    if (opts.list_methods) {
        print_method_list();
        return EXIT_SUCCESS;
    }

    if (!opts.od_rows_provided && !opts.od_cols_provided && opts.make_grid && opts.grid_spec_provided) {
        opts.od_rows = opts.grid_spec.rows;
        opts.od_cols = opts.grid_spec.cols;
        opts.od_rows_cols_inherited = true;
    } else {
        opts.od_rows_cols_inherited = false;
    }

    if (opts.make_grid && !opts.chain_run) {
        std::string grid_err;
        if (!write_grid_edge_list(opts.grid_spec, opts.emit_edges_path, grid_err)) {
            std::cerr << "[error] " << grid_err << '\n';
            return EXIT_FAILURE;
        }
        const std::string paths_info = paths_log.str();
        if (!paths_info.empty()) {
            std::cout << paths_info;
        }
        std::cout << "[info] grid edge list written to " << opts.emit_edges_path << '\n';
        return EXIT_SUCCESS;
    }

    const MethodSpec* method_spec = find_method_spec(opts.method_name);
    if (!method_spec) {
        std::cerr << "[error] unknown method '" << opts.method_name
                  << "'; run --list-methods to see supported options" << '\n';
        return EXIT_FAILURE;
    }

    std::vector<std::string> method_parse_warnings;
    std::unordered_map<std::string, std::string> method_args =
        parse_method_args(opts.method_args_raw, method_parse_warnings);

    std::ostringstream log;                     // Accumulates stdout.txt contents before flushing
    log << paths_log.str();
    Clock::time_point start_time = Clock::now(); // Start wall-clock timing for total runtime

    // ## Step 1: CLI parsing already performed (opts populated above).

    // ## Step 2: Load graph --------------------------------------------------------
    Graph G;
    std::vector<NodeId> candidates;
    std::vector<char> blocked_mask;
    const std::vector<char>* blocked_ptr = nullptr;

    GraphBinMeta graph_bin_meta{};
    std::vector<NodeId> bin_candidates;
    std::vector<char> bin_blocked;
    const bool using_graph_bin = !opts.graph_bin_in.empty();
    const bool using_grid = opts.make_grid;

    if (using_graph_bin) {
        auto load_start = Clock::now();
        std::string err;
        if (!read_graph_bin(opts.graph_bin_in, G, bin_candidates, bin_blocked, &graph_bin_meta, err)) {
            std::cerr << "[error] " << err << '\n';
            return EXIT_FAILURE;
        }
        auto load_elapsed = std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
        if (graph_bin_meta.n > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            std::cerr << "[error] graph bin node count exceeds supported range\n";
            return EXIT_FAILURE;
        }
        opts.n = static_cast<int>(graph_bin_meta.n);
        opts.n_provided = true;
        opts.undirected = !graph_bin_meta.directed;
        opts.edges_display = opts.graph_bin_in + " [bin]";
        std::ios::fmtflags old_flags = std::cerr.flags();
        std::streamsize old_precision = std::cerr.precision();
        std::cerr << "[info] graphbin: loaded in " << std::fixed << std::setprecision(3)
                  << load_elapsed << " ms (n=" << graph_bin_meta.n
                  << ", m=" << graph_bin_meta.m << ")\n";
        std::cerr.flags(old_flags);
        std::cerr.precision(old_precision);
        log << "[info] Graph source: graph_bin_in=" << opts.graph_bin_in
            << " (n=" << opts.n << ", m=" << graph_bin_meta.m << ")\n";
    } else if (using_grid) {
        opts.n = opts.grid_spec.rows * opts.grid_spec.cols;
        opts.n_provided = true;
        opts.undirected = true;
        G = build_grid_graph(opts.grid_spec);
        opts.edges_display = "grid:" + std::to_string(opts.grid_spec.rows) + "x" + std::to_string(opts.grid_spec.cols);
        log << "[info] Generated grid graph rows=" << opts.grid_spec.rows
            << ", cols=" << opts.grid_spec.cols
            << (opts.grid_spec.diag ? ", diag=1" : ", diag=0")
            << (opts.grid_spec.torus ? ", torus=1" : ", torus=0") << '\n';
        if (!opts.emit_edges_path.empty()) {
            std::string err;
            if (!write_grid_edge_list(opts.grid_spec, opts.emit_edges_path, err)) {
                std::cerr << "[error] " << err << '\n';
                return EXIT_FAILURE;
            }
        }
    } else {
        if (!opts.n_provided) {
            std::string err;
            if (!infer_node_count(opts.edges_path, opts.one_based, opts.n, err)) {
                std::cerr << "[error] " << err << '\n';
                return EXIT_FAILURE;
            }
            log << "[info] Inferred node count n=" << opts.n << '\n';
        }
        if (opts.n <= 0) {
            std::cerr << "[error] Node count must be positive; supply --n or ensure the edge list is not empty" << '\n';
            return EXIT_FAILURE;
        }
        G = Graph(opts.n);
        if (!load_edge_list(opts.edges_path, opts.n, G, opts.undirected, opts.one_based)) {
            std::cerr << "[error] Failed to load edge list; check --edges path and ensure 'u v' integers per line" << '\n';
            return EXIT_FAILURE;
        }
        opts.edges_display = opts.edges_path;
    }
    if (opts.edges_display.empty()) {
        opts.edges_display = using_graph_bin ? opts.graph_bin_in
                                             : (using_grid ? "grid" : opts.edges_path);
    }

    // ## Step 3: Load candidates & blocked nodes ------------------------------------
    opts.candidates_display = "<all nodes>";
    opts.blocked_display = "<none>";

    if (opts.has_candidates) {
        candidates = load_candidates(opts.candidates_path, opts.n, opts.one_based);
        if (candidates.empty()) {
            std::cerr << "[error] Candidate file produced zero usable nodes; verify node ids and --one_based flag" << '\n';
            return EXIT_FAILURE;
        }
        opts.candidates_display = opts.candidates_path;
    } else if (using_graph_bin && !bin_candidates.empty()) {
        candidates = bin_candidates;
        opts.candidates_display = !graph_bin_meta.candidates_src.empty()
                                  ? graph_bin_meta.candidates_src
                                  : opts.graph_bin_in + " [bin]";
    } else if (using_grid) {
        candidates = make_all_nodes_as_candidates(opts.grid_spec.rows, opts.grid_spec.cols);
    } else {
        candidates.resize(opts.n);
        std::iota(candidates.begin(), candidates.end(), 0);
    }

    if (opts.has_blocked) {
        std::string err;
        if (!load_blocked_nodes(opts.blocked_path, opts.n, opts.one_based, blocked_mask, err)) {
            std::cerr << "[error] " << err << '\n';
            return EXIT_FAILURE;
        }
        opts.blocked_display = opts.blocked_path;
    } else if (using_graph_bin && bin_blocked.size() == static_cast<size_t>(opts.n)) {
        blocked_mask = bin_blocked;
        if (!graph_bin_meta.blocked_source.empty()) {
            opts.blocked_display = graph_bin_meta.blocked_source;
        } else {
            opts.blocked_display = opts.graph_bin_in + " [bin]";
        }
    }

    const int runtime_blocked_count = static_cast<int>(std::count(blocked_mask.begin(), blocked_mask.end(), 1));
    if (!blocked_mask.empty()) {
        if (runtime_blocked_count > 0) {
            blocked_ptr = &blocked_mask;
        } else {
            blocked_mask.clear();
            opts.blocked_display = "<none>";
        }
    }

    opts.blocked_count = runtime_blocked_count;
    if (runtime_blocked_count > 0) {
        if (opts.has_blocked) {
            opts.blocked_source_label = opts.blocked_path;
        } else if (using_graph_bin && !graph_bin_meta.blocked_source.empty()) {
            opts.blocked_source_label = graph_bin_meta.blocked_source;
        } else if (!opts.blocked_display.empty() && opts.blocked_display != "<none>") {
            opts.blocked_source_label = opts.blocked_display;
        } else {
            opts.blocked_source_label = "<unknown>";
        }
    } else {
        opts.blocked_source_label = "<none>";
    }

    if (candidates.empty()) {
        std::cerr << "[error] Candidate set is empty; provide --cands or ensure the graph definition includes candidates" << '\n';
        return EXIT_FAILURE;
    }

    if (!opts.od_hotspots.empty()) {
        for (const auto& hw : opts.od_hotspots) {
            if (hw.first < 0 || hw.first >= opts.n) {
                std::cerr << "[error] hotspot node " << hw.first << " exceeds graph node range 0.." << (opts.n - 1) << '\n';
                return EXIT_FAILURE;
            }
            if (!(hw.second > 0.0)) {
                std::cerr << "[error] hotspot weight must be positive (got " << hw.second << ")" << '\n';
                return EXIT_FAILURE;
            }
        }
    }

    if (opts.k > static_cast<int>(candidates.size())) {
        log << "[warn] k(" << opts.k << ") exceeds candidate count " << candidates.size()
            << "; using all candidates.\n";
        opts.k = static_cast<int>(candidates.size());
    }

    if (opts.log_level == LogLevel::Debug) {
        long long edge_count_directed = 0;
        for (const auto& nbrs : G.adj) edge_count_directed += nbrs.size();
        const long long undirected_edges = opts.undirected ? edge_count_directed / 2 : edge_count_directed;
        const long long blocked_count = std::count(blocked_mask.begin(), blocked_mask.end(), 1);
        log << "[debug] dataset_summary: nodes=" << opts.n
            << ", edges=" << undirected_edges
            << ", candidates=" << candidates.size()
            << ", blocked=" << blocked_count
            << (blocked_ptr ? " (active)" : " (none)")
            << '\n';
    }

    // Optional: pack graph/candidates/blocked into binary container when requested.
    std::vector<NodeId> pack_candidates = candidates;
    if (!opts.pack_candidates_path.empty()) {
        pack_candidates = load_candidates(opts.pack_candidates_path, opts.n, opts.one_based);
        if (pack_candidates.empty()) {
            std::cerr << "[error] pack_candidates file produced zero usable nodes" << '\n';
            return EXIT_FAILURE;
        }
    }
    if (!pack_candidates.empty()) {
        std::sort(pack_candidates.begin(), pack_candidates.end());
        pack_candidates.erase(std::unique(pack_candidates.begin(), pack_candidates.end()), pack_candidates.end());
    }

    std::vector<char> pack_blocked = blocked_mask;
    bool pack_blocked_was_empty = false;
    BlockedMetaFileData pack_blocked_meta{};
    if (!opts.pack_blocked_path.empty()) {
        std::string err;
        if (!load_blocked_nodes(opts.pack_blocked_path, opts.n, opts.one_based, pack_blocked, err, true, &pack_blocked_was_empty)) {
            std::cerr << "[error] " << err << '\n';
            return EXIT_FAILURE;
        }
        pack_blocked_meta = load_blocked_meta_file(opts.pack_blocked_path);
    }
    if (pack_blocked.size() != static_cast<size_t>(opts.n)) {
        pack_blocked.assign(static_cast<size_t>(opts.n), 0);
    }

    const int pack_blocked_count = static_cast<int>(std::count(pack_blocked.begin(), pack_blocked.end(), 1));
    if (!opts.pack_blocked_path.empty()) {
        opts.blocked_count = pack_blocked_count;
        if (pack_blocked_was_empty) {
            opts.blocked_source_label = "empty_file";
        } else if (pack_blocked_meta.has_source) {
            opts.blocked_source_label = pack_blocked_meta.source;
        } else {
            opts.blocked_source_label = opts.pack_blocked_path;
        }
        if (pack_blocked_meta.has_seed) {
            opts.blocked_seed_has_value = true;
            opts.blocked_seed_value = pack_blocked_meta.seed;
        }
        if (pack_blocked_meta.has_fraction) {
            opts.blocked_fraction_has_value = true;
            opts.blocked_fraction_value = pack_blocked_meta.fraction;
        }
    }

    if (!opts.graph_bin_out.empty()) {
        GraphBinMeta meta;
        meta.directed = !opts.undirected;
        meta.candidates_src = !opts.pack_candidates_path.empty() ? opts.pack_candidates_path
                                  : (!opts.candidates_display.empty() ? opts.candidates_display : (opts.has_candidates ? opts.candidates_path : "<all>"));
        if (!opts.pack_blocked_path.empty()) {
            if (pack_blocked_was_empty) {
                meta.blocked_source = "empty_file";
            } else if (pack_blocked_meta.has_source) {
                meta.blocked_source = pack_blocked_meta.source;
            } else {
                meta.blocked_source = opts.pack_blocked_path;
            }
        } else if (!opts.blocked_display.empty() && opts.blocked_display != "<none>") {
            meta.blocked_source = opts.blocked_display;
        } else if (opts.has_blocked) {
            meta.blocked_source = opts.blocked_path;
        } else {
            meta.blocked_source = "<none>";
        }
        meta.blocked_count = static_cast<uint64_t>(pack_blocked_count);
        if (pack_blocked_meta.has_seed) {
            meta.blocked_seed_valid = true;
            meta.blocked_seed = pack_blocked_meta.seed;
        }
        if (pack_blocked_meta.has_fraction) {
            meta.blocked_fraction_valid = true;
            meta.blocked_fraction = pack_blocked_meta.fraction;
        }
        meta.build_cmdline = opts.raw_cmdline;
        meta.note.clear();
        meta.created_utc.clear();
        std::string err;
        const std::string bin_path = opts.graph_bin_out;
        const std::string json_path = opts.graph_bin_out + ".json";
        if (!write_graph_bin(bin_path, json_path, G, pack_candidates, pack_blocked, meta, err)) {
            std::cerr << "[error] " << err << '\n';
            return EXIT_FAILURE;
        }
        log << "[info] graphbin: wrote " << bin_path << " (+ " << json_path << ")\n";
    }

    // ## Step 4: Prepare output directories and neighbourhood weights ---------------
    const fs::path out_root = fs::path(opts.out_dir) / opts.tag; // Output directory <out>/<tag>
    if (opts.weights_path.empty()) {
        opts.weights_path = (out_root / "weights.bin").string(); // Default cache location inside run directory
    }

    SandwichConfig base_cfg = to_sandwich_config(opts);
    CapResolution cap_info_preview = resolve_cap_strategy(base_cfg);
    opts.cap_mode_effective = cap_info_preview.effective_mode;
    opts.cap_topk_effective = cap_info_preview.topk_k;
    opts.cap_mode_auto_resolved = cap_info_preview.auto_selected;
    opts.cap_mode_psi_bounded = cap_info_preview.psi_bounded;
    opts.cap_mode_pathlen_fallback = cap_info_preview.pathlen_fallback;
    if (cap_info_preview.auto_selected) {
        std::ostringstream cap_msg;
        cap_msg << "[info] cap_mode: input=Auto, effective="
                << cap_mode_name(cap_info_preview.effective_mode)
                << " (" << (cap_info_preview.psi_bounded ? "psi bounded" : "psi unbounded") << ")";
        opts.cap_mode_log_line = cap_msg.str();
    } else if (cap_info_preview.pathlen_fallback) {
        opts.cap_mode_log_line = "[warn] cap_mode PathlenOnly requested with unbounded psi; falling back to TopkLocal";
    } else {
        opts.cap_mode_log_line.clear();
    }

    // Echo config after defaults are stabilised ------------------------------------
    log_config_echo(opts, log);
    if (!opts.cap_mode_log_line.empty()) {
        log << opts.cap_mode_log_line << '\n';
    }
    if (opts.ub_mode == UBMode::ProxyCorridor && opts.proxy_r == 1 && opts.corridor_r != 1) {
        log << "[info] r=1 proxy requested; add --corridor_r 1 to select certified objective\n";
    }
    if (opts.dry_run) {
        if (!flush_log(log.str(), out_root / "stdout.txt")) {
            std::cerr << "[error] Failed to write dry_run stdout log\n";
            return EXIT_FAILURE;
        }
        if (opts.write_json_meta && !write_run_config_json(opts, out_root / "run_config.json")) {
            std::cerr << "[error] Failed to write run_config.json\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    if (!ensure_parent_dir((out_root / "stub").string())) {
        std::cerr << "[error] Cannot create output directory \"" << out_root.string()
                  << "\"; check path permissions or specify --out_dir" << '\n';
        return EXIT_FAILURE;
    }

    // Build or load weights ---------------------------------------------------------
    WeightsStore W;             // Stores candidate neighbourhood weights w_{v,u}
    CacheHeader hdr;            // Metadata for weight persistence / validation
    bool weights_loaded = false; // Track whether cache load succeeded
    double weights_time = 0.0;  // Wall-clock seconds spent building/loading weights
    if (opts.load_weights) {
        CacheHeader hdr_in;
        auto timer = Timer::measure(load_weights_binary, opts.weights_path, std::ref(W), std::ref(hdr_in));
        weights_time = timer.first.count();
        if (!timer.second) {
            std::cerr << "[error] Failed to load weights cache " << opts.weights_path
                      << "; regenerate with --save_weights 1 or remove the stale file" << '\n';
            return EXIT_FAILURE;
        }
        if (hdr_in.n != opts.n) {
            std::cerr << "[error] Weight cache n=" << hdr_in.n << " does not match graph n=" << opts.n << '\n';
            return EXIT_FAILURE;
        }
        hdr = hdr_in;
        weights_loaded = true;
        log << "[info] Loaded weights cache in " << weights_time << " s\n";
    }
    if (!weights_loaded) {
        Timer weights_timer{"build_weights"};
        if (opts.use_hard_cutoff) {
            build_weights_hard_cutoff(G, candidates, opts.hard_cutoff, W);
            hdr.D = opts.hard_cutoff;
            hdr.alpha = 0.0;
            hdr.R = -1;
            hdr.eps = 0.0;
            hdr.r = std::max(1, opts.hard_cutoff);
        } else {
            // Auto-cap default: bounded psi => PathlenOnly, unbounded => TopkLocal (handled downstream).
            build_weights_soft_decay(G, candidates, opts.soft.alpha, opts.soft.radius, opts.soft.eps, W);
            hdr.D = -1;
            hdr.alpha = opts.soft.alpha;
            hdr.R = opts.soft.radius;
            hdr.eps = opts.soft.eps;
            hdr.r = std::max(1, opts.soft.radius);
        }
        weights_time = weights_timer.stop().count();
        hdr.n = opts.n;
        hdr.rows = 0;
        hdr.cols = 0;
        hdr.rng_seed = opts.seed;
        hdr.num_agents = opts.M_agents;
        hdr.K_paths_per_agent = opts.K_paths_per_agent;
        log << "[info] Built weights in " << weights_time << " s\n";
        if (opts.save_weights) {
            if (!save_weights_binary(opts.weights_path, W, hdr)) {
                std::cerr << "[error] Failed to save weights cache to " << opts.weights_path
                          << "; check disk space or disable --save_weights" << '\n';
                return EXIT_FAILURE;
            }
            log << "[info] Saved weights cache to " << opts.weights_path << '\n';
        }
    }

    // ## Step 5: Dispatch selected method -------------------------------------------
    MethodContext method_ctx{G, candidates, blocked_ptr, W, opts, base_cfg};
    MethodRunResult method_run;
    method_run.applied_cfg = method_ctx.base_cfg;
    if (!method_spec->runner(method_ctx, method_args, method_run) || !method_run.success) {
        std::cerr << "[error] method '" << method_spec->name << "' failed" << '\n';
        return EXIT_FAILURE;
    }

    // Log margs parsing warnings (if any)
    for (const std::string& warn : method_parse_warnings) {
        log << "[warn] " << warn << '\n';
    }
    for (const std::string& warn : method_run.warnings) {
        log << "[warn] " << warn << '\n';
    }

    SandwichResult& result = method_run.result;
    SandwichConfig& cfg_used = method_run.applied_cfg;
    ReportStats pick_stats = method_run.stats_pick;
    ReportStats ub_stats = method_run.stats_UB;
    ReportStats lb_stats = method_run.stats_LB;

    std::vector<std::pair<std::string, std::string>> extra_meta = method_run.extra_meta;
    append_meta(extra_meta, "candidate_count", static_cast<int>(candidates.size()));
    append_meta(extra_meta, "blocked_count", opts.blocked_count);
    if (opts.blocked_fraction_has_value) {
        append_meta(extra_meta, "blocked_fraction", opts.blocked_fraction_value);
    }
    if (opts.blocked_seed_has_value) {
        append_meta(extra_meta, "blocked_seed", std::to_string(opts.blocked_seed_value));
    }
    append_meta(extra_meta, "blocked_source", opts.blocked_source_label.empty() ? "<none>" : opts.blocked_source_label);
    append_meta(extra_meta, "corridor_band_r_used", result.corridor_band_r_used);
    append_meta(extra_meta, "agents_dropped_due_corridor", result.agents_dropped_due_corridor);
    append_meta(extra_meta, "corridor_resample_attempts", result.corridor_resample_attempts);
    append_meta(extra_meta, "time_build_weights_sec", weights_time);
    append_meta(extra_meta, "proxy_baseline_time_sec", result.proxy_baseline_time_sec);
    append_meta(extra_meta, "proxy_solution_size", static_cast<int>(result.proxy_solution.size()));
    append_meta(extra_meta, "certified_union_value_pick", result.certified_union_value);
    append_meta(extra_meta, "capped_union_value_pick", result.capped_union_value);
    append_meta(extra_meta, "certified_ub_value_pick", result.certified_ub_value);
    append_meta(extra_meta, "ub_objective_total_S_UB", result.ub_objective_total_S_UB);
    append_meta(extra_meta, "ub_objective_mean_S_UB", result.ub_objective_mean_S_UB);
    append_meta(extra_meta, "ub_objective_mean_pick", result.ub_objective_mean_pick);
    append_meta(extra_meta, "proxy_score", result.proxy_score);
    append_meta(extra_meta, "selection_mode_effective", ub_mode_name(result.selection_mode_effective));
    append_meta(extra_meta, "selection_objective_type", result.selection_objective_type);
    append_meta(extra_meta, "selection_proxy_r", result.selection_proxy_r);
    append_meta(extra_meta, "routed_r_equals_one", result.routed_r_equals_one);
    append_meta(extra_meta, "sum_deprecated_used", result.sum_deprecated_used);
    append_meta(extra_meta, "cap_mode_used_runtime", cap_mode_name(result.cap_mode_used));
    append_meta(extra_meta, "cap_topk_used_runtime", result.cap_k_used);
    append_meta(extra_meta, "cap_min_runtime", result.cap_min);
    append_meta(extra_meta, "cap_median_runtime", result.cap_median);
    append_meta(extra_meta, "cap_max_runtime", result.cap_max);
    append_meta(extra_meta, "avg_agents_per_node_runtime", result.avg_agents_per_node);
    append_meta(extra_meta, "avg_agents_per_node", result.avg_agents_per_node);
    append_meta(extra_meta, "selection_time_sec", result.selection_time_sec);
    append_meta(extra_meta, "selection_iter_count", static_cast<int>(result.selection_iter_times.size()));
    double iter_sum = std::accumulate(result.selection_iter_times.begin(), result.selection_iter_times.end(), 0.0);
    append_meta(extra_meta, "selection_iter_total_sec", iter_sum);
    {
        std::ostringstream iter_stream;
        for (size_t i = 0; i < result.selection_iter_times.size(); ++i) {
            if (i) iter_stream << ',';
            iter_stream << to_string_compact(result.selection_iter_times[i]);
        }
        append_meta(extra_meta, "selection_iter_sec_list", iter_stream.str());
    }
    append_meta(extra_meta, "lb3_run_mode", result.lb3_run_mode.empty() ? "none" : result.lb3_run_mode);
    append_meta(extra_meta, "lb3_iters_done", result.lb3_iters_done);
    append_meta(extra_meta, "lb3_improved", result.lb3_improved);
    append_meta(extra_meta, "lb3_best_value", result.lb3_best_value);
    append_meta(extra_meta, "lb3_base_value", result.lb3_base_value);
    append_meta(extra_meta, "cap_cache_key_hash", result.cap_cache_key_hash);
    const std::string od_params_json = build_od_params_json(cfg_used.od);
    append_meta(extra_meta, "od_dist_used", od_distribution_cli_name(cfg_used.od.dist));
    append_meta(extra_meta, "od_params_json", od_params_json);
    append_meta(extra_meta, "od_rows", cfg_used.od.rows);
    append_meta(extra_meta, "od_cols", cfg_used.od.cols);
    append_meta(extra_meta, "od_rows_cols_inherited", opts.od_rows_cols_inherited);
    if (!opts.od_hotspots_hash.empty()) {
        append_meta(extra_meta, "od_hotspots_hash", opts.od_hotspots_hash);
    }
    append_meta(extra_meta, "method", method_spec->name);
    append_meta(extra_meta, "margs_raw", opts.method_args_raw.empty() ? "" : opts.method_args_raw);
    if (!method_run.note.empty()) {
        append_meta(extra_meta, "note", method_run.note);
    }
    bool has_time_run = false;
    for (const auto& kv : extra_meta) {
        if (kv.first == "time_run_sandwich_sec") { has_time_run = true; break; }
    }
    if (!has_time_run) {
        append_meta(extra_meta, "time_run_sandwich_sec", to_string_compact(result.selection_time_sec));
    }

    std::vector<std::pair<std::string, std::string>> meta;
    build_sandwich_metadata(cfg_used, candidates, meta, &extra_meta);

    ReportStats ub_stats_write = ub_stats;
    ReportStats lb_stats_write = lb_stats;
    if (cfg_used.M_eval == 0) {
        // When no MC evaluation was run, ensure fields remain zero (already default).
    }

    // ## Step 7: Persist report and optional JSON -----------------------------------
    const fs::path report_path = out_root / "report.tsv";
    ReportStats pick_stats_write = pick_stats;
    if (!save_eval_report(report_path.string(), opts.tag, meta,
                          result.S_pick, pick_stats_write,
                          &result.S_UB, &ub_stats_write,
                          &result.S_LB, &lb_stats_write)) {
        std::cerr << "[error] Failed to write report.tsv; check disk space and write permissions" << '\n';
        return EXIT_FAILURE;
    }

    if (opts.write_json_meta) {
        if (!write_run_config_json(opts, out_root / "run_config.json")) {
            std::cerr << "[error] Failed to write run_config.json; ensure the output directory is writable" << '\n';
            return EXIT_FAILURE;
        }
    }

    // ## Step 8: Compose console summary & diagnostics ------------------------------
    double total_time = std::chrono::duration<double>(Clock::now() - start_time).count();
    log << "[info] method='" << method_spec->name << "' selection_time="
        << to_string_compact(result.selection_time_sec)
        << " s (total wall clock " << to_string_compact(total_time) << " s)\n";

    const bool iter_mismatch = std::fabs(iter_sum - result.selection_time_sec) > kEpsilonCompare * std::max(1.0, result.selection_time_sec);
    if (iter_mismatch) {
        log << "[warn] selection_iter_total_sec differs from recorded selection_time_sec ("
            << iter_sum << " vs " << result.selection_time_sec << ")\n";
    }

    if (opts.log_level == LogLevel::Debug) {
        std::ostringstream iter_stream;
        for (size_t i = 0; i < result.selection_iter_times.size(); ++i) {
            if (i) iter_stream << ',';
            iter_stream << to_string_compact(result.selection_iter_times[i]);
        }
        log << "[debug] selection_iter_sec_list=" << iter_stream.str() << '\n';
    }

    log << "[info] UB-objective totals/means: S_UB total="
        << to_string_compact(result.ub_objective_total_S_UB)
        << ", mean=" << to_string_compact(result.ub_objective_mean_S_UB)
        << "; S_pick mean=" << to_string_compact(result.ub_objective_mean_pick) << '\n';
    log << "[info] corridor_band_r_used=" << result.corridor_band_r_used
        << ", agents_dropped_due_corridor=" << result.agents_dropped_due_corridor
        << ", corridor_resample_attempts=" << result.corridor_resample_attempts << '\n';

    log << "Summary: method=" << method_spec->name
        << " effective=" << ub_mode_name(result.selection_mode_effective)
        << " type=" << result.selection_objective_type
        << " routed_r1=" << bool_to_string(result.routed_r_equals_one)
        << " sum_deprecated=" << bool_to_string(result.sum_deprecated_used)
        << " proxy_score=" << to_string_compact(result.proxy_score)
        << " certified_ub_value=" << to_string_compact(result.certified_ub_value)
        << " iter_count=" << result.selection_iter_times.size()
        << " total_time=" << to_string_compact(total_time) << '\n';

    // ## Step 9: Flush stdout.txt and exit ------------------------------------------
    if (!flush_log(log.str(), out_root / "stdout.txt")) {
        std::cerr << "[error] Failed to write stdout.txt; ensure the output directory is writable" << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
