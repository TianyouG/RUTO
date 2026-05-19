#pragma once
#include "stdafx.h"

static inline bool ensure_parent_dir(const std::string& file) {
    namespace fs = std::filesystem;
    fs::path p(file);
    if (p.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        if (ec) {
            std::cerr << "[ensure_parent_dir] create_directories failed for "
                      << p.parent_path().string() << " : " << ec.message() << "\n";
            return false;
        }
    }
    return true;
}

// ---------------------------
// PathSet: lightweight container used by offline tests or debugging.
// Large-scale experiments sample paths on the fly; this struct keeps I/O simple.
// ---------------------------
struct PathSet {
    // paths[i] holds candidate paths for agent i; each path is a sequence of node ids
    std::vector<std::vector<Path>> paths;
    // lengths[i][j] stores |path|-1 (number of edges) for paths[i][j]
    std::vector<std::vector<int>>  lengths;
};

// -------------- Binary stream helpers --------------
static inline bool write_stream_to_file(const std::string& file, const StreamType& s) {
    if (!ensure_parent_dir(file)) return false;
    std::ofstream fout(file, std::ios::binary);
    if (!fout) return false;
    fout.write(reinterpret_cast<const char*>(s.data()), static_cast<std::streamsize>(s.size()));
    return bool(fout);
}
static inline bool read_stream_from_file(const std::string& file, StreamType& s) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin) return false;
    fin.seekg(0, std::ios::end);
    std::streamsize sz = fin.tellg();
    if (sz < 0) return false;
    fin.seekg(0, std::ios::beg);
    s.resize(static_cast<size_t>(sz));
    fin.read(reinterpret_cast<char*>(s.data()), sz);
    return bool(fin);
}

// ---------------------- 1) Load a graph from an edge list ----------------------
static inline bool load_edge_list(const std::string& file, int n, Graph& G,
                                  bool undirected=true, bool one_based=false, char comment='#') {
    G = Graph(n);
    std::ifstream fin(file);
    if (!fin) {
        std::cerr << "[load_edge_list] cannot open: " << file << "\n";
        return false;
    }
    long long lines=0, ok=0, oob=0;
    int u,v;
    std::string line;
    while (std::getline(fin, line)) {
        ++lines;
        if (line.empty()) continue;
        if (comment && !line.empty() && line[0]==comment) continue;
        std::istringstream iss(line);
        if (!(iss >> u >> v)) continue;
        if (one_based) { --u; --v; }
        if (u<0 || u>=n || v<0 || v>=n) { ++oob; continue; }
        G.add_edge(u, v, undirected);
        ++ok;
    }
    std::cerr << "[load_edge_list] lines="<<lines<<", loaded edges="<<ok
              <<", dropped(oob)="<<oob<<"\n";
    return true;
}

// ---------------------- 2) Load a candidate set ----------------------
static inline std::vector<NodeId> load_candidates(const std::string& file, int n,
                                                  bool one_based=false, char comment='#',
                                                  bool dedup=true) {
    std::ifstream fin(file);
    std::vector<NodeId> res;
    if (!fin) {
        std::cerr << "[load_candidates] cannot open: " << file << " ; fallback to empty.\n";
        return res;
    }
    std::vector<char> seen; 
    if (dedup) seen.assign(n, 0);
    long long ok=0, oob=0, dup=0;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (comment && line[0]==comment) continue;
        std::istringstream iss(line);
        int x; if (!(iss >> x)) continue;
        if (one_based) --x;
        if (x<0 || x>=n) { ++oob; continue; }
        if (dedup) {
            if (seen[x]) { ++dup; continue; }
            seen[x]=1;
        }
        res.push_back(x); ++ok;
    }
    std::cerr << "[load_candidates] loaded="<<ok<<", oob="<<oob<<", dup="<<dup<<"\n";
    return res;
}

// ---------------------- 3) Load candidate paths (small instances / debugging) ----------------------
static inline PathSet load_paths(const std::string& file, int n,
                                 bool one_based=false, char path_sep='|',
                                 char comment='#', bool drop_invalid=true) {
    PathSet P;
    std::ifstream fin(file);
    if (!fin) {
        std::cerr << "[load_paths] cannot open: " << file << " ; return empty.\n";
        return P;
    }
    std::string line;
    long long agents=0, total_paths=0, kept=0, dropped=0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (comment && line[0]==comment) continue;
        ++agents;

        // Split the line on the path separator into candidate paths
        std::vector<std::string> chunks;
        {
            std::string cur;
            for (char ch: line) {
                if (ch==path_sep) { if(!cur.empty()){chunks.push_back(cur); cur.clear();} }
                else cur.push_back(ch);
            }
            if (!cur.empty()) chunks.push_back(cur);
        }
        std::vector<Path> agent_paths;
        std::vector<int>   agent_lengths;

        for (auto &ps: chunks) {
            std::istringstream ss(ps);
            Path nodes; int x;
            while (ss >> x) {
                if (one_based) --x;
                nodes.push_back(x);
            }
            ++total_paths;

            // Validate the path: every node must be within bounds and length >= 1
            bool ok = true;
            if ((int)nodes.size()<2) ok=false;
            else {
                for (int id: nodes) if (id<0 || id>=n) { ok=false; break; }
            }
            if (!ok) {
                if (drop_invalid) { ++dropped; continue; }
            }
            agent_paths.push_back(std::move(nodes));
            agent_lengths.push_back((int)agent_paths.back().size()-1);
            ++kept;
        }
        P.paths.push_back(std::move(agent_paths));
        P.lengths.push_back(std::move(agent_lengths));
    }
    std::cerr << "[load_paths] agents="<<agents<<", paths="<<total_paths
              <<", kept="<<kept<<", dropped="<<dropped<<"\n";
    return P;
}

// ---------------------- 4) Dataset statistics / sanity checks ----------------------
static inline void print_dataset_summary(const Graph& G,
                                         const std::vector<NodeId>& cands,
                                         const PathSet* Popt = nullptr,
                                         std::ostream& os = std::cerr) {
    const int n = G.n;
    long long sumdeg = 0;
    for (int u=0; u<n; ++u) sumdeg += (long long)G.adj[u].size();
    long long edges_stored = sumdeg; // equals 2|E| when the graph is undirected

    os << "=== Dataset Summary ===\n";
    os << "|V|=" << n << ", sum(deg)=" << edges_stored
       << " (undirected E≈" << (edges_stored/2) << ")\n";
    os << "|Candidates|=" << cands.size() << "\n";
    if (Popt) {
        long long agents = (long long)Popt->paths.size();
        long long total_paths = 0;
        long long minL = LLONG_MAX, maxL = LLONG_MIN, sumL = 0;
        long long cntL = 0;
        for (size_t i=0;i<Popt->lengths.size();++i) {
            total_paths += (long long)Popt->lengths[i].size();
            for (int L: Popt->lengths[i]) {
                minL = std::min<long long>(minL, L);
                maxL = std::max<long long>(maxL, L);
                sumL += L; ++cntL;
            }
        }
        os << "#agents=" << agents << ", #paths=" << total_paths;
        if (cntL>0) {
            os << ", pathLen[min/avg/max]=[" << minL << "/"
               << std::fixed << std::setprecision(2)
               << (double)sumL/cntL << "/" << maxL << "]";
        }
        os << "\n";
    }
    os << "=======================\n";
}

// Variant of the summary that also reports blocked nodes (obstacle-aware datasets)
static inline void print_dataset_summary(const Graph& G,
    const std::vector<NodeId>& cands,
    const std::vector<char>* blockedMask,
    const PathSet* Popt,
    std::ostream& os = std::cerr)
{
    const int n = G.n;
    long long sumdeg = 0;
    for (int u=0; u<n; ++u) sumdeg += (long long)G.adj[u].size();
    os << "=== Dataset Summary (with obstacles) ===\n";
    os << "|V|=" << n << ", sum(deg)=" << sumdeg << " (undirected E≈" << (sumdeg/2) << ")\n";
    os << "|Candidates|=" << cands.size() << "\n";
    if (blockedMask && (int)blockedMask->size()==n) {
        long long n_blocked=0;
        for (int i=0;i<n;++i) if ((*blockedMask)[i]) ++n_blocked;
        os << "#obstacles=" << n_blocked << " ("
           << std::fixed << std::setprecision(2)
           << (100.0 * n_blocked / std::max(1,n)) << "%), walkable=" << (n - n_blocked) << "\n";
    }
    if (Popt) {
        long long agents = (long long)Popt->paths.size();
        long long total_paths = 0;
        long long minL = LLONG_MAX, maxL = LLONG_MIN, sumL = 0, cntL = 0;
        for (size_t i=0;i<Popt->lengths.size();++i) {
            total_paths += (long long)Popt->lengths[i].size();
            for (int L: Popt->lengths[i]) {
                minL = std::min<long long>(minL,L);
                maxL = std::max<long long>(maxL,L);
                sumL += L;
                ++cntL;
            }
        }
        os << "#agents="<<agents<<", #paths="<<total_paths;
        if (cntL>0) os << ", pathLen[min/avg/max]=["<<minL<<"/"<<(double)sumL/cntL<<"/"<<maxL<<"]";
        os << "\n";
    }
    os << "=======================\n";
}

// ---------------------- 5) Persist a solution and optional gain profile ----------------------
static inline bool save_solution(const std::string& file,
                                 const std::vector<NodeId>& selected,
                                 double value,
                                 const std::vector<double>* gains_per_step=nullptr) {
    if (!ensure_parent_dir(file)) return false;
    std::ofstream fout(file);
    if (!fout) return false;
    fout.setf(std::ios::fixed); fout<<std::setprecision(6);
    fout << "# value\n" << value << "\n";
    fout << "# selected (NodeId)\n";
    for (size_t i=0;i<selected.size();++i) {
        if (i) fout << ' ';
        fout << selected[i];
    }
    fout << "\n";
    if (gains_per_step) {
        fout << "# gains_per_step\n";
        for (size_t i=0;i<gains_per_step->size();++i) {
            if (i) fout << ' ';
            fout << (*gains_per_step)[i];
        }
        fout << "\n";
    }
    return bool(fout);
}

// ---------------------- X) Evaluation report (TSV) ----------------------
// Lightweight statistics container decoupled from alg.h::EvalStats
struct ReportStats {
    double mean = 0.0;
    double var  = 0.0;
    double se   = 0.0;
    double ci_low  = 0.0;
    double ci_high = 0.0;
    int    M    = 0;
};

static inline std::string __join_ids(const std::vector<NodeId>& ids, char sep=' ') {
    std::ostringstream os;
    for (size_t i=0;i<ids.size();++i){ if(i) os<<sep; os<<ids[i]; }
    return os.str();
}

/**
 * Write a single evaluation report as TSV.
 *
 * @param file   Output file (parent directories are created automatically)
 * @param tag    Free-form label (dataset name, experiment id, ...)
 * @param meta   Additional key-value metadata pairs
 * @param S_pick Final selection (the better solution chosen by the sandwich)
 * @param stats_pick Evaluation statistics for S_pick
 * @param S_UB/S_LB   Optional upper/lower-bound solutions
 * @param stats_UB/LB Optional statistics for the UB/LB solutions
 *
 * Format:
 *  1) Header: one `key\tvalue` per line (includes tag and metadata)
 *  2) Blank line separator
 *  3) Sections for each solution (pick/UB/LB):
 *       section \t key \t value
 *     where S is a space-separated list of node ids.
 */
static inline bool save_eval_report(const std::string& file,
                                    const std::string& tag,
                                    const std::vector<std::pair<std::string,std::string>>& meta,
                                    const std::vector<NodeId>& S_pick,
                                    const ReportStats& stats_pick,
                                    const std::vector<NodeId>* S_UB = nullptr,
                                    const ReportStats*   stats_UB = nullptr,
                                    const std::vector<NodeId>* S_LB = nullptr,
                                    const ReportStats*   stats_LB = nullptr)
{
    if (!ensure_parent_dir(file)) return false; // reuse existing directory helper
    std::ofstream fout(file);
    if (!fout) return false;

    fout.setf(std::ios::fixed);
    fout << std::setprecision(6);

    auto put_kv = [&](const std::string& k, const std::string& v){
        fout << k << '\t' << v << "\n";
    };
    auto put_stat_block = [&](const char* sec,
                              const std::vector<NodeId>& S,
                              const ReportStats& st){
        put_kv(sec, "BEGIN");
        fout << sec << "\tS\t"       << __join_ids(S) << "\n";
        fout << sec << "\tmean\t"    << st.mean    << "\n";
        fout << sec << "\tvar\t"     << st.var     << "\n";
        fout << sec << "\tse\t"      << st.se      << "\n";
        fout << sec << "\tci_low\t"  << st.ci_low  << "\n";
        fout << sec << "\tci_high\t" << st.ci_high << "\n";
        fout << sec << "\tM\t"       << st.M       << "\n";
        put_kv(sec, "END");
    };

    // 1) Header: tag + metadata
    put_kv("tag", tag);
    for (const auto& kv : meta) put_kv(kv.first, kv.second);

    // 2) Blank separator line
    fout << "\n";

    // 3) Section: pick
    put_stat_block("pick", S_pick, stats_pick);

    // 4) Section: UB / LB (when present)
    if (S_UB && stats_UB) {
        fout << "\n";
        put_stat_block("UB", *S_UB, *stats_UB);
    }
    if (S_LB && stats_LB) {
        fout << "\n";
        put_stat_block("LB", *S_LB, *stats_LB);
    }

    return bool(fout);
}


// ---------------------- 6) Cache header used for parameter validation ----------------------
struct CacheHeader {
    uint32_t magic   = 0x54524545; // 'TREE'
    uint32_t version = 1;
    int n = 0;          // number of nodes
    int r = 1;          // UB(r)
    // weight parameters (only one family is active at a time)
    int D = -1;               // hard cutoff radius
    double alpha = 0.0;       // soft-decay factor
    int R = -1; double eps=0; // soft truncation radius and epsilon threshold

    // Optional: grid metadata when make_grid_2d is used
    int rows = 0, cols = 0;
    // Sampling parameters for reproducibility
    uint64_t rng_seed = 0;
    int num_agents = 0;
    int K_paths_per_agent = 0;
};

// ---------------------- 7) Binary persistence for WeightsStore ----------------------
// serialize.h does not provide a std::pair specialization; store pairs as tuples.
static inline bool save_weights_binary(const std::string& file,
                                       const WeightsStore& W,
                                       const CacheHeader& hdr) {
    try {
        StreamType s;
        // Write header first
        serialize(hdr, s);
        // Candidate ids
        serialize(W.candId, s);
        // Persist neighborhoods (each entry becomes a vector of tuples)
        std::vector<std::vector<std::tuple<NodeId,double>>> tmp;
        tmp.resize(W.W.size());
        for (size_t i=0;i<W.W.size();++i) {
            tmp[i].reserve(W.W[i].vw.size());
            for (auto &p : W.W[i].vw) tmp[i].emplace_back(p.first, p.second);
        }
        serialize(tmp, s);
        return write_stream_to_file(file, s);
    } catch (...) {
        std::cerr << "[save_weights_binary] exception\n";
        return false;
    }
}

static inline bool load_weights_binary(const std::string& file,
                                       WeightsStore& W,
                                       CacheHeader& hdr_out) {
    try {
        StreamType s;
        if (!read_stream_from_file(file, s)) return false;
        StreamType::const_iterator it = s.begin();   // iterate with a const iterator
        hdr_out = deserialize<CacheHeader>(it, s.end());
        W.candId = deserialize<std::vector<NodeId>>(it, s.end());
        auto tmp = deserialize<std::vector<std::vector<std::tuple<NodeId,double>>>>(it, s.end());

        W.W.assign(tmp.size(), {});
        for (size_t i=0;i<tmp.size();++i) {
            auto &vec = W.W[i].vw;
            vec.reserve(tmp[i].size());
            for (auto &t : tmp[i]) vec.emplace_back(std::get<0>(t), std::get<1>(t));
        }
        return true;
    } catch (...) {
        std::cerr << "[load_weights_binary] exception\n";
        return false;
    }
}

// ---------------------- 8) Binary persistence for corridor statistics ----------------------
// Store individual fields instead of depending on alg.hpp definitions.

static inline bool save_corridor_binary(const std::string& file,
                                        const std::vector<int>& b_r,
                                        long long Lsum_r,
                                        const std::vector<int>& in_union,
                                        const std::vector<int>& c_total,
                                        long long Lmin1,
                                        long long Lsum_all,
                                        const CacheHeader& hdr) {
    try {
        StreamType s;
        serialize(hdr, s);
        serialize(b_r, s);
        serialize(Lsum_r, s);
        serialize(in_union, s);
        serialize(c_total, s);
        serialize(Lmin1, s);
        serialize(Lsum_all, s);
        return write_stream_to_file(file, s);
    } catch (...) {
        std::cerr << "[save_corridor_binary] exception\n";
        return false;
    }
}

static inline bool load_corridor_binary(const std::string& file,
                                        std::vector<int>& b_r,
                                        long long& Lsum_r,
                                        std::vector<int>& in_union,
                                        std::vector<int>& c_total,
                                        long long& Lmin1,
                                        long long& Lsum_all,
                                        CacheHeader& hdr_out) {
    try {
        StreamType s;
        if (!read_stream_from_file(file, s)) return false;
        StreamType::const_iterator it = s.begin();   // iterate with const access
        hdr_out  = deserialize<CacheHeader>(it, s.end());
        b_r      = deserialize<std::vector<int>>(it, s.end());
        Lsum_r   = deserialize<long long>(it, s.end());
        in_union = deserialize<std::vector<int>>(it, s.end());
        c_total  = deserialize<std::vector<int>>(it, s.end());
        Lmin1    = deserialize<long long>(it, s.end());
        Lsum_all = deserialize<long long>(it, s.end());

        return true;
    } catch (...) {
        std::cerr << "[load_corridor_binary] exception\n";
        return false;
    }
}

// ---------------------- 9) Export helpers (optional) ----------------------
static inline bool write_edge_list(const std::string& file, const Graph& G, bool undirected=true) {
    if (!ensure_parent_dir(file)) return false;
    std::ofstream fout(file);
    if (!fout) return false;
    for (int u=0; u<G.n; ++u) {
        for (auto v: G.adj[u]) {
            if (undirected && v < u) continue; // skip the mirrored half of undirected edges
            fout << u << " " << v << "\n";
        }
    }
    return bool(fout);
}

static inline bool write_candidates(const std::string& file, const std::vector<NodeId>& cands) {
    if (!ensure_parent_dir(file)) return false;
    std::ofstream fout(file);
    if (!fout) return false;
    for (auto x: cands) fout << x << "\n";
    return bool(fout);
}

static inline bool write_paths(const std::string& file, const PathSet& P, char sep='|') {
    if (!ensure_parent_dir(file)) return false;
    std::ofstream fout(file);
    if (!fout) return false;
    for (size_t i=0;i<P.paths.size();++i) {
        const auto& arr = P.paths[i];
        for (size_t j=0;j<arr.size();++j) {
            if (j) fout << " " << sep << " ";
            const auto& path = arr[j];
            for (size_t k=0;k<path.size();++k) {
                if (k) fout << ' ';
                fout << path[k];
            }
        }
        fout << "\n";
    }
    return bool(fout);
}

// ==== Obstacle I/O helpers ====
// 1) Load obstacles listed as node ids (one id per line)
static inline std::vector<NodeId> load_obstacles_nodeids(
    const std::string& file, int n, bool one_based=false, char comment='#', bool dedup=true)
{
    std::ifstream fin(file);
    std::vector<NodeId> obs;
    if (!fin) { std::cerr<<"[load_obstacles_nodeids] cannot open "<<file<<"\n"; return obs; }
    std::vector<char> seen; if (dedup) seen.assign(n,0);
    long long ok=0,oob=0,dup=0;
    std::string line;
    while (std::getline(fin,line)) {
        if (line.empty()) continue;
        if (comment && line[0]==comment) continue;
        std::istringstream ss(line);
        int x; if(!(ss>>x)) continue;
        if (one_based) --x;
        if (x<0 || x>=n) { ++oob; continue; }
        if (dedup && seen[x]) { ++dup; continue; }
        if (dedup) seen[x]=1;
        obs.push_back(x); ++ok;
    }
    std::cerr<<"[load_obstacles_nodeids] loaded="<<ok<<", oob="<<oob<<", dup="<<dup<<"\n";
    return obs;
}

// 2) Load obstacles listed as coordinates (one "r c" per line)
static inline std::vector<NodeId> load_obstacles_coords(
    const std::string& file, int rows, int cols, bool one_based=false, char comment='#', bool dedup=true)
{
    const int n = rows*cols;
    std::ifstream fin(file);
    std::vector<NodeId> obs;
    if (!fin) { std::cerr<<"[load_obstacles_coords] cannot open "<<file<<"\n"; return obs; }
    std::vector<char> seen; if (dedup) seen.assign(n,0);
    long long ok=0,oob=0,dup=0;
    std::string line;
    while (std::getline(fin,line)) {
        if (line.empty()) continue;
        if (comment && line[0]==comment) continue;
        std::istringstream ss(line);
        int r,c; if(!(ss>>r>>c)) continue;
        if (one_based) { --r; --c; }
        if (!in_bounds(r,c,rows,cols)) { ++oob; continue; }
        NodeId id = rc_to_id(r,c,cols);
        if (dedup && seen[id]) { ++dup; continue; }
        if (dedup) seen[id]=1;
        obs.push_back(id); ++ok;
    }
    std::cerr<<"[load_obstacles_coords] loaded="<<ok<<", oob="<<oob<<", dup="<<dup<<"\n";
    return obs;
}

// 3) Write obstacle node ids
static inline bool save_obstacles_nodeids(const std::string& file, const std::vector<NodeId>& obs)
{
    if (!ensure_parent_dir(file)) return false;
    std::ofstream fout(file);
    if (!fout) return false;
    for (auto id: obs) fout<<id<<"\n";
    return bool(fout);
}

// 4) Write obstacle coordinates (r c)
static inline bool save_obstacles_coords(const std::string& file,
                                         const std::vector<NodeId>& obs,
                                         [[maybe_unused]] int rows, int cols)
{
    if (!ensure_parent_dir(file)) return false;
    std::ofstream fout(file);
    if (!fout) return false;
    for (auto id: obs) {
        auto rc = id_to_rc(id, cols);
        fout<<rc.first<<" "<<rc.second<<"\n";
    }
    return bool(fout);
}
