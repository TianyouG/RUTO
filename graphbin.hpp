#pragma once
#include "stdafx.h"

struct GraphBinMeta {
    uint64_t n = 0;
    uint64_t m = 0;
    bool directed = false;
    std::string hash_edges;
    std::string created_utc;
    std::string note;
    std::string candidates_src;
    std::string blocked_source;
    uint64_t blocked_count = 0;
    bool blocked_seed_valid = false;
    long long blocked_seed = 0;
    bool blocked_fraction_valid = false;
    double blocked_fraction = 0.0;
    std::string build_cmdline;
    std::string json_raw;
};

bool write_graph_bin(const std::string& bin_path,
                     const std::string& json_path,
                     const Graph& G,
                     const std::vector<NodeId>& candidates,
                     const std::vector<char>& blocked,
                     const GraphBinMeta& meta,
                     std::string& error_out);

bool read_graph_bin(const std::string& bin_path,
                    Graph& G,
                    std::vector<NodeId>& candidates,
                    std::vector<char>& blocked,
                    GraphBinMeta* meta_out,
                    std::string& error_out);
