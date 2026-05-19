#include "stdafx.h"

bool parse_grid_spec(const std::string& spec,
                     GridBuildSpec& out,
                     std::string& error_out) {
    error_out.clear();
    GridBuildSpec parsed;
    parsed.rows = 0;
    parsed.cols = 0;
    parsed.diag = false;
    parsed.torus = false;

    size_t start = 0;
    while (start < spec.size()) {
        size_t comma = spec.find(',', start);
        std::string token = (comma == std::string::npos) ? spec.substr(start)
                                                         : spec.substr(start, comma - start);
        start = (comma == std::string::npos) ? spec.size() : comma + 1;
        if (token.empty()) continue;
        size_t eq = token.find('=');
        if (eq == std::string::npos) {
            error_out = "missing '=' in --make_grid spec segment: " + token;
            return false;
        }
        std::string key = token.substr(0, eq);
        std::string value = token.substr(eq + 1);
        auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
        key.erase(std::remove_if(key.begin(), key.end(), is_space), key.end());
        value.erase(std::remove_if(value.begin(), value.end(), is_space), value.end());
        if (key == "rows" || key == "cols") {
            int parsed_value = 0;
            try {
                parsed_value = std::stoi(value);
            } catch (const std::exception&) {
                error_out = "invalid integer in --make_grid spec: " + value;
                return false;
            }
            if (key == "rows") parsed.rows = parsed_value;
            else parsed.cols = parsed_value;
        } else if (key == "diag") {
            parsed.diag = (value == "1" || value == "true" || value == "True");
        } else if (key == "torus") {
            parsed.torus = (value == "1" || value == "true" || value == "True");
        } else {
            error_out = "unknown key in --make_grid spec: " + key;
            return false;
        }
    }

    if (parsed.rows <= 0 || parsed.cols <= 0) {
        error_out = "--make_grid requires positive rows and cols";
        return false;
    }

    out = parsed;
    return true;
}

Graph build_grid_graph(const GridBuildSpec& spec) {
    return make_grid_2d(spec.rows, spec.cols,
                        /*undirected=*/true,
                        spec.diag,
                        spec.torus,
                        spec.torus);
}

bool write_grid_edge_list(const GridBuildSpec& spec,
                          const std::string& path,
                          std::string& error_out) {
    error_out.clear();
    Graph G = build_grid_graph(spec);
    if (!ensure_parent_dir(path)) {
        error_out = "failed to create directories for grid edge list";
        return false;
    }
    std::ofstream fout(path);
    if (!fout) {
        error_out = "cannot open output file for grid edges: " + path;
        return false;
    }
    std::set<std::pair<NodeId, NodeId>> seen;
    for (int u = 0; u < G.n; ++u) {
        for (NodeId v : G.adj[u]) {
            if (v < 0 || v >= G.n) continue;
            NodeId a = std::min(u, static_cast<int>(v));
            NodeId b = std::max(u, static_cast<int>(v));
            if (a == b) continue;
            if (seen.insert({a, b}).second) {
                fout << a << ' ' << b << '\n';
            }
        }
    }
    if (!fout) {
        error_out = "failed while writing grid edges";
        return false;
    }
    return true;
}
