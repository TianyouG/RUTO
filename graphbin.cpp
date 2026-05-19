#include "stdafx.h"

namespace {

#pragma pack(push, 1)
struct GraphBinHeader {
    char magic[8];
    uint32_t version;
    uint64_t n;
    uint64_t m;
    uint8_t directed;
    uint8_t reserved[7];
    uint64_t offs_size;
    uint64_t idx_size;
    uint64_t meta_len;
};
#pragma pack(pop)

static_assert(sizeof(GraphBinHeader) == 60, "GraphBinHeader must be packed to 60 bytes");

static constexpr const char kMagic[8] = { 'G', 'B', 'I', 'N', 'v', '1', '\0', '\0' };
static constexpr uint32_t kVersion = 1u;

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(ch));
                    out += buf;
                } else {
                    out.push_back(ch);
                }
        }
    }
    return out;
}

static uint64_t hash_edges_directed(const Graph& G) {
    uint64_t h = 1469598103934665603ull;
    for (int u = 0; u < G.n; ++u) {
        for (NodeId v : G.adj[u]) {
            h ^= static_cast<uint64_t>(static_cast<uint32_t>(u));
            h *= 1099511628211ull;
            h ^= static_cast<uint64_t>(static_cast<uint32_t>(v));
            h *= 1099511628211ull;
        }
    }
    return h;
}

static std::string make_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

static bool write_meta_json(const std::string& path,
                            const GraphBinMeta& meta,
                            const std::string& hash_hex) {
    std::ofstream fout(path);
    if (!fout) return false;

    std::vector<std::string> fields;
    fields.emplace_back("\"hash_edges\": \"" + json_escape(hash_hex) + "\"");
    fields.emplace_back("\"created_utc\": \"" + json_escape(meta.created_utc) + "\"");
    fields.emplace_back("\"note\": \"" + json_escape(meta.note) + "\"");
    fields.emplace_back("\"candidates_src\": \"" + json_escape(meta.candidates_src) + "\"");
    fields.emplace_back("\"blocked_source\": \"" + json_escape(meta.blocked_source) + "\"");
    fields.emplace_back("\"blocked_count\": " + std::to_string(meta.blocked_count));
    if (meta.blocked_seed_valid) {
        fields.emplace_back("\"blocked_seed\": " + std::to_string(meta.blocked_seed));
    }
    if (meta.blocked_fraction_valid) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(6) << meta.blocked_fraction;
        fields.emplace_back("\"blocked_fraction\": " + oss.str());
    }
    fields.emplace_back("\"build_cmdline\": \"" + json_escape(meta.build_cmdline) + "\"");
    fields.emplace_back("\"n\": " + std::to_string(meta.n));
    fields.emplace_back("\"m\": " + std::to_string(meta.m));

    fout << "{\n";
    for (size_t i = 0; i < fields.size(); ++i) {
        fout << "  " << fields[i];
        if (i + 1 < fields.size()) fout << ',';
        fout << "\n";
    }
    fout << "}\n";
    return static_cast<bool>(fout);
}

static std::string to_hex(uint64_t value) {
    std::ostringstream oss;
    oss << std::hex << value;
    return oss.str();
}

static bool parse_json_field(const std::string& json,
                             const std::string& key,
                             std::string& value_out) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size()) return false;
    if (json[pos] == '"') {
        ++pos;
        std::string value;
        while (pos < json.size()) {
            char ch = json[pos++];
            if (ch == '"') break;
            if (ch == '\\' && pos < json.size()) {
                char esc = json[pos++];
                switch (esc) {
                    case '"': value.push_back('"'); break;
                    case '\\': value.push_back('\\'); break;
                    case 'n': value.push_back('\n'); break;
                    case 'r': value.push_back('\r'); break;
                    case 't': value.push_back('\t'); break;
                    case 'u': {
                        if (pos + 3 < json.size()) {
                            std::string hex = json.substr(pos, 4);
                            char* endptr = nullptr;
                            long code = std::strtol(hex.c_str(), &endptr, 16);
                            if (endptr == hex.c_str() + 4) {
                                if (code >= 0 && code <= 0x7f) value.push_back(static_cast<char>(code));
                            }
                            pos += 4;
                        }
                        break;
                    }
                    default: value.push_back(esc); break;
                }
            } else {
                value.push_back(ch);
            }
        }
        value_out = value;
        return true;
    } else {
        size_t end = pos;
        while (end < json.size() && !std::isspace(static_cast<unsigned char>(json[end])) && json[end] != ',') ++end;
        value_out = json.substr(pos, end - pos);
        return true;
    }
}

} // namespace

bool write_graph_bin(const std::string& bin_path,
                     const std::string& json_path,
                     const Graph& G,
                     const std::vector<NodeId>& candidates,
                     const std::vector<char>& blocked,
                     const GraphBinMeta& meta,
                     std::string& error_out) {
    error_out.clear();
    if (!ensure_parent_dir(bin_path) || !ensure_parent_dir(json_path)) {
        error_out = "failed to create parent directories for graph bin outputs";
        return false;
    }

    const uint64_t n = static_cast<uint64_t>(G.n);
    uint64_t m = 0;
    for (const auto& nbrs : G.adj) m += static_cast<uint64_t>(nbrs.size());

    std::vector<uint64_t> offsets;
    offsets.resize(static_cast<size_t>(n) + 1, 0);
    uint64_t acc = 0;
    for (uint64_t u = 0; u < n; ++u) {
        offsets[u] = acc;
        acc += static_cast<uint64_t>(G.adj[static_cast<int>(u)].size());
    }
    offsets[n] = acc;

    std::vector<uint32_t> indices;
    indices.reserve(static_cast<size_t>(m));
    for (int u = 0; u < G.n; ++u) {
        for (NodeId v : G.adj[u]) {
            if (v < 0) {
                error_out = "negative node id encountered while writing graph bin";
                return false;
            }
            indices.push_back(static_cast<uint32_t>(v));
        }
    }

    const size_t bitset_bytes = static_cast<size_t>((n + 7ull) / 8ull);
    std::vector<uint8_t> cand_bits(bitset_bytes, 0);
    for (NodeId v : candidates) {
        if (v < 0 || static_cast<uint64_t>(v) >= n) continue;
        size_t byte = static_cast<size_t>(v) / 8;
        size_t bit = static_cast<size_t>(v) % 8;
        cand_bits[byte] |= static_cast<uint8_t>(1u << bit);
    }

    std::vector<uint8_t> blocked_bits(bitset_bytes, 0);
    if (blocked.size() == static_cast<size_t>(G.n)) {
        for (int v = 0; v < G.n; ++v) {
            if (!blocked[v]) continue;
            size_t byte = static_cast<size_t>(v) / 8;
            size_t bit = static_cast<size_t>(v) % 8;
            blocked_bits[byte] |= static_cast<uint8_t>(1u << bit);
        }
    }

    GraphBinHeader hdr{};
    std::memcpy(hdr.magic, kMagic, sizeof(kMagic));
    hdr.version = kVersion;
    hdr.n = n;
    hdr.m = m;
    hdr.directed = meta.directed ? 1 : 0;
    std::fill(std::begin(hdr.reserved), std::end(hdr.reserved), 0);
    hdr.offs_size = offsets.size();
    hdr.idx_size = indices.size();

    const uint64_t hash_value = hash_edges_directed(G);
    const std::string hash_hex = to_hex(hash_value);

    GraphBinMeta meta_out = meta;
    meta_out.n = n;
    meta_out.m = m;
    meta_out.hash_edges = hash_hex;
    meta_out.created_utc = meta.created_utc.empty() ? make_iso_timestamp() : meta.created_utc;
    meta_out.blocked_seed_valid = meta.blocked_seed_valid;
    meta_out.blocked_seed = meta.blocked_seed;
    meta_out.blocked_fraction_valid = meta.blocked_fraction_valid;
    meta_out.blocked_fraction = meta.blocked_fraction;
    if (meta_out.blocked_source.empty()) {
        meta_out.blocked_source = "<none>";
    }
    meta_out.blocked_count = static_cast<uint64_t>(std::count(blocked.begin(), blocked.end(), static_cast<char>(1)));

    std::vector<std::string> meta_fields;
    meta_fields.emplace_back("\"hash_edges\": \"" + json_escape(hash_hex) + "\"");
    meta_fields.emplace_back("\"created_utc\": \"" + json_escape(meta_out.created_utc) + "\"");
    meta_fields.emplace_back("\"note\": \"" + json_escape(meta_out.note) + "\"");
    meta_fields.emplace_back("\"candidates_src\": \"" + json_escape(meta_out.candidates_src) + "\"");
    meta_fields.emplace_back("\"blocked_source\": \"" + json_escape(meta_out.blocked_source) + "\"");
    meta_fields.emplace_back("\"blocked_count\": " + std::to_string(meta_out.blocked_count));
    if (meta_out.blocked_seed_valid) {
        meta_fields.emplace_back("\"blocked_seed\": " + std::to_string(meta_out.blocked_seed));
    }
    if (meta_out.blocked_fraction_valid) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(6) << meta_out.blocked_fraction;
        meta_fields.emplace_back("\"blocked_fraction\": " + oss.str());
    }
    meta_fields.emplace_back("\"build_cmdline\": \"" + json_escape(meta_out.build_cmdline) + "\"");
    meta_fields.emplace_back("\"n\": " + std::to_string(meta_out.n));
    meta_fields.emplace_back("\"m\": " + std::to_string(meta_out.m));

    std::ostringstream meta_stream;
    meta_stream << "{\n";
    for (size_t i = 0; i < meta_fields.size(); ++i) {
        meta_stream << "  " << meta_fields[i];
        if (i + 1 < meta_fields.size()) meta_stream << ',';
        meta_stream << "\n";
    }
    meta_stream << "}\n";
    const std::string meta_json = meta_stream.str();
    hdr.meta_len = static_cast<uint64_t>(meta_json.size());

    std::ofstream fout(bin_path, std::ios::binary);
    if (!fout) {
        error_out = "failed to open graph bin for writing: " + bin_path;
        return false;
    }
    fout.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    fout.write(reinterpret_cast<const char*>(offsets.data()), static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
    fout.write(reinterpret_cast<const char*>(indices.data()), static_cast<std::streamsize>(indices.size() * sizeof(uint32_t)));
    fout.write(reinterpret_cast<const char*>(cand_bits.data()), static_cast<std::streamsize>(cand_bits.size()));
    fout.write(reinterpret_cast<const char*>(blocked_bits.data()), static_cast<std::streamsize>(blocked_bits.size()));
    fout.write(meta_json.data(), static_cast<std::streamsize>(meta_json.size()));
    if (!fout) {
        error_out = "failed while writing graph bin payload";
        return false;
    }

    if (!write_meta_json(json_path, meta_out, hash_hex)) {
        error_out = "failed to write graph bin JSON metadata";
        return false;
    }

    return true;
}

bool read_graph_bin(const std::string& bin_path,
                    Graph& G,
                    std::vector<NodeId>& candidates,
                    std::vector<char>& blocked,
                    GraphBinMeta* meta_out,
                    std::string& error_out) {
    error_out.clear();
    std::ifstream fin(bin_path, std::ios::binary);
    if (!fin) {
        error_out = "cannot open graph bin: " + bin_path;
        return false;
    }

    GraphBinHeader hdr{};
    fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!fin) {
        error_out = "graph bin header read failed";
        return false;
    }
    if (std::memcmp(hdr.magic, kMagic, sizeof(kMagic)) != 0) {
        error_out = "graph bin magic mismatch";
        return false;
    }
    if (hdr.version != kVersion) {
        error_out = "graph bin version unsupported";
        return false;
    }

    const uint64_t n = hdr.n;
    const uint64_t m = hdr.m;
    const size_t bitset_bytes = static_cast<size_t>((n + 7ull) / 8ull);

    std::vector<uint64_t> offsets;
    offsets.resize(static_cast<size_t>(hdr.offs_size));
    fin.read(reinterpret_cast<char*>(offsets.data()), static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
    if (!fin) {
        error_out = "graph bin offsets read failed";
        return false;
    }
    if (offsets.size() != static_cast<size_t>(n) + 1) {
        error_out = "graph bin offsets size mismatch";
        return false;
    }

    std::vector<uint32_t> indices;
    indices.resize(static_cast<size_t>(hdr.idx_size));
    fin.read(reinterpret_cast<char*>(indices.data()), static_cast<std::streamsize>(indices.size() * sizeof(uint32_t)));
    if (!fin) {
        error_out = "graph bin indices read failed";
        return false;
    }

    if (indices.size() != static_cast<size_t>(m)) {
        error_out = "graph bin indices size mismatch";
        return false;
    }

    std::vector<uint8_t> cand_bits(bitset_bytes, 0);
    fin.read(reinterpret_cast<char*>(cand_bits.data()), static_cast<std::streamsize>(cand_bits.size()));
    if (!fin) {
        error_out = "graph bin candidate bitset read failed";
        return false;
    }

    std::vector<uint8_t> blocked_bits(bitset_bytes, 0);
    fin.read(reinterpret_cast<char*>(blocked_bits.data()), static_cast<std::streamsize>(blocked_bits.size()));
    if (!fin) {
        error_out = "graph bin blocked bitset read failed";
        return false;
    }

    std::string meta_json;
    meta_json.resize(static_cast<size_t>(hdr.meta_len));
    fin.read(meta_json.data(), static_cast<std::streamsize>(meta_json.size()));
    if (!fin) {
        error_out = "graph bin meta section read failed";
        return false;
    }

    G = Graph(static_cast<int>(n));
    for (uint64_t u = 0; u < n; ++u) {
        uint64_t begin = offsets[u];
        uint64_t end = offsets[u + 1];
        if (end < begin || end > indices.size()) {
            error_out = "graph bin offsets corrupt";
            return false;
        }
        for (uint64_t idx = begin; idx < end; ++idx) {
            uint32_t v = indices[static_cast<size_t>(idx)];
            if (v >= n) {
                error_out = "graph bin neighbor out of range";
                return false;
            }
            G.add_edge(static_cast<NodeId>(u), static_cast<NodeId>(v), /*undirected=*/false);
        }
    }

    candidates.clear();
    for (uint64_t v = 0; v < n; ++v) {
        size_t byte = static_cast<size_t>(v) / 8;
        size_t bit = static_cast<size_t>(v) % 8;
        if (byte < cand_bits.size() && (cand_bits[byte] & (1u << bit))) {
            candidates.push_back(static_cast<NodeId>(v));
        }
    }

    blocked.assign(static_cast<size_t>(n), 0);
    for (uint64_t v = 0; v < n; ++v) {
        size_t byte = static_cast<size_t>(v) / 8;
        size_t bit = static_cast<size_t>(v) % 8;
        if (byte < blocked_bits.size() && (blocked_bits[byte] & (1u << bit))) {
            blocked[static_cast<size_t>(v)] = 1;
        }
    }

    if (meta_out) {
        meta_out->n = n;
        meta_out->m = m;
        meta_out->directed = hdr.directed != 0;
        meta_out->json_raw = meta_json;
        parse_json_field(meta_json, "hash_edges", meta_out->hash_edges);
        parse_json_field(meta_json, "created_utc", meta_out->created_utc);
        parse_json_field(meta_json, "note", meta_out->note);
        parse_json_field(meta_json, "candidates_src", meta_out->candidates_src);
        if (!parse_json_field(meta_json, "blocked_source", meta_out->blocked_source)) {
            parse_json_field(meta_json, "blocked_src", meta_out->blocked_source);
        }
        if (meta_out->blocked_source.empty()) {
            meta_out->blocked_source = "<none>";
        }
        std::string blocked_count_str;
        if (parse_json_field(meta_json, "blocked_count", blocked_count_str)) {
            meta_out->blocked_count = static_cast<uint64_t>(std::strtoull(blocked_count_str.c_str(), nullptr, 10));
        } else {
            meta_out->blocked_count = static_cast<uint64_t>(std::count(blocked.begin(), blocked.end(), static_cast<char>(1)));
        }
        std::string blocked_seed_str;
        if (parse_json_field(meta_json, "blocked_seed", blocked_seed_str)) {
            meta_out->blocked_seed = std::strtoll(blocked_seed_str.c_str(), nullptr, 10);
            meta_out->blocked_seed_valid = true;
        } else {
            meta_out->blocked_seed_valid = false;
        }
        std::string blocked_fraction_str;
        if (parse_json_field(meta_json, "blocked_fraction", blocked_fraction_str)) {
            meta_out->blocked_fraction = std::strtod(blocked_fraction_str.c_str(), nullptr);
            meta_out->blocked_fraction_valid = true;
        } else {
            meta_out->blocked_fraction_valid = false;
        }
        parse_json_field(meta_json, "build_cmdline", meta_out->build_cmdline);
    }

    return true;
}
