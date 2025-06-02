#pragma once
#include "flockmtl/core/common.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <shared_mutex>

namespace flockmtl {

enum CacheableFunction {LlmComplete};

// Convert CacheableFunction enum to string
std::string to_string(CacheableFunction function);

// Structure to hold cached results in buffer blocks
struct CacheEntry {
    duckdb::BufferHandle buffer_handle;
    idx_t string_size;     // Size of the stored string in bytes
    
    CacheEntry(duckdb::BufferHandle handle, idx_t size) 
        : buffer_handle(std::move(handle)), string_size(size) {}
    
    // Get the result from the buffer
    std::string get_result() const;
    
    // Store result in the buffer
    static CacheEntry create(duckdb::BufferManager &buffer_manager, const std::string& result);
};

// Individual cache table for a specific provider/function combination
class CacheTable {
private:
    std::unordered_map<std::string, std::unique_ptr<CacheEntry>> cache_map;
    mutable std::shared_mutex cache_mutex;

public:
    CacheTable() = default;

    // Store result in cache using buffer manager
    void put(const std::string& key, const std::string& result, duckdb::BufferManager &buffer_manager);
    
    // Retrieve result from cache (returns nullptr if not found)
    std::unique_ptr<std::string> get(const std::string& key);
    
    // Check if key exists
    bool contains(const std::string& key);
    
    // Get cache statistics
    size_t size() const;
    void clear();
};

class CacheManager {
private:
    static std::unordered_map<std::string, std::unique_ptr<CacheTable>> cache_tables;
    static std::shared_mutex tables_mutex;
    
    // Helper methods
    static std::string serialize_data_chunk(const duckdb::DataChunk& args);
    static CacheTable* get_or_create_table(const std::string& table_name);

public:
    // Get table name for a specific provider/model/function combination
    static std::string get_table_name(const std::string& provider, const std::string& model, CacheableFunction function);
    
    // Store function result in cache
    static void store_result(const std::string& provider, const std::string& model, CacheableFunction function,
                             const duckdb::DataChunk& args, const std::string& result,
                             duckdb::ExpressionState& state);
    
    // Retrieve cached result if available
    static std::unique_ptr<std::string> get_cached_result(const std::string& provider, 
                                                         const std::string& model, CacheableFunction function,
                                                         const duckdb::DataChunk& args);
    
    // Check if result is cached
    static bool is_cached(const std::string& provider, const std::string& model, CacheableFunction function,
                         const duckdb::DataChunk& args);
    
    // Cache management methods
    static void clear_cache(const std::string& provider, const std::string& model, CacheableFunction function);
    static void clear_all_caches();
    
    // Stats
    static size_t get_cache_size(const std::string& provider, const std::string& model, CacheableFunction function);
    static size_t get_total_cache_size();
};

} // namespace flockmtl