#pragma once
#include "flockmtl/core/common.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "flockmtl/functions/batch_response_builder.hpp"

#include <unordered_map>
#include <string>
#include <memory>
#include <shared_mutex>
#include <mutex>

namespace flockmtl {

enum CacheableFunction {LlmComplete};

// Convert CacheableFunction enum to string
std::string to_string(CacheableFunction function);

// Structure to hold cached results in buffer blocks
struct CacheEntry {
    duckdb::shared_ptr<duckdb::BlockHandle> block_handle_pointer;
    idx_t string_size;     // Size of the stored string in bytes
    
    CacheEntry(duckdb::shared_ptr<duckdb::BlockHandle> handle, idx_t size)
        : block_handle_pointer(std::move(handle)), string_size(size) {}
    
    // Get the result from the buffer
    std::string get_result(duckdb::BufferManager &buffer_manager);
    
    // Store result in the buffer
    static CacheEntry create(duckdb::BufferManager &buffer_manager, const std::string& result);
};

// Individual cache table for a specific provider/function combination
class CacheTable {
private:
    /// provides a mapping between each prompt and a mapping of its tuples entries/results
    std::unordered_map<std::string, std::unordered_map<std::string, std::unique_ptr<CacheEntry>>> cache_map;
    mutable std::shared_mutex cache_mutex;

public:
    CacheTable() = default;

    // Store result in cache using buffer manager
    void put(const std::string& prompt, const std::string& tuple, const std::string& result, duckdb::BufferManager &buffer_manager);
    
    // Retrieve result from cache (returns nullptr if not found)
    std::unique_ptr<std::string> get(const std::string& prompt, const std::string& tuple, duckdb::BufferManager &buffer_manager);
    
    // Check if key exists
    bool contains_prompt(const std::string& prompt);
    bool contains(const std::string& prompt, const std::string& tuple);

    // Get cache statistics
    size_t size() const;
    void clear();
};

class CacheManager {
private:
    static std::unordered_map<std::string, std::unique_ptr<CacheTable>> cache_tables;
    // Maps model_name to the provider. WARNING: If the user changes the model using the same name, this won't update.
    static std::unordered_map<std::string, std::string> model_providers;
    static std::shared_mutex tables_mutex;
    static std::shared_mutex providers_mutex;
    
    // Automatic cleanup initialization
    static std::once_flag cleanup_initialized;
    static void register_cleanup_callback(duckdb::ExpressionState& state);
    
    // Inline initialization check for performance
    static inline void ensure_cleanup_initialized(duckdb::ExpressionState& state) {
        std::call_once(cleanup_initialized, register_cleanup_callback, std::ref(state));
    }
    
    // Helper methods
    /// Extracts each prompt/tuple combination from the aggregate of inputs so they can be cached separately
    static std::pair<std::string, std::vector<std::string>> get_prompt_and_tuples(const duckdb::DataChunk& args);
    static CacheTable* get_or_create_table(const std::string& table_name);

    // first is the provider, second is the model
    static inline std::pair<std::string, std::string> get_provider_and_model(const duckdb::DataChunk& args) {
        nlohmann::json model_details_json = CastVectorOfStructsToJson(args.data[0], 1)[0];
        std::string model_name = model_details_json["model_name"].get<std::string>();
        std::shared_lock<std::shared_mutex> read_lock(providers_mutex);

        auto it = model_providers.find(model_name);

        // cached, no need for database query, faster
        if (it != model_providers.end()) {
            return {it->second, model_name};
        }

        read_lock.unlock();

        // retrieve provider name using database query and cache it
        Model model(model_details_json);
        std::string provider_name = model.GetModelDetails().provider_name;

        std::unique_lock<std::shared_mutex> write_lock(providers_mutex);
        // entry could have changed while we didn't have the lock, we need to check again
        if (model_providers.find(model_name) == model_providers.end()) {
            model_providers[model_name] = provider_name;
        } else {
            // if model_name was added to model_providers, we wanna use this information for consistency
            provider_name = model_providers[model_name];
        }
        
        return {provider_name, model_name};
    }

public:
    // Get table name for a specific provider/model/function combination
    static std::string get_table_name(const std::string& provider, const std::string& model, CacheableFunction function);
    
    // Store function result in cache
    static void store_results(CacheableFunction function, const duckdb::DataChunk& args,
                              const std::vector<std::string>& results, duckdb::ExpressionState& state);
    
    // Retrieve cached result if available
    static std::vector<std::unique_ptr<std::string>>
    get_cached_results(CacheableFunction function, const duckdb::DataChunk& args, duckdb::ExpressionState& state);
    
    // Check if result is cached
    static std::vector<bool> is_cached(CacheableFunction function, const duckdb::DataChunk& args);
    static bool all_is_cached(CacheableFunction function, const duckdb::DataChunk& args);
    
    // Cache management methods
    static void clear_cache(const std::string& provider, const std::string& model, CacheableFunction function);
    static void clear_all_caches();
    
    // Stats
    static size_t get_cache_size(const std::string& provider, const std::string& model, CacheableFunction function);
    static size_t get_total_cache_size();
};

} // namespace flockmtl