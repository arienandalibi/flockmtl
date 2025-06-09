#include "flockmtl/caching/cache_manager.hpp"
#include "flockmtl/functions/batch_response_builder.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/common/enums/memory_tag.hpp"
#include "duckdb/planner/extension_callback.hpp"
#include <cstring>
#include <sstream>
#include <mutex>
#include <tuple>

namespace flockmtl {

// Static member definitions
std::unordered_map<std::string, std::unique_ptr<CacheTable>> CacheManager::cache_tables;
std::unordered_map<std::string, std::string> CacheManager::model_providers;
std::shared_mutex CacheManager::tables_mutex;
std::once_flag CacheManager::cleanup_initialized;

// Extension callback for cleanup
class CacheCleanupCallback : public duckdb::ExtensionCallback {
public:
    ~CacheCleanupCallback() override {
        // Cleanup when the callback itself is destroyed (during database shutdown)
        CacheManager::clear_all_caches();
    }
};

// Initialize the cleanup callback (called via std::call_once)
void CacheManager::register_cleanup_callback(duckdb::ExpressionState& state) {
    auto &db = duckdb::DatabaseInstance::GetDatabase(state.GetContext());
    auto &config = duckdb::DBConfig::GetConfig(db);
    config.extension_callbacks.push_back(std::make_unique<CacheCleanupCallback>());
}

// Convert CacheableFunction enum to string
std::string to_string(CacheableFunction function) {
    switch (function) {
        case LlmComplete: return "LlmComplete";
        default: return "Unknown";
    }
}

// CacheEntry implementation
std::string CacheEntry::get_result(duckdb::BufferManager &buffer_manager) {
    if (string_size == 0) {
        return "";
    }
    
    // Pin the block to get access to the buffer
    auto buffer_handle = buffer_manager.Pin(block_handle_pointer);
    
    // Read the data from the buffer
    const char* buffer_data = reinterpret_cast<const char*>(buffer_handle.Ptr());
    std::string result(buffer_data, string_size);
    
    // No need to unpin the block once again, it happens automatically when buffer_handle goes out of scope
    
    return result;
}

CacheEntry CacheEntry::create(duckdb::BufferManager &buffer_manager, const std::string& result) {
    idx_t buffer_size = result.size();
    
    // Allocate buffer using DuckDB's buffer manager
    auto buffer_handle = buffer_manager.Allocate(duckdb::MemoryTag::EXTENSION, buffer_size, false);
    
    if (buffer_size > 0) {
        char* buffer_data = reinterpret_cast<char*>(buffer_handle.Ptr());
        std::memcpy(buffer_data, result.data(), buffer_size);
    }
    
    // Get the block handle from the buffer handle
    auto block_handle = buffer_handle.GetBlockHandle();
    
    // No need to unpin the block handle because it is automatically managed. Unpin happens when it goes out of scope
    
    return CacheEntry(std::move(block_handle), buffer_size);
}

// CacheTable implementation
void CacheTable::put(const std::string& prompt, const std::string& tuple, const std::string& result, duckdb::BufferManager &buffer_manager) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex);
    
    // Create cache entry using buffer manager
    auto entry = std::make_unique<CacheEntry>(CacheEntry::create(buffer_manager, result));
    cache_map[prompt][tuple] = std::move(entry);
}

std::unique_ptr<std::string> CacheTable::get(const std::string& prompt, const std::string& tuple, duckdb::BufferManager &buffer_manager) {
    std::shared_lock<std::shared_mutex> lock(cache_mutex);
    
    auto it = cache_map.find(prompt);
    if (it == cache_map.end()) {
        return nullptr;
    }

    auto it2 = cache_map[prompt].find(tuple);
    if (it2 == cache_map[prompt].end()) {
        return nullptr;
    }
    
    // Get result from buffer using pin/unpin pattern
    auto result = it2->second->get_result(buffer_manager);
    return std::make_unique<std::string>(std::move(result));
}

bool CacheTable::contains_prompt(const std::string& prompt) {
    std::shared_lock<std::shared_mutex> lock(cache_mutex);
    return cache_map.find(prompt) != cache_map.end();
}

bool CacheTable::contains(const std::string& prompt, const std::string& tuple) {
    std::shared_lock<std::shared_mutex> lock(cache_mutex);
    return cache_map.find(prompt) != cache_map.end() ? cache_map[prompt].find(tuple) != cache_map[prompt].end() : false;
}

size_t CacheTable::size() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex);
    return cache_map.size();
}

void CacheTable::clear() {
    std::unique_lock<std::shared_mutex> lock(cache_mutex);
    cache_map.clear();
}

// CacheManager implementation
std::string CacheManager::get_table_name(const std::string& provider, const std::string& model, CacheableFunction function) {
    return provider + "_" + model + "_" + to_string(function);
}

// Extracts each prompt/tuple combination from the aggregate of inputs so they can be cached separately
std::pair<std::string, std::vector<std::string>> CacheManager::get_prompt_and_tuples(const duckdb::DataChunk& args) {
    std::vector<std::string> tuples_strings;
    
    // Extract prompt details from args.data[1]
    const auto prompt_details_json = CastVectorOfStructsToJson(args.data[1], 1)[0];
    const std::string prompt = PromptManager::CreatePromptDetails(prompt_details_json).prompt;

    
    // Extract input tuples from args.data[2] if present
    if (args.ColumnCount() > 2) {
        auto tuples_json = CastVectorOfStructsToJson(args.data[2], args.size());
        tuples_strings.reserve(tuples_json.size());

        // populate the tuples_strings vector. We keep the whole json because the key can change as well, which might change semantics
        std::transform(tuples_json.begin(), tuples_json.end(),
               std::back_inserter(tuples_strings),
               [](const nlohmann::json& json_obj) -> std::string {
                   return json_obj.dump();
               });
    }
    
    return {prompt, tuples_strings};
}

CacheTable* CacheManager::get_or_create_table(const std::string& table_name) {
    std::unique_lock<std::shared_mutex> lock(tables_mutex);

    auto it = cache_tables.find(table_name);
    if (it == cache_tables.end()) {
        cache_tables[table_name] = std::make_unique<CacheTable>();
        return cache_tables[table_name].get();
    }

    return it->second.get();
}

void CacheManager::store_results(CacheableFunction function, const duckdb::DataChunk& args,
                                 const std::vector<std::string>& results, duckdb::ExpressionState& state) {
    // Ensure cleanup callback is initialized (inline check)
    ensure_cleanup_initialized(state);

    auto [provider, model] = get_provider_and_model(args);

    std::string table_name = get_table_name(provider, model, function);
    auto [prompt, tuples] = get_prompt_and_tuples(args);

    // Get BufferManager from the state's context
    auto& buffer_manager = duckdb::BufferManager::GetBufferManager(state.GetContext());

    CacheTable* table = get_or_create_table(table_name);

    // if there are no tuples, there is only a prompt and so there should only be one result
    if (tuples.empty()) {
        if (results.size() != 1) {
            throw std::runtime_error("CacheManager::store_results: Only expected 1 result for 1 prompt");
        }
        table->put(prompt, "", results[0], buffer_manager);
    } else {
        if (results.size() != tuples.size()) {
            const std::string err_msg = duckdb_fmt::format("CacheManager::store_results: There should the same number "
                                                           "of tuples as results. Found {} tuples and {} results",
                                                           tuples.size(), results.size());
            throw std::runtime_error(err_msg);
        }
        for (size_t i = 0; i < tuples.size(); ++i) {
            table->put(prompt, tuples[i], results[i], buffer_manager);
            std::cout << "Testing, storing result." << std::endl;
            std::cout << "Prompt: " << prompt << " ; Tuple: " << tuples[i] << std::endl;
            std::cout << "Result: " << results[i] << std::endl;
        }
    }
}

std::vector<std::unique_ptr<std::string>> CacheManager::get_cached_results(CacheableFunction function,
                                                                           const duckdb::DataChunk& args,
                                                                           duckdb::ExpressionState& state) {
    // Ensure cleanup callback is initialized (inline check)
    ensure_cleanup_initialized(state);

    auto [provider, model] = get_provider_and_model(args);

    std::string table_name = get_table_name(provider, model, function);

    auto [prompt, tuples] = get_prompt_and_tuples(args);

    std::shared_lock<std::shared_mutex> lock(tables_mutex);
    auto it = cache_tables.find(table_name);
    if (it == cache_tables.end()) {
        // Table doesn't exist, no cached results
        if (tuples.empty()) {
            return std::vector<std::unique_ptr<std::string>>(1);
        } else {
            return std::vector<std::unique_ptr<std::string>>(tuples.size());
        }
    }

    // Get BufferManager from the state's context
    auto& buffer_manager = duckdb::BufferManager::GetBufferManager(state.GetContext());
    std::vector<std::unique_ptr<std::string>> results;

    if (tuples.empty()) {
        results.push_back(it->second->get(prompt, "", buffer_manager));
    } else {
        results.reserve(tuples.size());
        for (size_t i = 0; i < tuples.size(); ++i) {
            results.push_back(it->second->get(prompt, tuples[i], buffer_manager));
        }
    }
    
    return results;
}

std::vector<bool> CacheManager::is_cached(CacheableFunction function, const duckdb::DataChunk& args) {
    auto [provider, model] = get_provider_and_model(args);

    std::string table_name = get_table_name(provider, model, function);

    auto [prompt, tuples] = get_prompt_and_tuples(args);

    std::shared_lock<std::shared_mutex> lock(tables_mutex);
    auto it = cache_tables.find(table_name);
    if (it == cache_tables.end()) {
        if (tuples.empty()) {
            return std::vector<bool>(1, false);
        } else {
            return std::vector<bool>(tuples.size(), false); // Table doesn't exist
        }
    }

    std::vector<bool> results;

    if (tuples.empty()) {
        results.push_back(it->second->contains(prompt, ""));
    } else {
        results.reserve(tuples.size());
        for (size_t i = 0; i < tuples.size(); ++i) {
            results.push_back(it->second->contains(prompt, tuples[i]));
        }
    }

    return results;
}

bool CacheManager::all_is_cached(CacheableFunction function, const duckdb::DataChunk& args) {
    std::vector<bool> cached_vec = CacheManager::is_cached(function, args);
    for (const bool is_cached : cached_vec) {
        if (!is_cached) {
            return false;
        }
    }
    return true;
}

// No need to mark blocks as can_destroy = True, the fact that no pointers to them exist anymore is enough to delete them (destructor called)
void CacheManager::clear_cache(const std::string& provider, const std::string& model, CacheableFunction function) {
    std::string table_name = get_table_name(provider, model, function);
    
    std::shared_lock<std::shared_mutex> lock(tables_mutex);
    auto it = cache_tables.find(table_name);
    if (it != cache_tables.end()) {
        it->second->clear();
    }
}

// No need to mark blocks as can_destroy = True, the fact that no pointers to them exist anymore is enough to delete them (destructor called)
void CacheManager::clear_all_caches() {
    std::unique_lock<std::shared_mutex> lock(tables_mutex);
    cache_tables.clear();
}

size_t CacheManager::get_cache_size(const std::string& provider, const std::string& model, CacheableFunction function) {
    std::string table_name = get_table_name(provider, model, function);
    
    std::shared_lock<std::shared_mutex> lock(tables_mutex);
    auto it = cache_tables.find(table_name);
    if (it != cache_tables.end()) {
        return it->second->size();
    }
    return 0;
}

size_t CacheManager::get_total_cache_size() {
    std::shared_lock<std::shared_mutex> lock(tables_mutex);
    size_t total = 0;
    for (const auto& [name, table] : cache_tables) {
        total += table->size();
    }
    return total;
}

} // namespace flockmtl
