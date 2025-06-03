#include "flockmtl/caching/cache_manager.hpp"
#include "flockmtl/functions/batch_response_builder.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/common/enums/memory_tag.hpp"
#include <cstring>
#include <sstream>

namespace flockmtl {

// Static member definitions
std::unordered_map<std::string, std::unique_ptr<CacheTable>> CacheManager::cache_tables;
std::shared_mutex CacheManager::tables_mutex;

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
    
    // Unpin the block when done
    buffer_manager.Unpin(block_handle_pointer);
    
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
    
    // Unpin the buffer since we're storing the block handle
    buffer_manager.Unpin(block_handle);
    
    return CacheEntry(std::move(block_handle), buffer_size);
}

// CacheTable implementation
void CacheTable::put(const std::string& key, const std::string& result, duckdb::BufferManager &buffer_manager) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex);
    
    // Create cache entry using buffer manager
    auto entry = std::make_unique<CacheEntry>(CacheEntry::create(buffer_manager, result));
    cache_map[key] = std::move(entry);
}

std::unique_ptr<std::string> CacheTable::get(const std::string& key, duckdb::BufferManager &buffer_manager) {
    std::shared_lock<std::shared_mutex> lock(cache_mutex);
    
    auto it = cache_map.find(key);
    if (it == cache_map.end()) {
        return nullptr;
    }
    
    // Get result from buffer using pin/unpin pattern
    auto result = it->second->get_result(buffer_manager);
    return std::make_unique<std::string>(std::move(result));
}

bool CacheTable::contains(const std::string& key) {
    std::shared_lock<std::shared_mutex> lock(cache_mutex);
    return cache_map.find(key) != cache_map.end();
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

// TODO: Fix this to extract each prompt/data combination from the aggregate of inputs so they can be cached separately
std::string CacheManager::serialize_data_chunk(const duckdb::DataChunk& args) {
    std::ostringstream oss;
    
    // Extract model details from args.data[0]
    if (args.ColumnCount() > 0) {
        auto model_details_json = CastVectorOfStructsToJson(args.data[0], 1)[0];
        oss << "model:" << model_details_json.dump() << ";";
    }
    std::cout << "Testing, after model: " << oss.str() << std::endl;
    
    // Extract prompt details from args.data[1]
    if (args.ColumnCount() > 1) {
        auto prompt_details_json = CastVectorOfStructsToJson(args.data[1], 1)[0];
        oss << "prompt:" << prompt_details_json.dump() << ";";
    }
    std::cout << "Testing, after prompt: " << oss.str() << std::endl;
    
    // Extract input tuples from args.data[2] if present
    if (args.ColumnCount() > 2) {
        auto tuples = CastVectorOfStructsToJson(args.data[2], args.size());
        oss << "tuples:";
        for (size_t i = 0; i < tuples.size(); i++) {
            oss << tuples[i].dump();
            if (i < tuples.size() - 1) {
                oss << ",";
            }
        }
        oss << ";";
        std::cout << "Testing, after input tuples: " << oss.str() << std::endl;
    }
    
    return oss.str();
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

void CacheManager::store_result(const std::string& provider, const std::string& model, CacheableFunction function,
                                const duckdb::DataChunk& args, const std::string& result,
                                duckdb::ExpressionState& state) {
    std::string table_name = get_table_name(provider, model, function);
    std::string key = serialize_data_chunk(args);
    
    // Get BufferManager from the state's context
    auto& buffer_manager = duckdb::BufferManager::GetBufferManager(state.GetContext());
    
    CacheTable* table = get_or_create_table(table_name);
    std::cout << "Testing, storing result." << std::endl;
    std::cout << "Key: " << key << std::endl;
    std::cout << "Result: " << result << std::endl;
    table->put(key, result, buffer_manager);
}

std::unique_ptr<std::string> CacheManager::get_cached_result(const std::string& provider,
                                                            const std::string& model, CacheableFunction function,
                                                            const duckdb::DataChunk& args,
                                                            duckdb::ExpressionState& state) {
    std::string table_name = get_table_name(provider, model, function);
    
    {
        std::shared_lock<std::shared_mutex> lock(tables_mutex);
        auto it = cache_tables.find(table_name);
        if (it == cache_tables.end()) {
            return nullptr; // Table doesn't exist, no cached result
        }
        
        std::string key = serialize_data_chunk(args);
        
        // Get BufferManager from the state's context
        auto& buffer_manager = duckdb::BufferManager::GetBufferManager(state.GetContext());
        
        return it->second->get(key, buffer_manager);
    }
}

bool CacheManager::is_cached(const std::string& provider, const std::string& model, CacheableFunction function,
                           const duckdb::DataChunk& args) {
    std::string table_name = get_table_name(provider, model, function);
    
    {
        std::shared_lock<std::shared_mutex> lock(tables_mutex);
        auto it = cache_tables.find(table_name);
        if (it == cache_tables.end()) {
            return false; // Table doesn't exist
        }
        
        std::string key = serialize_data_chunk(args);
        
        return it->second->contains(key);
    }
}

//TODO: Mark all stored cache entries as can_destroy = True before clearing
void CacheManager::clear_cache(const std::string& provider, const std::string& model, CacheableFunction function) {
    std::string table_name = get_table_name(provider, model, function);
    
    std::shared_lock<std::shared_mutex> lock(tables_mutex);
    auto it = cache_tables.find(table_name);
    if (it != cache_tables.end()) {
        it->second->clear();
    }
}

//TODO: Mark all stored cache entries as can_destroy = True before clearing
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
