#pragma once

#include "flockmtl/functions/scalar/scalar.hpp"

namespace flockmtl {

class LlmEmbedding : public ScalarFunctionBase {
public:
    static void ValidateArguments(duckdb::DataChunk& args);
    static std::vector<duckdb::vector<duckdb::Value>> Operation(duckdb::DataChunk& args);
    static void Execute(duckdb::DataChunk& args, duckdb::ExpressionState& state, duckdb::Vector& result);
private:
    static void CacheListOfEmbeddings(const std::string& filename, const std::vector<std::string>& prompts,
                                      const std::vector<std::vector<double>>& embeddings);
    static std::vector<std::string> GetInputs(duckdb::DataChunk& args);
    static bool IsCached(const std::string& filename, const std::vector<std::string>& prompts);
    static std::vector<duckdb::vector<duckdb::Value>> GetEmbeddingsFromCache(const std::string& filename, const std::vector<std::string> prompts);
};

} // namespace flockmtl
