#pragma once

#include "flockmtl/functions/scalar/scalar.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace flockmtl {

class FusionRRF : public ScalarFunctionBase {
public:
    static void ValidateArguments(duckdb::DataChunk& args);
    static std::vector<int> Operation(duckdb::DataChunk& args);
    static void Execute(duckdb::DataChunk& args, duckdb::ExpressionState& state, duckdb::Vector& result);
};

} // namespace flockmtl
