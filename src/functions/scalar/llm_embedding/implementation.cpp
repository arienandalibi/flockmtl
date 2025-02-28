#include "flockmtl/functions/scalar/llm_embedding.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

namespace flockmtl {

void LlmEmbedding::ValidateArguments(duckdb::DataChunk& args) {
    if (args.ColumnCount() < 2 || args.ColumnCount() > 2) {
        throw std::runtime_error("LlmEmbedScalarParser: Invalid number of arguments.");
    }
    if (args.data[0].GetType().id() != duckdb::LogicalTypeId::STRUCT) {
        throw std::runtime_error("LlmEmbedScalarParser: Model details must be a struct.");
    }
    if (args.data[1].GetType().id() != duckdb::LogicalTypeId::STRUCT) {
        throw std::runtime_error("LlmEmbedScalarParser: Inputs must be a struct.");
    }
}

std::vector<duckdb::vector<duckdb::Value>> LlmEmbedding::Operation(duckdb::DataChunk& args) {
    LlmEmbedding::ValidateArguments(args);

    auto inputs = CastVectorOfStructsToJson(args.data[1], args.size());
    auto model_details_json = CastVectorOfStructsToJson(args.data[0], 1)[0];
    Model model(model_details_json);

    std::vector<std::string> prepared_inputs;
    for (auto& row : inputs) {
        std::string concat_input;
        for (auto& item : row.items()) {
            concat_input += item.value().get<std::string>() + " ";
        }
        prepared_inputs.push_back(concat_input);
    }

    auto embeddings = model.CallEmbedding(prepared_inputs);
    std::vector<duckdb::vector<duckdb::Value>> results;
    for (size_t index = 0; index < embeddings.size(); index++) {
        duckdb::vector<duckdb::Value> embedding;
        for (auto& value : embeddings[index]) {
            embedding.push_back(duckdb::Value(static_cast<double>(value)));
        }
        results.push_back(embedding);
    }
    return results;
}

void LlmEmbedding::Execute(duckdb::DataChunk& args, duckdb::ExpressionState& state, duckdb::Vector& result) {
    // use different json files for different models
    auto model_details_json = CastVectorOfStructsToJson(args.data[0], 1)[0];
    Model model(model_details_json);
    auto model_details = model.GetModelDetails();
    std::string filename = model_details.provider_name + "_" + model_details.model + ".json";

    // const std::string& filename = "test1.json";
    const auto prompts = LlmEmbedding::GetInputs(args);
    std::vector<duckdb::vector<duckdb::Value>> results;

    // if the embeddings are already cached, load them, or else retrieve and store them
    if (LlmEmbedding::IsCached(filename, prompts)) {
        results = LlmEmbedding::GetEmbeddingsFromCache(filename, prompts);
    } else {
        results = LlmEmbedding::Operation(args);    // execute the operation
        // store results as vector
        std::vector<std::vector<double>> embeddings;
        for (const auto& res : results) {
            std::vector<double> tmp_embedding;
            for (auto& tmp_num : res) {
                tmp_embedding.push_back(tmp_num.GetValue<double>());
            }
            embeddings.push_back(tmp_embedding);
        }
        LlmEmbedding::CacheListOfEmbeddings(filename, prompts, embeddings);
    }

    auto index = 0;
    for (const auto& res : results) {
        result.SetValue(index++, duckdb::Value::LIST(res));
    }
}

std::vector<duckdb::vector<duckdb::Value>> LlmEmbedding::GetEmbeddingsFromCache(const std::string& filename, const std::vector<std::string> prompts) {
    // will store the embeddings to be returned
    std::vector<duckdb::vector<duckdb::Value>> embeddings;
    using json = nlohmann::json;
    json json_object;
    std::ifstream input_file("../../cache_embeddings/" + filename);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
    }

    try {
        input_file >> json_object; // load the json file
        input_file.close();
    } catch (const json::parse_error& e) {
        std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
        input_file.close();
    }

    // populate the vector to be returned
    for (const std::string& prompt : prompts) {
        // get the embedding from the json file
        auto tmp_embedding = json_object[prompt].get<std::vector<double>>();
        duckdb::vector<duckdb::Value> tmp_embedding_duckdb = duckdb::vector<duckdb::Value>();
        // transform all doubles to duckdb values
        for (double x : tmp_embedding) {
            tmp_embedding_duckdb.push_back(duckdb::Value::DOUBLE(x));
        }
        embeddings.push_back(tmp_embedding_duckdb);
    }
    return embeddings;

}

bool LlmEmbedding::IsCached(const std::string& filename, const std::vector<std::string>& prompts) {
    using json = nlohmann::json;
    json json_object;
    std::ifstream input_file("../../cache_embeddings/" + filename);
    if (!input_file.is_open()) {
        return false; // file can't be opened
    }

    try {
        input_file >> json_object; // load the json file
        input_file.close();
    } catch (const json::parse_error& e) {
        std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
        input_file.close();
        return false; // json file can't be loaded
    }

    // make sure ALL prompts exist in the json file
    for (const std::string& prompt : prompts) {
        if (json_object.find(prompt) == json_object.end()) {
            return false; // at least one prompt is missing
        }
    }

    // all prompts were found in the json file
    return true;
}

std::vector<std::string> LlmEmbedding::GetInputs(duckdb::DataChunk& args) {
    auto inputs = CastVectorOfStructsToJson(args.data[1], args.size());
    std::vector<std::string> prepared_inputs;
    for (auto& row : inputs) {
        std::string concat_input;
        for (auto& item : row.items()) {
            concat_input += item.value().get<std::string>() + " ";
        }
        prepared_inputs.push_back(concat_input);
    }
    return prepared_inputs;
}

void LlmEmbedding::CacheListOfEmbeddings(const std::string& filename, const std::vector<std::string>& prompts,
                                         const std::vector<std::vector<double>>& embeddings) {
    using json = nlohmann::json;
    json json_object;

    // make sure we have the same number of prompts and embeddings
    if (prompts.size() != embeddings.size()) {
        std::cout << "ERROR: prompts and embeddings not of same size" << std::endl;
    }

    // open the json file so we update it instead of overwrite it
    std::ifstream input_file("../../cache_embeddings/" + filename);
    if (input_file.is_open()) {
        try {
            input_file >> json_object;
        } catch (const json::parse_error& e) {
            std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
            json_object = json::object(); // create new empty json object if it failed to ensure the rest of the code works
        }
        input_file.close();
    } else {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
    }

    // update the json object by storing the received embeddings
    for (int i = 0; i < prompts.size(); i++) {
        json_object[prompts[i]] = embeddings[i];
    }

    std::ofstream file("../../cache_embeddings/"+filename);
    if (file.is_open()) {
        file << json_object.dump(4);
        file.close();
        std::cout << "Updated cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

} // namespace flockmtl
