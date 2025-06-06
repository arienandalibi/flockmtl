#include "flockmtl-test/functions/scalar/test_fusion.hpp"
#include "flockmtl/caching/cache_manager.hpp"

using namespace duckdb;

void setup_db(const unique_ptr<Connection>& con) {
    con->Query("CREATE SECRET (TYPE OLLAMA, API_URL '10.0.0.230:11434');" );
    con->Query(R"(CREATE MODEL('deepseek', 'deepseek-r1', 'ollama', {"context_window": 8192, "max_output_tokens": 8000});)");
    con->Query(R"(CREATE TABLE products (product_name VARCHAR);)");
    con->Query(R"(INSERT INTO products (product_name) VALUES ('shoes'), ('computer'), ('bottle');)");
}

TEST_CASE("Test caching prompt only", "[caching][flockmtl]") {
    // Initialize an in-memory DuckDB instanceAdd commentMore actions
    auto db = make_uniq<DuckDB>(nullptr);
    auto con = make_uniq<Connection>(*db);

    // Set up the database
    setup_db(con);

    // Should cache the response
    auto result = con->Query(R"(SELECT llm_complete(
            {'model_name': 'deepseek'},
            {'prompt': 'Explain the purpose of FlockMTL. Limit your response to 10 words. No need to think before answering, if you do not know what it is just say that.'}
        ) AS flockmtl_purpose;)");

    // Should use cached response
    auto result2 = con->Query(R"(SELECT llm_complete(
            {'model_name': 'deepseek'},
            {'prompt': 'Explain the purpose of FlockMTL. Limit your response to 10 words. No need to think before answering, if you do not know what it is just say that.'}
        ) AS flockmtl_purpose;)");

    // Ensure query executed successfully
    REQUIRE(!result->HasError());
}

TEST_CASE("Test caching with input tuples", "[caching][flockmtl]") {
    // Initialize an in-memory DuckDB instanceAdd commentMore actions
    auto db = make_uniq<DuckDB>(nullptr);
    auto con = make_uniq<Connection>(*db);

    // Set up the database
    setup_db(con);

    // Should cache the response
    auto result = con->Query(R"(SELECT llm_complete(
            {'model_name': 'deepseek'},
            {'prompt': 'Generate a description for the product.'},
            {'product_name': product_name}
        ) AS product_description
        FROM products;)");

    // Should use cached response
    auto result2 = con->Query(R"(SELECT llm_complete(
            {'model_name': 'deepseek'},
            {'prompt': 'Generate a description for the product.'},
            {'product_name': product_name}
        ) AS product_description
        FROM products;)");

    // Ensure query executed successfully
    REQUIRE(!result->HasError());
}