# Proposed design for caching system
## Important concepts
* Use DataChunks extensively. They are used by the core DuckDB engine directly, so this will minimizing parsing and conversion of data.
* Keep multithreading in mind. DuckDB is multithreaded application. Writes to cache can be handled by a different thread to avoid slowing down a currently executing query.
* We need an eviction strategy. The most common ones are Least Recently Used (LRU) and Least Frequently Used (LFU). We can even implement a hybrid of both, where the most frequently used cached entries are prioritized and kept. Consider if evictions happen on individual DataChunks, individual queries, or some other construct.
* I believe I remember seeing serialization in DuckDB's code, so we can potentially use the already implemented serialization (if I am correct).
* Might want a configurable cap on memory usage, where the default is some percentage of total available memory, or different discrete levels (low, medium, high)
* We will want to keep locking in mind. If multiple threads are accessing the cache at once, lock contention can become a bottleneck. We need as little thread locking as possible.
* Evaluate the cost/benefit ratio. What things should we be caching to observe noticeable improvements. We should consider the cost of recomputation versus the cost of maintaining the cache entry.
* We should start by caching things which are slower by nature (anything LLM related, computation heavy functions, ...).

## Questions to adress
### How will we deal with staleness? 
If caching is only implemented for our (flockmtl) functions, then we might not have access to DuckDB functions. If the underlying table changes, how will we know? Since DuckDB is already well optimized (I'm assuming), there might not be much benefit in caching here. Maybe we can use some hashing method or verify signatures to see if underlying data has changed.

This would not be a problem for things which don't have to do with DuckDB, which is when it would be mostly beneficial. Retrieving LLM Embeddings of prompts is a good example. LLM output in general is a good example. Computation heavy functionalities are a good example. 

### At what level does the caching happen?
If we want caching for our functions to be integrated with core DuckDB, we might have to implement caching somewhere along the query processing pipeline instead of directly into our functions.

SQL Parser -> Logical Planner -> Logical Optimizer -> Physical Optimizer -> Vectorized Execution Engine -> Result Materialization

In theory, a good plan would be to generate keys based on the query plan (before execution) that includes the plan, parameters, and if possible the relevant state. If this is feasible we can avoid the execution part of the query. However, how realistic is it to compare states (which are very general) faster than executing the query. The state would have to include a lot of information, including all relevant inputs such as table data.

## Technical details
### Entry information to keep track of
* A unique key calculated based on the query (ie a hash of the query plan, execution parameters, state, ...)
* A pointer to the data (the DataChunk or it's serialized form for example)
* Metadata like the timestamp for last access or usage count for eviction purposes
* Size information so we can track it's memory consumption

### General DuckDB constructs to (potentially) utilize
* DuckDB's Allocator
    * Already designed to be efficient in multithreaded scenarios and reduces fragmentation.
    * Allocations are tracked by DuckDB so overall memory usage can be monitored and constrained
    * We can benefit from the same optimizations
* DuckDB DataChunks
    * DataChunks are already in a "binary-friendly" format.
    * If we can manage the lifecycle of DataChunks, we can store them directly without having to copy them. Ideally, we simply make them "live" in the cache once they have already been .retrieved/generated
    * Serialization is mainly for persisting on-disk apparently
    * We will probably need to store additional metadata alongside the DataChunk
* DuckDB's Buffer Manager
    * Used for managing in-memory pages read from disk. Already implements caching for on-disk data.
    * Has an eviction policy, ensures cached pages don't exceed limits, and manages asynchronous read/write operations to ensure consistency
    * We will need to find a way to make it's memory tracking support our cached data
    * I think it is best to make use of this. It seems like it already implements all the important aspects we are interested in.
* Block Manager / Storage Manager
    * Manages blocks stored on-disk. 
* Query planner and executor
    * We might need to implement our caching at these levels.

#### Buffer manager functions/types we will probably need to use
* BufferManager::Pin/Unpin: A pinned page is protected from eviction while it is in use.
* BufferManager::Prefetch: used for performance suggestions
* BufferHandle: a handle to a page so that we can access it. Contains pointers to the data
* RegisterSmallMemory(), ReserveMemory(), FreeReservedMemory(), GetMemoryUsageInfo()
* BufferManager::ConstructManagedBuffer, BufferManager::GetBufferManager
* The buffer manager has direct support for swap space

#### There is also a BlockManager class to manage Blocks.
* Allows granular control over blocks
* Read, write, mark as free, mark as used, mark as modified, convert from in-memory buffer to persistent disk-backed block
* Manages locks