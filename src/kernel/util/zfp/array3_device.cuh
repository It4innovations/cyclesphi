#ifndef CUDA_ZFP_ARRAY3_DEVICE_CUH
#define CUDA_ZFP_ARRAY3_DEVICE_CUH

/* Allow compilation on non-CUDA platforms (CPU kernel path). */
#ifndef __CUDACC__
#  ifndef __device__
#    define __device__
#    define __host__
#  endif
#endif

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {

/**
 * GPU-side block store for 3D compressed arrays
 * Provides access to compressed block data stored in device memory
 */
template <typename Scalar, int BlockSize = 64>
class DeviceBlockStore3 {
public:
    // Compressed data storage
    Word* d_compressed_data;       // Compressed block stream
    size_t* d_block_offsets;       // Byte offset for each block (optional, for variable-rate)
    
    // Array dimensions
    uint3 dims;                    // Actual array dimensions (nx, ny, nz)
    uint3 block_dims;              // Number of blocks in each dimension
    uint maxbits;                  // Maximum bits per block
    size_t total_blocks;           // Total number of blocks
    bool fixed_rate;               // True if fixed-rate compression
    
    // Default constructor
    __device__ __host__
    DeviceBlockStore3() 
        : d_compressed_data(nullptr)
        , d_block_offsets(nullptr)
        , maxbits(0)
        , total_blocks(0)
        , fixed_rate(true)
    {
        dims = make_uint3(0, 0, 0);
        block_dims = make_uint3(0, 0, 0);
    }
    
    /**
     * Get flat block index from 3D coordinates
     */
    __device__ __host__ inline
    size_t get_block_index(size_t x, size_t y, size_t z) const {
        size_t bx = x / 4;
        size_t by = y / 4;
        size_t bz = z / 4;
        return bx + block_dims.x * (by + block_dims.y * bz);
    }
    
    /**
     * Get pointer to compressed data for a specific block
     * For fixed-rate: simple offset calculation
     * For variable-rate: use offset table
     */
    __device__ inline
    Word* get_block_ptr(size_t block_idx) const {
        if (fixed_rate) {
            // Fixed rate: each block at predictable offset
            size_t word_index = (block_idx * maxbits) / (sizeof(Word) * 8);
            return d_compressed_data + word_index;
        } else {
            // Variable rate: use offset table
            return d_compressed_data + d_block_offsets[block_idx];
        }
    }
    
    /**
     * Get local coordinates within a block (0-3 for each dimension)
     */
    __device__ __host__ inline
    void get_local_coords(size_t x, size_t y, size_t z, 
                         uint& lx, uint& ly, uint& lz) const {
        lx = x % 4;
        ly = y % 4;
        lz = z % 4;
    }
    
    /**
     * Convert local 3D coords to flat index within block
     */
    __device__ __host__ inline
    uint local_index(uint lx, uint ly, uint lz) const {
        return lx + 4 * (ly + 4 * lz);
    }
};

/**
 * Simple per-thread cache - no caching, just decompress on demand
 * This is the simplest implementation for random access
 */
template <typename Scalar, int BlockSize = 64>
class NoCache {
public:
    __host__ __device__ inline
    Scalar get(DeviceBlockStore3<Scalar, BlockSize>& store,
               size_t block_idx,
               uint local_idx) {
        // Must be zero-initialized: zfp_decode skips writing fblock when s_cont=0
        // (all-zero block), leaving it unchanged; uninitialized memory would be returned.
        Scalar block[BlockSize] = {};
        
        // Pass the base pointer so BlockReader computes the block offset once.
        // get_block_ptr() already adds the same word_index that BlockReader
        // computes internally, so using a pre-offset pointer would double it.
        BlockReader<BlockSize> reader(store.d_compressed_data, store.maxbits,
                                     block_idx, store.total_blocks);
        
        // Decode the block
        zfp_decode<Scalar, BlockSize>(reader, block, store.maxbits);
        
        // Return the requested element
        return block[local_idx];
    }
};

/**
 * Per-warp shared memory cache
 * Each warp maintains a small cache of recently accessed blocks
 * Best for spatially coherent access patterns within a warp
 */
template <typename Scalar, int BlockSize = 64, int CacheLines = 4>
class WarpCache {
private:
    // Cache storage in shared memory (allocated per-warp)
    Scalar cache_data[CacheLines][BlockSize];
    uint32_t cache_tags[CacheLines];  // Block indices (-1 = invalid)
    uint32_t lru_counter;
    
public:
    __device__
    WarpCache() : lru_counter(0) {
        // Initialize cache as empty
        for (int i = 0; i < CacheLines; i++) {
            cache_tags[i] = 0xFFFFFFFF; // Invalid tag
        }
    }
    
    __device__ inline
    Scalar get(DeviceBlockStore3<Scalar, BlockSize>& store,
               size_t block_idx,
               uint local_idx) {
        // Check if block is in cache
        for (int i = 0; i < CacheLines; i++) {
            if (cache_tags[i] == block_idx) {
                // Cache hit
                return cache_data[i][local_idx];
            }
        }
        
        // Cache miss - find LRU entry (simple round-robin for now)
        int evict_idx = (lru_counter++) % CacheLines;

        // Zero before decode: s_cont=0 (all-zero block) skips writing fblock.
        memset(cache_data[evict_idx], 0, BlockSize * sizeof(Scalar));

        // Pass base pointer; BlockReader computes the block offset internally.
        BlockReader<BlockSize> reader(store.d_compressed_data, store.maxbits,
                                     block_idx, store.total_blocks);
        zfp_decode<Scalar, BlockSize>(reader, cache_data[evict_idx], store.maxbits);
        
        // Update tag
        cache_tags[evict_idx] = block_idx;
        
        // Return requested element
        return cache_data[evict_idx][local_idx];
    }
};

/**
 * Device-side random access view of a 3D compressed array
 * Template parameters:
 *   Scalar: data type (float, double, int32_t, int64_t)
 *   BlockSize: block size (64 for 3D)
 *   Cache: caching strategy (NoCache or WarpCache)
 */
template <typename Scalar, int BlockSize = 64, typename Cache = NoCache<Scalar, BlockSize>>
class DeviceArray3View {
private:
    DeviceBlockStore3<Scalar, BlockSize> store;
    Cache cache;
    
public:
    /**
     * Constructor - typically called from host to setup device view
     */
    __device__ __host__
    DeviceArray3View() {}
    
    /**
     * Initialize the view (called from host)
     */
    __host__
    void init(Word* d_compressed_data,
              uint3 dims,
              uint maxbits,
              size_t* d_block_offsets = nullptr) {
        store.d_compressed_data = d_compressed_data;
        store.d_block_offsets = d_block_offsets;
        store.dims = dims;
        store.maxbits = maxbits;
        store.fixed_rate = (d_block_offsets == nullptr);
        
        // Calculate block dimensions
        store.block_dims.x = (dims.x + 3) / 4;
        store.block_dims.y = (dims.y + 3) / 4;
        store.block_dims.z = (dims.z + 3) / 4;
        store.total_blocks = store.block_dims.x * store.block_dims.y * store.block_dims.z;
    }
    
    /**
     * Random access operator - read a value at (x, y, z)
     */
    __device__ inline
    Scalar operator()(size_t x, size_t y, size_t z) const {
        // Bounds checking (can be disabled for performance)
        #ifdef CUDA_ZFP_BOUNDS_CHECK
        if (x >= store.dims.x || y >= store.dims.y || z >= store.dims.z) {
            return Scalar(0);  // Or handle error differently
        }
        #endif
        
        // Calculate block index
        size_t block_idx = store.get_block_index(x, y, z);
        
        // Get local coordinates within block
        uint lx, ly, lz;
        store.get_local_coords(x, y, z, lx, ly, lz);
        uint local_idx = store.local_index(lx, ly, lz);
        
        // Access through cache
        return const_cast<Cache&>(cache).get(
            const_cast<DeviceBlockStore3<Scalar, BlockSize>&>(store),
            block_idx,
            local_idx
        );
    }
    
    /**
     * Get array dimensions
     */
    __device__ __host__ inline
    uint3 get_dims() const { return store.dims; }
    
    __device__ __host__ inline
    size_t size_x() const { return store.dims.x; }
    
    __device__ __host__ inline
    size_t size_y() const { return store.dims.y; }
    
    __device__ __host__ inline
    size_t size_z() const { return store.dims.z; }
};

// Type aliases for convenience
template <typename Scalar>
using DeviceArray3ViewNoCache = DeviceArray3View<Scalar, 64, NoCache<Scalar, 64>>;

template <typename Scalar, int CacheLines = 4>
using DeviceArray3ViewCached = DeviceArray3View<Scalar, 64, WarpCache<Scalar, 64, CacheLines>>;

} // namespace cuZFP

#endif // CUDA_ZFP_ARRAY3_DEVICE_CUH
