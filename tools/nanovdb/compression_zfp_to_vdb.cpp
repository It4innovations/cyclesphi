#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/tools/Dense.h>
#include <zfp.h>
#include <zfp/array3.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Type aliases for clarity
using uchar = unsigned char;
using uint = unsigned int;

// Global dimensions of the volume data
struct Dims3 {
    size_t x, y, z;
} g_dims;

// ZFP compression rate in bytes per value (e.g., 0.5 means 4 bits per value)
double g_compressionRate = 0.5;

// Output file names
std::string g_outFileName;
std::string g_compressedFileName;
std::string g_vdbOutputFileName;

// /**
//  * Compresses a 3D float array using ZFP compression algorithm with accuracy mode.
//  * 
//  * @param input The input float vector containing the 3D volume data in row-major order
//  * @param tolerance The accuracy tolerance for compression
//  * @param dim_size The dimension size (assumes cubic volume)
//  * @return A vector of bytes containing the compressed data
//  */
// std::vector<uchar> zfpCompress_accuracy(std::vector<float> &input, double tolerance, size_t dim_size)
// {
//     zfp_type type = zfp_type_float;
//     zfp_field* field = zfp_field_3d(input.data(), type, dim_size, dim_size, dim_size);
//     zfp_stream* zfp = zfp_stream_open(NULL);
//     //zfp_stream_set_rate(zfp, compression_rate*8/*8 bits*/, type, dims, 0);
//     //zfp_stream_set_precision(zfp, precision, type);
//     zfp_stream_set_accuracy(zfp, tolerance);
//     size_t bufsize = zfp_stream_maximum_size(zfp, field);
//     std::vector<uchar> buffer(bufsize);
//     bitstream* stream = stream_open(buffer.data(), bufsize);
//     zfp_stream_set_bit_stream(zfp, stream);
//     zfp_stream_rewind(zfp);
//     std::cout << "compress...\n";
//     size_t size = zfp_compress(zfp, field);
//     buffer.resize(size); // adjust to actual size
//     return buffer;
// }

// /**
//  * Decompresses ZFP-compressed data back to a 3D float array using accuracy mode.
//  * 
//  * @param buffer The compressed data buffer
//  * @param tolerance The accuracy tolerance used during compression
//  * @param dim_size The dimension size (assumes cubic volume)
//  * @return A vector of floats containing the decompressed 3D volume data
//  */
// std::vector<float> zfpDecompress_accuracy(std::vector<uchar> &buffer, double tolerance, size_t dim_size)
// {
//     std::vector<float> result(dim_size*size_t(dim_size)*dim_size);
//     zfp_type type = zfp_type_float;
//     zfp_field* field = zfp_field_3d(result.data(), type, dim_size, dim_size, dim_size);
//     zfp_stream* zfp = zfp_stream_open(NULL);
//     zfp_stream_set_accuracy(zfp, tolerance);
//     bitstream* stream = stream_open(buffer.data(), buffer.size());
//     zfp_stream_set_bit_stream(zfp, stream);
//     zfp_stream_rewind(zfp);
//     zfp_decompress(zfp, field);
//     return result;
// }

std::vector<uchar> zfpCompress(std::vector<float> &input)
{
  uint dims = 3;
  zfp_type type = zfp_type_float;
  zfp_field* field = zfp_field_3d(input.data(), type, g_dims.x, g_dims.y, g_dims.z); 
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, g_compressionRate*8/*8 bits*/, type, dims, 0);
  //zfp_stream_set_precision(zfp, precision, type);
  //zfp_stream_set_accuracy(zfp, tolerance, type);
  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  std::vector<uchar> buffer(bufsize);
  bitstream* stream = stream_open(buffer.data(), bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);
  std::cout << "compress...\n";
  size_t size = zfp_compress(zfp, field);
  buffer.resize(size); // adjust to actual size
  return buffer;
}

/**
 * Compresses a 3D float array using ZFP with header for ZFPImageLoader compatibility.
 * Format: [nx:8][ny:8][nz:8][rate:8][compressed_size:8][compressed_data]
 * 
 * @param input The input float vector containing the 3D volume data
 * @return A vector of bytes containing the header and compressed data
 */
std::vector<uchar> zfpCompressWithHeader(std::vector<float> &input)
{
  // Create zfp::array3f with the data
  double rate = g_compressionRate * 8;  // Convert bytes to bits
  size_t cache_size = 64 * 1024 * 1024;  // 64MB cache for compression
  
  std::cout << "Creating ZFP compressed array...\n";
  std::cout << "  Dimensions: " << g_dims.x << " x " << g_dims.y << " x " << g_dims.z << "\n";
  std::cout << "  Rate: " << rate << " bits/value\n";
  
  // Create array and populate it
  zfp::array3f array(g_dims.x, g_dims.y, g_dims.z, rate, 0, cache_size);
  
  std::cout << "Populating array...\n";
  for (size_t z = 0; z < g_dims.z; ++z) {
    for (size_t y = 0; y < g_dims.y; ++y) {
      for (size_t x = 0; x < g_dims.x; ++x) {
        array(x, y, z) = input[x + g_dims.x * (y + g_dims.y * z)];
      }
    }
  }
  
  // Flush cache to compress all data
  array.flush_cache();
  
  // Get compressed data and size
  size_t compressed_size = array.compressed_size();
  void* compressed_data = array.compressed_data();
  
  std::cout << "Compressed size: " << compressed_size << " bytes\n";
  
  // Create output buffer: header (5 * 8 bytes) + compressed data
  size_t header_size = 5 * sizeof(size_t);
  size_t total_size = header_size + compressed_size;
  std::vector<uchar> buffer(total_size);
  
  // Write header: dimensions, rate, compressed size
  size_t* header = reinterpret_cast<size_t*>(buffer.data());
  header[0] = g_dims.x;
  header[1] = g_dims.y;
  header[2] = g_dims.z;
  // Store rate as uint64 (will be converted back to double)
  std::memcpy(&header[3], &rate, sizeof(double));
  header[4] = compressed_size;
  
  // Copy compressed data after header
  std::memcpy(buffer.data() + header_size, compressed_data, compressed_size);
  
  std::cout << "  Total serialized size: " << total_size << " bytes\n";
  std::cout << "  Compression ratio: " << (double)(g_dims.x * g_dims.y * g_dims.z * sizeof(float)) / (double)total_size << ":1\n";
  
  return buffer;
}

std::vector<float> zfpDecompress(std::vector<uchar> &buffer)
{
  uint dims = 3;
  std::vector<float> result(g_dims.x*size_t(g_dims.y)*g_dims.z);
  zfp_type type = zfp_type_float;
  zfp_field* field = zfp_field_3d(result.data(), type, g_dims.x, g_dims.y, g_dims.z); 
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, g_compressionRate*8/* bits*/, type, dims, 0);
  bitstream* stream = stream_open(buffer.data(), buffer.size());
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);
  size_t size = zfp_decompress(zfp, field);
  return result;
}

/**
 * Converts an OpenVDB grid to a dense 3D float array.
 * 
 * @param grid The input OpenVDB FloatGrid
 * @param outData Output vector that will contain the dense data
 * @param bbox Output bounding box of the active region
 * @return true if successful, false otherwise
 */
bool convertVDBToDenseArray(openvdb::FloatGrid::Ptr grid, std::vector<float> &outData, 
                            openvdb::CoordBBox &bbox)
{
    // Get the bounding box of active voxels
    bbox = grid->evalActiveVoxelBoundingBox();
    
    if (bbox.empty()) {
        std::cerr << "Error: VDB grid is empty!" << std::endl;
        return false;
    }
    
    // Calculate dimensions
    openvdb::Coord dim = bbox.dim();
    g_dims.x = dim.x();
    g_dims.y = dim.y();
    g_dims.z = dim.z();
    
    std::cout << "VDB grid dimensions: " << g_dims.x << " x " << g_dims.y 
              << " x " << g_dims.z << std::endl;
    std::cout << "Bounding box: min=" << bbox.min() << ", max=" << bbox.max() << std::endl;
    
    // Allocate dense array
    size_t totalVoxels = g_dims.x * g_dims.y * g_dims.z;
    outData.resize(totalVoxels);
    
    // Create a dense grid accessor
    openvdb::tools::Dense<float> dense(bbox, outData.data());
    
    // Copy VDB sparse data to dense array
    std::cout << "Converting VDB to dense array..." << std::endl;
    openvdb::tools::copyToDense(*grid, dense);
    
    return true;
}

/**
 * Converts a dense 3D float array back to an OpenVDB grid.
 * 
 * @param data The input dense float array
 * @param bbox The bounding box defining the volume extent
 * @return Pointer to the created FloatGrid
 */
openvdb::FloatGrid::Ptr convertDenseArrayToVDB(const std::vector<float> &data, 
                                                const openvdb::CoordBBox &bbox)
{
    std::cout << "Converting dense array back to VDB..." << std::endl;
    
    // Create a dense grid accessor from the data
    openvdb::tools::Dense<const float> dense(bbox, data.data());
    
    // Create an empty FloatGrid
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    
    // Copy dense data to VDB grid
    // The tolerance parameter determines which values are considered "background"
    // and will be stored sparsely (default background is 0.0)
    openvdb::tools::copyFromDense(dense, *grid, /*tolerance=*/0.0f);
    
    std::cout << "VDB grid created with " << grid->activeVoxelCount() 
              << " active voxels" << std::endl;
    
    return grid;
}

/**
 * Saves compressed binary data to disk.
 * 
 * @param data The compressed byte buffer
 * @param filename Output file path
 */
void saveCompressedData(const std::vector<uchar> &data, const std::string &filename)
{
    std::cout << "Saving compressed data to: " << filename << std::endl;
    std::ofstream os(filename, std::ios::out | std::ios::binary);
    
    if (!os) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    os.write((const char *)data.data(), data.size());
    os.close();
    
    std::cout << "Saved " << data.size() << " bytes" << std::endl;
}

/**
 * Saves decompressed float data to disk as a raw binary file.
 * 
 * @param data The decompressed float buffer
 * @param filename Output file path
 */
void saveDecompressedData(const std::vector<float> &data, const std::string &filename)
{
    std::cout << "Saving decompressed data to: " << filename << std::endl;
    std::ofstream os(filename, std::ios::out | std::ios::binary);
    
    if (!os) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return;
    }
    
    os.write((const char *)data.data(), g_dims.x * size_t(g_dims.y) * g_dims.z * sizeof(float));
    os.close();
    
    std::cout << "Saved " << (g_dims.x * g_dims.y * g_dims.z * sizeof(float)) 
              << " bytes" << std::endl;
}

/**
 * Saves an OpenVDB grid to a .vdb file.
 * 
 * @param grid The grid to save
 * @param filename Output VDB file path
 */
void saveVDBGrid(openvdb::FloatGrid::Ptr grid, const std::string &filename)
{
    std::cout << "Saving VDB grid to: " << filename << std::endl;
    
    // Create a VDB file with the grid
    openvdb::io::File file(filename);
    
    // Set grid name
    grid->setName("density");
    
    // Create a grid pointer vector
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    
    // Write the grid to file
    file.write(grids);
    file.close();
    
    std::cout << "VDB file saved successfully" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.vdb> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --rate <value>        Compression rate in bytes per value (default: 0.5 = 4 bits/value)" << std::endl;
        std::cerr << "  --output <prefix>     Output file prefix (default: compressed_output)" << std::endl;
        std::cerr << "  --zfp-loader          Create ZFP file with header for ZFPImageLoader" << std::endl;
        std::cerr << "  --skip-vdb            Skip VDB reconstruction (faster, only creates ZFP)" << std::endl;
        std::cerr << "  --cache-size <bytes>  Cache size for ZFP array (default: 4096)" << std::endl;
        std::cerr << "\nExamples:" << std::endl;
        std::cerr << "  " << argv[0] << " input.vdb --rate 0.5 --output myvolume" << std::endl;
        std::cerr << "  " << argv[0] << " input.vdb --zfp-loader --skip-vdb --output volume" << std::endl;
        return 1;
    }
    
    std::string inputVDBFile = argv[1];
    bool zfpLoaderMode = false;
    bool skipVDBReconstruction = false;
    size_t cacheSize = 4096;
    
    // Parse command-line arguments
    std::string outputPrefix = "compressed_output";
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--rate" && i + 1 < argc) {
            g_compressionRate = std::atof(argv[++i]);
            if (g_compressionRate <= 0) {
                std::cerr << "Warning: Invalid compression rate, using default 0.5" << std::endl;
                g_compressionRate = 0.5;
            }
        }
        else if (arg == "--output" && i + 1 < argc) {
            outputPrefix = argv[++i];
        }
        else if (arg == "--zfp-loader") {
            zfpLoaderMode = true;
        }
        else if (arg == "--skip-vdb") {
            skipVDBReconstruction = true;
        }
        else if (arg == "--cache-size" && i + 1 < argc) {
            cacheSize = std::atoll(argv[++i]);
        }
        // Legacy positional arguments support
        else if (i == 2 && arg[0] != '-') {
            g_compressionRate = std::atof(argv[i]);
            if (g_compressionRate <= 0) {
                std::cerr << "Warning: Invalid compression rate, using default 0.5" << std::endl;
                g_compressionRate = 0.5;
            }
        }
        else if (i == 3 && arg[0] != '-') {
            outputPrefix = argv[i];
        }
    }
    
    // Set output filenames
    g_compressedFileName = outputPrefix + (zfpLoaderMode ? "_loader.zfp" : ".zfp");
    g_outFileName = outputPrefix + "_decompressed.raw";
    g_vdbOutputFileName = outputPrefix + "_reconstructed.vdb";
    
    // Initialize OpenVDB
    openvdb::initialize();
    
    std::cout << "========================================" << std::endl;
    std::cout << "VDB ZFP Compression Pipeline" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Input VDB: " << inputVDBFile << std::endl;
    std::cout << "Compression rate: " << g_compressionRate << " bytes/value (" 
              << (g_compressionRate * 8) << " bits/value)" << std::endl;
    std::cout << "ZFPImageLoader mode: " << (zfpLoaderMode ? "enabled" : "disabled") << std::endl;
    std::cout << "Skip VDB reconstruction: " << (skipVDBReconstruction ? "yes" : "no") << std::endl;
    if (zfpLoaderMode) {
        std::cout << "Cache size: " << cacheSize << " bytes" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    try {
        // Step 1: Load VDB file
        std::cout << "\n[1/6] Loading VDB file..." << std::endl;
        openvdb::io::File file(inputVDBFile);
        file.open();
        
        // Read the first grid as a FloatGrid
        openvdb::GridBase::Ptr baseGrid;
        for (openvdb::io::File::NameIterator nameIter = file.beginName();
             nameIter != file.endName(); ++nameIter)
        {
            baseGrid = file.readGrid(nameIter.gridName());
            std::cout << "Found grid: " << nameIter.gridName() << std::endl;
            break; // Use first grid
        }
        file.close();
        
        if (!baseGrid) {
            std::cerr << "Error: No grids found in VDB file!" << std::endl;
            return 1;
        }
        
        openvdb::FloatGrid::Ptr floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        if (!floatGrid) {
            std::cerr << "Error: Grid is not a FloatGrid!" << std::endl;
            return 1;
        }
        
        // Step 2: Convert VDB to dense array
        std::cout << "\n[2/6] Converting VDB to dense array..." << std::endl;
        std::vector<float> denseData;
        openvdb::CoordBBox bbox;
        
        if (!convertVDBToDenseArray(floatGrid, denseData, bbox)) {
            return 1;
        }
        
        // Step 3: Compress with ZFP
        int totalSteps = skipVDBReconstruction ? 3 : 6;
        std::cout << "\n[3/" << totalSteps << "] Compressing with ZFP..." << std::endl;
        
        std::vector<uchar> compressedData;
        if (zfpLoaderMode) {
            // Use compression with header for ZFPImageLoader
            compressedData = zfpCompressWithHeader(denseData);
            if (compressedData.empty()) {
                std::cerr << "Error: Compression failed!" << std::endl;
                return 1;
            }
        } else {
            // Use compression without header (legacy mode)
            compressedData = zfpCompress(denseData);
        }
        
        // Save compressed data
        saveCompressedData(compressedData, g_compressedFileName);
        
        // Calculate compression ratio
        size_t originalSize = g_dims.x * g_dims.y * g_dims.z * sizeof(float);
        double compressionRatio = (double)originalSize / (double)compressedData.size();
        std::cout << "Compression ratio: " << compressionRatio << ":1" << std::endl;
        std::cout << "Original size: " << originalSize << " bytes" << std::endl;
        std::cout << "Compressed size: " << compressedData.size() << " bytes" << std::endl;
        
        if (skipVDBReconstruction) {
            std::cout << "\nSkipping VDB reconstruction (--skip-vdb enabled)" << std::endl;
        } else {
            // Step 4: Decompress with ZFP
            std::cout << "\n[4/6] Decompressing with ZFP..." << std::endl;
            std::vector<float> decompressedData = zfpDecompress(compressedData);
            
            // Save decompressed raw data
            saveDecompressedData(decompressedData, g_outFileName);
            
            // Step 5: Convert back to VDB
            std::cout << "\n[5/6] Converting back to VDB..." << std::endl;
            openvdb::FloatGrid::Ptr reconstructedGrid = convertDenseArrayToVDB(decompressedData, bbox);
            
            // Step 6: Save VDB file
            std::cout << "\n[6/6] Saving VDB file..." << std::endl;
            saveVDBGrid(reconstructedGrid, g_vdbOutputFileName);
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Pipeline completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Output files:" << std::endl;
        std::cout << "  - Compressed data: " << g_compressedFileName << std::endl;
        if (zfpLoaderMode) {
            std::cout << "    (Compatible with ZFPImageLoader)" << std::endl;
        }
        if (!skipVDBReconstruction) {
            std::cout << "  - Decompressed raw: " << g_outFileName << std::endl;
            std::cout << "  - Reconstructed VDB: " << g_vdbOutputFileName << std::endl;
        }
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
