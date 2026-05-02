#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/tools/Dense.h>
#include <zfp.h>
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
        std::cerr << "Usage: " << argv[0] << " <input.vdb> [compression_rate] [output_prefix]" << std::endl;
        std::cerr << "  compression_rate: bytes per value (default: 0.5 = 4 bits/value)" << std::endl;
        std::cerr << "  output_prefix: prefix for output files (default: compressed_output)" << std::endl;
        return 1;
    }
    
    std::string inputVDBFile = argv[1];
    
    // Parse optional compression rate
    if (argc >= 3) {
        g_compressionRate = std::atof(argv[2]);
        if (g_compressionRate <= 0) {
            std::cerr << "Warning: Invalid compression rate, using default 0.5" << std::endl;
            g_compressionRate = 0.5;
        }
    }
    
    // Parse optional output prefix
    std::string outputPrefix = "compressed_output";
    if (argc >= 4) {
        outputPrefix = argv[3];
    }
    
    // Set output filenames
    g_compressedFileName = outputPrefix + ".zfp";
    g_outFileName = outputPrefix + "_decompressed.raw";
    g_vdbOutputFileName = outputPrefix + "_reconstructed.vdb";
    
    // Initialize OpenVDB
    openvdb::initialize();
    
    std::cout << "========================================" << std::endl;
    std::cout << "VDB ZFP Compression Pipeline" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Input VDB: " << inputVDBFile << std::endl;
    std::cout << "Compression rate: " << g_compressionRate << " bytes/value" << std::endl;
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
        std::cout << "\n[3/6] Compressing with ZFP..." << std::endl;
        // Note: Using g_compressionRate as tolerance and assumes cubic dimensions
        size_t maxDim = std::max({g_dims.x, g_dims.y, g_dims.z});
        // std::vector<uchar> compressedData = zfpCompress_accuracy(denseData, g_compressionRate, maxDim);
        std::vector<uchar> compressedData = zfpCompress(denseData);
        
        // Save compressed data
        saveCompressedData(compressedData, g_compressedFileName);
        
        // Step 4: Decompress with ZFP
        std::cout << "\n[4/6] Decompressing with ZFP..." << std::endl;
        // std::vector<float> decompressedData = zfpDecompress_accuracy(compressedData, g_compressionRate, maxDim);
        std::vector<float> decompressedData = zfpDecompress(compressedData);
        
        // Save decompressed raw data
        saveDecompressedData(decompressedData, g_outFileName);
        
        // Step 5: Convert back to VDB
        std::cout << "\n[5/6] Converting back to VDB..." << std::endl;
        openvdb::FloatGrid::Ptr reconstructedGrid = convertDenseArrayToVDB(decompressedData, bbox);
        
        // Step 6: Save VDB file
        std::cout << "\n[6/6] Saving VDB file..." << std::endl;
        saveVDBGrid(reconstructedGrid, g_vdbOutputFileName);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Pipeline completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Output files:" << std::endl;
        std::cout << "  - Compressed data: " << g_compressedFileName << std::endl;
        std::cout << "  - Decompressed raw: " << g_outFileName << std::endl;
        std::cout << "  - Reconstructed VDB: " << g_vdbOutputFileName << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
