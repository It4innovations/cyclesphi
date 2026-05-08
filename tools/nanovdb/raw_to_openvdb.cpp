#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>
#include <openvdb/tools/Dense.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

// Global dimensions of the volume data
struct Dims3 {
    size_t x, y, z;
} g_dims;

/**
 * Loads raw binary float data from disk.
 * 
 * @param filename Input file path
 * @param data Output vector that will contain the loaded data
 * @return true if successful, false otherwise
 */
bool loadRawData(const std::string &filename, std::vector<float> &data)
{
    std::cout << "Loading raw data from: " << filename << std::endl;
    std::ifstream is(filename, std::ios::in | std::ios::binary);
    
    if (!is) {
        std::cerr << "Error: Cannot open file for reading: " << filename << std::endl;
        return false;
    }
    
    // Get file size
    is.seekg(0, std::ios::end);
    size_t fileSize = is.tellg();
    is.seekg(0, std::ios::beg);
    
    // Calculate expected size
    size_t expectedSize = g_dims.x * g_dims.y * g_dims.z * sizeof(float);
    
    std::cout << "File size: " << fileSize << " bytes" << std::endl;
    std::cout << "Expected size: " << expectedSize << " bytes" << std::endl;
    
    if (fileSize != expectedSize) {
        std::cerr << "Warning: File size does not match expected size based on dimensions!" << std::endl;
        std::cerr << "  File has " << fileSize << " bytes, expected " << expectedSize << " bytes" << std::endl;
        std::cerr << "  Proceeding anyway..." << std::endl;
    }
    
    // Allocate and read data
    size_t numElements = g_dims.x * g_dims.y * g_dims.z;
    data.resize(numElements);
    
    is.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(float));
    
    if (!is) {
        std::cerr << "Error: Failed to read data from file" << std::endl;
        return false;
    }
    
    is.close();
    
    std::cout << "Loaded " << numElements << " float values" << std::endl;
    
    // Print some statistics
    float minVal = data[0], maxVal = data[0];
    double sum = 0.0;
    for (size_t i = 0; i < numElements; ++i) {
        float val = data[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
        sum += val;
    }
    double avg = sum / numElements;
    
    std::cout << "Data statistics:" << std::endl;
    std::cout << "  Min: " << minVal << std::endl;
    std::cout << "  Max: " << maxVal << std::endl;
    std::cout << "  Avg: " << avg << std::endl;
    
    return true;
}

/**
 * Converts a dense 3D float array to an OpenVDB grid.
 * 
 * @param data The input dense float array
 * @param bbox The bounding box defining the volume extent
 * @param tolerance Values within this tolerance of background are stored sparsely
 * @return Pointer to the created FloatGrid
 */
openvdb::FloatGrid::Ptr convertDenseArrayToVDB(const std::vector<float> &data, 
                                                const openvdb::CoordBBox &bbox,
                                                float tolerance = 0.0f)
{
    std::cout << "\nConverting dense array to OpenVDB..." << std::endl;
    
    // Create a dense grid accessor from the data
    openvdb::tools::Dense<const float, openvdb::tools::LayoutXYZ> dense(bbox, data.data());
    
    // Create an empty FloatGrid
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    
    // Set grid name
    grid->setName("density");
    
    // Copy dense data to VDB grid using copyFromDense
    // The tolerance parameter determines which values are considered "background"
    // and will be stored sparsely (default background is 0.0)
    std::cout << "Copying from dense with tolerance: " << tolerance << std::endl;
    openvdb::tools::copyFromDense(dense, *grid, tolerance);
    
    std::cout << "OpenVDB grid created:" << std::endl;
    std::cout << "  Active voxel count: " << grid->activeVoxelCount() << std::endl;
    std::cout << "  Memory size: " << grid->memUsage() << " bytes" << std::endl;
    
    return grid;
}

/**
 * Saves an OpenVDB grid to a .vdb file.
 * 
 * @param grid The FloatGrid to save
 * @param filename Output file path
 */
void saveVDBGrid(openvdb::FloatGrid::Ptr grid, const std::string &filename)
{
    std::cout << "\nSaving OpenVDB grid to: " << filename << std::endl;
    
    // Create a VDB file object
    openvdb::io::File file(filename);
    
    // Create a grid list and add our grid
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    
    // Write the grids to the file
    file.write(grids);
    file.close();
    
    std::cout << "OpenVDB file saved successfully!" << std::endl;
}

/**
 * Main function - converts raw binary float data to OpenVDB format.
 * 
 * Usage: raw_to_openvdb <input.raw> <output.vdb> <dim_x> <dim_y> <dim_z> [tolerance]
 */
int main(int argc, char* argv[])
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input.raw> <output.vdb> <dim_x> <dim_y> <dim_z> [tolerance]\n";
        std::cerr << "\n";
        std::cerr << "  input.raw   - Input raw binary file containing float data\n";
        std::cerr << "  output.vdb  - Output OpenVDB file\n";
        std::cerr << "  dim_x       - X dimension of the volume\n";
        std::cerr << "  dim_y       - Y dimension of the volume\n";
        std::cerr << "  dim_z       - Z dimension of the volume\n";
        std::cerr << "  tolerance   - Optional: tolerance for background values (default: 0.0)\n";
        std::cerr << "\n";
        std::cerr << "Example:\n";
        std::cerr << "  " << argv[0] << " data.raw data.vdb 512 512 512 0.001\n";
        return EXIT_FAILURE;
    }

    const std::string inputFile = argv[1];
    const std::string outputFile = argv[2];
    
    // Parse dimensions
    g_dims.x = std::atoll(argv[3]);
    g_dims.y = std::atoll(argv[4]);
    g_dims.z = std::atoll(argv[5]);
    
    // Parse optional tolerance
    float tolerance = 0.0f;
    if (argc >= 7) {
        tolerance = std::atof(argv[6]);
    }
    
    std::cout << "============================================\n";
    std::cout << "Raw to OpenVDB Converter\n";
    std::cout << "============================================\n";
    std::cout << "Input:      " << inputFile << "\n";
    std::cout << "Output:     " << outputFile << "\n";
    std::cout << "Dimensions: " << g_dims.x << " x " << g_dims.y << " x " << g_dims.z << "\n";
    std::cout << "Tolerance:  " << tolerance << "\n";
    std::cout << "============================================\n\n";

    try {
        // Initialize OpenVDB library
        openvdb::initialize();
        
        // Load raw binary data
        std::vector<float> data;
        if (!loadRawData(inputFile, data)) {
            return EXIT_FAILURE;
        }
        
        // Define bounding box (assumes data starts at origin)
        openvdb::CoordBBox bbox(
            openvdb::Coord(0, 0, 0),
            openvdb::Coord(g_dims.x - 1, g_dims.y - 1, g_dims.z - 1)
        );
        
        std::cout << "\nBounding box: min=" << bbox.min() << ", max=" << bbox.max() << std::endl;
        
        // Convert dense array to OpenVDB
        openvdb::FloatGrid::Ptr grid = convertDenseArrayToVDB(data, bbox, tolerance);
        
        // Save to VDB file
        saveVDBGrid(grid, outputFile);
        
        std::cout << "\n============================================\n";
        std::cout << "Conversion completed successfully!\n";
        std::cout << "============================================\n";
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return EXIT_FAILURE;
    }
}
