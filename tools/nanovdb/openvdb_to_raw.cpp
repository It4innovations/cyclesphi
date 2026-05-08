#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Dense.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

/**
 * Generates output filename with dimensions and type.
 * Format: volume_gridname_dimX_dimY_dimZ_float.raw
 * 
 * @param baseName Base name for the output file
 * @param gridName Name of the grid being exported
 * @param dims Dimensions of the volume
 * @param dataType Type of data (e.g., "float", "double")
 * @return Formatted filename
 */
std::string generateOutputFilename(const std::string &baseName, 
                                   const std::string &gridName,
                                   const openvdb::Coord &dims,
                                   const std::string &dataType)
{
    std::ostringstream oss;
    
    // Remove .raw extension if present in baseName
    std::string cleanBaseName = baseName;
    if (cleanBaseName.length() >= 4 && 
        cleanBaseName.substr(cleanBaseName.length() - 4) == ".raw") {
        cleanBaseName = cleanBaseName.substr(0, cleanBaseName.length() - 4);
    }
    
    // Sanitize grid name (replace spaces and special chars with underscores)
    std::string sanitizedGridName = gridName;
    for (char &c : sanitizedGridName) {
        if (!std::isalnum(c)) {
            c = '_';
        }
    }
    
    // Build filename: volume_gridname_dimX_dimY_dimZ_type.raw
    oss << cleanBaseName << "_" << sanitizedGridName 
        << "_" << dims.x() << "_" << dims.y() << "_" << dims.z() 
        << "_" << dataType << ".raw";
    
    return oss.str();
}

/**
 * Saves raw binary float data to disk.
 * 
 * @param filename Output file path
 * @param data Vector containing the float data
 * @return true if successful, false otherwise
 */
bool saveRawData(const std::string &filename, const std::vector<float> &data)
{
    std::cout << "Saving raw data to: " << filename << std::endl;
    std::ofstream os(filename, std::ios::out | std::ios::binary);
    
    if (!os) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write data
    os.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    
    if (!os) {
        std::cerr << "Error: Failed to write data to file" << std::endl;
        return false;
    }
    
    os.close();
    
    std::cout << "Saved " << data.size() << " float values (" 
              << (data.size() * sizeof(float)) << " bytes)" << std::endl;
    
    return true;
}

/**
 * Converts an OpenVDB FloatGrid to a dense array.
 * 
 * @param grid The input FloatGrid
 * @param data Output vector that will contain the dense data
 * @param bbox Output bounding box of the active region
 * @return true if successful, false otherwise
 */
bool convertVDBGridToDenseArray(openvdb::FloatGrid::Ptr grid,
                                std::vector<float> &data,
                                openvdb::CoordBBox &bbox)
{
    std::cout << "\nConverting OpenVDB grid to dense array..." << std::endl;
    
    // Get the active voxel bounding box
    bbox = grid->evalActiveVoxelBoundingBox();
    
    if (bbox.empty()) {
        std::cerr << "Error: Grid has no active voxels!" << std::endl;
        return false;
    }
    
    std::cout << "Bounding box: min=" << bbox.min() << ", max=" << bbox.max() << std::endl;
    
    // Calculate dimensions
    openvdb::Coord dims = bbox.extents();
    size_t numElements = dims.x() * dims.y() * dims.z();
    
    std::cout << "Dimensions: " << dims.x() << " x " << dims.y() << " x " << dims.z() << std::endl;
    std::cout << "Total voxels: " << numElements << std::endl;
    std::cout << "Active voxels: " << grid->activeVoxelCount() << std::endl;
    
    // Allocate output buffer
    data.resize(numElements);
    
    // Create a dense grid accessor
    openvdb::tools::Dense<float> dense(bbox, data.data());
    
    // Copy VDB grid to dense array
    std::cout << "Copying from sparse VDB to dense array..." << std::endl;
    openvdb::tools::copyToDense(*grid, dense);
    
    // Print some statistics
    float minVal = data[0], maxVal = data[0];
    double sum = 0.0;
    size_t nonZeroCount = 0;
    
    for (size_t i = 0; i < numElements; ++i) {
        float val = data[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
        sum += val;
        if (val != 0.0f) nonZeroCount++;
    }
    double avg = sum / numElements;
    
    std::cout << "\nData statistics:" << std::endl;
    std::cout << "  Min: " << minVal << std::endl;
    std::cout << "  Max: " << maxVal << std::endl;
    std::cout << "  Avg: " << avg << std::endl;
    std::cout << "  Non-zero voxels: " << nonZeroCount << " (" 
              << (100.0 * nonZeroCount / numElements) << "%)" << std::endl;
    
    return true;
}

/**
 * Loads an OpenVDB file and returns the first FloatGrid found.
 * 
 * @param filename Input VDB file path
 * @param gridName Optional: specific grid name to load. If empty, loads first FloatGrid
 * @return Pointer to the loaded FloatGrid, or nullptr on error
 */
openvdb::FloatGrid::Ptr loadVDBGrid(const std::string &filename, const std::string &gridName = "")
{
    std::cout << "Loading OpenVDB file: " << filename << std::endl;
    
    openvdb::io::File file(filename);
    
    try {
        file.open();
    } catch (const std::exception &e) {
        std::cerr << "Error: Cannot open VDB file: " << e.what() << std::endl;
        return nullptr;
    }
    
    // Get list of available grids
    openvdb::GridPtrVecPtr grids = file.getGrids();
    
    if (!grids || grids->empty()) {
        std::cerr << "Error: No grids found in VDB file!" << std::endl;
        file.close();
        return nullptr;
    }
    
    std::cout << "Found " << grids->size() << " grid(s) in file:" << std::endl;
    for (size_t i = 0; i < grids->size(); ++i) {
        std::cout << "  [" << i << "] " << (*grids)[i]->getName() 
                  << " (" << (*grids)[i]->valueType() << ")" << std::endl;
    }
    
    // Find the requested grid or first FloatGrid
    openvdb::FloatGrid::Ptr floatGrid;
    
    if (!gridName.empty()) {
        // Search for specific grid by name
        for (auto &grid : *grids) {
            if (grid->getName() == gridName) {
                floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
                if (!floatGrid) {
                    std::cerr << "Error: Grid '" << gridName << "' is not a FloatGrid!" << std::endl;
                    file.close();
                    return nullptr;
                }
                break;
            }
        }
        if (!floatGrid) {
            std::cerr << "Error: Grid '" << gridName << "' not found!" << std::endl;
            file.close();
            return nullptr;
        }
    } else {
        // Find first FloatGrid
        for (auto &grid : *grids) {
            floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
            if (floatGrid) {
                break;
            }
        }
        if (!floatGrid) {
            std::cerr << "Error: No FloatGrid found in VDB file!" << std::endl;
            file.close();
            return nullptr;
        }
    }
    
    std::cout << "\nUsing grid: " << floatGrid->getName() << std::endl;
    std::cout << "  Value type: " << floatGrid->valueType() << std::endl;
    std::cout << "  Active voxels: " << floatGrid->activeVoxelCount() << std::endl;
    std::cout << "  Memory: " << floatGrid->memUsage() << " bytes" << std::endl;
    
    file.close();
    
    return floatGrid;
}

/**
 * Main function - converts OpenVDB format to raw binary float data.
 * 
 * Usage: openvdb_to_raw <input.vdb> <output_base> [grid_name]
 */
int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.vdb> <output_base> [grid_name]\n";
        std::cerr << "\n";
        std::cerr << "  input.vdb   - Input OpenVDB file\n";
        std::cerr << "  output_base - Output base name (dimensions and type will be appended)\n";
        std::cerr << "  grid_name   - Optional: specific grid name to export (default: first FloatGrid)\n";
        std::cerr << "\n";
        std::cerr << "Output filename format: output_base_gridname_dimX_dimY_dimZ_float.raw\n";
        std::cerr << "\n";
        std::cerr << "Example:\n";
        std::cerr << "  " << argv[0] << " data.vdb volume\n";
        std::cerr << "  Output: volume_density_512_512_512_float.raw\n";
        std::cerr << "\n";
        std::cerr << "  " << argv[0] << " data.vdb volume density\n";
        std::cerr << "  Output: volume_density_512_512_512_float.raw\n";
        return EXIT_FAILURE;
    }

    const std::string inputFile = argv[1];
    const std::string outputBase = argv[2];
    const std::string gridName = (argc >= 4) ? argv[3] : "";
    
    std::cout << "============================================\n";
    std::cout << "OpenVDB to Raw Converter\n";
    std::cout << "============================================\n";
    std::cout << "Input:  " << inputFile << "\n";
    std::cout << "Output: " << outputBase << "_<gridname>_<dims>_float.raw\n";
    if (!gridName.empty()) {
        std::cout << "Grid:   " << gridName << "\n";
    }
    std::cout << "============================================\n\n";

    try {
        // Initialize OpenVDB library
        openvdb::initialize();
        
        // Load VDB grid
        openvdb::FloatGrid::Ptr grid = loadVDBGrid(inputFile, gridName);
        if (!grid) {
            return EXIT_FAILURE;
        }
        
        // Convert to dense array
        std::vector<float> data;
        openvdb::CoordBBox bbox;
        if (!convertVDBGridToDenseArray(grid, data, bbox)) {
            return EXIT_FAILURE;
        }
        
        // Generate output filename with dimensions and type
        openvdb::Coord dims = bbox.extents();
        std::string outputFile = generateOutputFilename(outputBase, grid->getName(), dims, "float");
        
        // Save raw binary data
        if (!saveRawData(outputFile, data)) {
            return EXIT_FAILURE;
        }
        
        std::cout << "\n============================================\n";
        std::cout << "Conversion completed successfully!\n";
        std::cout << "Output file: " << outputFile << "\n";
        std::cout << "Dimensions: " << dims.x() << " x " << dims.y() << " x " << dims.z() << "\n";
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
