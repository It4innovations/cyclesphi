#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/math/Transform.h>

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <sys/stat.h>

/**
 * @brief Downsamples an OpenVDB grid by a given integer factor
 * 
 * Reads every Nth voxel from the original grid, where N is the downsample factor.
 * Uses three nested loops over the original dimensions to sample the grid.
 * 
 * @param inputGrid The input OpenVDB FloatGrid to downsample
 * @param factor The downsample factor (e.g., 2 means read every 2nd voxel)
 * @param keepTransform If true, keeps the original transformation matrix; 
 *                      if false, scales the transform to maintain physical size
 * @return Pointer to the downsampled FloatGrid
 */
openvdb::FloatGrid::Ptr downsampleGrid(openvdb::FloatGrid::Ptr inputGrid, 
                                       int factor, 
                                       bool keepTransform)
{
    if (!inputGrid) {
        std::cerr << "Error: Input grid is null!" << std::endl;
        return nullptr;
    }
    
    if (factor < 1) {
        std::cerr << "Error: Downsample factor must be >= 1!" << std::endl;
        return nullptr;
    }
    
    // Get the bounding box of active voxels
    openvdb::CoordBBox bbox = inputGrid->evalActiveVoxelBoundingBox();
    
    if (bbox.empty()) {
        std::cerr << "Error: Input grid is empty!" << std::endl;
        return nullptr;
    }
    
    openvdb::Coord minCoord = bbox.min();
    openvdb::Coord maxCoord = bbox.max();
    openvdb::Coord dim = bbox.dim();
    
    std::cout << "Input grid dimensions: " << dim.x() << " x " << dim.y() 
              << " x " << dim.z() << std::endl;
    std::cout << "Bounding box: min=" << minCoord << ", max=" << maxCoord << std::endl;
    std::cout << "Downsample factor: " << factor << std::endl;
    std::cout << "Keep original transform: " << (keepTransform ? "yes" : "no") << std::endl;
    
    // Calculate downsampled dimensions
    int newDimX = (dim.x() + factor - 1) / factor;  // Ceiling division
    int newDimY = (dim.y() + factor - 1) / factor;
    int newDimZ = (dim.z() + factor - 1) / factor;
    
    std::cout << "Output grid dimensions: " << newDimX << " x " << newDimY 
              << " x " << newDimZ << std::endl;
    
    // Create output grid with same background value as input
    openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create(inputGrid->background());
    
    // Copy metadata from input grid
    outputGrid->insertMeta(*inputGrid);
    
    // Get accessor for input grid (read-only)
    openvdb::FloatGrid::ConstAccessor inputAccessor = inputGrid->getConstAccessor();
    
    // Get accessor for output grid (write)
    openvdb::FloatGrid::Accessor outputAccessor = outputGrid->getAccessor();
    
    std::cout << "Downsampling..." << std::endl;
    
    // Sample every Nth voxel using three nested loops over original dimensions
    int outputZ = 0;
    for (int z = minCoord.z(); z <= maxCoord.z(); z += factor) {
        int outputY = 0;
        for (int y = minCoord.y(); y <= maxCoord.y(); y += factor) {
            int outputX = 0;
            for (int x = minCoord.x(); x <= maxCoord.x(); x += factor) {
                // Read value from input grid at current position
                openvdb::Coord inputCoord(x, y, z);
                float value = inputAccessor.getValue(inputCoord);
                
                // Write to output grid at downsampled position
                openvdb::Coord outputCoord(outputX, outputY, outputZ);
                outputAccessor.setValue(outputCoord, value);
                
                outputX++;
            }
            outputY++;
        }
        outputZ++;
    }
    
    std::cout << "Downsampling complete. Output has " << outputGrid->activeVoxelCount() 
              << " active voxels" << std::endl;
    
    // Handle transformation matrix
    if (keepTransform) {
        // Keep the original transformation matrix
        // This means the downsampled grid will represent a smaller physical volume
        outputGrid->setTransform(inputGrid->transform().copy());
        std::cout << "Using original transformation matrix (physical size will be smaller)" << std::endl;
    }
    else {
        // Scale the transformation matrix to maintain the same physical size
        // This adjusts the voxel size to compensate for the reduced resolution
        openvdb::math::Transform::Ptr newTransform = inputGrid->transform().copy();
        
        // Scale the transform by the downsample factor
        // This increases voxel size, keeping the physical dimensions the same
        newTransform->preScale(openvdb::Vec3d(factor, factor, factor));
        
        outputGrid->setTransform(newTransform);
        std::cout << "Scaled transformation matrix by factor " << factor 
                  << " (physical size maintained)" << std::endl;
    }
    
    return outputGrid;
}

/**
 * @brief Prints usage information
 */
void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " <input.vdb> <output.vdb> <factor> <keep_transform>" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  input.vdb       - Input OpenVDB file to downsample" << std::endl;
    std::cout << "  output.vdb      - Output OpenVDB file for downsampled result" << std::endl;
    std::cout << "  factor          - Integer downsample factor (e.g., 2 = every 2nd voxel)" << std::endl;
    std::cout << "  keep_transform  - 0 or 1:" << std::endl;
    std::cout << "                    1 = keep original transformation matrix (smaller physical size)" << std::endl;
    std::cout << "                    0 = scale transformation matrix (maintain physical size)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << programName << " input.vdb output.vdb 2 1" << std::endl;
    std::cout << "  (downsample by factor 2, keeping original transform)" << std::endl;
}

/**
 * @brief Main function
 */
int main(int argc, char* argv[])
{
    // Check command line arguments
    if (argc != 5) {
        std::cerr << "Error: Invalid number of arguments!" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    int factor = std::atoi(argv[3]);
    int keepTransformArg = std::atoi(argv[4]);
    bool keepTransform = (keepTransformArg != 0);
    
    // Validate arguments
    if (factor < 1) {
        std::cerr << "Error: Factor must be a positive integer!" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    if (keepTransformArg != 0 && keepTransformArg != 1) {
        std::cerr << "Warning: keep_transform should be 0 or 1, using " 
                  << (keepTransform ? "1" : "0") << std::endl;
    }
    
    std::cout << "============================================" << std::endl;
    std::cout << "OpenVDB Downsampler" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Input file:  " << inputFile << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    std::cout << "Factor:      " << factor << std::endl;
    std::cout << "Keep transform: " << (keepTransform ? "yes" : "no") << std::endl;
    std::cout << "============================================" << std::endl;
    
    // Initialize OpenVDB
    openvdb::initialize();
    
    try {
        // Open input file
        std::cout << "Opening input file: " << inputFile << std::endl;
        
        // Check if file exists and get its size
        struct stat fileStat;
        if (stat(inputFile.c_str(), &fileStat) != 0) {
            std::cerr << "Error: Input file does not exist or cannot be accessed: " << inputFile << std::endl;
            return 1;
        }
        std::cout << "File size: " << fileStat.st_size << " bytes (" 
                  << (fileStat.st_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
        
        openvdb::io::File file(inputFile);
        file.open();
        
        // Read all grids from the file
        openvdb::GridPtrVecPtr grids = file.getGrids();
        file.close();
        
        if (grids->empty()) {
            std::cerr << "Error: No grids found in input file!" << std::endl;
            return 1;
        }
        
        std::cout << "Found " << grids->size() << " grid(s) in input file" << std::endl;
        
        // Open output file for writing
        openvdb::io::File outFile(outputFile);
        openvdb::GridPtrVec outputGrids;
        
        // Process each grid
        for (size_t i = 0; i < grids->size(); ++i) {
            openvdb::GridBase::Ptr baseGrid = (*grids)[i];
            std::cout << std::endl << "Processing grid " << (i + 1) << " of " << grids->size() 
                      << ": " << baseGrid->getName() << std::endl;
            std::cout << "Grid type: " << baseGrid->type() << std::endl;
            
            // Check if it's a FloatGrid
            if (baseGrid->isType<openvdb::FloatGrid>()) {
                openvdb::FloatGrid::Ptr floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
                
                // Downsample the grid
                openvdb::FloatGrid::Ptr downsampledGrid = downsampleGrid(floatGrid, factor, keepTransform);
                
                if (downsampledGrid) {
                    // Preserve grid name
                    downsampledGrid->setName(floatGrid->getName());
                    outputGrids.push_back(downsampledGrid);
                    std::cout << "Successfully downsampled grid: " << downsampledGrid->getName() << std::endl;
                }
                else {
                    std::cerr << "Warning: Failed to downsample grid " << baseGrid->getName() << std::endl;
                }
            }
            else {
                std::cerr << "Warning: Skipping non-FloatGrid: " << baseGrid->getName() 
                          << " (type: " << baseGrid->type() << ")" << std::endl;
                std::cerr << "  Only FloatGrid is currently supported." << std::endl;
            }
        }
        
        // Write output file
        if (outputGrids.empty()) {
            std::cerr << "Error: No grids to write to output file!" << std::endl;
            return 1;
        }
        
        std::cout << std::endl << "Writing " << outputGrids.size() 
                  << " grid(s) to output file: " << outputFile << std::endl;
        outFile.write(outputGrids);
        outFile.close();
        
        std::cout << "============================================" << std::endl;
        std::cout << "Downsampling complete!" << std::endl;
        std::cout << "Output saved to: " << outputFile << std::endl;
        std::cout << "============================================" << std::endl;
        
        return 0;
    }
    catch (const std::runtime_error& e) {
        std::string errorMsg(e.what());
        std::cerr << "\n============================================" << std::endl;
        std::cerr << "RUNTIME ERROR" << std::endl;
        std::cerr << "============================================" << std::endl;
        std::cerr << "Error message: " << errorMsg << std::endl;
        
        // Check if it's a Blosc decompression error
        if (errorMsg.find("Blosc") != std::string::npos || 
            errorMsg.find("blosc") != std::string::npos ||
            errorMsg.find("BLOSC") != std::string::npos) {
            std::cerr << "\n*** BLOSC DECOMPRESSION ERROR DETECTED ***" << std::endl;
            std::cerr << "\nPossible causes:" << std::endl;
            std::cerr << "  1. Corrupted VDB file" << std::endl;
            std::cerr << "  2. Incomplete file transfer" << std::endl;
            std::cerr << "  3. File created with incompatible Blosc version" << std::endl;
            std::cerr << "  4. Insufficient memory for decompression" << std::endl;
            std::cerr << "  5. File metadata corruption" << std::endl;
            std::cerr << "\nDebugging steps:" << std::endl;
            std::cerr << "  - Verify file integrity (checksum if available)" << std::endl;
            std::cerr << "  - Check available system memory" << std::endl;
            std::cerr << "  - Try reading with vdb_print or vdb_view utilities" << std::endl;
            std::cerr << "  - Verify Blosc library version compatibility" << std::endl;
            std::cerr << "\nInput file: " << inputFile << std::endl;
            
            // Get file info again
            struct stat fileStat;
            if (stat(inputFile.c_str(), &fileStat) == 0) {
                std::cerr << "File size: " << fileStat.st_size << " bytes" << std::endl;
            }
        }
        std::cerr << "============================================" << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "\n============================================" << std::endl;
        std::cerr << "EXCEPTION" << std::endl;
        std::cerr << "============================================" << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Input file: " << inputFile << std::endl;
        std::cerr << "============================================" << std::endl;
        return 1;
    }
}
