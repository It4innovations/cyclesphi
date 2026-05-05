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

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Downsamples an OpenVDB grid by a given integer factor
 * 
 * Uses copyToDense to convert grid to dense data, downsamples using OpenMP,
 * then uses copyFromDense to create the output sparse grid.
 * 
 * @param inputGrid The input OpenVDB FloatGrid to downsample
 * @param factor The downsample factor (e.g., 2 means read every 2nd voxel)
 * @param keepTransform If true, keeps the original transformation matrix; 
 *                      if false, sets transform to identity matrix
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
    
#ifdef _OPENMP
    std::cout << "OpenMP enabled with " << omp_get_max_threads() << " threads" << std::endl;
#else
    std::cout << "OpenMP not enabled (serial execution)" << std::endl;
#endif
    
    // Step 1: Copy input grid to dense array
    std::cout << "Copying input grid to dense array..." << std::endl;
    openvdb::tools::Dense<float> inputDense(bbox, inputGrid->background());
    openvdb::tools::copyToDense(*inputGrid, inputDense);
    
    // Get pointer to input dense data
    float* inputData = inputDense.data();
    
    // Step 2: Create output dense array for downsampled data
    std::cout << "Creating downsampled dense array..." << std::endl;
    openvdb::CoordBBox outputBBox(openvdb::Coord(0, 0, 0), 
                                   openvdb::Coord(newDimX - 1, newDimY - 1, newDimZ - 1));
    openvdb::tools::Dense<float> outputDense(outputBBox, inputGrid->background());
    float* outputData = outputDense.data();
    
    // Step 3: Downsample using OpenMP
    std::cout << "Downsampling with OpenMP..." << std::endl;
    
    const size_t inputDimX = dim.x();
    const size_t inputDimY = dim.y();
    const size_t inputDimZ = dim.z();
    
    // Parallel loop over output voxels
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int oz = 0; oz < newDimZ; ++oz) {
        for (int oy = 0; oy < newDimY; ++oy) {
            for (int ox = 0; ox < newDimX; ++ox) {
                // Calculate input coordinates
                size_t ix = static_cast<size_t>(ox) * factor;
                size_t iy = static_cast<size_t>(oy) * factor;
                size_t iz = static_cast<size_t>(oz) * factor;
                
                // Bounds check
                if (ix < inputDimX && iy < inputDimY && iz < inputDimZ) {
                    // Calculate linear indices (prevent integer overflow)
                    size_t inputIdx = iz * (inputDimY * inputDimX) + iy * inputDimX + ix;
                    size_t outputIdx = static_cast<size_t>(oz) * (static_cast<size_t>(newDimY) * static_cast<size_t>(newDimX)) 
                                      + static_cast<size_t>(oy) * static_cast<size_t>(newDimX) + static_cast<size_t>(ox);
                    
                    // Copy value
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        }
    }
    
    std::cout << "Downsampling complete." << std::endl;
    
    // Step 4: Create output grid and copy from dense
    std::cout << "Creating output grid from downsampled dense data..." << std::endl;
    openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create(inputGrid->background());
    
    // Copy metadata from input grid
    outputGrid->insertMeta(*inputGrid);
    
    // Copy from dense to output grid
    openvdb::tools::copyFromDense(outputDense, *outputGrid, /*tolerance=*/0.0f);
    
    std::cout << "Output has " << outputGrid->activeVoxelCount() 
              << " active voxels" << std::endl;
    
    // Handle transformation matrix
    if (keepTransform) {
        // Keep the original transformation matrix
        outputGrid->setTransform(inputGrid->transform().copy());
        std::cout << "Using original transformation matrix" << std::endl;
    }
    else {
        // Set transformation matrix to identity
        openvdb::math::Transform::Ptr identityTransform = openvdb::math::Transform::createLinearTransform();
        outputGrid->setTransform(identityTransform);
        std::cout << "Set transformation matrix to identity" << std::endl;
    }
    
    return outputGrid;
}

/**
 * @brief Cuts a region from an OpenVDB grid using a bounding box
 * 
 * Extracts a specific region defined by a bounding box from the input grid.
 * Uses copyToDense to read the region and copyFromDense to create the output grid.
 * 
 * @param inputGrid The input OpenVDB FloatGrid to cut from
 * @param bboxMin Minimum coordinates of the bounding box
 * @param bboxMax Maximum coordinates of the bounding box
 * @param keepTransform If true, keeps the original transformation matrix; 
 *                      if false, sets transform to identity
 * @return Pointer to the cut FloatGrid
 */
openvdb::FloatGrid::Ptr cutGrid(openvdb::FloatGrid::Ptr inputGrid,
                                const openvdb::Coord& bboxMin,
                                const openvdb::Coord& bboxMax,
                                bool keepTransform)
{
    if (!inputGrid) {
        std::cerr << "Error: Input grid is null!" << std::endl;
        return nullptr;
    }
    
    // Validate bounding box
    if (bboxMin.x() > bboxMax.x() || bboxMin.y() > bboxMax.y() || bboxMin.z() > bboxMax.z()) {
        std::cerr << "Error: Invalid bounding box! Min coordinates must be <= max coordinates." << std::endl;
        return nullptr;
    }
    
    std::cout << "Cut bounding box: min=" << bboxMin << ", max=" << bboxMax << std::endl;
    
    // Create bounding box
    openvdb::CoordBBox bbox(bboxMin, bboxMax);
    openvdb::Coord dim = bbox.dim();
    
    std::cout << "Cut region dimensions: " << dim.x() << " x " << dim.y() 
              << " x " << dim.z() << std::endl;
    std::cout << "Keep original transform: " << (keepTransform ? "yes" : "no (set to identity)") << std::endl;
    
    // Create dense grid to hold the cut region
    std::cout << "Copying region to dense grid..." << std::endl;
    openvdb::tools::Dense<float> dense(bbox, inputGrid->background());
    openvdb::tools::copyToDense(*inputGrid, dense);
    
    std::cout << "Creating output grid from dense data..." << std::endl;
    
    // Create output grid with same background value as input
    openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create(inputGrid->background());
    
    // Copy metadata from input grid
    outputGrid->insertMeta(*inputGrid);
    
    // Copy from dense to output grid, starting at origin (0,0,0)
    openvdb::Coord outputOrigin(0, 0, 0);
    openvdb::tools::copyFromDense(dense, *outputGrid, /*tolerance=*/0.0f);
    
    std::cout << "Cut complete. Output has " << outputGrid->activeVoxelCount() 
              << " active voxels" << std::endl;
    
    // Handle transformation matrix
    if (keepTransform) {
        // Keep the original transformation matrix
        outputGrid->setTransform(inputGrid->transform().copy());
        std::cout << "Using original transformation matrix" << std::endl;
    }
    else {
        // Set transform to identity
        openvdb::math::Transform::Ptr identityTransform = openvdb::math::Transform::createLinearTransform();
        outputGrid->setTransform(identityTransform);
        std::cout << "Set transformation matrix to identity" << std::endl;
    }
    
    return outputGrid;
}

/**
 * @brief Prints usage information
 */
void printUsage(const char* programName)
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  Downsample mode:" << std::endl;
    std::cout << "    " << programName << " <input.vdb> <output.vdb> <factor> <keep_transform> [compression]" << std::endl;
    std::cout << std::endl;
    std::cout << "  Cut mode:" << std::endl;
    std::cout << "    " << programName << " <input.vdb> <output.vdb> --cut-bbox <lowX> <lowY> <lowZ> <highX> <highY> <highZ> <keep_transform> [compression]" << std::endl;
    std::cout << std::endl;
    std::cout << "Downsample mode arguments:" << std::endl;
    std::cout << "  input.vdb       - Input OpenVDB file to downsample" << std::endl;
    std::cout << "  output.vdb      - Output OpenVDB file for downsampled result" << std::endl;
    std::cout << "  factor          - Integer downsample factor (e.g., 2 = every 2nd voxel)" << std::endl;
    std::cout << "  keep_transform  - 0 or 1:" << std::endl;
    std::cout << "                    1 = keep original transformation matrix" << std::endl;
    std::cout << "                    0 = set transformation matrix to identity" << std::endl;
    std::cout << "  compression     - Optional compression type (default: BLOSC):" << std::endl;
    std::cout << "                    NONE  - No compression" << std::endl;
    std::cout << "                    BLOSC - Blosc compression (recommended)" << std::endl;
    std::cout << "                    ZIP   - ZIP/ZLIB compression" << std::endl;
    std::cout << std::endl;
    std::cout << "Cut mode arguments:" << std::endl;
    std::cout << "  input.vdb       - Input OpenVDB file to cut from" << std::endl;
    std::cout << "  output.vdb      - Output OpenVDB file for cut result" << std::endl;
    std::cout << "  --cut-bbox      - Flag to enable cut mode" << std::endl;
    std::cout << "  lowX lowY lowZ  - Minimum coordinates of bounding box (integers)" << std::endl;
    std::cout << "  highX highY highZ - Maximum coordinates of bounding box (integers)" << std::endl;
    std::cout << "  keep_transform  - 0 or 1:" << std::endl;
    std::cout << "                    1 = keep original transformation matrix" << std::endl;
    std::cout << "                    0 = set transformation matrix to identity" << std::endl;
    std::cout << "  compression     - Optional compression type (default: BLOSC):" << std::endl;
    std::cout << "                    NONE  - No compression" << std::endl;
    std::cout << "                    BLOSC - Blosc compression (recommended)" << std::endl;
    std::cout << "                    ZIP   - ZIP/ZLIB compression" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " input.vdb output.vdb 2 1" << std::endl;
    std::cout << "  (downsample by 2x, keep original transform, default BLOSC compression)" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << programName << " input.vdb output.vdb 2 0 NONE" << std::endl;
    std::cout << "  (downsample by 2x, set identity transform, no compression)" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << programName << " input.vdb output.vdb 1 0 NONE" << std::endl;
    std::cout << "  (copy grid with identity transform, no compression)" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << programName << " input.vdb output.vdb --cut-bbox 0 0 0 100 100 100 0 ZIP" << std::endl;
    std::cout << "  (cut region from (0,0,0) to (100,100,100), identity transform, ZIP compression)" << std::endl;
}

/**
 * @brief Main function
 */
int main(int argc, char* argv[])
{
    // Determine operation mode
    bool cutMode = false;
    std::string inputFile;
    std::string outputFile;
    int factor = 0;
    openvdb::Coord bboxMin, bboxMax;
    bool keepTransform = false;
    std::string compressionType = "BLOSC";  // Default compression
    
    // Check for cut mode (11 or 12 arguments)
    if ((argc == 11 || argc == 12) && std::string(argv[3]) == "--cut-bbox") {
        cutMode = true;
        inputFile = argv[1];
        outputFile = argv[2];
        // Parse bounding box coordinates
        int lowX = std::atoi(argv[4]);
        int lowY = std::atoi(argv[5]);
        int lowZ = std::atoi(argv[6]);
        int highX = std::atoi(argv[7]);
        int highY = std::atoi(argv[8]);
        int highZ = std::atoi(argv[9]);
        bboxMin = openvdb::Coord(lowX, lowY, lowZ);
        bboxMax = openvdb::Coord(highX, highY, highZ);
        int keepTransformArg = std::atoi(argv[10]);
        keepTransform = (keepTransformArg != 0);
        
        if (keepTransformArg != 0 && keepTransformArg != 1) {
            std::cerr << "Warning: keep_transform should be 0 or 1, using " 
                      << (keepTransform ? "1" : "0") << std::endl;
        }
        
        // Parse optional compression argument
        if (argc == 12) {
            compressionType = argv[11];
        }
    }
    // Check for downsample mode (5 or 6 arguments)
    else if (argc == 5 || argc == 6) {
        cutMode = false;
        inputFile = argv[1];
        outputFile = argv[2];
        factor = std::atoi(argv[3]);
        int keepTransformArg = std::atoi(argv[4]);
        keepTransform = (keepTransformArg != 0);
        
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
        
        // Parse optional compression argument
        if (argc == 6) {
            compressionType = argv[5];
        }
    }
    else {
        std::cerr << "Error: Invalid number of arguments!" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // Validate compression type
    if (compressionType != "NONE" && compressionType != "BLOSC" && compressionType != "ZIP") {
        std::cerr << "Error: Invalid compression type '" << compressionType << "'!" << std::endl;
        std::cerr << "Valid options are: NONE, BLOSC, ZIP" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    std::cout << "============================================" << std::endl;
    if (cutMode) {
        std::cout << "OpenVDB Cut Tool" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Input file:  " << inputFile << std::endl;
        std::cout << "Output file: " << outputFile << std::endl;
        std::cout << "Mode:        Cut bounding box" << std::endl;
        std::cout << "BBox min:    " << bboxMin << std::endl;
        std::cout << "BBox max:    " << bboxMax << std::endl;
        std::cout << "Keep transform: " << (keepTransform ? "yes" : "no (identity)") << std::endl;
        std::cout << "Compression: " << compressionType << std::endl;
    }
    else {
        std::cout << "OpenVDB Downsampler" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Input file:  " << inputFile << std::endl;
        std::cout << "Output file: " << outputFile << std::endl;
        std::cout << "Mode:        Downsample" << std::endl;
        std::cout << "Factor:      " << factor << std::endl;
        std::cout << "Keep transform: " << (keepTransform ? "yes" : "no (identity)") << std::endl;
        std::cout << "Compression: " << compressionType << std::endl;
    }
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
                
                openvdb::FloatGrid::Ptr processedGrid;
                
                if (cutMode) {
                    // Cut the grid
                    processedGrid = cutGrid(floatGrid, bboxMin, bboxMax, keepTransform);
                }
                else {
                    // Downsample the grid
                    processedGrid = downsampleGrid(floatGrid, factor, keepTransform);
                }
                
                if (processedGrid) {
                    // Preserve grid name
                    processedGrid->setName(floatGrid->getName());
                    outputGrids.push_back(processedGrid);
                    std::cout << "Successfully processed grid: " << processedGrid->getName() << std::endl;
                }
                else {
                    std::cerr << "Warning: Failed to process grid " << baseGrid->getName() << std::endl;
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
        
        // Set compression flags
        uint32_t compressionFlags = openvdb::io::COMPRESS_ACTIVE_MASK;
        if (compressionType == "NONE") {
            compressionFlags = openvdb::io::COMPRESS_NONE;
            std::cout << std::endl << "Using no compression" << std::endl;
        }
        else if (compressionType == "BLOSC") {
            compressionFlags |= openvdb::io::COMPRESS_BLOSC;
            std::cout << std::endl << "Using BLOSC compression" << std::endl;
        }
        else if (compressionType == "ZIP") {
            compressionFlags |= openvdb::io::COMPRESS_ZIP;
            std::cout << std::endl << "Using ZIP compression" << std::endl;
        }
        
        std::cout << "Writing " << outputGrids.size() 
                  << " grid(s) to output file: " << outputFile << std::endl;
        outFile.setCompression(compressionFlags);
        outFile.write(outputGrids);
        outFile.close();
        
        std::cout << "============================================" << std::endl;
        std::cout << "Processing complete!" << std::endl;
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
