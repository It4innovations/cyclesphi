#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <omp.h>

struct float3 {
    float x, y, z;
};

// Helper function to print usage
void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " <x_field.vdb> <y_field.vdb> <z_field.vdb> [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --mode <magnitude|vector>     Calculate magnitude (scalar) or keep vector (default: vector)" << std::endl;
    std::cerr << "  --format <raw|vdb>            Output format: raw binary or VDB (default: raw)" << std::endl;
    std::cerr << "  --output <filename>           Output filename (optional, auto-generated if not provided)" << std::endl;
    std::cerr << "  --normalize                   Normalize vectors (only applies to vector mode)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  " << progName << " x.vdb y.vdb z.vdb --mode vector --format raw" << std::endl;
    std::cerr << "  " << progName << " x.vdb y.vdb z.vdb --mode magnitude --format vdb --output mag.vdb" << std::endl;
    std::cerr << "  " << progName << " x.vdb y.vdb z.vdb --mode vector --format vdb --normalize" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    // Parse input files
    std::string filenames[3] = {argv[1], argv[2], argv[3]};
    
    // Default options
    std::string mode = "vector";        // "magnitude" or "vector"
    std::string format = "raw";          // "raw" or "vdb"
    std::string outputFile = "";
    bool normalize = false;
    
    // Parse command-line arguments
    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            if (mode != "magnitude" && mode != "vector") {
                std::cerr << "Error: mode must be 'magnitude' or 'vector'" << std::endl;
                return 1;
            }
        }
        else if (arg == "--format" && i + 1 < argc) {
            format = argv[++i];
            if (format != "raw" && format != "vdb") {
                std::cerr << "Error: format must be 'raw' or 'vdb'" << std::endl;
                return 1;
            }
        }
        else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        }
        else if (arg == "--normalize") {
            normalize = true;
        }
        else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Validate options
    if (mode == "magnitude" && normalize) {
        std::cerr << "Warning: --normalize is ignored in magnitude mode" << std::endl;
        normalize = false;
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Mode: " << mode << std::endl;
    std::cout << "  Format: " << format << std::endl;
    std::cout << "  Normalize: " << (normalize ? "yes" : "no") << std::endl;
    std::cout << std::endl;

    // Initialize OpenVDB
    openvdb::initialize();

    openvdb::GridBase::Ptr grids[3];
    openvdb::CoordBBox bboxes[3];
    openvdb::Coord dims[3];

    std::cout << "Reading VDB files..." << std::endl;

    // Read the three VDB files in parallel
    #pragma omp parallel for num_threads(3)
    for (int i = 0; i < 3; i++) {
        try {
            openvdb::io::File file(filenames[i]);
            file.open();
            
            // Read the first grid
            openvdb::GridPtrVecPtr gridsVec = file.getGrids();
            if (gridsVec->empty()) {
                #pragma omp critical
                {
                    std::cerr << "Error: No grids found in " << filenames[i] << std::endl;
                }
                continue;
            }

            grids[i] = (*gridsVec)[0];
            file.close();

            #pragma omp critical
            {
                std::cout << "  Loaded " << filenames[i] << " - Grid: " << grids[i]->getName() 
                          << " (Type: " << grids[i]->type() << ")" << std::endl;
            }
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Error reading " << filenames[i] << ": " << e.what() << std::endl;
            }
        }
    }

    // Verify all grids were loaded
    for (int i = 0; i < 3; i++) {
        if (!grids[i]) {
            std::cerr << "Failed to load grid from " << filenames[i] << std::endl;
            return 1;
        }
    }

    // Get the bounding boxes and verify they match
    openvdb::CoordBBox bbox;
    for (int i = 0; i < 3; i++) {
        bboxes[i] = grids[i]->evalActiveVoxelBoundingBox();
        dims[i] = bboxes[i].dim();
        
        if (i == 0) {
            bbox = bboxes[i];
        } else {
            if (dims[i] != dims[0]) {
                std::cerr << "Warning: Grid dimensions don't match!" << std::endl;
                std::cerr << "  Grid 0: " << dims[0] << std::endl;
                std::cerr << "  Grid " << i << ": " << dims[i] << std::endl;
            }
        }
    }

    openvdb::Coord dim = dims[0];
    size_t totalVoxels = static_cast<size_t>(dim.x()) * static_cast<size_t>(dim.y()) * static_cast<size_t>(dim.z());
    
    std::cout << "Bounding box: " << bbox.min() << " to " << bbox.max() << std::endl;
    std::cout << "Grid resolution: " << dim.x() << "x" << dim.y() << "x" << dim.z() << std::endl;
    std::cout << "Total voxels: " << totalVoxels << std::endl;

    // Convert grids to FloatGrid for processing
    openvdb::FloatGrid::Ptr floatGrids[3];
    for (int i = 0; i < 3; i++) {
        floatGrids[i] = openvdb::gridPtrCast<openvdb::FloatGrid>(grids[i]);
        if (!floatGrids[i]) {
            std::cerr << "Error: Grid " << i << " is not a FloatGrid" << std::endl;
            return 1;
        }
    }

    // Create dense representations
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> denseX(bbox);
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> denseY(bbox);
    openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> denseZ(bbox);

    std::cout << "Converting to dense representation..." << std::endl;
    openvdb::tools::copyToDense(*floatGrids[0], denseX);
    openvdb::tools::copyToDense(*floatGrids[1], denseY);
    openvdb::tools::copyToDense(*floatGrids[2], denseZ);

    openvdb::Coord minCoord = bbox.min();
    openvdb::Coord maxCoord = bbox.max();

    // Process based on mode and format
    if (mode == "magnitude") {
        // Calculate magnitude (scalar output)
        std::cout << "Calculating magnitude..." << std::endl;
        std::cout << "Memory required: " << (totalVoxels * sizeof(float)) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        
        std::vector<float> magnitudes(totalVoxels);
        
        #pragma omp parallel for collapse(2)
        for (int k = minCoord.z(); k <= maxCoord.z(); k++) {
            for (int j = minCoord.y(); j <= maxCoord.y(); j++) {
                for (int i = minCoord.x(); i <= maxCoord.x(); i++) {
                    openvdb::Coord coord(i, j, k);
                    int ix = i - minCoord.x();
                    int iy = j - minCoord.y();
                    int iz = k - minCoord.z();
                    size_t idx = static_cast<size_t>(ix) + static_cast<size_t>(iy) * dim.x() + static_cast<size_t>(iz) * dim.x() * dim.y();
                    
                    float x = denseX.getValue(coord);
                    float y = denseY.getValue(coord);
                    float z = denseZ.getValue(coord);
                    
                    magnitudes[idx] = std::sqrt(x * x + y * y + z * z);
                }
            }
        }
        
        // Output magnitude
        if (format == "raw") {
            // Generate output filename if not provided
            if (outputFile.empty()) {
                std::ostringstream oss;
                oss << dim.x() << "_" << dim.y() << "_" << dim.z() << "_magnitude.raw";
                outputFile = oss.str();
            }
            
            std::cout << "Writing to " << outputFile << "..." << std::endl;
            std::ofstream outFile(outputFile, std::ios::binary);
            if (!outFile) {
                std::cerr << "Error: Cannot create output file " << outputFile << std::endl;
                return 1;
            }
            
            outFile.write(reinterpret_cast<const char*>(magnitudes.data()), totalVoxels * sizeof(float));
            outFile.close();
            
            std::cout << "Successfully wrote " << totalVoxels * sizeof(float) << " bytes" << std::endl;
        }
        else { // vdb format
            // Generate output filename if not provided
            if (outputFile.empty()) {
                outputFile = "magnitude.vdb";
            }
            
            std::cout << "Writing to " << outputFile << "..." << std::endl;
            
            // Create a new FloatGrid
            openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create();
            outputGrid->setName("magnitude");
            outputGrid->setGridClass(openvdb::GRID_FOG_VOLUME);
            outputGrid->setTransform(floatGrids[0]->transformPtr());
            
            // Use accessor for efficient grid access
            openvdb::FloatGrid::Accessor accessor = outputGrid->getAccessor();
            
            // Copy data from magnitude array to grid
            for (int k = minCoord.z(); k <= maxCoord.z(); k++) {
                for (int j = minCoord.y(); j <= maxCoord.y(); j++) {
                    for (int i = minCoord.x(); i <= maxCoord.x(); i++) {
                        openvdb::Coord coord(i, j, k);
                        int ix = i - minCoord.x();
                        int iy = j - minCoord.y();
                        int iz = k - minCoord.z();
                        size_t idx = static_cast<size_t>(ix) + static_cast<size_t>(iy) * dim.x() + static_cast<size_t>(iz) * dim.x() * dim.y();
                        
                        accessor.setValue(coord, magnitudes[idx]);
                    }
                }
            }
            
            // Write VDB file
            openvdb::io::File file(outputFile);
            openvdb::GridPtrVec gridsToWrite;
            gridsToWrite.push_back(outputGrid);
            file.write(gridsToWrite);
            file.close();
            
            std::cout << "Successfully wrote VDB file with magnitude grid" << std::endl;
        }
    }
    else { // vector mode
        std::cout << "Processing vector data..." << std::endl;
        std::cout << "Memory required: " << (totalVoxels * sizeof(float3)) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        
        std::vector<float3> vectors(totalVoxels);
        
        #pragma omp parallel for collapse(2)
        for (int k = minCoord.z(); k <= maxCoord.z(); k++) {
            for (int j = minCoord.y(); j <= maxCoord.y(); j++) {
                for (int i = minCoord.x(); i <= maxCoord.x(); i++) {
                    openvdb::Coord coord(i, j, k);
                    int ix = i - minCoord.x();
                    int iy = j - minCoord.y();
                    int iz = k - minCoord.z();
                    size_t idx = static_cast<size_t>(ix) + static_cast<size_t>(iy) * dim.x() + static_cast<size_t>(iz) * dim.x() * dim.y();
                    
                    float x = denseX.getValue(coord);
                    float y = denseY.getValue(coord);
                    float z = denseZ.getValue(coord);
                    
                    if (normalize) {
                        float magnitude = std::sqrt(x * x + y * y + z * z);
                        if (magnitude > 1e-10f) {
                            vectors[idx].x = x / magnitude;
                            vectors[idx].y = y / magnitude;
                            vectors[idx].z = z / magnitude;
                        } else {
                            vectors[idx].x = 0.0f;
                            vectors[idx].y = 0.0f;
                            vectors[idx].z = 0.0f;
                        }
                    } else {
                        vectors[idx].x = x;
                        vectors[idx].y = y;
                        vectors[idx].z = z;
                    }
                }
            }
        }
        
        // Output vector
        if (format == "raw") {
            // Generate output filename if not provided
            if (outputFile.empty()) {
                std::ostringstream oss;
                oss << dim.x() << "_" << dim.y() << "_" << dim.z() << "_float3.raw";
                outputFile = oss.str();
            }
            
            std::cout << "Writing to " << outputFile << "..." << std::endl;
            std::ofstream outFile(outputFile, std::ios::binary);
            if (!outFile) {
                std::cerr << "Error: Cannot create output file " << outputFile << std::endl;
                return 1;
            }
            
            outFile.write(reinterpret_cast<const char*>(vectors.data()), totalVoxels * sizeof(float3));
            outFile.close();
            
            std::cout << "Successfully wrote " << totalVoxels * sizeof(float3) << " bytes" << std::endl;
        }
        else { // vdb format
            // Generate output filename if not provided
            if (outputFile.empty()) {
                outputFile = "vector.vdb";
            }
            
            std::cout << "Writing to " << outputFile << "..." << std::endl;
            
            // Create a new Vec3fGrid for vector data
            openvdb::Vec3fGrid::Ptr outputGrid = openvdb::Vec3fGrid::create();
            outputGrid->setName("vector");
            outputGrid->setGridClass(openvdb::GRID_STAGGERED);
            outputGrid->setTransform(floatGrids[0]->transformPtr());
            
            // Use accessor for efficient grid access
            openvdb::Vec3fGrid::Accessor accessor = outputGrid->getAccessor();
            
            // Copy data from vector array to grid
            for (int k = minCoord.z(); k <= maxCoord.z(); k++) {
                for (int j = minCoord.y(); j <= maxCoord.y(); j++) {
                    for (int i = minCoord.x(); i <= maxCoord.x(); i++) {
                        openvdb::Coord coord(i, j, k);
                        int ix = i - minCoord.x();
                        int iy = j - minCoord.y();
                        int iz = k - minCoord.z();
                        size_t idx = static_cast<size_t>(ix) + static_cast<size_t>(iy) * dim.x() + static_cast<size_t>(iz) * dim.x() * dim.y();
                        
                        openvdb::Vec3f vec(vectors[idx].x, vectors[idx].y, vectors[idx].z);
                        accessor.setValue(coord, vec);
                    }
                }
            }
            
            // Write VDB file
            openvdb::io::File file(outputFile);
            openvdb::GridPtrVec gridsToWrite;
            gridsToWrite.push_back(outputGrid);
            file.write(gridsToWrite);
            file.close();
            
            std::cout << "Successfully wrote VDB file with vector grid" << std::endl;
        }
    }

    return 0;
}
