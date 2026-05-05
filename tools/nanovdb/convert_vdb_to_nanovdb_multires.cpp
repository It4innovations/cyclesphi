// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <vector>
#include <iostream>

# define NANOVDB_USE_OPENVDB

#include <nanovdb/io/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>


#define ALIGNMENT_SIZE 32
#define ALIGN_SIZE(size) (((size) + (ALIGNMENT_SIZE - 1)) & ~(ALIGNMENT_SIZE - 1))

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.vdb> <output.bin> [asc|desc]" << std::endl;
        std::cerr << "  Reads OpenVDB file and converts all grids to NanoVDB format" << std::endl;
        std::cerr << "  and saves them to a binary file (one grid per level)" << std::endl;
        std::cerr << "  Optional sort order: 'asc' (default) or 'desc' by resolution" << std::endl;
        return EXIT_FAILURE;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    
    // Parse sort order (default is ascending)
    bool sortAscending = true;
    if (argc >= 4) {
        std::string sortOrder = argv[3];
        std::transform(sortOrder.begin(), sortOrder.end(), sortOrder.begin(), ::tolower);
        if (sortOrder == "desc") {
            sortAscending = false;
        } else if (sortOrder != "asc") {
            std::cerr << "Warning: Unknown sort order '" << argv[3] << "', using 'asc' by default" << std::endl;
        }
    }
    std::cout << "Sort order: " << (sortAscending ? "ascending" : "descending") << " by resolution" << std::endl;

    try {
        // Initialize OpenVDB
        openvdb::initialize();

        std::cout << "Opening OpenVDB file: " << inputFile << std::endl;

        // Open the OpenVDB file
        openvdb::io::File file(inputFile);
        file.open(false); // disable delayed loading

        // Get all grids from the file
        auto grids = file.getGrids();
        
        if (!grids || grids->empty()) {
            std::cerr << "No grids found in the OpenVDB file" << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Found " << grids->size() << " grid(s) in the file" << std::endl;

        // Create a vector of grids with their resolutions for sorting
        struct GridInfo {
            openvdb::GridBase::Ptr grid;
            size_t resolution;
        };
        
        std::vector<GridInfo> gridInfos;
        for (auto& grid : *grids) {
            // Calculate resolution as the maximum dimension of the active bounding box
            openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
            openvdb::Coord dim = bbox.dim();
            size_t resolution = std::max({dim.x(), dim.y(), dim.z()});
            
            std::cout << "Grid: " << grid->getName() 
                      << " (type: " << grid->type() 
                      << ", resolution: " << resolution << ")" << std::endl;
            
            gridInfos.push_back({grid, resolution});
        }
        
        // Sort grids by resolution
        std::sort(gridInfos.begin(), gridInfos.end(), 
            [sortAscending](const GridInfo& a, const GridInfo& b) {
                return sortAscending ? (a.resolution < b.resolution) : (a.resolution > b.resolution);
            });
        
        std::cout << "\nSorted grid order:" << std::endl;
        for (size_t i = 0; i < gridInfos.size(); ++i) {
            std::cout << "  Level " << i << ": " << gridInfos[i].grid->getName() 
                      << " (resolution: " << gridInfos[i].resolution << ")" << std::endl;
        }
        std::cout << std::endl;

        // Convert each OpenVDB grid to NanoVDB
        std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> nanoHandles;
        
        for (auto& gridInfo : gridInfos) {
            auto& grid = gridInfo.grid;
            std::cout << "Converting grid: " << grid->getName() 
                      << " (type: " << grid->type() 
                      << ", resolution: " << gridInfo.resolution << ")" << std::endl;
            
            // Convert OpenVDB grid to NanoVDB
            auto nanoHandle = nanovdb::tools::openToNanoVDB(grid);
            
            if (!nanoHandle) {
                std::cerr << "Failed to convert grid: " << grid->getName() << std::endl;
                continue;
            }
            
            nanoHandles.push_back(std::move(nanoHandle));
        }

        if (nanoHandles.empty()) {
            std::cerr << "No grids were successfully converted" << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Successfully converted " << nanoHandles.size() << " grid(s)" << std::endl;

        // Write to binary file
        std::cout << "Writing to binary file: " << outputFile << std::endl;
        std::ofstream outFile(outputFile, std::ios::binary);
        
        if (!outFile) {
            std::cerr << "Failed to open output file: " << outputFile << std::endl;
            return EXIT_FAILURE;
        }

        ///////////////////// NanoVDBMultiResImageLoader
        // format description of bin file:
        // size_t : number of levels (aligned to 32 bytes)
        // size_t : offset to grid1
        // grid0 data (aligned to 32 bytes)
        // size_t : offset to grid2
        // grid1 data (aligned to 32 bytes)
        // ...

        // Write number of levels (grids)
        size_t numLevels = nanoHandles.size();
        outFile.write(reinterpret_cast<const char*>(&numLevels), sizeof(size_t));
        std::cout << "Number of levels: " << numLevels << std::endl;

        // Pad to 32 bytes after num_levels
        size_t currentPos = sizeof(size_t);
        size_t paddingSize = ALIGN_SIZE(currentPos) - currentPos;
        std::vector<char> padding(paddingSize, 0);
        if (paddingSize > 0) {
            outFile.write(padding.data(), paddingSize);
        }
        currentPos = ALIGN_SIZE(currentPos);

        // Calculate offsets for each grid
        std::vector<size_t> offsets;
        for (size_t i = 0; i < numLevels; ++i) {
            if (i > 0) {
                // Current position includes: previous offset field + previous grid data (aligned)
                currentPos += sizeof(size_t); // offset field
                currentPos += ALIGN_SIZE(nanoHandles[i-1].size()); // previous grid aligned
            }
            offsets.push_back(currentPos + sizeof(size_t)); // position after the next offset field
        }

        // Write grids with offsets
        for (size_t i = 0; i < numLevels; ++i) {
            // Write offset to next grid (except for the last grid)
            if (i < numLevels - 1) {
                outFile.write(reinterpret_cast<const char*>(&offsets[i + 1]), sizeof(size_t));
                std::cout << "Offset to grid " << (i + 1) << ": " << offsets[i + 1] << std::endl;
            }

            // Write grid data
            const void* gridData = nanoHandles[i].data();
            size_t gridSize = nanoHandles[i].size();
            outFile.write(reinterpret_cast<const char*>(gridData), gridSize);
            std::cout << "Written grid " << i << " data (" << gridSize << " bytes)" << std::endl;

            // Pad to 32 bytes (except for the last grid, though padding it doesn't hurt)
            size_t alignedSize = ALIGN_SIZE(gridSize);
            paddingSize = alignedSize - gridSize;
            if (paddingSize > 0) {
                padding.resize(paddingSize, 0);
                outFile.write(padding.data(), paddingSize);
                std::cout << "Added " << paddingSize << " bytes padding for grid " << i << std::endl;
            }
        }

        outFile.close();
        std::cout << "Successfully saved multi-resolution NanoVDB data to: " << outputFile << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}