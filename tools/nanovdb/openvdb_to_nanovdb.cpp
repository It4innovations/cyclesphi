// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <iostream>
#include <cstdlib>

#define NANOVDB_USE_OPENVDB

#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include "w.openvdb.array.h"

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.vdb> <output.nvdb>\n";
        std::cerr << "  Converts the first grid from an OpenVDB file to a NanoVDB file.\n";
        return EXIT_FAILURE;
    }

    const std::string inputFile = argv[1];
    const std::string outputFile = argv[2];

    try {
        // Initialize OpenVDB library
        openvdb::initialize();
        
        // Register VDBArrayWrapper grid types
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<4, float>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<4, double>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<10, float>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<10, double>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<20, float>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<20, double>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<35, float>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<35, double>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<56, float>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<56, double>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<84, float>, 5, 4, 3>::Type>::registerGrid();
        openvdb::Grid<openvdb::tree::Tree4<it4i::mesio::VDBArrayWrapper<84, double>, 5, 4, 3>::Type>::registerGrid();

        std::cout << "============================================\n";
        std::cout << "OpenVDB to NanoVDB Converter\n";
        std::cout << "============================================\n";
        std::cout << "Input:  " << inputFile << "\n";
        std::cout << "Output: " << outputFile << "\n\n";

        // Open the OpenVDB file
        openvdb::io::File file(inputFile);
        file.open(false); // disable delayed loading

        // Get all grids from the file
        auto openvdbGrids = file.getGrids();
        
        if (!openvdbGrids || openvdbGrids->empty()) {
            std::cerr << "Error: No grids found in the OpenVDB file\n";
            return EXIT_FAILURE;
        }

        std::cout << "Found " << openvdbGrids->size() << " grid(s) in file\n";
        std::cout << "Converting first grid...\n\n";

        // Get the first grid
        auto openvdbGrid = (*openvdbGrids)[0];
        const std::string gridName = openvdbGrid->getName();
        
        std::cout << "Grid name: " << gridName << "\n";
        std::cout << "Grid type: " << openvdbGrid->type() << "\n";
        
        // Get bounding box information
        openvdb::CoordBBox bbox = openvdbGrid->evalActiveVoxelBoundingBox();
        openvdb::Coord bboxMin = bbox.min();
        openvdb::Coord bboxMax = bbox.max();
        openvdb::Coord dims = bbox.dim();
        
        std::cout << "Bounding box: (" << bboxMin.x() << "," << bboxMin.y() << "," << bboxMin.z() << ") -> "
                  << "(" << bboxMax.x() << "," << bboxMax.y() << "," << bboxMax.z() << ")\n";
        std::cout << "Dimensions: " << dims.x() << " x " << dims.y() << " x " << dims.z() << "\n";
        std::cout << "Active voxel count: " << openvdbGrid->activeVoxelCount() << "\n\n";
        
        // Convert OpenVDB grid to NanoVDB
        std::cout << "Converting to NanoVDB...\n";
        auto nanoHandle = nanovdb::tools::openToNanoVDB(openvdbGrid);
        
        if (!nanoHandle) {
            std::cerr << "Error: Failed to convert grid to NanoVDB\n";
            return EXIT_FAILURE;
        }
        
        std::cout << "Conversion successful!\n";
        std::cout << "NanoVDB grid size: " << nanoHandle.size() << " bytes\n\n";
        
        // Write NanoVDB grid to file
        std::cout << "Writing NanoVDB file...\n";
        nanovdb::io::writeGrid(outputFile, nanoHandle);
        
        std::cout << "============================================\n";
        std::cout << "Conversion completed successfully!\n";
        std::cout << "Output written to: " << outputFile << "\n";
        std::cout << "============================================\n";

        file.close();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Error: Unknown error occurred\n";
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
