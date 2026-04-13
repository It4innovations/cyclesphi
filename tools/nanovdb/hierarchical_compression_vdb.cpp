

/*
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Portions of this code are derived from dense2vdb:
 * https://github.com/szellmann/dense2vdb
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Hierarchical VDB Compression Tool
 * 
 * This tool compresses OpenVDB grids using hierarchical compression based on
 * the paper "GPU Volume Rendering with Hierarchical Compression Using VDB".
 * DOI: https://doi.org/10.2312/pgv.20251152
 * 
 * The compression algorithm works by:
 * 1. Dividing the volume into hierarchical bricks (leaf nodes)
 * 2. Computing value ranges for each brick
 * 3. Sorting bricks by their distance from the background value
 * 4. Keeping only the most important bricks based on compression ratio
 * 5. Inactive bricks are represented by the background value
 */

#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>
#include <iostream>
#include <vector>
#include <cstdlib>

// OpenVDB
#include <openvdb/openvdb.h>
#include <openvdb/tools/DenseSparseTools.h>
#include <openvdb/math/Math.h>
#include <openvdb/io/File.h>

// Simple math types
struct int3 {
  int x, y, z;
  int3(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}
  int3 operator*(int s) const { return int3(x*s, y*s, z*s); }
};

struct float2 {
  float x, y;
  float2(float x_ = 0.f, float y_ = 0.f) : x(x_), y(y_) {}
};

std::ostream& operator<<(std::ostream& os, const float2& v) {
  return os << "[" << v.x << ", " << v.y << "]";
}

/**
 * Compute the value range (min, max) for a given region of a VDB grid
 */
float2 computeValueRange(openvdb::FloatGrid::Ptr grid, 
                        const openvdb::CoordBBox& region)
{
  auto accessor = grid->getConstAccessor();
  float minVal = 1e31f;
  float maxVal = -1e31f;
  
  for (auto iter = region.begin(); iter; ++iter) {
    float value = accessor.getValue(*iter);
    minVal = std::min(minVal, value);
    maxVal = std::max(maxVal, value);
  }
  
  return float2(minVal, maxVal);
}

/**
 * Find the background value by computing a histogram and finding
 * the most frequently occurring value
 */
float computeBackgroundValue(openvdb::FloatGrid::Ptr grid)
{
  std::cout << "Computing background value from histogram..." << std::endl;
  
  // First pass: compute value range
  auto accessor = grid->getConstAccessor();
  float minVal = 1e31f;
  float maxVal = -1e31f;
  size_t count = 0;
  
  for (auto iter = grid->beginValueOn(); iter; ++iter) {
    float value = iter.getValue();
    minVal = std::min(minVal, value);
    maxVal = std::max(maxVal, value);
    count++;
  }
  
  std::cout << "  Value range: " << float2(minVal, maxVal) << std::endl;
  std::cout << "  Active voxels: " << count << std::endl;
  
  // Build histogram
  const int numBins = 1024;
  std::vector<uint64_t> histogram(numBins, 0);
  
  for (auto iter = grid->beginValueOn(); iter; ++iter) {
    float value = iter.getValue();
    float normalized = (value - minVal) / (maxVal - minVal);
    int bin = std::min(int(normalized * (numBins - 1)), numBins - 1);
    histogram[bin]++;
  }
  
  // Find the bin with maximum count
  int maxBin = 0;
  uint64_t maxCount = 0;
  for (int i = 0; i < numBins; ++i) {
    if (histogram[i] > maxCount) {
      maxBin = i;
      maxCount = histogram[i];
    }
  }
  
  // Convert bin back to value
  float backgroundValue = minVal + (maxBin / float(numBins - 1)) * (maxVal - minVal);
  std::cout << "  Background value: " << backgroundValue << std::endl;
  
  return backgroundValue;
}

/**
 * Apply hierarchical compression to an OpenVDB grid
 * 
 * This implements the hierarchical compression algorithm from the paper
 * "GPU Volume Rendering with Hierarchical Compression Using VDB"
 * 
 * The algorithm:
 * 1. Divides the active region into bricks at the leaf level (8x8x8 voxels)
 * 2. Computes the value range for each brick
 * 3. Sorts bricks by their distance from the background value
 * 4. Keeps only the top N% of bricks based on the compression ratio
 * 5. Inactive bricks are pruned, leaving only the background value
 */
openvdb::FloatGrid::Ptr compressGrid(openvdb::FloatGrid::Ptr inputGrid, 
                                     double compressionRatio)
{
  std::cout << "\n=== Hierarchical Compression ===" << std::endl;
  std::cout << "Target compression ratio: " << compressionRatio << std::endl;
  
  // Get the background value (most common value in the grid)
  float backgroundValue = computeBackgroundValue(inputGrid);
  
  // Get the bounding box of active voxels
  openvdb::CoordBBox bbox = inputGrid->evalActiveVoxelBoundingBox();
  openvdb::Coord minCoord = bbox.min();
  openvdb::Coord maxCoord = bbox.max();
  
  int3 dims(maxCoord.x() - minCoord.x() + 1,
            maxCoord.y() - minCoord.y() + 1,
            maxCoord.z() - minCoord.z() + 1);
  
  std::cout << "Active bounding box: [" << minCoord << "] to [" << maxCoord << "]" << std::endl;
  std::cout << "Dimensions: " << dims.x << " x " << dims.y << " x " << dims.z << std::endl;
  
  // Create output grid with the background value
  openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create(backgroundValue);
  outputGrid->setName(inputGrid->getName());
  outputGrid->setTransform(inputGrid->transform().copy());
  
  // Handle full compression (ratio >= 1.0) - keep all active voxels
  if (compressionRatio >= 1.0) {
    std::cout << "Compression ratio >= 1.0, copying all active voxels..." << std::endl;
    outputGrid->tree().merge(inputGrid->tree());
    std::cout << "Active voxel count: " << outputGrid->tree().activeVoxelCount() << std::endl;
    return outputGrid;
  }
  
  // Hierarchical compression using brick-based approach
  // VDB uses a tree structure: we work at the leaf level (8x8x8 = 512 voxels per leaf)
  const int LEAF_SIZE = 8;  // VDB leaf nodes are 8x8x8
  
  auto divUp = [](int a, int b) { return (a + b - 1) / b; };
  
  int3 numBricks(divUp(dims.x, LEAF_SIZE),
                 divUp(dims.y, LEAF_SIZE),
                 divUp(dims.z, LEAF_SIZE));
  
  std::cout << "Number of bricks: " << numBricks.x << " x " 
            << numBricks.y << " x " << numBricks.z 
            << " = " << (numBricks.x * size_t(numBricks.y) * numBricks.z) << std::endl;
  
  // Structure to track brick information
  struct BrickInfo {
    int3 brickID;
    openvdb::CoordBBox region;
    float2 valueRange;
    float distance;  // Distance from background value
  };
  
  std::vector<BrickInfo> bricks;
  bricks.reserve(numBricks.x * size_t(numBricks.y) * numBricks.z);
  
  // Compute value range for each brick
  std::cout << "Computing value ranges for bricks..." << std::endl;
  
  for (int bz = 0; bz < numBricks.z; ++bz) {
    for (int by = 0; by < numBricks.y; ++by) {
      for (int bx = 0; bx < numBricks.x; ++bx) {
        openvdb::Coord lower(minCoord.x() + bx * LEAF_SIZE,
                            minCoord.y() + by * LEAF_SIZE,
                            minCoord.z() + bz * LEAF_SIZE);
        
        openvdb::Coord upper(std::min(lower.x() + LEAF_SIZE, maxCoord.x() + 1),
                            std::min(lower.y() + LEAF_SIZE, maxCoord.y() + 1),
                            std::min(lower.z() + LEAF_SIZE, maxCoord.z() + 1));
        
        openvdb::CoordBBox region(lower, upper.offsetBy(-1, -1, -1));
        float2 valueRange = computeValueRange(inputGrid, region);
        
        // Compute distance as maximum distance from background to either extreme
        float dist = std::max(std::abs(valueRange.x - backgroundValue),
                             std::abs(valueRange.y - backgroundValue));
        
        BrickInfo brick;
        brick.brickID = int3(bx, by, bz);
        brick.region = region;
        brick.valueRange = valueRange;
        brick.distance = dist;
        
        bricks.push_back(brick);
      }
    }
  }
  
  // Sort bricks by distance (ascending)
  // Bricks with values closer to background are less important
  std::cout << "Sorting bricks by importance..." << std::endl;
  std::sort(bricks.begin(), bricks.end(),
            [](const BrickInfo& a, const BrickInfo& b) {
              return a.distance < b.distance;
            });
  
  // Determine how many bricks to keep based on compression ratio
  size_t numBricksToKeep = size_t(bricks.size() * compressionRatio);
  numBricksToKeep = std::max(size_t(1), numBricksToKeep);  // Keep at least one brick
  
  std::cout << "Keeping " << numBricksToKeep << " out of " << bricks.size() 
            << " bricks" << std::endl;
  
  // Copy voxels from the most important bricks
  auto accessor = inputGrid->getConstAccessor();
  auto outputAccessor = outputGrid->getAccessor();
  
  size_t voxelsCopied = 0;
  for (size_t i = bricks.size() - numBricksToKeep; i < bricks.size(); ++i) {
    const BrickInfo& brick = bricks[i];
    
    // Copy all voxels in this brick's region
    for (auto iter = brick.region.begin(); iter; ++iter) {
      float value = accessor.getValue(*iter);
      outputAccessor.setValue(*iter, value);
      voxelsCopied++;
    }
  }
  
  // Prune to remove redundant nodes
  outputGrid->tree().prune();
  
  std::cout << "Voxels copied: " << voxelsCopied << std::endl;
  std::cout << "Active voxels after pruning: " << outputGrid->tree().activeVoxelCount() << std::endl;
  
  size_t originalCount = inputGrid->tree().activeVoxelCount();
  size_t compressedCount = outputGrid->tree().activeVoxelCount();
  double actualRatio = double(compressedCount) / double(originalCount);
  
  std::cout << "Actual compression ratio: " << actualRatio 
            << " (" << compressedCount << " / " << originalCount << ")" << std::endl;
  
  return outputGrid;
}

/**
 * Print usage information
 */
void printUsage(const char* programName)
{
  std::cout << "Usage: " << programName << " [options] <input.vdb> <output.vdb>" << std::endl;
  std::cout << "\nOptions:" << std::endl;
  std::cout << "  -r, --ratio <value>    Compression ratio (0.0-1.0, default: 0.5)" << std::endl;
  std::cout << "                         1.0 = no compression, 0.5 = 50% voxels kept" << std::endl;
  std::cout << "  -g, --grid <name>      Grid name to compress (default: first grid)" << std::endl;
  std::cout << "  -h, --help             Show this help message" << std::endl;
  std::cout << "\nExample:" << std::endl;
  std::cout << "  " << programName << " -r 0.3 input.vdb compressed.vdb" << std::endl;
}

/**
 * Main function
 */
int main(int argc, char** argv)
{
  // Default parameters
  double compressionRatio = 0.5;
  std::string inputFile;
  std::string outputFile;
  std::string gridName;
  
  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    
    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0]);
      return 0;
    }
    else if (arg == "-r" || arg == "--ratio") {
      if (i + 1 < argc) {
        compressionRatio = std::atof(argv[++i]);
        if (compressionRatio <= 0.0 || compressionRatio > 1.0) {
          std::cerr << "Error: Compression ratio must be between 0.0 and 1.0" << std::endl;
          return 1;
        }
      }
      else {
        std::cerr << "Error: --ratio requires a value" << std::endl;
        return 1;
      }
    }
    else if (arg == "-g" || arg == "--grid") {
      if (i + 1 < argc) {
        gridName = argv[++i];
      }
      else {
        std::cerr << "Error: --grid requires a name" << std::endl;
        return 1;
      }
    }
    else if (arg[0] == '-') {
      std::cerr << "Error: Unknown option: " << arg << std::endl;
      printUsage(argv[0]);
      return 1;
    }
    else {
      if (inputFile.empty()) {
        inputFile = arg;
      }
      else if (outputFile.empty()) {
        outputFile = arg;
      }
      else {
        std::cerr << "Error: Too many arguments" << std::endl;
        printUsage(argv[0]);
        return 1;
      }
    }
  }
  
  // Validate arguments
  if (inputFile.empty() || outputFile.empty()) {
    std::cerr << "Error: Input and output files are required" << std::endl;
    printUsage(argv[0]);
    return 1;
  }
  
  try {
    // Initialize OpenVDB
    openvdb::initialize();
    
    std::cout << "=== Hierarchical VDB Compression Tool ===" << std::endl;
    std::cout << "Input file:  " << inputFile << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    
    // Open input VDB file
    openvdb::io::File file(inputFile);
    file.open();
    
    openvdb::GridBase::Ptr baseGrid;
    
    // Get the grid to compress
    if (!gridName.empty()) {
      baseGrid = file.readGrid(gridName);
      if (!baseGrid) {
        std::cerr << "Error: Grid '" << gridName << "' not found in file" << std::endl;
        file.close();
        return 1;
      }
    }
    else {
      // Use the first grid
      if (file.getGrids()->empty()) {
        std::cerr << "Error: No grids found in file" << std::endl;
        file.close();
        return 1;
      }
      baseGrid = file.getGrids()->front();
    }
    
    file.close();
    
    std::cout << "Grid name: " << baseGrid->getName() << std::endl;
    std::cout << "Grid type: " << baseGrid->valueType() << std::endl;
    
    // Check if it's a FloatGrid
    openvdb::FloatGrid::Ptr floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
    if (!floatGrid) {
      std::cerr << "Error: Only FloatGrid is supported (grid type: " 
                << baseGrid->valueType() << ")" << std::endl;
      return 1;
    }
    
    std::cout << "Original active voxels: " << floatGrid->tree().activeVoxelCount() << std::endl;
    
    // Compress the grid
    openvdb::FloatGrid::Ptr compressedGrid = compressGrid(floatGrid, compressionRatio);
    
    // Write output file
    std::cout << "\nWriting compressed grid to: " << outputFile << std::endl;
    openvdb::io::File outFile(outputFile);
    openvdb::GridPtrVec grids;
    grids.push_back(compressedGrid);
    outFile.write(grids);
    outFile.close();
    
    std::cout << "\n=== Compression Complete ===" << std::endl;
    
    return 0;
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}




