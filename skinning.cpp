#include "skinning.h"
#include "vec3d.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li
#include <Eigen/Core>
#include <Eigen/Geometry>

Eigen::Quaterniond rotationMatrixToQuaternion(const Eigen::Matrix3d& R)
{
    double trace = R.trace();

    if (trace > 0) {
        double s = sqrt(trace + 1.0);
        double w = 0.5 * s;
        double x = (R(2, 1) - R(1, 2)) / (2.0 * s);
        double y = (R(0, 2) - R(2, 0)) / (2.0 * s);
        double z = (R(1, 0) - R(0, 1)) / (2.0 * s);
        return Eigen::Quaterniond(w, x, y, z);
    } else {
        int i = 0;
        if (R(1, 1) > R(0, 0)) i = 1;
        if (R(2, 2) > R(i, i)) i = 2;

        int j = (i + 1) % 3;
        int k = (j + 1) % 3;

        double s = sqrt(R(i, i) - R(j, j) - R(k, k) + 1.0);
        double q[4];
        q[i+1] = 0.5 * s;
        q[0] = (R(k, j) - R(j, k)) / (2.0 * s);
        q[j+1] = (R(j, i) + R(i, j)) / (2.0 * s);
        q[k+1] = (R(k, i) + R(i, k)) / (2.0 * s);
        return Eigen::Quaterniond(q[0], q[1], q[2], q[3]);
    }
}

Eigen::Quaterniond homogeneousMatrixToQuaternion(const Eigen::Matrix4d& H)
{
    Eigen::Matrix3d R = H.block(0, 0, 3, 3);
    return rotationMatrixToQuaternion(R);
}

Eigen::Quaterniond dualQuaternionFromHomogeneousMatrix(const Eigen::Matrix4d& H)
{
    Eigen::Vector3d T = H.block(0, 3, 3, 1);
    Eigen::Quaterniond q = homogeneousMatrixToQuaternion(H);

    Eigen::Quaterniond eps(0, T.x(), T.y(), T.z());
    // Eigen::Quaterniond dq = q + (0.5 * eps) * q.conjugate();

    // Compute the dual quaternion
    Eigen::Quaterniond dq;
    for (int i = 0; i < 4; i++) {
        dq.coeffs()[i] = q.coeffs()[i] + 0.5 * eps.coeffs()[i] * q.coeffs()[0];
    }

    return dq;
}

Eigen::Matrix4d rigidTransform4dToMatrix4d(const RigidTransform4d& T)
{
    Eigen::Matrix4d M;
    M.setIdentity();
    // Copy all the elements from T into M
    for(int i = 0; i < 4; i++){
      for(int j = 0; j < 4; j ++){
        M(i,j) = T[i][j];
      }
    }    
    
    return M;
}

int numJoints;
Skinning::Skinning(int numMeshVertices, const double * restMeshVertexPositions,
    const std::string & meshSkinningWeightsFilename)
{
  this->numMeshVertices = numMeshVertices;
  this->restMeshVertexPositions = restMeshVertexPositions;

  cout << "Loading skinning weights..." << endl;
  ifstream fin(meshSkinningWeightsFilename.c_str());
  assert(fin);
  int numWeightMatrixRows = 0, numWeightMatrixCols = 0;
  fin >> numWeightMatrixRows >> numWeightMatrixCols;
  assert(fin.fail() == false);
  assert(numWeightMatrixRows == numMeshVertices);
  numJoints = numWeightMatrixCols;

  vector<vector<int>> weightMatrixColumnIndices(numWeightMatrixRows);
  vector<vector<double>> weightMatrixEntries(numWeightMatrixRows);
  fin >> ws;
  while(fin.eof() == false)
  {
    int rowID = 0, colID = 0;
    double w = 0.0;
    fin >> rowID >> colID >> w;
    weightMatrixColumnIndices[rowID].push_back(colID);
    weightMatrixEntries[rowID].push_back(w);
    assert(fin.fail() == false);
    fin >> ws;
  }
  fin.close();

  // Build skinning joints and weights.
  numJointsInfluencingEachVertex = 0;
  for (int i = 0; i < numMeshVertices; i++)
    numJointsInfluencingEachVertex = std::max(numJointsInfluencingEachVertex, (int)weightMatrixEntries[i].size());
  assert(numJointsInfluencingEachVertex >= 2);

  // Copy skinning weights from SparseMatrix into meshSkinningJoints and meshSkinningWeights.
  meshSkinningJoints.assign(numJointsInfluencingEachVertex * numMeshVertices, 0);
  meshSkinningWeights.assign(numJointsInfluencingEachVertex * numMeshVertices, 0.0);
  for (int vtxID = 0; vtxID < numMeshVertices; vtxID++)
  {
    vector<pair<double, int>> sortBuffer(numJointsInfluencingEachVertex);
    for (size_t j = 0; j < weightMatrixEntries[vtxID].size(); j++)
    {
      int frameID = weightMatrixColumnIndices[vtxID][j];
      double weight = weightMatrixEntries[vtxID][j];
      sortBuffer[j] = make_pair(weight, frameID);
    }
    sortBuffer.resize(weightMatrixEntries[vtxID].size());
    assert(sortBuffer.size() > 0);
    sort(sortBuffer.rbegin(), sortBuffer.rend()); // sort in descending order using reverse_iterators
    for(size_t i = 0; i < sortBuffer.size(); i++)
    {
      meshSkinningJoints[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].second;
      meshSkinningWeights[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].first;
    }

    // Note: When the number of joints used on this vertex is smaller than numJointsInfluencingEachVertex,
    // the remaining empty entries are initialized to zero due to vector::assign(XX, 0.0) .
  }
}

void Skinning::applySkinning(const RigidTransform4d * jointSkinTransforms, double * newMeshVertexPositions) const
{
  // Students should implement this

  int skinningMethod = 0; // 0: linear blend skinning, 1: dual quaternion skinning
  if (skinningMethod == 0)
  {
    // Convert restMeshVertexPositions to a 4d vector of homogeneous coordinates.
    vector<Vec4d> restMeshVertexPositions4d(numMeshVertices);
    vector<Vec4d> newMeshVertexPositions4d(numMeshVertices);
    for(int i=0; i<numMeshVertices; i++)
    {
      restMeshVertexPositions4d[i] = Vec4d(restMeshVertexPositions[3 * i + 0],
                                          restMeshVertexPositions[3 * i + 1],
                                          restMeshVertexPositions[3 * i + 2],
                                          1.0);
      newMeshVertexPositions4d[i] = Vec4d(0.0, 0.0, 0.0, 0.0);
    }

    // Apply Linear Blend Skinning to each vertex.
    for(int i=0; i<numMeshVertices; i++)
    {
      for(int j=0; j<numJointsInfluencingEachVertex; j++)
      {
        // Get the joint ID and weight.
        int jointID = meshSkinningJoints[i * numJointsInfluencingEachVertex + j];
        double weight = meshSkinningWeights[i * numJointsInfluencingEachVertex + j];

        // Apply the joint transform to the vertex.
        newMeshVertexPositions4d[i] += weight * jointSkinTransforms[jointID] * restMeshVertexPositions4d[i];
      }
    }


    // Store the result in newMeshVertexPositions.
    for(int i=0; i<numMeshVertices; i++)
    {
      newMeshVertexPositions[3 * i + 0] = newMeshVertexPositions4d[i][0];
      newMeshVertexPositions[3 * i + 1] = newMeshVertexPositions4d[i][1];
      newMeshVertexPositions[3 * i + 2] = newMeshVertexPositions4d[i][2];
    }
  }
  else{
    // Implement dual quaternion skinning here.
    // Convert jointSkinTransforms to dual quaternion.
    vector<Eigen::Quaterniond> jointSkinTransformsDualQuaternion(numJoints);  
    for(int i=0; i<numJoints; i++)
    {
      jointSkinTransformsDualQuaternion[i] = dualQuaternionFromHomogeneousMatrix(rigidTransform4dToMatrix4d(jointSkinTransforms[i]));
    }

    // Print Dual Quaternions
    for(int i = 0; i < numJoints; i++)
    {
      cout << i << " - " << jointSkinTransformsDualQuaternion[i].coeffs()<<"\n";
    }

    // Apply Dual Quaternion Skinning to each vertex.
  
  }
}



