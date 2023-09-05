#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{

// Converts degrees to radians.
template<typename real>
inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

template<typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch(order)
  {
    case RotateOrder::XYZ:
      return RZ * RY * RX;
    case RotateOrder::YZX:
      return RX * RZ * RY;
    case RotateOrder::ZXY:
      return RY * RX * RZ;
    case RotateOrder::XZY:
      return RY * RZ * RX;
    case RotateOrder::YXZ:
      return RZ * RX * RY;
    case RotateOrder::ZYX:
      return RX * RY * RZ;
  }
  assert(0);
}

// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
//   IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
template<typename real>
void forwardKinematicsFunction(
    int numIKJoints, const int * IKJointIDs, const FK & fk,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
  // Students should implement this.
  // The implementation of this function is very similar to function computeLocalAndGlobalTransforms in the FK class.
  // The recommended approach is to first implement FK::computeLocalAndGlobalTransforms.
  // Then, implement the same algorithm into this function. To do so,
  // you can use fk.getJointUpdateOrder(), fk.getJointRestTranslation(), and fk.getJointRotateOrder() functions.
  // Also useful is the multiplyAffineTransform4ds function in minivectorTemplate.h .
  // It would be in principle possible to unify this "forwardKinematicsFunction" and FK::computeLocalAndGlobalTransforms(),
  // so that code is only written once. We considered this; but it is actually not easily doable.
  // If you find a good approach, feel free to document it in the README file, for extra credit.
  int numberOfJoints;
  numberOfJoints = fk.getNumJoints();
  vector<Mat3<real>> localTransforms(numberOfJoints);
  vector<Mat3<real>> globalTransforms(numberOfJoints);
  vector<Vec3<real>> globalTranslations(numberOfJoints);
  vector<Vec3<real>> localTranslations(numberOfJoints);

  double rotationMatrix[9];
  Mat3<real> jointOrientationRotationMatrix, eulerAnglesRotationMatrix, rotationMatrix3d;
  Vec3<real> translations[numberOfJoints];
  
  for (int i = 0; i < numberOfJoints; i++) {
    // First, compute the localTransform for each joint, using eulerAngles and jointOrientationEulerAngles,
    // and the "euler2Rotation" function.
  
    int currentJoint = fk.getJointUpdateOrder(i);
    // Get the rotation matrix from the joint orientation i.e. orientation of the joint w.r.t. the parent joint
    real angle[3];
    angle[0] = fk.getJointOrient(currentJoint)[0];
    angle[1] = fk.getJointOrient(currentJoint)[1];
    angle[2] = fk.getJointOrient(currentJoint)[2];
    rotationMatrix3d = Euler2Rotation(angle, fk.getJointRotateOrder(currentJoint));
    // Get the rotation matrix from the current joint's rotation
    angle[0] = eulerAngles[currentJoint * 3];
    angle[1] = eulerAngles[currentJoint * 3 + 1];
    angle[2] = eulerAngles[currentJoint * 3 + 2];
    
    // Final rotation = joint orientation * current joint rotation
    localTransforms[currentJoint] = rotationMatrix3d * Euler2Rotation(angle, fk.getJointRotateOrder(currentJoint));


    // Then, recursively compute the globalTransforms, from the root to the leaves of the hierarchy.
    // Use the jointParents and jointUpdateOrder arrays to do so.
    // Also useful are the Mat3d and RigidTransform4d classes defined in the Vega folder.
    localTranslations[currentJoint][0] = fk.getJointRestTranslation(currentJoint)[0]; // the global translations here are temp, they will be calculated later
    localTranslations[currentJoint][1] = fk.getJointRestTranslation(currentJoint)[1];
    localTranslations[currentJoint][2] = fk.getJointRestTranslation(currentJoint)[2];
      
    
    int current = fk.getJointUpdateOrder(i); // Get the joint that appears at position "i" in a linear joint update order.
    int parentOfCurrent = fk.getJointParent(current);
    // If the current joint is the root, then the global transform is the same as the local transform
    if(parentOfCurrent == -1){ // means current is root
      globalTranslations[currentJoint][0] = localTranslations[currentJoint][0];
      globalTranslations[currentJoint][1] = localTranslations[currentJoint][1];
      globalTranslations[currentJoint][2] = localTranslations[currentJoint][2];
    
      globalTransforms[current] = localTransforms[current];
    }
    // Otherwise, the global transform is the global transform of the parent joint multiplied by the local transform of the current joint
    else
    {
      multiplyAffineTransform4ds(globalTransforms[parentOfCurrent], globalTranslations[parentOfCurrent], localTransforms[current], localTranslations[current], globalTransforms[current], globalTranslations[current]);
    }
  }


  // Update the handle positions
  for (int i = 0; i < numIKJoints; i++) {
    handlePositions[i * 3] = globalTranslations[IKJointIDs[i]][0];
    handlePositions[i * 3 + 1] = globalTranslations[IKJointIDs[i]][1];
    handlePositions[i * 3 + 2] = globalTranslations[IKJointIDs[i]][2];
  }

  
} // end anonymous namespaces
}

IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc()
{
  // Students should implement this.
  // Here, you should setup adol_c:
  //   Define adol_c inputs and outputs. 

  int n = FKInputDim;
  int m = FKOutputDim;

  trace_on(adolc_tagID);

  vector<adouble> x(n); // define the input of the function f
  for(int i = 0; i < n; i++)
    x[i] <<= 0.0; // The <<= syntax tells ADOL-C that these are the input variables.

  vector<adouble> y(m); // define the output of the function f

  //   Use the "forwardKinematicsFunction" as the function that will be computed by adol_c.
  //   This will later make it possible for you to compute the gradient of this function in IK::doIK
  //   (in other words, compute the "Jacobian matrix" J).
  // See ADOLCExample.cpp .

  forwardKinematicsFunction(numIKJoints, IKJointIDs, *fk, x, y);

  vector<double> output(m);
  for(int i = 0; i < m; i++)
    y[i] >>= output[i]; // Use >>= to tell ADOL-C that y[i] are the output variables

  // Finally, call trace_off to stop recording the function f.
  trace_off(); // ADOL-C tracking finished

}

void IK::doIK(const Vec3d * targetHandlePositions, Vec3d * jointEulerAngles)
{
  // You may find the following helpful:
  int IKMethod = 2; // 0 = Tikhonov, 1 = PseudoInverse, 2 = Transpose Jacobian
  int divideIK = 1; // 0 = no, 1 = yes
  int numJoints = fk->getNumJoints(); // Note that is NOT the same as numIKJoints!
  double max_change_tolerable = 0.2; // This is the maximum change in delta_p that we will allow in one iteration of the IK solver.
  // Students should implement this.
  // Use adolc to evalute the forwardKinematicsFunction and its gradient (Jacobian). It was trained in train_adolc().
  // Specifically, use ::function, and ::jacobian .
  // See ADOLCExample.cpp .
  int n = FKInputDim;
  int m = FKOutputDim;

  double handle_positions[m];
  ::function(adolc_tagID, m, n, jointEulerAngles->data(), handle_positions);

  // cout<<"Maybe here?"<<endl;
  // cout<<"n: "<<n<<endl;
  // cout<<"m: "<<m<<endl;
  // for (int i = 0; i < m; i++) {
  //   cout << i << ": " << handle_positions[i] <<"Some"<< endl;
  // }
  // cout<<"Not here";


  vector<double> jacobianMatrix(m * n);
  vector<double *> jacobianMatrixEachRow(n);
  // cout<<"Not here";
  // change this
  for(int i = 0; i < m; i++)
    jacobianMatrixEachRow[i] = &jacobianMatrix[n*i];
  
  ::jacobian(adolc_tagID, m, n, jointEulerAngles->data(), jacobianMatrixEachRow.data()); 
    
  // ->  Declare Eigen matrix J and save the values from jacobianMatrix
  Eigen::MatrixXd J(m, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      J(i, j) = jacobianMatrix[i * n + j];
    }
  }
  // // print the jacobian matrix
  // for (int i = 0; i < m; i++) {
  //   for (int j = 0; j < n; j++) {
  //     cout <<"Jacob" <<jacobianMatrix[i * n + j] << " ";
  //   }
  //   cout << endl;
  // }
  // Divide IK innto smaller steps if the change in delta_p is more than max_change_tolerable.
  // ->  Declare Eigen matrix delta_p
  Eigen::MatrixXd delta_p(m, 1);
  double max_delta_p = 0.0;
  // -> Compute delta_p = target - current
  for (int i = 0; i < m; i++) {
    delta_p(i, 0) = targetHandlePositions[i / 3][i % 3] - handle_positions[i];
    if (abs(delta_p(i, 0)) > max_delta_p) {
      max_delta_p = abs(delta_p(i, 0));
    }
  }
  // cout<<"max_delta_p: "<<max_delta_p<<endl;

  int num_steps = 1;

  if(divideIK == 1){
    if (max_delta_p > max_change_tolerable) {
      num_steps = ceil(max_delta_p / max_change_tolerable);
    }
  }
  
  while(num_steps > 0) {
    // cout<<"Iters >>>>>>>>>>>>>>>>> "<<num_steps<<endl;
    double inner_max_delta_p = 0.0;
    // Use it implement the Tikhonov IK method (or the pseudoinverse method for extra credit).
    // Note that at entry, "jointEulerAngles" contains the input Euler angles. 
    // Upon exit, jointEulerAngles should contain the new Euler angles.
    double handle_positions[m];
    ::function(adolc_tagID, m, n, jointEulerAngles->data(), handle_positions);

    vector<double> jacobianMatrix(m * n);
    vector<double *> jacobianMatrixEachRow(n);
    // cout<<"Not here";
    // change this
    for(int i = 0; i < m; i++)
      jacobianMatrixEachRow[i] = &jacobianMatrix[n*i];
    
    ::jacobian(adolc_tagID, m, n, jointEulerAngles->data(), jacobianMatrixEachRow.data()); 
      
    // ->  Declare Eigen matrix J and save the values from jacobianMatrix
    Eigen::MatrixXd J(m, n);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        J(i, j) = jacobianMatrix[i * n + j];
      }
    }

    // Declare Eigen matrix delta_theta
    Eigen::MatrixXd delta_theta(n, 1);


    // Compute delta_p
    // ->  Declare Eigen matrix delta_p
    Eigen::MatrixXd delta_p(m, 1);

    // -> Compute delta_p = (target - current)/num_steps
    for (int i = 0; i < m; i++) {
      delta_p(i, 0) = (targetHandlePositions[i / 3][i % 3] - handle_positions[i])/ num_steps;
      // Calc max delta_p
      if (abs(delta_p(i, 0)) > inner_max_delta_p) {
        inner_max_delta_p = abs(delta_p(i, 0));
      }
    }

    if(IKMethod == 0){

      // Tikhonov IK method
      // Tikhonov equation: 
      // (J^T * J + lambda * I)*delta_theta = J^T * delta_p
      // (J^T * J + lambda * I)*delta_theta = J^T * (target - current)
      // Compute J^T * J + lambda * I


      // ->  Declare Eigen matrix J^T
      Eigen::MatrixXd J_transpose = J.transpose();

      // ->  Declare Eigen matrix J^T * J
      Eigen::MatrixXd JTJ = J_transpose * J;

      // ->  Declare Eigen matrix lambda * I
      double lambda = 0.01;
      Eigen::MatrixXd lambdaI = Eigen::MatrixXd::Identity(n, n) * lambda;

      // ->  Declare Eigen matrix J^T * J + lambda * I
      Eigen::MatrixXd JTJ_lambdaI = JTJ + lambdaI;

      // Compute J^T * delta_p and store it in Eigen matrix J_transpose_delta_p
      Eigen::MatrixXd J_transpose_delta_p = J_transpose * delta_p;

      // Now solve using ldlt() to get delta_theta
      delta_theta = JTJ_lambdaI.ldlt().solve(J_transpose_delta_p);  
      
      // print inner_max_delta_p
      // cout<<"inner_max_delta_p: "<<inner_max_delta_p<<endl;
    }
    else if(IKMethod == 1){
      // Pseudoinverse method
      // delta_theta = J^T * (J * J^T)^-1 * delta_p
      // delta_theta = J^T * (J * J^T)^-1 * (target - current)

      // ->  Declare Eigen matrix J^T
      Eigen::MatrixXd J_transpose = J.transpose();

      // -> Declare Eigen matrix J * J^T
      Eigen::MatrixXd JJT = J * J_transpose;

      // Compute (J * J^T)^-1
      Eigen::MatrixXd JJT_inverse = JJT.inverse();

      // Compute J^T * (J * J^T)^-1
      Eigen::MatrixXd J_transpose_JJT_inverse = J_transpose * JJT_inverse;

      // Compute J^T * (J * J^T)^-1 * delta_p and store it in Eigen matrix delta_theta
      delta_theta = J_transpose_JJT_inverse * delta_p;

    }
    else if(IKMethod == 2){
      // Jacobian Transpose method
      // delta_theta = alpha * J^T * delta_p
      // delta_theta = alpha * J^T * (target - current)

      // ->  Declare Eigen matrix J^T
      Eigen::MatrixXd J_transpose = J.transpose();

      // Compute J^T * delta_p and store it in Eigen matrix delta_theta
      delta_theta = J_transpose * delta_p;
      
      // Compute alpha as close to delta_p as possible
      double alpha = delta_p.norm() / delta_theta.norm();
      delta_theta = alpha * delta_theta;
    }
    else{
      cout<<"Invalid IK method"<<endl;
      exit(1);
    }
    // Update jointEulerAngles
    for (int i = 0; i < n; i++) {
      jointEulerAngles->data()[i] += delta_theta(i, 0);
    }

    num_steps -= 1;

  }
  
  

}
