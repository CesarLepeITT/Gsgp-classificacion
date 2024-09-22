/*<one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2020  José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//! \file     GsgpCuda.cpp
//! \brief    file containing the definition of the modules (kernels) used to create the population of individuals, evaluate them, the search operator and read data
//! \author   Jose Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
//! \date     created on 25/01/2020

#include "GsgpCuda.h"
#include <cmath>
#include <cstdio>
using namespace std; 

/*!
* \fn       string currentDateTime()
* \brief    function to capture the date and time of the host, this allows to define the exact moment of each GSGP-CUDA run,
            this allows us to name the output files by date and time.
* \return   char: return date and time from the host computer
* \date     25/01/2020
* \author   Jose Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
const std::string currentDateTimeM() { 
  time_t now = time(0); struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);
  return buf;    
}

/*!
* \fn       string currentDateTime()
* \brief    function to capture the date and time of the host, this allows to define the exact moment of each GSGP-CUDA run,
            this allows us to name the output files by date and time.
* \return   char: return date and time from the host computer
* \date     25/01/2020
* \author   Jose Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
const std::string currentDateTime(){
  time_t     now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&now);
  strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
  return buf;
}

/*!
* \fn       void cudaErrorCheck(const char* functionName)
* \brief    This function catches the error detected by the compiler and prints the user-friendly error message.
* \param    char funtionName: pointer to the name of the kernel executed to verify if there was an error 
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
void cudaErrorCheck(const char* functionName){
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess){
    printf("CUDA error (%s): %s\n", functionName, cudaGetErrorString(error));
    exit(-1);
  }
}

/*!
* \fn       __global__ void init(unsigned int seed, curandState_t* states)
* \brief    This kernel is used to initialize the random states to generate random numbers with a different pseudo sequence in each thread
* \param    int seed: used to generate a random number for each core  
* \param    curandState_t states: pointer to store a random state for each thread
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__global__ void init(unsigned int seed, curandState_t* states){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  curand_init(seed, tid, 0, &states[tid]);
}

/*!
* \fn       __device__ int push(float val,int *pushGenes, float *stackInd)
* \brief    push() function is used to insert an element at the top of the stack. The element is added to the stack container and the size of the stack is increased by 1.
* \param    float val: variable that stores a value resulting from a valid operation in the interpreter
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    float *stackInd: auxiliary pointer that stores the values ​​resulting from the interpretation of each individual
* \return   int: auxiliary pointer that stores the positions of individuals + 1
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/
__device__ int push(float val, int *pushGenes, float *stackInd){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  stackInd[pushGenes[tid]] = val;
  return pushGenes[tid]+1;
}

__device__ int pushMo(float val, int *pushGenes, float *stackInd){
	const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
	stackInd[pushGenes[tid]] = val;
	return pushGenes[tid]+1;
}

/*!
* \fn       __device__ float pop(int *pushGenes, float *stackInd)
* \brief    pop() function is used to remove an element from the top of the stack(newest element in the stack). The element is removed to the stack container and the size of the stack is decreased by 1.
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    float *stackInd: auxiliary pointer that stores the values ​​resulting from the interpretation of each individual
* \return   float: returns the stackInd without the value positioned in pushGenes [tid]
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__device__ float pop(int *pushGenes, float *stackInd){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  pushGenes[tid]--;
  return stackInd[pushGenes[tid]];
}

/*!
* \fn       __device__ bool isEmpty(int *pushGenes, unsigned int sizeMaxDepthIndividual)     
* \brief    Check if a stack is empty
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    int sizeMaxDepthIndividual: variable thar stores maximum depth for individuals
* \return   bool - true if the stack is empty, false otherwise
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     gsgpMalloc.cpp
*/
__device__ bool isEmpty(int *pushGenes, unsigned int sizeMaxDepthIndividual){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (pushGenes[tid] <= tid * sizeMaxDepthIndividual)
    return true;
  else
    return false;
}

/*!
* \fn       __device__ void clearStack(int *pushGenes, unsigned int sizeMaxDepthIndividual, float *stackInd)
* \brief    remove all elements from the stack so that in the next evaluations there are no previous values of other individuals
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    int sizeMaxDepthIndividual: variable thar stores maximum depth for individuals
* \param    float *stackInd: auxiliary pointer that stores the values ​​resulting from the evaluation of each individual
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/
__device__ void clearStack(int *pushGenes, unsigned int sizeMaxDepthIndividual, float *stackInd){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  pushGenes[tid] = tid*sizeMaxDepthIndividual;
  for(int i = 0; i< sizeMaxDepthIndividual; i++)
    stackInd[tid*sizeMaxDepthIndividual+i] = 0;
}

/*!
* \fn       __global__ void initializePopulation(float* dInitialPopulation, int nvar, int maxIndividualLength, curandState_t* states, int maxRandomConstant)
* \brief    The initializePopulation kernel creates the population of programs T and the set of random trees R uses by the GSM kernel, based on the desired population
            size and maximun program length. The individuals are representd using a linear genome, composed of valid terminals (inputs to the program) and functions 
            (basic elements with which programs can be built).
* \param    float *dInitialPopulation: vector pointers to store the individuals of the initial population
* \param    int nvar: variable containing the number of columns (excluding the target) of the training dataset
* \param    int maxIndividualLength: number of genes in the genome
* \param    curandState_t *states: random status pointer to generate random numbers for each thread
* \param    int maxRandomConstant: variable containing the maximum number to generate ephemeral constants
* \param    int funtion: variable containing the number of functions
* \param    float functionRatio: variable containing the ratio of functions
* \param    float variableRatio: variable containing the ratio of terminals
* \return   void
* \date     09/11/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__global__ void initializePopulation(float* dInitialPopulation, int nvar, int maxIndividualLength, curandState_t* states, int maxRandomConstant, int functions, float functionRatio, float variableRatio){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  for (unsigned int j = 0; j < maxIndividualLength; j++){
    if(curand_uniform(&states[tid])<functionRatio){
      dInitialPopulation[tid*maxIndividualLength+j] = (curand(&states[tid]) % 5 + 10000)*(float)(-1);
    }else{
      if (curand_uniform(&states[tid])<variableRatio){
          dInitialPopulation[tid*maxIndividualLength+j] = (curand(&states[tid]) % nvar+1000)*(float)(-1);
      }else{
        if(curand_uniform(&states[tid])<0.5){
          dInitialPopulation[tid*maxIndividualLength+j] = (curand(&states[tid]) % maxRandomConstant+1);
        }else{
          dInitialPopulation[tid*maxIndividualLength+j] = (curand(&states[tid]) % maxRandomConstant+1)*(float)(-1);
        }
      }
    }
  }
}

/*!
* \fn       __global__ void computeSemantics    
* \brief    The ComputeSemantics kernel is an interpreter, that decodes each individual and evaluates it over all fitness cases,
            producing as output the semantic vector of each individual. The chromosome is interpreted linearly, using an auxiliary LIFO stack D that stores 
            terminals from the chromosome and the output from valid operations.
* \param    float *inputPopulation: vector pointers to store the individuals of the population
* \param    float *outSemantic: vector pointers to store the semantics of each individual in the population
* \param    int sizeMaxDepthIndividual: variable thar stores maximum depth for individuals
* \param    float *data: pointer vector containing training or test data
* \param    int nrow: variable containing the number of rows (instances) of the training dataset
* \param    int nvar: variable containing the number of columns (excluding the target) of the training dataset
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    float *stackInd: auxiliary pointer that stores the values ​​resulting from the interpretation of each individual
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/
__global__ void computeSemantics(float *inputPopulation, float *outSemantic, unsigned int sizeMaxDepthIndividual, float *data,
 int nrow, int nvar, int *pushGenes, float *stackInd, int protected_division){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  pushGenes[tid] = tid * sizeMaxDepthIndividual;
  int t,t_;
  float tmp,tmp2,out;
  for(int k=0; k<nrow; k++){
    out=0, t=0, t_=0 ,tmp=0, tmp2=0;
    clearStack(pushGenes,sizeMaxDepthIndividual, stackInd);
    for(int i=0; i < sizeMaxDepthIndividual; i++){
      if(inputPopulation[tid*sizeMaxDepthIndividual+i] > 0){
        pushGenes[tid] = push(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenes,stackInd);
      }
      else if(inputPopulation[tid*sizeMaxDepthIndividual+i] < 0 && inputPopulation[tid*sizeMaxDepthIndividual+i] > -1000 && inputPopulation[tid*sizeMaxDepthIndividual+i] > -10000){
        pushGenes[tid] = push(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenes,stackInd);
      }
      else if(inputPopulation[tid*sizeMaxDepthIndividual+i] <= -1000 && inputPopulation[tid*sizeMaxDepthIndividual+i] > -10000){
          t=inputPopulation[tid*sizeMaxDepthIndividual+i];
          t_=(t+1000)*(-1);
          pushGenes[tid] = push(data[t_+nvar*k],pushGenes,stackInd);
        }
        else if (inputPopulation[tid*sizeMaxDepthIndividual+i] == -10001){
          if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
            tmp = pop(pushGenes,stackInd);
            if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
              tmp2 = pop(pushGenes,stackInd);
              if (!isnan(tmp) && !isinf(tmp) && !isnan(tmp2) && !isinf(tmp2)){
                pushGenes[tid] = push(tmp2 + tmp,pushGenes,stackInd);
                out = tmp2+tmp;
              }
            }else
            pushGenes[tid] = push(tmp,pushGenes,stackInd);
          }
        }
        else if (inputPopulation[tid*sizeMaxDepthIndividual+i] == -10002){
          if(!isEmpty(pushGenes,sizeMaxDepthIndividual)){
            tmp = pop(pushGenes,stackInd);
            if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
              tmp2 = pop(pushGenes,stackInd);
              if (!isnan(tmp) && !isinf(tmp) && !isnan(tmp2) && !isinf(tmp2)){
                pushGenes[tid] = push(tmp2 - tmp,pushGenes,stackInd);
                out = tmp2-tmp;
              }
            }else
            pushGenes[tid] = push(tmp,pushGenes,stackInd);
          }
        }
        else if (inputPopulation[tid*sizeMaxDepthIndividual+i] == -10003){
          if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
            tmp = pop(pushGenes,stackInd);
            if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
              tmp2 = pop(pushGenes,stackInd);
              if (!isnan(tmp) && !isinf(tmp) && !isnan(tmp2) && !isinf(tmp2)){
                pushGenes[tid] = push(tmp2 * tmp,pushGenes,stackInd);
                out = tmp2*tmp;
              }
            }else
            pushGenes[tid] = push(tmp,pushGenes,stackInd);
          }
        }
        else if (inputPopulation[tid*sizeMaxDepthIndividual+i] == -10004){
          if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
            tmp = pop(pushGenes,stackInd);
            if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
              tmp2 = pop(pushGenes,stackInd);
              if (!isnan(tmp) && !isinf(tmp) && !isnan(tmp2) && !isinf(tmp2)){
                if (protected_division==0) {
                  if (tmp!=0) {
                    pushGenes[tid] = push(tmp2 / tmp,pushGenes,stackInd);
                    out = tmp2 / tmp;
                  }else {
                    pushGenes[tid] = push(tmp2 / sqrtf(1+tmp*tmp),pushGenes,stackInd);
                    out = tmp2 / sqrtf(1+tmp*tmp);
                  }
                }else {
                    pushGenes[tid] = push(tmp2 / sqrtf(1+tmp*tmp),pushGenes,stackInd);
                    out = tmp2 / sqrtf(1+tmp*tmp);
                }
                
              }
            }else
            pushGenes[tid] = push(tmp,pushGenes,stackInd);
          }
        }
    }
  outSemantic[(tid*nrow+k)] = out;
  }
}
/*!
* \fn       __global__ void computeError(float *semantics, float *targetValues, float *fit, int nrow)
* \brief    The computeError kernel computes the RMSE between each row of the semantic matrix ST,m×n and the target vector t, computing the
            fitness of each individual in the population.
* \param    float *semantics: vector of pointers that contains the semantics of the individuals of the initial population 
* \param    float *targetValues: pointer containing the target values of train or test
* \param    float *fit: vector that will store the error of each individual in the population
* \param    int nrow: variable containing the number of rows (instances) of the training and test dataset
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     gsgpMalloc.cpp
*/

__global__ void computeError(float *semantics, float *targetValues, float *fit, int nrow){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  float temp = 0;
  for(int i=0; i<nrow; i++){
    temp += (semantics[tid*nrow+i]-targetValues[i])*(semantics[tid*nrow+i]-targetValues[i]);  
  }
  temp = sqrt(temp/nrow);
  fit[tid] = temp;
}

float findMean(float *targetValues, int nrow){
  float tmp =0.0,mean=0.0;
  for (size_t i = 0; i < nrow; i++){
    tmp += targetValues[i];
  }
  mean = tmp/nrow;
  return mean;
}

__global__ void computeRo(float *semantics, float *targetValues, float *fit, int nrow, float y_mean){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  
  double residual1=0, residual2=0;
  double sum_squared_residual=0;
  double sum_squared_total=0;

  /* Calculate targets ans sematics means*/
  /* Calculate residual_sum_of_square and total_sum_of_square*/
  /* Version 1 de calculo de R2  */
  /* Calculate residual_sum_of_square */
  /* Calculate total_sum_of_square*/
  for(int i=0; i < nrow; i++) {
    residual1 = (targetValues[i] - semantics[tid*nrow+i]);
    sum_squared_residual = sum_squared_residual + (residual1 * residual1);
    residual2 = (targetValues[i] - y_mean);
    sum_squared_total = sum_squared_total + (residual2 * residual2);
  }

  fit[tid] = (1 - (sum_squared_residual / sum_squared_total));

  if(isnan(fit[tid]) || isinf(fit[tid]) ||  fit[tid] > 2.0 || fit[tid] < -4000)
    fit[tid] = -4000;

    
  fit[tid] = fit[tid] + (4000);

  // printf("Final tid: %i , - nRows: %i Fit: %lf , y_mean: %lf , semantics_mean: %lf  \n", tid, nrow, fit[tid], y_mean, semantics_mean);
  
}


__global__ void diferenceRt(int nrow, int populationSize ,float *rtSemantics, float *indexRts, float *rtsDifference, int sigmoid, int normalize){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int nSeed = (tid/nrow);
  int firstTree = indexRts[nSeed], secondTree = indexRts[populationSize + nSeed];
  int r = tid - (int)(tid/nrow)*nrow;

  double s1=0,s2=0,tmp1=0,tmp2=0;
  if(sigmoid==0&&normalize==0){
    // s1=((rtSemantics[firstTree*nrow+r])  * (1/(1+exp(-(rtSemantics[firstTree*nrow+r])))));
    // s2=((rtSemantics[secondTree*nrow+r]) * (1/(1+exp(-(rtSemantics[secondTree*nrow+r])))));
    tmp1 = rtSemantics[firstTree*nrow+r];
    tmp2 = rtSemantics[secondTree*nrow+r];
    if(isnan(tmp1) || isinf(tmp1) ){
      tmp1=1;
    }else if(tmp1 < 0){
      tmp1 = -1000;
    }else{
      tmp1 = 1000;
    }
    if(isnan(tmp2) || isinf(tmp2) ){
      tmp2=1;
    }else if(tmp2 < 0){
      tmp2 = -1000;
    }else{
      tmp2 = 1000;
    }
    s1=((tmp1) * (1/(1+exp(-(tmp1)))));
    s2=((tmp2) * (1/(1+exp(-(tmp2)))));

  }else if(sigmoid==1&&normalize==0){
    s1 = (1.0/(1+exp(-(rtSemantics[firstTree*nrow+r]))));
    s2 = (1.0/(1+exp(-(rtSemantics[secondTree*nrow+r]))));
  }else if(sigmoid==2&&normalize==0){
    s1 = ((2.0/(1+exp(-2*(rtSemantics[firstTree*nrow+r]))))-1);
    s2 = ((2.0/(1+exp(-2*(rtSemantics[secondTree*nrow+r]))))-1); 
  }
  if (normalize==0) {
    rtsDifference[tid] = s1-s2;
  }else{
    rtsDifference[tid] = rtSemantics[firstTree*nrow+r];
  }

  
}

__global__ void rt2(float *rtSemantics, float *updateRts){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  updateRts[tid] = pow(rtSemantics[tid],2);
}

__global__ void inverseVector(float *rtSemantics, float *updateRts, int nrow ,float *inverse){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  float tmp = 0;
  int nSeed = (tid/nrow);
  for(int i=0;i<nrow;i++){
    tmp += updateRts[nSeed*nrow+i];
  }
  inverse[tid] = ((rtSemantics[tid])/(sqrtf(1+tmp*tmp)));
  if(isnan(inverse[tid]) || isinf(inverse[tid])){
    inverse[tid]=1.0;
  }
}

__global__ void makeOms(float *semantics,float *randomTreesSemantics, float *targetValues, float *oms, int nrow, int sizePopulation, float *ms){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  float tmp=0;
  for(int i=0;i<nrow;i++){
      tmp+= ((targetValues[i]-semantics[tid*nrow+i])*(randomTreesSemantics[tid*nrow+i]));
  }
  
  if(isnan(tmp) || isinf(tmp) || tmp < -5000 || tmp > 5000){
    if(isnan(tmp) || isinf(tmp) ){
      tmp=ms[tid];
    }else if(tmp < 0){
      tmp = -1000;
    }else{
      tmp = 1000;
    }
  }
  
  oms[tid]=tmp;
}

/*!
* \fn       __device__ float sigmoid(float n)
* \brief    auxiliary function for the geometric semantic mutation operation
* \param    float n: semantic value of a random tree
* \return   float n: value resulting from the function
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/
__device__ float sigmoid(float n){
  return 1.0/(1+exp(-1*(n)));
}

/*!
* \fn       __global__ void initializeIndexRandomTrees(int sizePopulation, float *indexRandomTrees, curandState_t* states)
* \brief    this kernel generates random indexes for random trees that are used in the mutation operator to select two random trees.
* \param    int sizePopulation: this variable contains the number of individuals that the population has
* \param    float *indexRandomTrees: this pointer stores the indexes randomly for mutation
* \param    curandState_t* states: random status pointer to generate random numbers for each thread
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/              
__global__ void initializeIndexRandomTrees(int sizePopulation, float *indexRandomTrees, curandState_t* states){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  indexRandomTrees[tid] = (curand(&states[tid]) % sizePopulation);
}

/*!
* \fn       __global__ void initializeIndexRandomTrees(int sizePopulation, float *indexRandomTrees, curandState_t* states)
* \brief    this kernel generates random indexes for random trees that are used in the mutation operator to select two random trees.
* \param    int sizePopulation: this variable contains the number of individuals that the population has
* \param    float *mutationStep: this pointer stores the indexes randomly for mutation
* \param    curandState_t* states: random status pointer to generate random numbers for each thread
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/              
__global__ void initializeMutationStep(float *mutationStep, curandState_t* state){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  mutationStep[tid] = (curand_uniform(&state[tid]));
}

/*!
* \fn       __global__ void geometricSemanticMutation(float *initialPopulationSemantics, float *randomTreesSemantics, float *newSemanticsOffsprings, int sizePopulation, int nrow, int tElements, int generation, float *indexRandomTrees, entry_ *x)
* \brief    The GSM operator is basically a vector addition operation, that can be performed independently for each semantic element STi,j.
            However, it is necessary to select the semantics of two random trees R u and R v , and a random mutation step ms.
* \param    float *initialPopulationSemantics: this vector of pointers contains the semantics of the initial population
* \param    float *randomTreesSemantics: this vector of pointers contains the semantics of the random trees
* \param    float *newSemanticsOffsprings: this vector of pointers will store the semantics of the new offspring
* \param    int sizePopulation: this variable contains the number of individuals that the population has
* \param    int nrow: variable containing the number of rows (instances) of the training dataset
* \param    int tElements: variables containing the total number of semantic elements
* \param    int generation: number of generation
* \param    float *indexRandomTrees: this pointer stores the indexes randomly for mutation
* \param    struc *x: variable used to store training and test instances 
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__global__ void geometricSemanticMutation(float *initialPopulationSemantics, 
float *randomTreesSemantics, float *newSemanticsOffsprings, int sizePopulation,
int nrow, int tElements, int generation, float *indexRandomTrees, entry_ *y,
int index, float *mutationStep, int sigmoid, int normalize){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  double s1 =0;
  double s2 =0;
  int nSeed = (tid/nrow);
  int firstTree = indexRandomTrees[nSeed], secondTree = indexRandomTrees[sizePopulation + nSeed];
  if (normalize==0) {
    if (tid%nrow==0){
      y[(index*sizePopulation)+tid/nrow].firstParent=firstTree;
      y[(index*sizePopulation)+tid/nrow].secondParent=secondTree;
      y[(index*sizePopulation)+tid/nrow].number=tid/nrow;
      y[(index*sizePopulation)+tid/nrow].event=1;
      y[(index*sizePopulation)+tid/nrow].newIndividual=tid/nrow;
      y[(index*sizePopulation)+tid/nrow].mark=0;
      y[(index*sizePopulation)+tid/nrow].mutStep=mutationStep[tid/nrow];
    }
  }else{
    if (tid%nrow==0){
      y[(index*sizePopulation)+tid/nrow].firstParent=firstTree;
      y[(index*sizePopulation)+tid/nrow].secondParent=-1;
      y[(index*sizePopulation)+tid/nrow].number=tid/nrow;
      y[(index*sizePopulation)+tid/nrow].event=1;
      y[(index*sizePopulation)+tid/nrow].newIndividual=tid/nrow;
      y[(index*sizePopulation)+tid/nrow].mark=0;
      y[(index*sizePopulation)+tid/nrow].mutStep=mutationStep[tid/nrow];
    }
  }
  
  
  if(sigmoid==0 && normalize==0){
    double tmp1=0,tmp2=0;
    // s1=((randomTreesSemantics[firstTree*nrow+tid%nrow])  * (1/(1+exp(-(randomTreesSemantics[firstTree*nrow+tid%nrow])))));
    // s2=((randomTreesSemantics[secondTree*nrow+tid%nrow]) * (1/(1+exp(-(randomTreesSemantics[secondTree*nrow+tid%nrow])))));
    // s1=((rtSemantics[firstTree*nrow+r])  * (1/(1+exp(-(rtSemantics[firstTree*nrow+r])))));
    // s2=((rtSemantics[secondTree*nrow+r]) * (1/(1+exp(-(rtSemantics[secondTree*nrow+r])))));
    tmp1 = randomTreesSemantics[firstTree*nrow+tid%nrow];
    tmp2 = randomTreesSemantics[secondTree*nrow+tid%nrow];
    if(isnan(tmp1) || isinf(tmp1) ){
      tmp1=1;
    }else if(tmp1 < 0){
      tmp1 = -1000;
    }else{
      tmp1 = 1000;
    }
    if(isnan(tmp2) || isinf(tmp2) ){
      tmp2=1;
    }else if(tmp2 < 0){
      tmp2 = -1000;
    }else{
      tmp2 = 1000;
    }
    s1=((tmp1)  * (1/(1+exp(-(tmp1)))));
    s2=((tmp2)  * (1/(1+exp(-(tmp2)))));

  }else if(sigmoid==1 && normalize==0){
    s1 = (1.0/(1+exp(-(randomTreesSemantics[firstTree*nrow+tid%nrow]))));
    s2 = (1.0/(1+exp(-(randomTreesSemantics[secondTree*nrow+tid%nrow]))));
  }else if(sigmoid==2 && normalize==0){
    s1 = ((2.0/(1+exp(-2*(randomTreesSemantics[firstTree*nrow+tid%nrow]))))-1);
    s2 = ((2.0/(1+exp(-2*(randomTreesSemantics[secondTree*nrow+tid%nrow]))))-1); 
  }
  if (normalize==0) {
    newSemanticsOffsprings[tid] = initialPopulationSemantics[tid]+(mutationStep[tid/nrow])*(s1-s2);
  }else {
    newSemanticsOffsprings[tid] = initialPopulationSemantics[tid]+(mutationStep[tid/nrow])*(randomTreesSemantics[firstTree*nrow+tid%nrow]);
  }
}

/*!
* \fn        __host__ void readInpuData(char *train_file, char *test_file, float *dataTrain, float *dataTest, float *dataTrainTarget,float *dataTestTarget, int nrow, int nvar, int nrowTest, int nvarTest)
* \brief    This function that reads data from training and test files, also reads target values to store them in pointer vectors.
* \param    char *train_file: name of the file with training instances 
* \param    char *test_file: name of the file with test instances
* \param    float *dataTrain: vector pointers to store training data
* \param    float *dataTest: vector pointers to store test data
* \param    float *dataTrainTarget: vector pointers to store training target data
* \param    float *dataTestTarget: vector pointers to store test target data
* \param    int nrow: variable containing the number of rows (instances) of the training dataset
* \param    int nvar: variable containing the number of columns (excluding the target) of the training dataset
* \param    int nrowTest: variable containing the number of rows (instances) of the test dataset
* \param    int nvarTest: variable containing the number of columns (excluding the target) of the test dataset
* \return   void: 
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/
__host__ void readInpuData(char *trainFile, char *testFile, float *dataTrain, float *dataTest, float *dataTrainTarget, float *dataTestTarget, int nrow, int nvar, int nrowTest, int nvarTest){

  std::fstream in(trainFile,ios::in);
  if (!in.is_open()){
    cout<<endl<<"ERROR: TRAINING FILE NOT FOUND." << endl;
    exit(-1);
  }
  std::fstream inTest(testFile,ios::in);
  if (!in.is_open()){
    cout<<endl<<"ERROR: TEST FILE NOT FOUND." << endl;
    exit(-1);
  }
  char Str[1024];
  int max = nvar;
  for(int i=0;i<nrow;i++){
    for (int j=0; j<nvar+1; j++){
      if (j==max){
        in>>Str;
        dataTrainTarget[i]=atof(Str);
      }
      if (j<nvar){
        in>>Str;
        dataTrain[i*nvar+j] = atof(Str);
      }
    }
  }
  in.close();
  int maxTest = nvarTest;
  for(int i=0;i<nrowTest;i++){
    for (int j=0; j<nvarTest+1; j++){
      if (j==maxTest){
        inTest>>Str;
        dataTestTarget[i]=atof(Str);
      }
      if (j<nvarTest){
        inTest>>Str;
        dataTest[i*nvarTest+j] = atof(Str);
      }
    }
  }
  inTest.close();
}

/*!
* \fn        __host__ void readInpuData(char *train_file, char *test_file, float *dataTrain, float *dataTest, float *dataTrainTarget,float *dataTestTarget, int nrow, int nvar, int nrowTest, int nvarTest)
* \brief    This function that reads data from training and test files, also reads target values to store them in pointer vectors.
* \param    char *train_file: name of the file with training instances 
* \param    char *test_file: name of the file with test instances
* \param    float *dataTrain: vector pointers to store training data
* \param    float *dataTest: vector pointers to store test data
* \param    float *dataTrainTarget: vector pointers to store training target data
* \param    float *dataTestTarget: vector pointers to store test target data
* \param    int nrow: variable containing the number of rows (instances) of the training dataset
* \param    int nvar: variable containing the number of columns (excluding the target) of the training dataset
* \param    int nrowTest: variable containing the number of rows (instances) of the test dataset
* \param    int nvarTest: variable containing the number of columns (excluding the target) of the test dataset
* \return   void: 
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/
__host__ void readInpuDataTrain(char *trainFile, float *dataTrain, float *dataTrainTarget, int nrow, int nvar){

  std::fstream in(trainFile,ios::in);
  if (!in.is_open()){
    cout<<endl<<"ERROR: TRAINING FILE NOT FOUND." << endl;
    exit(-1);
  }
  char Str[1024];
  int max = nvar;
  for(int i=0;i<nrow;i++){
    for (int j=0; j<nvar+1; j++){
      if (j==max){
        in>>Str;
        dataTrainTarget[i]=atof(Str);
      }
      if (j<nvar){
        in>>Str;
        dataTrain[i*nvar+j] = atof(Str);
      }
    }
  }
  in.close();
}

/*!
* \fn        __host__ void countInputFile(std::string fileName, int &rows, int &cols)
* \brief     function that reads rows and colums of files to train and test
* \param     std::string fileName: This variable store the name of file data train or test
* \param     int rows: This variable store the number of rows of file data
* \param     int nvar: This variable store the number of colums of file data
* \return    void
* \date      05/10/2021
* \author    José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file      GsgpCuda.cpp
*/
void countInputFile(std::string fileName, int &rows, int &cols){
  std::ifstream f(fileName);
  string line;
  std::getline(f, line);
  stringstream s;
  s << line;                   //send the line to the stringstream object...
  int how_many_columns = 0;    
  double value;
  while(s >> value) how_many_columns++;  //while there's something in the line, increase the number of columns

  cols = how_many_columns;
  int how_many_rows =1;
  while (std::getline(f, line)) how_many_rows++;
  rows = how_many_rows;

  f.close();
}

/*!
* \fn        __host__ void readConfigFile(cfg *config)
* \brief     function that reads the configuration file
* \param     cfg *config: pointer to the struct containing the variables needed to run the program
* \return    void
* \date      05/10/2020
* \author    José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file      GsgpCuda.cpp
*/
__host__ void readConfigFile(string path,cfg *config){
  // std::fstream f("configuration.ini", ios::in);
  // printf("path desde el ini %s\n", path.c_str());
  std::fstream f(path, ios::in);
  if (!f.is_open()){
    cerr<<"CONFIGURATION FILE NOT FOUND." << endl;
    exit(-1);
  }
  int k=0;
  while(!f.eof()){
    char str[100]="";
    char str1[100]="";
    char str2[100]="";
    int j=0;
    f.getline(str,100);
    if(str[0]!='\0'){
      while(str[j]!='='){
    	  if (str[j]!=' ')
    		  str1[j] = str[j];
    	  else
    		  str1[j] = '\0';
    	  j++;
      }
      j++;
      int i=0;
      while(str[j]==' '){
        j++;
      }
      while(str[j]!='\0'){
        str2[i] = str[j];
        j++;
        i++;
      }
    }

    if (strcmp(str1, "numberGenerations")==0)
    	config->numberGenerations=atoi(str2);

    if (strcmp(str1, "populationSize")==0)
    	config->populationSize =atoi(str2);

    if (strcmp(str1, "maxIndividualLength")==0)
    	config->maxIndividualLength=atoi(str2);

    if (strcmp(str1, "functionRatio")==0)
      config->functionRatio =atof(str2);

    if (strcmp(str1, "variableRatio")==0)
      config->variableRatio =atof(str2);

    if (strcmp(str1, "maxRandomConstant")==0)
    	config->maxRandomConstant=atof(str2);

    if (strcmp(str1, "sigmoid")==0)
      config->sigmoid=atoi(str2);

    if (strcmp(str1, "errorFunction")==0)
      config->errorFunction=atoi(str2);

    if (strcmp(str1, "oms")==0)
    	config->oms=atoi(str2);

    if (strcmp(str1, "normalize")==0)
    	config->normalize=atoi(str2);

    if (strcmp(str1, "do_min_max")==0)
    	config->do_min_max=atoi(str2);
    
    if (strcmp(str1, "protected_division")==0)
    	config->protected_division=atoi(str2);

    if (strcmp(str1, "logPath")==0)
    	strcpy(config->logPath, str2);

    k++;
  } 
    f.close();
    if(config->populationSize<0 || config->maxIndividualLength<0 ){
        cout<<"ERROR: POPULATION SIZE AND MAX DEPTH MUST BE SMALLER THAN (OR EQUAL TO) 0 AND THEIR SUM SMALLER THAN (OR EQUAL TO) 1.";
        exit(-1);
    }
}

/*!
* \fn       bool IsPathExist(const std::string &s)
* \brief    function to check if exist a directory path.
* \param    string &s: name of path
* \return   void
* \date     27/02/2021
* \author   Luis Armando Cardenas Florido, José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     CudaGP.cpp
*/
bool IsPathExist(const std::string &s){
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}

/*!
* \fn       void checkDirectoryPath(string dirPath)
* \brief    function to check if exist a directory path, if not, create the directory path
* \param    string dirPath: name of path
* \return   void
* \date     27/02/2021
* \author   Luis Armando Cardenas Florido, José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
void checkDirectoryPath(std::string dirPath){
  if ((dirPath.length() > 0 ) && (!IsPathExist(dirPath))){
    if (mkdir(dirPath.c_str(), 0771) != 0 && errno != EEXIST){
      cout<<"ERROR: PATH "<<dirPath<<" NOT FOUND OR CANNOT BE CREATED.";
      exit(-1);
   	}
  }
}

/*!
* \fn       static void list_dir(std::string path, std::string nameFile, int useMultipleFiles, std::vector<string> &files)
* \brief    function for get directories files
* \param    string path: path where the algorithm read files needed to work
* \param    string name: nombre de los archivos a buscar en la ruta
* \param    int useMultipleFiles: variable to exit file search
* \param    vector &files: variable to store the names of the found files.
* \return   void
* \date     10/02/2021
* \author   Luis Armando Cardenas Florido, José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/

static void list_dir(std::string path, std::string nameFile, int useMultipleFiles, std::vector<string> &files){
    struct dirent *entry;
    DIR *dir = opendir(path.c_str());
    if (dir == NULL) {
        return;
    }
    while ((entry = readdir(dir)) != NULL) {
    	std::string cFile(entry->d_name);
		  if (cFile.compare(0, nameFile.length(), nameFile) == 0){
		  	files.push_back(cFile);
        }
      	if (useMultipleFiles == 0)
      		break;
      	//files.push_back(string(entry->d_name));
        //printf("%s\n",entry->d_name);
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
}

/*!
* \fn       __host__ void markTracesGeneration(entry *vectorTraces, int populationSize, int generationSize ,int bestIndividual)
* \brief    This function that implements the marking procedure used to store the structure of the optimal solution
* \param    entry *vectorTraces: variable used to store the information needed to evaluate the optimal individual on newly provided unseen data.
* \param    int populationSize: Number of individuals in the population
* \param    int generationSize: number of generations
* \param    int bestIndividual: index of best individual of population
* \return   void
* \date     10/02/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp
*/
__host__ void markTracesGeneration(entry *vectorTraces, int populationSize, int generationSize ,int bestIndividual){
  vectorTraces[(generationSize-1)*populationSize+bestIndividual].mark=1;
  int index,index2;
  int a = 0;
  for (int i = generationSize-1; i >0; i--){
    for (int j = 0; j < populationSize; j++){
      index =0,index2=0;
      a = i-1;
      if(vectorTraces[i*populationSize+j].mark==1 && vectorTraces[i*populationSize+j].event==1){ 
          index = vectorTraces[i*populationSize+j].number;
          vectorTraces[a*populationSize+index].mark=1;
        }
      if(vectorTraces[i*populationSize+j].mark==1 && vectorTraces[i*populationSize+j].event==-1){
          index2 = vectorTraces[i*populationSize+j].number;
          vectorTraces[a*populationSize+index2].mark=1;
        }
    }
  }
}

/*!
* \fn       __host__ void saveTrace(entry *structSurvivor, int generation) 
* \brief    Function that stores the information related to the evolutionary cycle and stores the indices of the individuals that were used in each generation to create new offspring,
            to later perform the reconstruction of the optimal solution in the trace.txt file
* \param    string path: path where the algorithm output files are stored.
* \param    struc *structSurvivor: pointer that stores information of the best individual throughout the generations
* \param    int generation: number of generations
* \param    int populationSize: Number of individuals in the population
* \return   void
* \date     08/10/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/
__host__ void saveTrace(std::string name, std::string path, entry *structSurvivor, int generation, int populationSize){
  cudaDeviceSynchronize();
  std::string tmpT = name;
  std::string tmpExt = ".csv";
  tmpT = path + tmpT  +tmpExt;  
  std::ofstream trace(tmpT,ios::out);
  int r=0;
  
  for(int i=0; i<generation; i++){
    r=generation-1;
    for (int j = 0; j < populationSize; j++){
      if (structSurvivor[i*populationSize+j].mark==1){
        trace << structSurvivor[i*populationSize+j].firstParent<<"\t"<<structSurvivor[i*populationSize+j].secondParent<<"\t"<<structSurvivor[i*populationSize+j].number<<"\t"<<structSurvivor[i*populationSize+j].event<<"\t"<<structSurvivor[i*populationSize+j].newIndividual<<"\t"<<structSurvivor[i*populationSize+j].mutStep<<endl;
      }
    }
    if(i<r)
      trace<<"***"<<endl;
    else
      trace<<"***";
  } 
}

/*!
* \fn       __host__ void saveTraceComplete(std::string path, entry *structSurvivor, int generation, int populationSize)
* \brief    Function that stores the information related to the evolutionary cycle and stores the indices of the individuals that were used in each generation to create new offspring,
            to later perform the reconstruction of the optimal solution in the trace.txt file
* \param    string path: path where the algorithm output files are stored.
* \param    struc *structSurvivor: pointer that stores information of the best individual throughout the generations
* \param    int generation: number of generations
* \param    int populationSize: : Number of individuals in the population
* \return   void
* \date     08/10/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/
__host__ void saveTraceComplete(std::string path, entry *structSurvivor, int generation, int populationSize){
  cudaDeviceSynchronize();
  std::string tmpT  = "traceCompleteIndividuals";
  std::string tmpExt = ".csv";
  std::string tmpTime = currentDateTime();
  tmpT = path + tmpT + tmpTime +tmpExt;  
  std::ofstream trace(tmpT,ios::out);
  for(int i=0; i<generation; i++){
    for (int j = 0; j < populationSize; j++){
      trace << structSurvivor[i*populationSize+j].firstParent<<"\t" << structSurvivor[i*populationSize+j].secondParent << "\t" << structSurvivor[i*populationSize+j].number << "\t" << structSurvivor[i*populationSize+j].event <<"\t"<< structSurvivor[i*populationSize+j].mark <<"\t"<<  structSurvivor[i*populationSize+j].newIndividual << "\t" <<structSurvivor[i*populationSize+j].mutStep<<endl;
    }
    if(i<generation-1)
      trace<<"***"<<endl;
    else
      trace<<"***";
  } 
}

/*!
* \fn       __host__ void saveIndividuals(std::string path, float *hInitialPopulation, std::string namePopulation ,int sizeMaxDepthIndividual, int populationSize)
* \brief    This function that stores the information related to the initial population on file initialPopulation.csv
* \param    string path: path where the algorithm output files are stored.
* \param    float *hInitialPopulation: This vector pointers to store the individuals of the initial population.
* \param    string namePopulation: This variable contein the name of the population o random trees.
* \param    int sizeMAxDepthIndividual: This variable thar stores maximum depth for individuals
* \param    int populationSize
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/

__host__ void saveIndividuals(std::string path, float *hInitialPopulation, std::string namePopulation ,int sizeMaxDepthIndividual, int populationSize){  
  namePopulation = path + namePopulation;       
  std::ofstream outIndividuals(namePopulation,ios::out);
  for (int i=0; i< (populationSize); i++){
      for (int j=0; j<sizeMaxDepthIndividual; j++){
        outIndividuals<< hInitialPopulation[i*sizeMaxDepthIndividual+j] << " ";      
      }
    outIndividuals<< endl;        
  }
}

/*!
* \fn       __host__ void saveDataNormalize(std::string path, float *hDataNormalize, int populationSize)
* \brief    This function that stores the information related to normalize semantic to random trees
* \param    string path: path where the algorithm output files are stored.
* \param    float *hDataNormalize: This vector pointers to store the individuals of the initial population.
* \param    int populationSize
* \return   void
* \date     28/03/2023
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/

__host__ void saveDataNormalize(std::string path, float *hDataNormalize, int populationSize, std::string outputNameFiles){  
  outputNameFiles = outputNameFiles + "_dataNormalize.csv";
  outputNameFiles = path + outputNameFiles;
  std::ofstream outData(outputNameFiles,ios::out);
  for (int i=0; i< populationSize; i++){
      for (int j=0; j<4; j++){
        outData<< hDataNormalize[i*4+j] << " ";      
      }
    outData<< endl;        
  }
}

/*!
* \fn       __host__ void readDataNormalize(std::string path, float *hDataNormalize, int populationSize, std::string outputNameFiles)
* \brief    This function that stores the information related to normalize semantic to random trees
* \param    string path: path where the algorithm output files are stored.
* \param    float *hDataNormalize: This vector pointers to store the individuals of the initial population.
* \param    int populationSize
* \return   void
* \date     28/03/2023
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp  
*/

__host__ void readDataNormalize(std::string path, float *hDataNormalize, int populationSize, std::string outputNameFiles){  
  outputNameFiles = outputNameFiles + "_dataNormalize.csv";
  outputNameFiles = path + outputNameFiles;
  
  std::fstream inTest(outputNameFiles,ios::in);
  if (!inTest.is_open()){
    cout<<endl<<"ERROR: NORMALIZE DATA FILE NOT FOUND." << endl;
    exit(-1);
  }

  char Str[1024];
  for(int i=0;i<populationSize;i++){
    for (int j=0; j<4; j++){
      inTest>>Str;
      hDataNormalize[i*4+j] = atof(Str);
      // printf("Data Normalizada %f ",hDataNormalize[i*4+j]);
    }
    // printf("\n");
  }

}

/*!
* \fn        __host__ void readInpuTestData(char *train_file, char *test_file, float *dataTrain, float *dataTest, float *dataTrainTarget,float *dataTestTarget, int nrow, int nvar, int nrowTest, int nvarTest)
* \brief    This function that reads data from test file, also reads target values to store them in pointer vectors.
* \param    char *test_file: name of the file with test instances
* \param    float *dataTest: vector pointers to store test data
* \param    float *dataTestTarget: vector pointers to store test target data
* \param    int nrowTest: variable containing the number of rows (instances) of the test dataset
* \param    int nvarTest: variable containing the number of columns (excluding the target) of the test dataset
* \return   void: 
* \date     8/10/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/

__host__ void readInpuTestData( char *test_file, float *dataTest, int nrowTest, int nvarTest){
  std::fstream inTest(test_file,ios::in);
  if (!inTest.is_open()){
    cout<<endl<<"ERROR: TEST FILE NOT FOUND." << endl;
    exit(-1);
  }
  char Str[1024];
  for(int i=0;i<nrowTest;i++){
    for (int j=0; j<nvarTest+1; j++){
      if (j<nvarTest){
        // printf("indice de lectura de datos test %i \n",j);
        inTest>>Str;
        dataTest[i*nvarTest+j] = atof(Str);
      }
    }
  }
  inTest.close();
}

/*!
* \fn      __host__ void readPopulation( float *initialPopulation, float *randomTrees, int sizePopulation, int depth, std::string log, std::string name, std::string nameR)
* \brief    This function that read the information related to the initial population from file initialPopulation.csv
* \param    float *initialPopulation: This vector pointers to store the individuals of the initial population.
* \param    float *randomTrees: vector of pointers storing random trees
* \param    int sizePopulation: Number of individuals in the population
* \param    int depth: This variable thar stores maximum depth for individuals
* \param    std::string log: path where the algorithm output files are stored.
* \param    std::string name: name of file the initial population
* \param    std::string nameR: name of file the random trees
* \return   void: 
* \date     8/10/2021
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/

__host__ void readPopulation( float *initialPopulation, float *randomTrees, int sizePopulation, int depth, std::string log, std::string name, std::string nameR){
 
  std::fstream inTest(name,ios::in);
  if (!inTest.is_open()){
    cout<<endl<<"ERROR: INITIAL POPULATION FILE NOT FOUND." << endl;
    exit(-1);
  }

  char Str[1024];
  for(int i=0;i<sizePopulation;i++){
    for (int j=0; j<depth; j++){
      inTest>>Str;
      initialPopulation[i*depth+j] = atof(Str);
    }
  }
  inTest.close();
  std::fstream in(nameR,ios::in);
  if (!in.is_open()){
    cout<<endl<<"ERROR: RANDOM TREE FILE NOT FOUND." << endl;
    exit(-1);
  }
  char tr[1024];
  for(int i=0;i<sizePopulation;i++){
    for (int j=0; j<depth; j++){
      in>>tr;
      randomTrees[i*depth+j] = atof(tr);
    }
  }
  in.close();
}

/*!
* \fn       __host__ void evaluate_data(std::string path, int generations, float *initialPopulation, float *randomTrees, std::ofstream& OUT, std::string log, int nrow, int numIndi, int nvarTest, float *salidas)
* \brief    This function that evaluates the best model stored in trace.txt over newly provided unseen data
* \param    std::string path: trace file name
* \param    int generations: number of generations.
* \param    const int sizeMaxDepthIndividual: ariable that stores maximum depth for individuals.
* \param    float *initialPopulation: vector of pointers storing the initial population
* \param    float *randomTrees: vector of pointers storing random trees
* \param    std::ofstream& OUT: file where the result of the evaluation of the best model with each fitness will be written.
* \param    std::string log: path where the algorithm output files are stored.
* \param    float *dataTest : vector pointers to store test data
* \param    int nrow: This variable contains the number of fitness cases.
* \param    int numIndi: number of individuals in the population
* \param    int nvarTest:This variable contains the number of features of problem.
* \return   void
* \date     05/12/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     testSemantic.cu
*/

__host__ void evaluate_data(std::string path, int generations, float *initialPopulation, 
float *randomTrees, std::ofstream& OUT, std::string log, int nrow, int nvarTest, int sigmoid,cfg config){
  // printf("path %s y log %s \n", path.c_str(), log.c_str());
  std::vector<string>filesRa = vector<string>();
  list_dir(log,path,1,filesRa);
  int tama = filesRa.size();
  std::string nameFile = filesRa[0];
  char tracePath[500] = "";
  strcat(tracePath,log.c_str());
  strcat(tracePath,nameFile.c_str());
  vector <double> eval_;
  vector <double> eval_new;
  int best=0;
  float r1,r2;
  cudaDeviceSynchronize();
  for (int i = 0; i < nrow; i++){
    r1=0;
    for(int j=0; j<config.populationSize; j++){
      eval_.push_back(-1);
    }
    fstream in(tracePath,ios::in);
    if(!in.is_open()) {
      cout<<endl<<"ERROR: FILE MODEL NOT FOUND." << endl;
      exit(-1);
    }else{
      char str[255];
      while(true){
        in >> str;
        if(strcmp(str,"***")==0){
          break;
        }
        int index1 = atoi(str); 
        in >> str;
        int index2 = atoi(str); 
        in >> str;
        int index3 = atoi(str); 
        in >> str;
        int index4 = atoi(str); 
        in >> str;
        int index5 = atoi(str);
        in >> str;
        double index6 = atof(str); 
        if(index4==1){
          if (index2==-1) {
            if (config.do_min_max==0) {
              r1 = randomTrees[index1*nrow+i];
              eval_[index5]= initialPopulation[index3*nrow+i]+(index6)*(r1);
              // printf("Entro al do min max 0 primer if %f %f %f \n",initialPopulation[index3*nrow+i], (index6), (r1));
            }else if (config.do_min_max==1) {
              r1 = randomTrees[index1*nrow+i];
              eval_[index5]= initialPopulation[index3*nrow+i]+(index6)*(r1);
              // printf("Entro al do min max 1 primer if %f %f %f \n",initialPopulation[index3*nrow+i], (index6), (r1));
            }else if (config.do_min_max==2) {
              r1 = randomTrees[index1*nrow+i];
              eval_[index5]= initialPopulation[index3*nrow+i]+(index6)*(r1);
              // printf("Entro al do min max 2 primer if %f %f %f %i\n",initialPopulation[index3*nrow+i], (index6), (r1),index1);
            }
          }else{
            r1=0;
            r2=0;
            if(sigmoid==0){     
              double tmp1=0,tmp2=0;
              tmp1 = randomTrees[index1*nrow+i];
              tmp2 = randomTrees[index2*nrow+i];
              if(isnan(tmp1) || isinf(tmp1) ){
                tmp1=1;
              }else if(tmp1 < 0){
                tmp1 = -1000;
              }else{
                tmp1 = 1000;
              }
              if(isnan(tmp2) || isinf(tmp2) ){
                tmp2=1;
              }else if(tmp2 < 0){
                tmp2 = -1000;
              }else{
                tmp2 = 1000;
              }
              r1=((tmp1)  * (1/(1+exp(-(tmp1)))));
              r2=((tmp2)  * (1/(1+exp(-(tmp2)))));
            }else if(sigmoid==1){
              r1 = ((1.0/(1+exp(-(randomTrees[index1*nrow+i])))));
              r2 = (1.0/(1+exp(-(randomTrees[index2*nrow+i]))));
            }else if(sigmoid==2){
              r1 = ((2.0/(1+exp(-2*(randomTrees[index1*nrow+i]))))-1);
              r2 = ((2.0/(1+exp(-2*(randomTrees[index2*nrow+i]))))-1);
            }
            eval_[index5]= initialPopulation[index3*nrow+i]+(index6)*(r1-r2);
          }
          best=index5;
        }
        if(index4==-1){
          eval_[index3] = initialPopulation[index3*nrow+i];
          best=index3;
        }
      }
      while(!in.eof()){
        for(int l=0; l<config.populationSize; l++){
            eval_new.push_back(-1);
          }
        while(true){
          in >> str;
          if(strcmp(str,"***")==0){
              break;
          }
          int index1 = atoi(str); 
          in >> str;
          int index2 = atoi(str); 
          in >> str;
          int index3 = atoi(str); 
          in >> str;
          int index4 = atoi(str); 
          in >> str;
          int index5 = atoi(str); 
				  in >> str;
				  double index6 = atof(str); 
          if(index4==1){
            r1=0;
            if (index2==-1) {
              if (config.do_min_max==0) {
                r1 = randomTrees[index1*nrow+i];
                eval_new[index5]= eval_[best]+(index6)*(r1);
                // printf("Entro al do min max 0 seg if %f %f %f \n",eval_[best], (index6), (r1));
              }else if (config.do_min_max==1) {
                r1 = randomTrees[index1*nrow+i];
                eval_new[index5]= eval_[best]+(index6)*(r1);
              }else if (config.do_min_max==2) {
                r1 = randomTrees[index1*nrow+i];
                eval_new[index5]= eval_[best]+(index6)*(r1);
                // printf("Entro al do min max 2 segundo if%f %f %f %f %i\n",eval_new[index5],eval_[best], (index6), (r1),index1);
              }
            }else{ 
            r1=0;
            r2=0;
            if(sigmoid==0){
              // r1 = ((randomTrees[index1*nrow+i])*(1/(1+exp(-(randomTrees[index1*nrow+i])))));
              // r2 = ((randomTrees[index2*nrow+i])*(1/(1+exp(-(randomTrees[index2*nrow+i])))));
              // r1 = ((randomTrees[index1*nrow+i])*(1/(1+exp(-(randomTrees[index1*nrow+i])))));
            // r2 = ((randomTrees[index2*nrow+i])*(1/(1+exp(-(randomTrees[index2*nrow+i])))));
            double tmp1=0,tmp2=0;
            // s1=((randomTreesSemantics[firstTree*nrow+tid%nrow])  * (1/(1+exp(-(randomTreesSemantics[firstTree*nrow+tid%nrow])))));
            // s2=((randomTreesSemantics[secondTree*nrow+tid%nrow]) * (1/(1+exp(-(randomTreesSemantics[secondTree*nrow+tid%nrow])))));
            // s1=((rtSemantics[firstTree*nrow+r])  * (1/(1+exp(-(rtSemantics[firstTree*nrow+r])))));
            // s2=((rtSemantics[secondTree*nrow+r]) * (1/(1+exp(-(rtSemantics[secondTree*nrow+r])))));
            tmp1 = randomTrees[index1*nrow+i];
            tmp2 = randomTrees[index2*nrow+i];
            if(isnan(tmp1) || isinf(tmp1) ){
              tmp1=1;
            }else if(tmp1 < 0){
              tmp1 = -1000;
            }else{
              tmp1 = 1000;
            }
            if(isnan(tmp2) || isinf(tmp2) ){
              tmp2=1;
            }else if(tmp2 < 0){
              tmp2 = -1000;
            }else{
              tmp2 = 1000;
            }
            r1=((tmp1)  * (1/(1+exp(-(tmp1)))));
            r2=((tmp2)  * (1/(1+exp(-(tmp2)))));
            }else if(sigmoid==1){
              r1 = ((1.0/(1+exp(-(randomTrees[index1*nrow+i])))));
              r2 = (1.0/(1+exp(-(randomTrees[index2*nrow+i]))));
            }else if(sigmoid==2){
              r1 = ((2.0/(1+exp(-2*(randomTrees[index1*nrow+i]))))-1);
              r2 = ((2.0/(1+exp(-2*(randomTrees[index2*nrow+i]))))-1);
            }
            eval_new[index5] = eval_[best]+(index6)*(r1-r2);
            }
            best=index5;
          }
          if(index4==-1){
            eval_new[index3] = eval_[best];
					  best=index3;
          }
        }
        eval_.clear();
        eval_.assign(eval_new.begin(),eval_new.end());
        eval_new.clear();        
      }
    }
    OUT<<eval_[best]<<endl;
  }
}

void validacion(std::string name, float *targer, int nrow){
    
    std::fstream in(name.c_str(),ios::in);
    char Str[1024];
    float RMSE=0;
    double t=0;
    float tmp[nrow];
    if (!in.is_open())
    {
      cout<<endl<<"ERROR: TRAINING FILE NOT FOUND." << endl;
      exit(-1);
    }

    for(int i=0;i<nrow;i++){
        in>>Str;
        tmp[i]=atof(Str);
        RMSE += (tmp[i]-targer[i])*(tmp[i]-targer[i]);
        //printf("Error %f diferencia %f semantica %f - targer %f \n", RMSE, (tmp[i]-targer[i])*(tmp[i]-targer[i]) ,tmp[i], targer[i]);
    }
    double p = RMSE/nrow;
    t = sqrt(p);
    printf("error %f nrows %i \n", t, nrow);
}

/*!
* \fn       __global__ void computeSemanticsExpressions    
* \brief    The computeSemanticsExpressions kernel is an interpreter, that decodes each individual and evaluates it over all fitness cases,
            producing as output the semantic vector of each individual. The chromosome is interpreted linearly, using an auxiliary LIFO stack D that stores 
            terminals from the chromosome and the output from valid operations.
* \param    float *inputPopulation: vector pointers to store the individuals of the population
* \param    float *outSemantic: vector pointers to store the semantics of each individual in the population
* \param    int sizeMaxDepthIndividual: variable thar stores maximum depth for individuals
* \param    float *data: pointer vector containing training or test data
* \param    int nrow: variable containing the number of rows (instances) of the training dataset
* \param    int nvar: variable containing the number of columns (excluding the target) of the training dataset
* \param    int *pushGenes: auxiliary pointer that stores the positions of individuals 
* \param    float *stackInd: auxiliary pointer that stores the values ​​resulting from the interpretation of each individual
* \return   void
* \date     01/25/2020
* \author   José Manuel Muñoz Contreras, Leonardo Trujillo, Daniel E. Hernandez, Perla Juárez Smith
* \file     GsgpCuda.cpp 
*/
__global__ void computeSemanticsExpressions(float *inputPopulation, float *outSemantic, 
	unsigned int sizeMaxDepthIndividual,int nrow, int nvar, int *pushGenes, int *pushGenesModel,
  float *stackInd, float *stackModel, int protected_division){
	
	 const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
	 pushGenes[tid] = tid * sizeMaxDepthIndividual;
	 pushGenesModel[tid] =  tid * sizeMaxDepthIndividual;
	 int t,t_;
	 float tmp,tmp2,out;
	 float maxVar = (float)(1000 + nvar -1) * (float)(-1);

	for(int k=0; k<nrow; k++){
    out=0, t=0, t_=0 ,tmp=0, tmp2=0;
    clearStack(pushGenes,sizeMaxDepthIndividual, stackInd);
    for(int i=0; i < sizeMaxDepthIndividual; i++){
      if(inputPopulation[tid*sizeMaxDepthIndividual+i] >= -999 && inputPopulation[tid*sizeMaxDepthIndividual+i] <= 999 ){
        pushGenes[tid] = push(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenes,stackInd);
        // pushGenes[tid] = pushMo(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenes,stackModel);
        pushGenesModel[tid] = pushMo(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenesModel,stackModel);
		   	// printf("pushGenes %i stack %f \n", pushGenes[tid],inputPopulation[tid*sizeMaxDepthIndividual+i]);
      }else if(inputPopulation[tid*sizeMaxDepthIndividual+i] >= maxVar && inputPopulation[tid*sizeMaxDepthIndividual+i] <=-1000){
        t=inputPopulation[tid*sizeMaxDepthIndividual+i];
			  t_=(t+1000)*(-1);
			  pushGenes[tid] = push(1.0,pushGenes,stackInd);
        pushGenesModel[tid] = pushMo(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenesModel,stackModel);
			  // pushGenes[tid] = pushMo(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenes,stackModel);
			  // printf("Variable pushGenes %i stack %f \n", pushGenes[tid],inputPopulation[tid*sizeMaxDepthIndividual+i]);
		  }else if (inputPopulation[tid*sizeMaxDepthIndividual+i] == -10001){
        if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
          tmp = pop(pushGenes,stackInd);
          if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
            tmp2 = pop(pushGenes,stackInd);
            if (!isnan(tmp) && !isinf(tmp) && !isnan(tmp2) && !isinf(tmp2)){
              pushGenes[tid] = push(tmp2 + tmp,pushGenes,stackInd);
              out = tmp2+tmp;
              pushGenesModel[tid] = pushMo(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenesModel,stackModel);
              // printf("Suma pushGenes %i stack %f \n", pushGenes[tid],inputPopulation[tid*sizeMaxDepthIndividual+i]);
            }
          }else
          pushGenes[tid] = push(tmp,pushGenes,stackInd);
        }
      }else if (inputPopulation[tid*sizeMaxDepthIndividual+i] == -10001){
        if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
          tmp = pop(pushGenes,stackInd);
          if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
            tmp2 = pop(pushGenes,stackInd);
            if (!isnan(tmp) && !isinf(tmp) && !isnan(tmp2) && !isinf(tmp2)){
              pushGenes[tid] = push(tmp2 + tmp,pushGenes,stackInd);
              out = tmp2+tmp;
              pushGenesModel[tid] = pushMo(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenesModel,stackModel);
				      // printf("Suma pushGenes %i stack %f \n", pushGenes[tid],inputPopulation[tid*sizeMaxDepthIndividual+i]);
				    }
			    }else
          pushGenes[tid] = push(tmp,pushGenes,stackInd);
		    }
			}else if (inputPopulation[tid*sizeMaxDepthIndividual+i] == -10002){
        if(!isEmpty(pushGenes,sizeMaxDepthIndividual)){
          tmp = pop(pushGenes,stackInd);
          if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
            tmp2 = pop(pushGenes,stackInd);
            if (!isnan(tmp) && !isinf(tmp) && !isnan(tmp2) && !isinf(tmp2)){
              pushGenes[tid] = push(tmp2 - tmp,pushGenes,stackInd);
              out = tmp2-tmp;
              pushGenesModel[tid] = pushMo(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenesModel,stackModel);
        			  //  printf("resta pushGenes %i stack %f \n", pushGenes[tid],inputPopulation[tid*sizeMaxDepthIndividual+i]);
        		}
        	}else
          pushGenes[tid] = push(tmp,pushGenes,stackInd);
        }
      }else if (inputPopulation[tid*sizeMaxDepthIndividual+i] == -10003){
        if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
          tmp = pop(pushGenes,stackInd);
          if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
            tmp2 = pop(pushGenes,stackInd);
            if (!isnan(tmp) && !isinf(tmp) && !isnan(tmp2) && !isinf(tmp2)){
              pushGenes[tid] = push(tmp2 * tmp,pushGenes,stackInd);
              out = tmp2*tmp;
              pushGenesModel[tid] = pushMo(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenesModel,stackModel);
              // printf("*** pushGenes %i stack %f \n", pushGenes[tid],inputPopulation[tid*sizeMaxDepthIndividual+i]);
				    }
	        }else
          pushGenes[tid] = push(tmp,pushGenes,stackInd);
        }
	    }else if (inputPopulation[tid*sizeMaxDepthIndividual+i] == -10004){
        if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
          tmp = pop(pushGenes,stackModel);
          if (!isEmpty(pushGenes,sizeMaxDepthIndividual)){
            tmp2 = pop(pushGenes,stackModel);
            if (!isnan(tmp) && !isinf(tmp) && !isnan(tmp2) && !isinf(tmp2)){
                if (protected_division==0) {
                  if (tmp!=0) {
                    pushGenes[tid] = push(tmp2 / tmp,pushGenes,stackInd);
                    out = tmp2 / tmp;
                  }else if(tmp==0) {
                    pushGenes[tid] = push(tmp2 / sqrtf(1+tmp*tmp),pushGenes,stackInd);
                    out = tmp2 / sqrtf(1+(tmp*tmp));
                  }
                }else {
                    pushGenes[tid] = push(tmp2 / sqrtf(1+tmp*tmp),pushGenes,stackInd);
                    out = tmp2 / sqrtf(1+(tmp*tmp));
                }
              // pushGenes[tid] = push(tmp2 / sqrtf(1+tmp*tmp),pushGenes,stackInd);
              // out = tmp2 / sqrtf(1+tmp*tmp);
              pushGenesModel[tid] = pushMo(inputPopulation[tid*sizeMaxDepthIndividual+i],pushGenesModel,stackModel);
				      // printf("/// pushGenes %i stack %f \n", pushGenes[tid],inputPopulation[tid*sizeMaxDepthIndividual+i]);
				    }
          }else
          pushGenes[tid] = push(tmp,pushGenes,stackInd);
        }
		  }
    }
  }
}


void clearCha(stack <string> s) 
{ 
	while (!s.empty()) 
	{ 
		s.pop(); 
	} 
}

void showstack(stack <float> s) 
{ 
	while (!s.empty()) 
	{ 
		cout << '\t' << s.top(); 
		s.pop(); 
	} 
	cout << '\n'; 
}

void showstackSt(stack <string> s) 
{ 
	while (!s.empty()) 
	{ 
		cout << '\t' << s.top(); 
		s.pop(); 
	} 
	cout << '\n'; 
}


void clear(stack <float> s) 
{ 
	while (!s.empty()) 
	{ 
		s.pop(); 
	} 
}

void initialize(stack <string> s, cfg config){ 
	for(int i=0;i<config.populationSize;i++){
    for(int j =0; j< config.maxIndividualLength;j++){
      s.push(""); 
    }
  }	 
}

__host__ string getGeneExp(cfg config, float gene) {
    string Expr = "";

	if (gene == -10001) {
		Expr.append("+");
	}
	if (gene == -10002) {
		Expr.append("-");
	}
	if (gene == -10003) {
		Expr.append("*");
	}
	if (gene == -10004) {
    Expr.append("/");
	}
	if (gene == -10005) {
		Expr.append("sin");
	}
	if (gene == -10006) {
		Expr.append("cos");
	}
	if (gene == -10007) {
		Expr.append("exp");
	}
	if (gene == -10008) {
		Expr.append("log");
	}
	if (gene == -10009) {
		Expr.append("abs");
	}
	if ((gene <= -1000) && (gene > -10000)) {
		Expr.append("X_");
		Expr.append(to_string((int)((gene+1000) * (-1))));
	}
	if ((gene >= (config.maxRandomConstant * (-1) )) && (gene <= config.maxRandomConstant)) {
		Expr.append(to_string(gene));
	}

    return Expr;
}

__host__ int getModel(cfg config, float *uStackModel, float *uStackModelRt, int pop [], int rt []){

  int modelLength = 0;
  int numOpIf = 0;
  int numVars = 0;
  int numConst = 0;
  int numOps = 0;
  int numOpSin = 0;
  int numOpCos = 0;
  int numOpExp = 0;
  int numOpLog = 0;
  int numOpAbs = 0;
  std::string Expr = "";
	std::string algo ="";
	std:: string geneExpr="";
	float gene = 0;
	std:: string tmp,tmp2, tmpExpr;
	std:: string expr = "";
	std:: string strTmp = "";
	std:: string strTmp2 = "";
	int lenIndiv = 0;
	string strCont = "";
	
	std::stack<std::string> stackModelo;
  std::stack<std::string> stackModeloRt;
	for (unsigned int i = 0; i <config.populationSize; i++) {
		Expr = "";
		modelLength = 0;
		algo = "";
		geneExpr="";
		tmpExpr="";
		lenIndiv = 0;
    numVars =0;
    numConst=0;
    numOps=0;

		for(unsigned int j=0; j<config.maxIndividualLength;j++){
      
			gene =0;
			gene = uStackModel[i*config.maxIndividualLength+j];
			tmpExpr="";
			if (gene == -11111)
				break;
			if (gene == 0)
				break;
			
			if ((gene <= -1000) && (gene > -10000)) {
				numVars++;
				Expr.append("X");
				Expr.append(to_string((int)((gene+1000) * (-1))));
				Expr.append("\t");
				geneExpr = getGeneExp(config, gene);
				stackModelo.push(geneExpr);
			}
			
			if ((gene >= (config.maxRandomConstant * (-1) )) && (gene <= config.maxRandomConstant)) {
				numConst++;
				Expr.append(" ");
				Expr.append(to_string(gene));
				Expr.append("\t");
				geneExpr = getGeneExp(config, gene);
				stackModelo.push(geneExpr);
			}

			if (gene == -10001) {
				numOps++;
				Expr.append("+\t");
				geneExpr = getGeneExp(config, gene);
				if (!stackModelo.empty()) {
					tmp = stackModelo.top();
					stackModelo.pop();
					if (!stackModelo.empty()) {
						lenIndiv++;
						tmp2 = stackModelo.top();
						stackModelo.pop();
		
						tmpExpr.append("(");
						tmpExpr.append(tmp2);
						tmpExpr.append(geneExpr);
            if(tmp.find('-')!= std::string::npos){
              tmpExpr.append("(");
						  tmpExpr.append(tmp);
						  tmpExpr.append("))");
            }else{
						  tmpExpr.append(tmp);
						  tmpExpr.append(")");
            }
            
						stackModelo.push(tmpExpr);		
					}
					else{				
						stackModelo.push(tmp);
					}
				}
			}
			if (gene == -10002) {
				numOps++;
				Expr.append("-\t");
				geneExpr = getGeneExp(config, gene);
				if (!stackModelo.empty()) {
					tmp = stackModelo.top();
					stackModelo.pop();
					if (!stackModelo.empty()) {
						tmp2 = stackModelo.top();
						stackModelo.pop();
						lenIndiv++;
						tmpExpr.append("(");
						tmpExpr.append(tmp2);
						tmpExpr.append(geneExpr);
            if(tmp.find('-')!= std::string::npos){
              tmpExpr.append("(");
						  tmpExpr.append(tmp);
						  tmpExpr.append("))");
            }else{
              tmpExpr.append(tmp);
						  tmpExpr.append(")");
            }
            
						stackModelo.push(tmpExpr);	
					}
					else{
						stackModelo.push(tmp);
					}
				}
			}
			if (gene == -10003) {
				numOps++;
				Expr.append("*\t");
				geneExpr = getGeneExp(config, gene);
				if (!stackModelo.empty()) {
					tmp = stackModelo.top();
					stackModelo.pop();
					if (!stackModelo.empty()) {
						lenIndiv++;
						tmp2 = stackModelo.top();
						stackModelo.pop();
						tmpExpr.append("(");
						tmpExpr.append(tmp2);
						tmpExpr.append(geneExpr);
            if(tmp.find('-')!= std::string::npos){
              tmpExpr.append("(");
						  tmpExpr.append(tmp);
						  tmpExpr.append("))");
            }else{
						  tmpExpr.append(tmp);
						  tmpExpr.append(")");
            }
						stackModelo.push(tmpExpr);
					}
					else{
						stackModelo.push(tmp);
					}
				}
			}if (gene == -10004) {
				numOps++;
				Expr.append("/sqrt\t");
				geneExpr = getGeneExp(config, gene);
				if (!stackModelo.empty()) {
					tmp = stackModelo.top();
					stackModelo.pop();
					if (!stackModelo.empty()) {
						tmp2 = stackModelo.top();
						stackModelo.pop();
						lenIndiv++;
            if (config.protected_division==0) {
              if (tmp!="0") {
                tmpExpr.append(tmp2);
                tmpExpr.append("/");
                tmpExpr.append(tmp); 
              }else {
                tmpExpr.append("protected_division(");
						    tmpExpr.append(tmp2);
						    tmpExpr.append(",");
                tmpExpr.append(tmp);
                tmpExpr.append(")");
              }
            }else {
                tmpExpr.append("protected_division(");
						    tmpExpr.append(tmp2);
						    tmpExpr.append(",");
                tmpExpr.append(tmp);
                tmpExpr.append(")");
            }
						stackModelo.push(tmpExpr);
					}else{		
						stackModelo.push(tmp);
					}
				}
			}if (gene == -10005) {
        numOpSin++;
				Expr.append("sin\t");
			}
			if (gene == -10006) {
				numOpCos++;
				Expr.append("cos\t");
			}
			if (gene == -10007) {
				numOpExp++;
				Expr.append("exp\t");
			}
			if (gene == -10008) {
				numOpLog++;
				Expr.append("log\t");
			}
			if (gene == -10009) {
				numOpAbs++;
				Expr.append("abs\t");
			}
			if ((gene == -10010) || (gene == -10011) || (gene == -10012)) {
				numOpIf++;
				Expr.append("if\t");
			}
		}

    pop[i] = numVars + numConst + numOps;
    if (!stackModelo.empty()) {
      initPop.push_back(lenIndiv);
			tmpExpr.append(stackModelo.top());
			stackModelo.pop();
		}
				
    sample.push_back(tmpExpr);
	}

  //This section recrate to random trees 
  for (unsigned int i = 0; i <config.populationSize; i++) {
		Expr = "";
		modelLength = 0;
		algo = "";
		geneExpr="";
		tmpExpr="";
		lenIndiv = 0;
    numVars =0;
    numConst=0;
    numOps=0;
		for(unsigned int j=0; j<config.maxIndividualLength;j++){
      
			gene =0;
			gene = uStackModelRt[i*config.maxIndividualLength+j];
			tmpExpr="";
			if (gene == -11111)
				break;
			if (gene == 0)
				break;
			
			if ((gene <= -1000) && (gene > -10000)) {
				numVars++;
				Expr.append("X");
				Expr.append(to_string((int)((gene+1000) * (-1))));
				Expr.append("\t");
				geneExpr = getGeneExp(config, gene);
				stackModeloRt.push(geneExpr);
			}
			
			if ((gene >= (config.maxRandomConstant * (-1) )) && (gene <= config.maxRandomConstant)) {
				numConst++;
				Expr.append(" ");
				Expr.append(to_string(gene));
				Expr.append("\t");
				geneExpr = getGeneExp(config, gene);
				stackModeloRt.push(geneExpr);
			}

			if (gene == -10001) {
				numOps++;
				Expr.append("+\t");
				geneExpr = getGeneExp(config, gene);
				if (!stackModeloRt.empty()) {
					tmp = stackModeloRt.top();
					stackModeloRt.pop();
					if (!stackModeloRt.empty()) {
						lenIndiv++;
						tmp2 = stackModeloRt.top();
						stackModeloRt.pop();
		
						tmpExpr.append("(");
						tmpExpr.append(tmp2);
						tmpExpr.append(geneExpr);
            if(tmp.find('-')!= std::string::npos){
              tmpExpr.append("(");
						  tmpExpr.append(tmp);
						  tmpExpr.append("))");
            }else{
						  tmpExpr.append(tmp);
						  tmpExpr.append(")");
            }
            
						stackModeloRt.push(tmpExpr);		
					}
					else{				
						stackModeloRt.push(tmp);
					}
				}
			}
			if (gene == -10002) {
				numOps++;
				Expr.append("-\t");
				geneExpr = getGeneExp(config, gene);
				if (!stackModeloRt.empty()) {
					tmp = stackModeloRt.top();
					stackModeloRt.pop();
					if (!stackModeloRt.empty()) {
						tmp2 = stackModeloRt.top();
						stackModeloRt.pop();
						lenIndiv++;
						tmpExpr.append("(");
						tmpExpr.append(tmp2);
						tmpExpr.append(geneExpr);
            if (tmp.find('-')!= std::string::npos){
              tmpExpr.append("(");
		  				tmpExpr.append(tmp);
			  			tmpExpr.append("))");
            }else{
              tmpExpr.append(tmp);
			  			tmpExpr.append(")");
            }
            
						stackModeloRt.push(tmpExpr);	
					}
					else{
						stackModeloRt.push(tmp);
					}
				}
			}
			if (gene == -10003) {
				numOps++;
				Expr.append("*\t");
				geneExpr = getGeneExp(config, gene);
				if (!stackModeloRt.empty()) {
					tmp = stackModeloRt.top();
					stackModeloRt.pop();
					if (!stackModeloRt.empty()) {
						lenIndiv++;
						tmp2 = stackModeloRt.top();
						stackModeloRt.pop();
						tmpExpr.append("(");
						tmpExpr.append(tmp2);
						tmpExpr.append(geneExpr);
            if(tmp.find('-')!= std::string::npos){
              tmpExpr.append("(");
						  tmpExpr.append(tmp);
						  tmpExpr.append("))");  
            }else{
              tmpExpr.append(tmp);
						  tmpExpr.append(")");
            }
            
						stackModeloRt.push(tmpExpr);
					}
					else{
						stackModeloRt.push(tmp);
					}
				}
			}if (gene == -10004) {
				numOps++;
				Expr.append("/\t");
				geneExpr = getGeneExp(config, gene);
				if (!stackModeloRt.empty()) {
					tmp = stackModeloRt.top();
					stackModeloRt.pop();
					if (!stackModeloRt.empty()) {
						tmp2 = stackModeloRt.top();
						stackModeloRt.pop();
						lenIndiv++;

            if (config.protected_division==0) {
              if (tmp!="0") {
                tmpExpr.append(tmp2);
                tmpExpr.append("/");
                tmpExpr.append(tmp); 
              }else {
                tmpExpr.append("protected_division(");
						    tmpExpr.append(tmp2);
						    tmpExpr.append(",");
                tmpExpr.append(tmp);
                tmpExpr.append(")");
              }
            }else {
                tmpExpr.append("protected_division(");
						    tmpExpr.append(tmp2);
						    tmpExpr.append(",");
                tmpExpr.append(tmp);
                tmpExpr.append(")");
            }
						stackModelo.push(tmpExpr);
					}else{		
						stackModeloRt.push(tmp);
					}
				}
			}if (gene == -10005) {
        numOpSin++;
				Expr.append("sin\t");
			}
			if (gene == -10006) {
				numOpCos++;
				Expr.append("cos\t");
			}
			if (gene == -10007) {
				numOpExp++;
				Expr.append("exp\t");
			}
			if (gene == -10008) {
				numOpLog++;
				Expr.append("log\t");
			}
			if (gene == -10009) {
				numOpAbs++;
				Expr.append("abs\t");
			}
			if ((gene == -10010) || (gene == -10011) || (gene == -10012)) {
				numOpIf++;
				Expr.append("if\t");
			}
		}
    rt[i] = numVars + numConst + numOps;
    if (!stackModeloRt.empty()) {
      sizerandomTree.push_back(lenIndiv);
			tmpExpr.append(stackModeloRt.top());
			stackModeloRt.pop();
		}
    if (!tmpExpr.empty()) {
      randomTrees.push_back(tmpExpr);
    }else {
      randomTrees.push_back("1");
    }
    
	}
	return modelLength;
}

void checkPopulations(){
  for(int i =0; i<sample.size();i++){
    if(i==30){
      cout << sample[i] << endl;
      cout << randomTrees[i] << endl;
    }
  }

}

void expre_model(std::string name, std::string path, int pop [],
 int rt [], cfg config, float *hNormalizeData){
  std:: string model="";
  std::string tmpT = name;
  std::string tmpExt = ".csv";
  tmpT = path +"/"+ name + tmpExt; 
  
  std::string nameModel = "_ModelExpression";
  std::string exMo = path +"/"+ name +nameModel  +tmpExt;  
  std::ofstream expre(exMo,ios::out);
  int tmp;
  fstream in(tmpT,ios::in);
  
  if(!in.is_open()) {
    cout<<endl<<"ERROR: FILE MODEL NOT FOUND." << endl;
    exit(-1);
    }else{
      char str[1024];
      while(true){
        in >> str;
        if(strcmp(str,"***")==0){
          break;
        }
        int index1 = atoi(str); 
        in >> str;
        int index2 = atoi(str); 
        in >> str;
        int index3 = atoi(str); 
        in >> str;
        int index4 = atoi(str); 
        in >> str;
        int index5 = atoi(str);
        in >> str;
        double index6 = atof(str); 
        if(index4==1){
          if (index2==-1) {
            if (config.do_min_max==0) {
              model = model.append(sample.at(index3));
              model.append("+");
              model.append("(");
              model.append(std::to_string(index6));
              model.append("*");
              model.append("((");
              model.append(randomTrees.at(index1));
              model.append("-(");
              model.append(std::to_string(hNormalizeData[index1*4]));
              model.append("))");
              model.append("/(");
              model.append(std::to_string(hNormalizeData[index1*4+1]));
              model.append("+0.0001");
              model.append("))");
              model.append("):");
            }else if (config.do_min_max==1) {
              model = model.append(sample.at(index3));
              model.append("+");
              model.append("(");
              model.append(std::to_string(index6));
              model.append(")");
              model.append("*");
              model.append("(2*(");
              model.append(randomTrees.at(index1));
              model.append("-(");
              model.append(std::to_string(hNormalizeData[index1*4]));
              model.append("))");
              model.append("/");
              model.append("(");
              model.append(std::to_string(hNormalizeData[index1*4+3]));
              model.append("-(");
              model.append(std::to_string(hNormalizeData[index1*4+2]));
              model.append("+0.0001");
              model.append("))):");
            }else if (config.do_min_max==2) {
              model = model.append(sample.at(index3));
              model.append("+");
              model.append("(");
              model.append(std::to_string(index6));
              model.append(")");
              model.append("*");
              model.append("(2*((((");
              model.append(randomTrees.at(index1));
              model.append("-");
              model.append(std::to_string(hNormalizeData[index1*4]));
              model.append(")");
              model.append("/");
              model.append("(");
              model.append(std::to_string(hNormalizeData[index1*4+1]));
              model.append("+0.0001");
              model.append("))");
              model.append("-");
              model.append(std::to_string(hNormalizeData[index1*4+3]));
              model.append(")");
              model.append("/");
              model.append("(");
              model.append(std::to_string(hNormalizeData[index1*4+3]));
              model.append("-");
              model.append(std::to_string(hNormalizeData[index1*4+2]));
              model.append("+0.0001");
              model.append("))-1):");
            }
          }else {
            model.append("(");
            model = model.append(sample.at(index3));
            model.append("+");
            model.append("((");
            model.append(std::to_string(index6));
            model.append(")");
            model.append("*");
            model.append("(");
            model.append("((1.0/(1+exp(-1-(");
            model.append(randomTrees.at(index1));
            model.append(")))))");
            model.append("-");
            model.append("((1.0/(1+exp(-1*(");
            model.append(randomTrees.at(index2));
            model.append(")))))))):");
            tmp = pop[index3]+rt[index1]+rt[index2];
          }
          
        }
        if(index4==-1){
          model.append("(");
          model = model.append(sample.at(index3));
          model.append("):");
        }

      }
      while(!in.eof()){

        while(true){
          in >> str;
          if(strcmp(str,"***")==0){
              break;
          }
          int index1 = atoi(str); 
          in >> str;
          int index2 = atoi(str); 
          in >> str;
          int index3 = atoi(str); 
          in >> str;
          int index4 = atoi(str); 
          in >> str;
          int index5 = atoi(str); 
				  in >> str;
				  double index6 = atof(str); 
          if(index4==1){
            if (index2==-1) {
              if (config.do_min_max==0) {
                model.append("+(");
                model = model.append(sample.at(index3));
                model.append(")+");
                model.append("(");
                model.append(std::to_string(index6));
                model.append("*");
                model.append("((");
                model.append(randomTrees.at(index1));
                model.append("-(");
                model.append(std::to_string(hNormalizeData[index1*4]));
                model.append("))");
                model.append("/(");
                model.append(std::to_string(hNormalizeData[index1*4+1]));
                model.append("+0.0001");
                model.append("))");
                model.append("):");
              }else if (config.do_min_max==1) {
                model.append("+(");
                model = model.append(sample.at(index3));
                model.append(")+");
                model.append("(");
                model.append(std::to_string(index6));
                model.append(")");
                model.append("*");
                model.append("(2*(");
                model.append(randomTrees.at(index1));
                model.append("-(");
                model.append(std::to_string(hNormalizeData[index1*4]));
                model.append("))");
                model.append("/");
                model.append("(");
                model.append(std::to_string(hNormalizeData[index1*4+3]));
                model.append("-(");
                model.append(std::to_string(hNormalizeData[index1*4+2]));
                model.append("+0.0001");
                model.append("))):");
              }else if (config.do_min_max==2) {
                model.append("+(");
                model = model.append(sample.at(index3));
                model.append(")+");
                model.append("(");
                model.append(std::to_string(index6));
                model.append(")");
                model.append("*");
                model.append("(2*((((");
                model.append(randomTrees.at(index1));
                model.append("-");
                model.append(std::to_string(hNormalizeData[index1*4]));
                model.append(")");
                model.append("/");
                model.append("(");
                model.append(std::to_string(hNormalizeData[index1*4+1]));
                model.append("+0.0001");
                model.append("))");
                model.append("-");
                model.append(std::to_string(hNormalizeData[index1*4+3]));
                model.append(")");
                model.append("/");
                model.append("(");
                model.append(std::to_string(hNormalizeData[index1*4+3]));
                model.append("-");
                model.append(std::to_string(hNormalizeData[index1*4+2]));
                model.append("+0.0001");
                model.append("))-1):");
              }
            }else{ 
              model.append("+((");
              model = model.append(sample.at(index3));
              model.append(")+");
              model.append("((");
              model.append(std::to_string(index6));
              model.append(")");
              model.append("*");
              model.append("(");
              model.append("((1.0/(1+exp(-1*(");
              model.append(randomTrees.at(index1));
              model.append(")))))");
              model.append("-");
              model.append("((1.0/(1+exp(-1*(");
              model.append(randomTrees.at(index2));
              model.append(")))))))):");
          }
          }
          else if(index4==-1){
            model.append("+(");
            model = model.append(sample.at(index3));
            model.append("):");
          }
        }
    }
    expre<< model.c_str() <<endl;
  }
}

__global__ void computeNormalizeData(float *rtSemantics, float *dataNormalize, int nrow){
  const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
  float mean = 0;
  float std = 0;
  float max=rtSemantics[tid*nrow];
  float min=rtSemantics[tid*nrow];
  for(int i=0; i<nrow; i++){
    mean += (rtSemantics[tid*nrow+i]);

    if(rtSemantics[tid*nrow+i]>max){
      max=rtSemantics[tid*nrow+i];
    }
    if(rtSemantics[tid*nrow+i]<min){
      min=rtSemantics[tid*nrow+i];
    }
  }
  mean = (mean/nrow);
  for(int i=0; i<nrow; i++){
    std += ((rtSemantics[tid*nrow+i]-mean)*(rtSemantics[tid*nrow+i]-mean));
  }
  std = sqrt(std/nrow);
  if (std==0) {
    std = 1;  
  }

  dataNormalize[tid*4]   = mean;
  dataNormalize[tid*4+1] = std; 
  dataNormalize[tid*4+2] = min;
  dataNormalize[tid*4+3] = max;
}

__global__ void computeNormalizeSemantic(float *rtSemantics, float *dataNormalize, int nrow, int min_max, int bandera){
 const unsigned int tid = threadIdx.x+blockIdx.x*blockDim.x;
 float tmp = 0,tmp2=0;
 if (min_max==0) {
    tmp  =dataNormalize[tid*4];
    tmp2 =dataNormalize[tid*4+1];

    if (tmp2==0) {
      tmp2 = 0.0001;
    }
    for(int i=0; i<nrow; i++){
      rtSemantics[tid*nrow+i] = (((rtSemantics[tid*nrow+i]) - tmp)/(tmp2));  
    }
 }else if(min_max==1){
  tmp=dataNormalize[tid*4+3]-dataNormalize[tid*4+2];
  if (tmp==0) {
    tmp=0.0001;
  }
  for(int i=0; i<nrow; i++) {
    rtSemantics[tid*nrow+i] = 2*(((rtSemantics[tid*nrow+i]) - dataNormalize[tid*4+2])/(tmp))-1;    
  }  
 }else if (min_max==2) {
    tmp  =dataNormalize[tid*4];
    tmp2 =dataNormalize[tid*4+1];
    
    if (tmp2==0) {
      tmp2 = 0.0001;
    }
    for(int i=0; i<nrow; i++) {
      rtSemantics[tid*nrow+i] = (((rtSemantics[tid*nrow+i]) - tmp)/(tmp2));  

    }
    if (bandera==1) {
      float max=rtSemantics[tid*nrow];
      float min=rtSemantics[tid*nrow];
      for(int i=0; i<nrow; i++){
        if(rtSemantics[tid*nrow+i]>max){
          max=rtSemantics[tid*nrow+i];
        }
        if(rtSemantics[tid*nrow+i]<min){
          min=rtSemantics[tid*nrow+i];
        }
      }
      dataNormalize[tid*4+2] = min;
      dataNormalize[tid*4+3] = max;
    }
    
    tmp=dataNormalize[tid*4+3]-dataNormalize[tid*4+2];

    if (tmp==0) {
      tmp=0.0001;
    }
    for(int i=0; i<nrow; i++){
      // rtSemantics[tid*nrow+i] = 2*(((rtSemantics[tid*nrow+i]) - dataNormalize[tid*4+2])/(tmp))-1;
      rtSemantics[tid*nrow+i] = 2*(((rtSemantics[tid*nrow+i]) - dataNormalize[tid*4+2])/(tmp))-1;
    }
  }
}

void saveSemantics(std::string path, float *hDataNormalize, int populationSize, std::string outputNameFiles, int nrow, int smNor){
  if (smNor==0) {
    outputNameFiles = outputNameFiles + "_semantic.csv";
    outputNameFiles = path + outputNameFiles;
    std::ofstream outData(outputNameFiles,ios::out);
    for (int i=0; i< populationSize; i++){
      for (int j=0; j<nrow; j++){
        outData<< hDataNormalize[i*nrow+j] << " ";      
      }
    outData<< endl;        
    }
  }else if (smNor==1) {
    outputNameFiles = outputNameFiles + "_semanticNormalize.csv";
    outputNameFiles = path + outputNameFiles;
    std::ofstream outData(outputNameFiles,ios::out);
    for (int i=0; i< populationSize; i++){
      for (int j=0; j<nrow; j++){
        outData<< hDataNormalize[i*nrow+j] << " ";      
      }
    outData<< endl;        
    }
  }else if (smNor==2) {
    outputNameFiles = outputNameFiles + "_semanticPredit.csv";
    outputNameFiles = path + outputNameFiles;
    std::ofstream outData(outputNameFiles,ios::out);
    for (int i=0; i< populationSize; i++){
      for (int j=0; j<nrow; j++){
        outData<< hDataNormalize[i*nrow+j] << " ";      
      }
    outData<< endl;        
    }
  }else if (smNor==3) {
    outputNameFiles = outputNameFiles + "_semanticPREDITNormalize.csv";
    outputNameFiles = path + outputNameFiles;
    std::ofstream outData(outputNameFiles,ios::out);
    for (int i=0; i< populationSize; i++){
      for (int j=0; j<nrow; j++){
        outData<< hDataNormalize[i*nrow+j] << " ";      
      }
    outData<< endl;        
    }
  }
  
}

void saveModelIndividuals(std::string path, std::string outputNameFiles, int i){
  if (i==0) {
    outputNameFiles = outputNameFiles + "_modelsInitialPopulation.csv";
    outputNameFiles = path + outputNameFiles; 
    std::ofstream outfile(outputNameFiles); 
    for (int i = 0; i < sample.size(); i++) {
      outfile << sample[i] << std::endl; // escribir cada elemento del vector en una línea del archivo
    }
  }else {
    outputNameFiles = outputNameFiles + "_modelsRtess.csv";
    outputNameFiles = path + outputNameFiles;
    std::ofstream outfile(outputNameFiles); 
    for (int i = 0; i < randomTrees.size(); i++) {
      outfile << randomTrees[i] << std::endl; // escribir cada elemento del vector en una línea del archivo
    }
  }
}