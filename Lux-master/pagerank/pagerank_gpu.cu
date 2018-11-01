/* Copyright 2018 Stanford, UT Austin, LANL
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

#include "../core/graph.h" 
#include "../core/cuda_helper.h"
#include "realm/runtime_impl.h"
#include "realm/cuda/cuda_module.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 512;
const int BLOCK_SIZE_LIMIT = 32768;
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}
/*  global
void check_queue()
{

}*/
__global__
void load_kernel(V_ID my_in_vtxs,
                 const V_ID* in_vtxs,
                 Vertex* old_pr_fb,
                 const Vertex* old_pr_zc,V_ID * d_currentQueue,int totalNum)
{
  d_currentQueue[0]=0;
//  d_parent[0]=INT_MAX;
  for (V_ID i = blockIdx.x * blockDim.x + threadIdx.x; i < my_in_vtxs;
       i+= blockDim.x * gridDim.x)
  {
    V_ID vtx = in_vtxs[i];
    Vertex my_pr = old_pr_zc[vtx];
    cub::ThreadStore<cub::STORE_CG>(old_pr_fb + vtx, my_pr);
  }
}
                                        
__global__
void pr_kernel(V_ID rowLeft,
               V_ID rowRight,
               E_ID colLeft,
               float initRank,
               const NodeStruct* row_ptrs,
               const EdgeStruct* col_idxs,
               Vertex* old_pr_fb,
               Vertex* new_pr_fb,int level,V_ID *distance, V_ID *d_parent,int *  queueSize,int * nextQueueSize, V_ID *d_currentQueue, V_ID *d_nextQueue)
{
  typedef cub::BlockScan<E_ID, CUDA_NUM_THREADS> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  //__shared__ bool visited[]
  //__shared__ float pr[CUDA_NUM_THREADS];
  __shared__ E_ID blkColStart;
    V_ID curVtx = blockIdx.x*blockDim.x+threadIdx.x;
    if(curVtx <*queueSize) 
    {
      //int valueChange=0;
 //     printf("curVtx:%d\n",curVtx);
      int u=d_currentQueue[curVtx];
     /* if(level==0)
          printf("u:%d\n",u);*/
//      printf("row_ptrs[u].index:%d\n",row_ptrs[u].index);
      {
          //int cc=0;
          int i=0;
          if(u==0)
            i=0;
          else
            i=row_ptrs[u-1].index;
          for (; i < row_ptrs[u].index; i++) 
          {  
          
              EdgeStruct es = col_idxs[i];
              int v=es.dst;
             // printf("v:%d\n",v);
             /*if(level==0){
                  printf("level%d u:%d distance[%d]:%d\n",level,u,v,distance[v]);
             }*/
//              printf("atomicMin(&distance[v], level + 1) == INT_MAX: %d\n",atomicMin(&distance[v], level + 1) == INT_MAX);
              if(distance[v] == INT_MAX && atomicMin(&distance[v], level + 1) == INT_MAX)
             {
                 /* if(level==0)
                      printf("v:%d come in\n",v);*/
                  //d_parent[v]=u;
                  /*if(level==0)
                      printf("after come in d_parent[%d]:%d\n",v,d_parent[v]);*/
                  int position=atomicAdd(nextQueueSize,1);
                  d_nextQueue[position]=v;
              }
             
          }
      }
    }
    __syncthreads();
}
//__device__ __managed_ int  nextQueueSize;
//device__ __managed__ int nextQueueSize;
/*static*/
void pull_app_task_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 9);//加了一个region
  assert(task->regions.size() == 9);//加了一个region
  const GraphPiece *piece = (GraphPiece*) task->local_args;

  const AccessorRO<NodeStruct, 1> acc_row_ptr(regions[0], FID_DATA);
  const AccessorRO<V_ID, 1> acc_in_vtx(regions[1], FID_DATA);
  const AccessorRO<EdgeStruct, 1> acc_col_idx(regions[2], FID_DATA);
  const AccessorRO<Vertex, 1> acc_old_pr(regions[3], FID_DATA);
  const AccessorWO<Vertex, 1> acc_new_pr(regions[4], FID_DATA);
  const AccessorRW<V_ID, 1> acc_input_lp(regions[5],FID_DATA);
  const AccessorRW<V_ID, 1> acc_d_distance_lp(regions[6],FID_DATA);
  const AccessorWO<V_ID, 1> acc_d_currentQueue_lp(regions[7],FID_DATA);
  const AccessorWO<V_ID, 1> acc_d_nextQueue_lp(regions[8],FID_DATA);

  Rect<1> rect_row_ptr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_in_vtx = runtime->get_index_space_domain(
                            ctx, task->regions[1].region.get_index_space());
  Rect<1> rect_col_idx = runtime->get_index_space_domain(
                             ctx, task->regions[2].region.get_index_space());
  Rect<1> rect_old_pr = runtime->get_index_space_domain(
                            ctx, task->regions[3].region.get_index_space());
  Rect<1> rect_new_pr = runtime->get_index_space_domain(
                            ctx, task->regions[4].region.get_index_space());
  Rect<1> rect_input_lp=runtime->get_index_space_domain(
                            ctx,task->regions[5].region.get_index_space());
  Rect<1> rect_d_distance_lp=runtime->get_index_space_domain(
                            ctx,task->regions[6].region.get_index_space());
  Rect<1> rect_d_currentQueue_lp=runtime->get_index_space_domain(
                            ctx,task->regions[7].region.get_index_space());
  Rect<1> rect_d_nextQueue_lp=runtime->get_index_space_domain(
                            ctx,task->regions[8].region.get_index_space());
  assert(acc_row_ptr.accessor.is_dense_arbitrary(rect_row_ptr));
  assert(acc_in_vtx.accessor.is_dense_arbitrary(rect_in_vtx));
  assert(acc_col_idx.accessor.is_dense_arbitrary(rect_col_idx));
  assert(acc_old_pr.accessor.is_dense_arbitrary(rect_old_pr));
  assert(acc_new_pr.accessor.is_dense_arbitrary(rect_new_pr));
  assert(acc_input_lp.accessor.is_dense_arbitrary(rect_input_lp));
  assert(acc_d_distance_lp.accessor.is_dense_arbitrary(rect_d_distance_lp));
  assert(acc_d_currentQueue_lp.accessor.is_dense_arbitrary(rect_d_currentQueue_lp));
  assert(acc_d_nextQueue_lp.accessor.is_dense_arbitrary(rect_d_nextQueue_lp));
  V_ID * input_lp_ptr=acc_input_lp.ptr(rect_input_lp);
  V_ID * d_distance_ptr=acc_d_distance_lp.ptr(rect_d_distance_lp);
  V_ID * d_currentQueue_ptr=acc_d_currentQueue_lp.ptr(rect_d_currentQueue_lp);
  V_ID * d_nextQueue_ptr=acc_d_nextQueue_lp.ptr(rect_d_nextQueue_lp);
  const NodeStruct* row_ptrs = acc_row_ptr.ptr(rect_row_ptr);
  const V_ID* in_vtxs = acc_in_vtx.ptr(rect_in_vtx);
  const EdgeStruct* col_idxs = acc_col_idx.ptr(rect_col_idx);
  const Vertex* old_pr = acc_old_pr.ptr(rect_old_pr);
  Vertex* new_pr = acc_new_pr.ptr(rect_new_pr);
  V_ID rowLeft = rect_row_ptr.lo[0], rowRight = rect_row_ptr.hi[0];
  E_ID colLeft = rect_col_idx.lo[0], colRight = rect_col_idx.hi[0];
 
  // int *queueNow;
  // cudaMalloc((void**)&queueNow, 10000*sizeof(int));
  // int *queueNext;
  // cudaMalloc((void**)&queueNext, 10000*sizeof(int));
  // const int n = 32;
  // const size_t sz = size_t(n) * sizeof(int);
  // int *dJunk;
  // cudaMalloc((void**)&dJunk, sz);
  // cudaMemset(dJunk, 0, sz/4);
  // cudaMemset(dJunk, 0x12,  8);

  // int *Junk = new int[n];

  // cudaMemcpy(Junk, dJunk, sz, cudaMemcpyDeviceToHost);

  // for(int i=0; i<n; i++) {
  //     printf("%d %x\n", i, Junk[i]);
  // }

  /*int *distance;//利用d_distance_ptr取代distance
  cudaMalloc((void**)&distance, (piece->nv)*sizeof(int));
  cudaMemset(distance, std::numeric_limits<int>::max(),(piece->nv));
  cudaMemset(distance,0,1);*/
  int level=0;
  //printf("nextQueueSize:%d\n",nextQueueSize);
 
    
   // int * d_currentQueue,* d_nextQueue;  //分别使用d_currentQueue_pt和d_nextQueue_ptr来取代。
   //cudaMalloc((void * *)&d_parent,(piece->nv)* sizeof(int));//利用input_ptr取代d_parent
//    cudaMalloc((void * *)&d_currentQueue, (piece->nv)* sizeof(int));
//    cudaMalloc((void * *)&d_nextQueue, (piece->nv)* sizeof(int));
    //cudaMemset(d_distance,2147483647,piece->nv);
    //cudaMemset(d_distance,0,1);
    //cudaMemset(d_parent,std::numeric_limits<int>::max(),piece->nv);
    //cudaMemset(d_parent,0,1);
//    int firstElementQueue=0;
//    cudaMemcpy(d_currentQueue,&firstElementQueue,sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemset(d_nextQueue,0,piece->nv);
    load_kernel<<<GET_BLOCKS(piece->myInVtxs), CUDA_NUM_THREADS>>>(
        piece->myInVtxs, in_vtxs, piece->oldPrFb, old_pr,d_currentQueue_ptr,rowRight);
  /*{
    printf("before bfs: ");
    int *vc=new int[piece->nv];
    cudaMemcpy(vc,input_lp_ptr, piece->nv*sizeof(int), cudaMemcpyDeviceToHost);
//   printf("input_lp_ptr[0]:%i\n",*input_lp_ptr);
    int count=0;
    for(int i=0; i<piece->nv; i++) {
        if(vc[i]==std::numeric_limits<int>::max())
              count++;
    }
    printf("%d\n",count);
  }*/
  //load_kernel<<<GET_BLOCKS(piece->myInVtxs), CUDA_NUM_THREADS>>>(
  //  piece->myInVtxs, in_vtxs, piece->oldPrFb, old_pr,input_lp_ptr);    
  int * queueSize;
  cudaMallocManaged(&queueSize,sizeof(int));
  *queueSize=1;
 // cuMemAllocHost((void * *)&queueSize,sizeof(int));
  *queueSize=1;
  /*int * checkQueue;
  cuMemAllocHost((void * *)&checkQueue,sizeof(int)*(rowRight-rowLeft));    */


  int * nextQueueSize;
//  cudaMalloc((void * *)&nextQueueSize,sizeof(int));
  cudaMallocManaged(&nextQueueSize,sizeof(int));
//  cudaMemset(nextQueueSize,0,1);
  *nextQueueSize=0;
//  printf("row_right-row_left:%d\n",rowRight-rowLeft);
  while(*queueSize>0)
  {
    pr_kernel<<<GET_BLOCKS(*queueSize),CUDA_NUM_THREADS>>>(
        rowLeft, rowRight, colLeft, (1 - ALPHA) / piece->nv,
        row_ptrs, col_idxs, piece->oldPrFb, piece->newPrFb,level,d_distance_ptr, input_lp_ptr,queueSize,nextQueueSize,d_currentQueue_ptr,d_nextQueue_ptr);
    cudaDeviceSynchronize();
    level=level+1;   
    *queueSize=*nextQueueSize; 
//    cudaMemcpy(queueSize,nextQueueSize,sizeof(int),cudaMemcpyDeviceToHost);
     printf("level:%d queueSize:%d\n",level,*queueSize);
  //  cudaMemset(nextQueueSize,0,1); 
    *nextQueueSize=0;
    std::swap(d_currentQueue_ptr, d_nextQueue_ptr);
  }
 /* {
    int * val=new int[piece->nv];
    cudaMemcpy(val,input_lp_ptr,sizeof(int)*(piece->nv),cudaMemcpyDeviceToHost);
    int count=0;
    for(int i=0;i<piece->nv;i++)
    {
        if(val[i]==1)
            count++;
    } 
    printf("count:%d\n",count);
  }*/
}

__global__
void init_kernel(V_ID rowLeft,
                 V_ID rowRight,
                 E_ID colLeft,
                 NodeStruct* row_ptrs,
                 EdgeStruct* col_idxs,
                 const E_ID* raw_rows,
                 const V_ID* degrees,
                 const V_ID* raw_cols)
{
  for (V_ID n = blockIdx.x * blockDim.x + threadIdx.x;
       n + rowLeft <= rowRight; n += blockDim.x * gridDim.x)
  {
    E_ID startColIdx, endColIdx = raw_rows[n];
    if (n == 0)
      startColIdx = colLeft;
    else
      startColIdx = raw_rows[n - 1];
    row_ptrs[n].index = endColIdx;
    if (degrees != NULL)
      row_ptrs[n].degree = degrees[n];
    for (E_ID e = startColIdx; e < endColIdx; e++)
    {
      col_idxs[e - colLeft].dst = raw_cols[e - colLeft];
      col_idxs[e - colLeft].src = n + rowLeft;
    }
  }
}

GraphPiece pull_init_task_impl(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime)
{
#ifndef VERTEX_DEGREE
  assert(false);
#endif
#ifdef EDGE_WEIGHT
  assert(false);
#endif
  assert(regions.size() == 7);
  assert(task->regions.size() == 7);
  const Graph *graph = (Graph*) task->args;
  const AccessorWO<NodeStruct, 1> acc_row_ptr(regions[0], FID_DATA);
  const AccessorWO<V_ID, 1> acc_in_vtx(regions[1], FID_DATA);
  const AccessorWO<EdgeStruct, 1> acc_col_idx(regions[2], FID_DATA);
  const AccessorWO<Vertex, 1> acc_new_pr(regions[3], FID_DATA);
  const AccessorRO<E_ID, 1> acc_raw_rows(regions[4], FID_DATA);
  const AccessorRO<V_ID, 1> acc_raw_cols(regions[5], FID_DATA);

  Rect<1> rect_row_ptr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_in_vtx = runtime->get_index_space_domain(
                            ctx, task->regions[1].region.get_index_space());
  Rect<1> rect_col_idx = runtime->get_index_space_domain(
                             ctx, task->regions[2].region.get_index_space());
  Rect<1> rect_new_pr = runtime->get_index_space_domain(
                            ctx, task->regions[3].region.get_index_space());
  Rect<1> rect_raw_rows = runtime->get_index_space_domain(
                              ctx, task->regions[4].region.get_index_space());
  Rect<1> rect_raw_cols = runtime->get_index_space_domain(
                              ctx, task->regions[5].region.get_index_space());

  assert(acc_row_ptr.accessor.is_dense_arbitrary(rect_row_ptr));
  assert(acc_in_vtx.accessor.is_dense_arbitrary(rect_in_vtx));
  assert(acc_col_idx.accessor.is_dense_arbitrary(rect_col_idx));
  assert(acc_new_pr.accessor.is_dense_arbitrary(rect_new_pr));
  assert(acc_raw_rows.accessor.is_dense_arbitrary(rect_raw_rows));
  assert(acc_raw_cols.accessor.is_dense_arbitrary(rect_raw_cols));
  NodeStruct* row_ptrs = acc_row_ptr.ptr(rect_row_ptr);
  V_ID* in_vtxs = acc_in_vtx.ptr(rect_in_vtx);
  EdgeStruct* col_idxs = acc_col_idx.ptr(rect_col_idx);
  Vertex* new_pr = acc_new_pr.ptr(rect_new_pr);
  const E_ID* raw_rows = acc_raw_rows.ptr(rect_raw_rows);
  const V_ID* raw_cols = acc_raw_cols.ptr(rect_raw_cols);
  V_ID rowLeft = rect_row_ptr.lo[0], rowRight = rect_row_ptr.hi[0];
  E_ID colLeft = rect_col_idx.lo[0], colRight = rect_col_idx.hi[0];
  std::vector<V_ID> edges(colRight - colLeft + 1);
  for (E_ID e = 0; e < colRight - colLeft + 1; e++)
    edges[e] = raw_cols[e];
  std::sort(edges.begin(), edges.end());
  V_ID curVtx = edges[0], myInVtx = 0;
  for (E_ID e = 0; e < colRight - colLeft + 1; e++) {
    if (curVtx != edges[e]) {
      edges[myInVtx++] = curVtx;
      curVtx = edges[e];
    }
  }
  edges[myInVtx++] = curVtx;
  checkCUDA(cudaMemcpy(in_vtxs, edges.data(), sizeof(V_ID) * myInVtx,
                       cudaMemcpyHostToDevice));
  // Add degree if regions.size() == 7
  const V_ID *degrees = NULL;
  if (regions.size() == 7) {
    const AccessorRO<V_ID, 1> acc_degrees(regions[6], FID_DATA);
    Rect<1> rect_degrees = runtime->get_index_space_domain(
                               ctx, task->regions[6].region.get_index_space());
    assert(acc_degrees.accessor.is_dense_arbitrary(rect_degrees));
    degrees = acc_degrees.ptr(rect_degrees.lo);
  }
  init_kernel<<<GET_BLOCKS(rowRight - rowLeft + 1), CUDA_NUM_THREADS>>>(
      rowLeft, rowRight, colLeft, row_ptrs, col_idxs, raw_rows, degrees, raw_cols);
  checkCUDA(cudaDeviceSynchronize());
  float rank = 1.0f / graph->nv;
  assert(sizeof(float) == sizeof(Vertex));
  for (V_ID n = 0; n + rowLeft <= rowRight; n++) {
    new_pr[n] = degrees[n] == 0 ? rank : rank / degrees[n];
  }
  GraphPiece piece;
  piece.myInVtxs = myInVtx;
  piece.nv = graph->nv;
  piece.ne = graph->ne;
  // Allocate oldPrFb/newPrFb on the same memory as row_ptr
  std::set<Memory> memFB;
  regions[0].get_memories(memFB);
  assert(memFB.size() == 1);
  assert(memFB.begin()->kind() == Memory::GPU_FB_MEM);
  Realm::MemoryImpl* memImpl =
      Realm::get_runtime()->get_memory_impl(*memFB.begin());
  Realm::Cuda::GPUFBMemory* memFBImpl = (Realm::Cuda::GPUFBMemory*) memImpl;
  off_t offset = memFBImpl->alloc_bytes(sizeof(Vertex) * graph->nv);
  assert(offset >= 0);
  piece.oldPrFb = (Vertex*) memFBImpl->get_direct_ptr(offset, 0);
  offset = memFBImpl->alloc_bytes(sizeof(Vertex) * (rowRight - rowLeft + 1));
  assert(offset >= 0);
  piece.newPrFb = (Vertex*) memFBImpl->get_direct_ptr(offset, 0);
  //checkCUDA(cudaMalloc(&(piece.oldPrFb), sizeof(float) * graph->nv));
  //checkCUDA(cudaMalloc(&(piece.newPrFb), sizeof(float) * (rowRight-rowLeft+1)));
  return piece;
}

