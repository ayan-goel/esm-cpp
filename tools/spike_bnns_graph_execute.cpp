// Phase 10 T3 linchpin: does C++ BNNSGraphCompileFromFile + ContextExecute on
// the fp16 fc1 mlmodelc hit ~11ms (the CoreML predict number) — proving the
// fp16 AMX win is reachable from the engine without Python? Compares to the new
// SDOT (~20.5ms fc1 M=2048) and cblas-fp32 (~19.6ms).
#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using clk = std::chrono::steady_clock;
static double ms(clk::time_point t){return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now()-t).count()/1e6;}
static std::vector<float> load(const char* p, size_t n){std::vector<float> v(n);FILE* f=fopen(p,"rb");if(!f){std::printf("no %s\n",p);exit(1);}fread(v.data(),4,n,f);fclose(f);return v;}

int main(){
  const int M=2048,K=1280,N=5120;
  auto X=load("/tmp/fc1_X.f32",size_t(M)*K);
  auto Yref=load("/tmp/fc1_Y.f32",size_t(M)*N);

  auto opts=BNNSGraphCompileOptionsMakeDefault();
  bnns_graph_t g=BNNSGraphCompileFromFile("/tmp/fc1_fp16.mlmodelc",nullptr,opts);
  std::printf("BNNSGraphCompileFromFile data=%p size=%zu\n",g.data,g.size);
  if(!g.data){std::printf("FAIL: compile\n");return 1;}

  size_t argc=BNNSGraphGetArgumentCount(g,nullptr);
  size_t xi=BNNSGraphGetArgumentPosition(g,nullptr,"x");
  size_t oi=BNNSGraphGetArgumentPosition(g,nullptr,"out");
  std::printf("argc=%zu  pos(x)=%zu pos(out)=%zu\n",argc,xi,oi);
  if(xi==(size_t)-1||oi==(size_t)-1||argc==(size_t)-1){std::printf("FAIL: arg positions\n");return 1;}

  bnns_graph_context_t ctx=BNNSGraphContextMake(g);
  if(!ctx.data){std::printf("FAIL: context\n");return 1;}
  size_t ws=BNNSGraphContextGetWorkspaceSize(ctx,nullptr);
  std::vector<char> wbuf(ws?ws:1);
  std::printf("workspace=%zu\n",ws);

  std::vector<float> out(size_t(M)*N,0);
  std::vector<bnns_graph_argument_t> args(argc);
  for(auto&a:args){a.data_ptr=nullptr;a.data_ptr_size=0;}
  args[xi].data_ptr=X.data(); args[xi].data_ptr_size=X.size()*4;
  args[oi].data_ptr=out.data(); args[oi].data_ptr_size=out.size()*4;

  int rc=BNNSGraphContextExecute(ctx,nullptr,argc,args.data(),ws,wbuf.data());
  std::printf("Execute rc=%d  out[0..2]=%.4f %.4f %.4f  (ref %.4f %.4f %.4f)\n",
              rc,out[0],out[1],out[2],Yref[0],Yref[1],Yref[2]);
  if(rc!=0){std::printf("FAIL: execute\n");return 1;}
  double maxrel=0; for(int i=0;i<M*N;i+=997){double r=std::abs(out[i]-Yref[i])/(std::abs(Yref[i])+1e-3); if(r>maxrel)maxrel=r;}
  std::printf("max rel err vs CoreML predict ref = %.4f\n",maxrel);

  for(int i=0;i<10;++i) BNNSGraphContextExecute(ctx,nullptr,argc,args.data(),ws,wbuf.data());
  auto t=clk::now(); const int it=50;
  for(int i=0;i<it;++i) BNNSGraphContextExecute(ctx,nullptr,argc,args.data(),ws,wbuf.data());
  std::printf("BNNSGraph fp16 Execute p_avg = %.2f ms  (CoreML predict 11.3, new-SDOT ~20.5, cblas 19.6)\n",ms(t)/it);
  BNNSGraphContextDestroy(ctx);
  return 0;
}
