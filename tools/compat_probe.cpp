// Public-API smoke/latency probe. Use one GPU per process on NVIDIA 390.
#include "scrapgpu/cpu.h"
#include "scrapgpu/compat.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/reference.h"
#if defined(SCRAPGPU_PROBE_OPENCL11)
#include "scrapgpu/opencl11.h"
#endif
#if defined(SCRAPGPU_PROBE_VULKAN)
#include "scrapgpu/vulkan.h"
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(scrapgpu_status s) {
    if (s != SCRAPGPU_STATUS_OK) {
        std::fprintf(stderr, "%s: %s\n", scrapgpu_status_string(s), scrapgpu_last_error()); std::exit(1);
    }
}
static scrapgpu_tensor_desc tensor(scrapgpu_dtype dt, int64_t k, int64_t m, scrapgpu_buffer_t * b) {
    scrapgpu_tensor_desc t{}; t.struct_size=sizeof(t); t.dtype=dt;
    t.ne[0]=k; t.ne[1]=m; t.ne[2]=t.ne[3]=1;
    t.nb[0]=scrapgpu_dtype_type_size(dt); t.nb[1]=scrapgpu_tensor_row_size(dt,k);
    t.nb[2]=t.nb[3]=t.nb[1]*m; t.buffer=b; return t;
}
int main(int argc, char ** argv) {
    const char * name = argc > 1 ? argv[1] : "cpu";
    if (!std::strcmp(name,"cpu")) check(scrapgpu_cpu_register());
#if defined(SCRAPGPU_PROBE_OPENCL11)
    else if (!std::strcmp(name,"opencl11")) check(scrapgpu_opencl11_register());
#endif
#if defined(SCRAPGPU_PROBE_VULKAN)
    else if (!std::strcmp(name,"vulkan")) check(scrapgpu_vulkan_register());
#endif
    else { std::fprintf(stderr,"backend not built: %s\n",name); return 2; }
    if (!scrapgpu_get_device_count()) { std::puts("SKIP: no device"); return 77; }
    scrapgpu_device_t * dev=nullptr; scrapgpu_stream_t * stream=nullptr;
    check(scrapgpu_get_device(0,&dev)); check(scrapgpu_stream_create(dev,&stream));
    scrapgpu_device_info info{}; check(scrapgpu_device_info_init(&info,sizeof(info))); check(scrapgpu_device_get_info(dev,&info));
    std::printf("device=%s backend=%s driver=%s used_api=%s tuning=%s\n", info.name,
        info.backend_name, info.driver_version ? info.driver_version : "", info.used_api_version ? info.used_api_version : "",
        info.tuning_profile_name ? info.tuning_profile_name : "");
    constexpr int K=256, M=64, N=16;
    std::vector<float> w(K*M), x(K*N), ref(M*N), out(M*N);
    for (size_t i=0;i<w.size();++i) w[i]=(static_cast<int>(i%13)-6)*0.125f;
    for (size_t i=0;i<x.size();++i) x[i]=(static_cast<int>(i%17)-8)*0.125f;
    for (int n=0;n<N;++n) for (int m=0;m<M;++m) {
        double sum=0; for (int k=0;k<K;++k) sum+=double(w[m*K+k])*x[n*K+k]; ref[n*M+m]=float(sum);
    }
    std::vector<uint16_t> bf(w.size());
    for (size_t i=0;i<w.size();++i) { uint32_t b; std::memcpy(&b,&w[i],4); bf[i]=uint16_t(b>>16); }
    for (auto dtype : {SCRAPGPU_DTYPE_F32,SCRAPGPU_DTYPE_BF16}) {
        scrapgpu_buffer_t * bw=nullptr,* bx=nullptr,* by=nullptr;
        const size_t wbytes=w.size()*scrapgpu_dtype_type_size(dtype);
        check(scrapgpu_buffer_alloc(dev,wbytes,&bw)); check(scrapgpu_buffer_alloc(dev,x.size()*4,&bx)); check(scrapgpu_buffer_alloc(dev,out.size()*4,&by));
        scrapgpu_tensor_desc ins[]={tensor(dtype,K,M,bw),tensor(SCRAPGPU_DTYPE_F32,K,N,bx)};
        auto dst=tensor(SCRAPGPU_DTYPE_F32,M,N,by);
        check(scrapgpu_tensor_upload(stream,&ins[0],dtype==SCRAPGPU_DTYPE_F32 ? static_cast<void *>(w.data()) : bf.data(),wbytes));
        check(scrapgpu_tensor_upload(stream,&ins[1],x.data(),x.size()*4));
        scrapgpu_mul_mat_params params={sizeof(params),1};
        scrapgpu_op_desc op={sizeof(op),SCRAPGPU_OP_MUL_MAT,ins,2,&dst,1,&params,sizeof(params)};
        scrapgpu_compat_options options={sizeof(options),1,8*1024*1024};
        scrapgpu_execution_report report{}; report.struct_size=sizeof(report);
        check(scrapgpu_op_execute_compatible(dev,stream,&op,&options,&report)); check(scrapgpu_stream_sync(stream));
        const auto start=std::chrono::steady_clock::now();
        for (int r=0;r<5;++r) { check(scrapgpu_op_execute_compatible(dev,stream,&op,&options,&report)); check(scrapgpu_stream_sync(stream)); }
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/5;
        check(scrapgpu_tensor_download(stream,&dst,out.data(),out.size()*4));
        double error=0, norm=0; for (size_t i=0;i<out.size();++i) { error+=(out[i]-ref[i])*(out[i]-ref[i]); norm+=ref[i]*ref[i]; }
        const double nmse=error/(norm+1e-30);
        std::printf("dtype=%s path=%s ms=%.6f staging=%llu download=%llu upload=%llu nmse=%.9g %s\n",
            dtype==SCRAPGPU_DTYPE_F32 ? "F32":"BF16", report.path==SCRAPGPU_EXECUTION_NATIVE ? "native":"cpu_fallback",
            ms,(unsigned long long)report.staging_bytes,(unsigned long long)report.downloaded_bytes,(unsigned long long)report.uploaded_bytes,
            nmse,std::isfinite(nmse)&&nmse<1e-6 ? "PASS":"FAIL");
        if (!std::isfinite(nmse)||nmse>=1e-6) return 1;
        check(scrapgpu_buffer_free(bw)); check(scrapgpu_buffer_free(bx)); check(scrapgpu_buffer_free(by));
    }
    check(scrapgpu_stream_free(stream));
    return 0;
}
