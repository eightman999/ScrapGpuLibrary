from pathlib import Path
import datetime, hashlib, json, os, shlex, subprocess, time
ROOT=Path('/Users/eightman/dev/tools/ScrapGpuLibrary-heterogeneous')
OUT=Path('/private/tmp/scrapgpu-heterogeneous-agents/mac-torch')
PYTHON='/private/tmp/scrapgpu-pytorch-venv/bin/python'
results=[]
paths=[ROOT/'CMakeLists.txt', ROOT/'src/core/attention.cpp', ROOT/'src/core/compat.cpp', ROOT/'src/core/ref/ops.cpp', ROOT/'src/backends/cpu/api.cpp']
paths.extend(p for p in (ROOT/'adapters/pytorch').rglob('*') if p.is_file() and (p.suffix in ('.cpp','.py','.md') or p.name=='CMakeLists.txt'))
(OUT/'source_sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')

def run(name,argv,extra=None):
    env=dict(os.environ)
    env.update(extra or {})
    env.update({'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    path=OUT/(name+'.log')
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    print('START',name,stamp,flush=True)
    start=time.monotonic()
    with path.open('w') as handle:
        handle.write('# '+stamp+'\n# '+shlex.join(argv)+'\n# environment overrides: '+json.dumps(extra or {})+'\n')
        handle.flush()
        try:
            p=subprocess.run(argv,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=240)
            rc=p.returncode
        except subprocess.TimeoutExpired:
            rc=124
    data=path.read_text()
    status='PASS' if rc==0 and 'SKIP:' not in data else 'SKIP' if rc==0 else 'FAIL'
    results.append({'stage':name,'status':status,'exit_code':rc,'elapsed_seconds':round(time.monotonic()-start,3),'log':str(path),'started_at_jst':stamp})
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(status,name,'rc='+str(rc),flush=True)
    if rc: print(data[-6000:],flush=True)
    return rc==0

cpu={'SCRAPGPU_TORCH_BACKEND':'cpu','SCRAPGPU_TORCH_DEVICE':'0','SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK':'0'}
gpu={'SCRAPGPU_TORCH_BACKEND':'opencl11','SCRAPGPU_TORCH_DEVICE':'0','SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK':'1'}
if not run('build_cpu',['cmake','--build','build-pytorch-cpu','--target','_C','-j','4']): raise SystemExit(1)
for test in ['test_ops','test_attention','test_backend_config']:
    run('cpu_'+test,[PYTHON,'-u','adapters/pytorch/tests/'+test+'.py'],cpu)
run('cpu_mlp',[PYTHON,'-u','adapters/pytorch/examples/mlp.py'],cpu)
if not run('build_opencl',['cmake','--build','build-pytorch-opencl','--target','_C','-j','4']): raise SystemExit(1)
for test in ['test_ops','test_attention']:
    run('opencl_'+test,[PYTHON,'-u','adapters/pytorch/tests/'+test+'.py'],gpu)
code='''import sys
sys.path.insert(0, 'adapters/pytorch')
import torch, scrapgpu_torch as sgt
sgt.load()
x=torch.ones(2,8)
f32=torch.classes.scrapgpu.Weight(torch.ones(3,8),'f32')
assert torch.equal(torch.ops.scrapgpu.linear(x,f32),torch.full((2,3),8.0))
bf16=torch.classes.scrapgpu.Weight(torch.ones(3,8),'bf16')
try:
    torch.ops.scrapgpu.linear(x,bf16)
except RuntimeError as exc:
    assert EXPECTED in str(exc),str(exc)
else:
    raise AssertionError('BF16 fallback unexpectedly accepted')
print('PASS: F32 native executes; BF16 correctly rejected:',EXPECTED)
'''
for fallback,budget,expected,name in [('0','268435456','UNSUPPORTED','fallback_disabled'),('1','0','staging','fallback_zero_budget')]:
    setting=dict(gpu,SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK=fallback,SCRAPGPU_TORCH_MAX_STAGING_BYTES=budget)
    run('opencl_'+name,[PYTHON,'-u','-c','EXPECTED='+repr(expected)+'\n'+code],setting)
print('DONE',sum(r['status']=='PASS' for r in results),'/',len(results),flush=True)
