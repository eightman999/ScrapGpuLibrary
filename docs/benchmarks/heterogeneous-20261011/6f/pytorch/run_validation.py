from pathlib import Path
import datetime, json, shlex, subprocess, time

OUT = Path('/private/tmp/scrapgpu-heterogeneous-agents/6f-torch')
ROOT = '/home/eightman/dev/tools/ScrapGpuLibrary-heterogeneous-20261011'
PYTHON = ROOT + '/torch-venv/bin/python'
records = []

def run(name, remote, timeout):
    path = OUT / (name + '.log')
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    print('START', name, stamp, flush=True)
    start = time.monotonic()
    with path.open('w') as handle:
        handle.write('# ' + stamp + '\n# ' + remote + '\n')
        handle.flush()
        try:
            p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '6f', remote], stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 255
            handle.write('\nSSH supervisor timeout; remote command has its own timeout.\n')
    elapsed = round(time.monotonic()-start, 3)
    data = path.read_text()
    status = 'PASS' if rc == 0 and 'SKIP:' not in data else 'SKIP' if rc == 0 else 'FAIL'
    records.append({'stage':name, 'status':status,'exit_code':rc,'elapsed_seconds':elapsed,'log':str(path),'remote_command':remote,'started_at_jst':stamp})
    (OUT/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(status,name,'rc='+str(rc),'seconds='+str(elapsed),flush=True)
    print(data[-5500:],flush=True)
    return rc == 0

install = ' && '.join([
    shlex.join(['python3', '-m', 'venv', ROOT+'/torch-venv']),
    shlex.join(['timeout','-k','10','600',PYTHON,'-m','pip','--isolated','install','--disable-pip-version-check','--no-input','--retries','0','--cache-dir',ROOT+'/pip-cache','--index-url','https://download.pytorch.org/whl/cpu','torch==2.14.1']),
    shlex.join([PYTHON,'-c','import torch,sys; print("torch", torch.__version__, "python", sys.version)']),
])
if not run('install',install,650):
    raise SystemExit('Installation failed; not retrying as requested.')
configure = shlex.join(['cmake','-S',ROOT+'/src','-B',ROOT+'/build-pytorch','-DCMAKE_BUILD_TYPE=Release','-DSCRAPGPU_BUILD_CORE=ON','-DSCRAPGPU_BUILD_OPENCL11=OFF','-DSCRAPGPU_BUILD_PYTORCH=ON','-DSCRAPGPU_BUILD_TESTS=OFF','-DSCRAPGPU_BUILD_EXAMPLES=OFF','-DPython3_EXECUTABLE='+PYTHON])
build = shlex.join(['timeout','-k','10','600','cmake','--build',ROOT+'/build-pytorch','--target','_C','-j','2'])
if not run('build',configure+' && '+build,660):
    raise SystemExit('CPU-only PyTorch build failed.')
for test in ['tests/test_ops.py','tests/test_attention.py','tests/test_backend_config.py','examples/mlp.py']:
    name=Path(test).stem
    command=shlex.join(['timeout','-k','10','180','env','PYTHONDONTWRITEBYTECODE=1','SCRAPGPU_TORCH_BACKEND=cpu','SCRAPGPU_TORCH_DEVICE=0','SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK=0','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',PYTHON,'-u',ROOT+'/src/adapters/pytorch/'+test])
    run(name,command,200)
print('DONE',sum(row['status']=='PASS' for row in records),'/',len(records),flush=True)
