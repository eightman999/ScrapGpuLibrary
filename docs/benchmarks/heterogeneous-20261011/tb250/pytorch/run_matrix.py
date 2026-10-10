from pathlib import Path
import datetime, json, shlex, subprocess, time

OUT = Path('/private/tmp/scrapgpu-heterogeneous-agents/tb250-torch')
ROOT = '/home/eightman/dev/tools/ScrapGpuLibrary-heterogeneous-20261011'
PYTHON = '/home/eightman/bench/conv-venv/bin/python'
LOCK = '/home/eightman/bench/gpu-bench.lock'
GPU_EXEC = ROOT + '/src/adapters/ggml/tools/kepler-sweep/gpu_exec.sh'
targets = [('cpu', None, None), ('gt730', 'GT 730', 'NVIDIA'), ('gt430', 'GT 430', 'NVIDIA'), ('gt710', 'GT 710', 'NVIDIA'), ('wx2100_rusticl', 'WX 2100 rusticl', 'rusticl'), ('rx6400_rusticl', 'RX 6400 rusticl', 'rusticl')]
results = []
for name, gpu, platform in targets:
    cases = ['ops', 'attention'] + (['backend_config'] if gpu is None else [])
    for case in cases:
        test = ROOT + '/src/adapters/pytorch/tests/test_' + case + '.py'
        env = ['env', 'SCRAPGPU_TORCH_BACKEND=' + ('cpu' if gpu is None else 'opencl11'), 'SCRAPGPU_TORCH_DEVICE=0', 'SCRAPGPU_TORCH_ALLOW_CPU_FALLBACK=' + ('0' if gpu is None else '1'), 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'THERMAL_WAIT_MAX=120']
        if gpu:
            env += ['GGML_CL11_PLATFORMS=' + platform]
            operation = env + ['bash', GPU_EXEC, gpu, PYTHON, '-u', test]
        else:
            operation = env + [PYTHON, '-u', test]
        argv = ['flock', '-w', '300', LOCK, 'timeout', '-k', '10', '180'] + operation
        command = shlex.join(argv)
        log = OUT / (name + '_' + case + '.log')
        stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
        print('START', name, case, stamp, flush=True)
        start = time.monotonic()
        with log.open('w') as handle:
            handle.write('# ' + stamp + '\n# ' + command + '\n')
            handle.flush()
            try:
                process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'tb250_tn', command], stdout=handle, stderr=subprocess.STDOUT, timeout=510)
                rc = process.returncode
            except subprocess.TimeoutExpired:
                rc = 255
                handle.write('\nSSH supervisor exceeded 510 seconds; remote timeout remains active.\n')
        elapsed = round(time.monotonic() - start, 3)
        output = log.read_text()
        status = 'PASS' if rc == 0 and 'SKIP:' not in output else 'SKIP' if rc == 0 else 'FAIL'
        row = {'target': name, 'test': case, 'status': status, 'exit_code': rc, 'elapsed_seconds_including_lock_wait': elapsed, 'log': str(log), 'remote_command': command, 'started_at_jst': stamp}
        results.append(row)
        (OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        print(status, name, case, 'rc=' + str(rc), 'seconds=' + str(elapsed), flush=True)
        if rc:
            print(output[-5000:], flush=True)
print('DONE', sum(r['status'] == 'PASS' for r in results), '/', len(results), flush=True)
