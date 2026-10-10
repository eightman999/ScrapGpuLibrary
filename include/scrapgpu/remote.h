#ifndef SCRAPGPU_REMOTE_H
#define SCRAPGPU_REMOTE_H

#include "scrapgpu/api.h"
#include "scrapgpu/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// The "remote" backend: the devices of a scrapgpu-device-server process show up next to the local
// ones in scrapgpu_get_device. Every backend call is forwarded over a Unix domain socket; bulk data
// (buffer read / write, tensor upload / download) goes through a shared memory region instead of
// the socket. This is how a device that cannot share a process with others (the Fermi GT 430 and
// the Kepler cards of TB250 under the NVIDIA 390 driver) joins a multi-GPU program. See
// docs/MULTI_GPU.md and the transport chapter of docs/API_CONTRACT.md.
typedef struct scrapgpu_remote scrapgpu_remote_t;

// fork + exec scrapgpu-device-server and connect to it. env_spec is a NULL-terminated array of
// "KEY=VALUE" environment overrides applied to the server process (e.g. "GGML_CL11_DEVICES=GT 430");
// everything else is inherited. The backend the server registers comes from SCRAPGPU_REMOTE_BACKEND
// in env_spec or the environment (a comma-separated list, default "opencl11"). The server binary is
// SCRAPGPU_DEVICE_SERVER when set, else "scrapgpu-device-server" next to the running executable,
// else found on PATH. The device(s) appear in scrapgpu_get_device with a "remote:" name prefix.
// Safe to call more than once: every call spawns one server and registers one backend.
SCRAPGPU_API scrapgpu_status scrapgpu_remote_spawn(const char * const * env_spec, scrapgpu_remote_t ** out);

// Connect to the socket of a scrapgpu-device-server that is already running
// (scrapgpu-device-server --socket PATH --backend opencl11). Same backend effect as
// scrapgpu_remote_spawn, without a child process to reap.
SCRAPGPU_API scrapgpu_status scrapgpu_remote_connect(const char * socket_path, scrapgpu_remote_t ** out);

// Disconnect and, for a spawned server, wait for it to exit (it stops when the client disconnects).
// The remote devices disappear from scrapgpu_get_device. Streams, buffers, plans and devices of
// this remote must not be used after close. Safe on NULL.
SCRAPGPU_API void scrapgpu_remote_close(scrapgpu_remote_t * remote);

#ifdef __cplusplus
}
#endif

#endif
