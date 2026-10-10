// Remote protocol regression tests (audit H-2): a real scrapgpu-device-server is spawned on a
// throwaway Unix socket and driven with raw wire messages (protocol.h). The payload a client may
// make the server allocate is bounded by sgr::kMaxPayload: an oversized header must produce an
// error response and a clean disconnect, not a multi-GB allocation or an abort. The shared-memory
// growth a client may request is bounded by sgr::kMaxShm; its refusal must leave the connection
// usable. GPU-free: the server starts with 0 devices fine and no backend call is exercised.

#include "protocol.h"

#include "scrapgpu/op.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n",        \
                         __FILE__, __LINE__, #cond);                 \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

#ifndef SCRAPGPU_TEST_SERVER
#define SCRAPGPU_TEST_SERVER "scrapgpu-device-server"
#endif

#ifndef SCRAPGPU_TEST_SERVER_BACKEND
#define SCRAPGPU_TEST_SERVER_BACKEND "opencl11"
#endif

bool write_all(int fd, const void * p, size_t n) {
    const uint8_t * q = static_cast<const uint8_t *>(p);
    while (n > 0) {
#ifdef MSG_NOSIGNAL
        const ssize_t w = send(fd, q, n, MSG_NOSIGNAL);
#else
        const ssize_t w = write(fd, q, n);
#endif
        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        q += w;
        n -= (size_t) w;
    }
    return true;
}

// 0 on a clean EOF, -1 on an error
ssize_t read_all(int fd, void * p, size_t n) {
    uint8_t * q = static_cast<uint8_t *>(p);
    size_t left = n;
    while (left > 0) {
        const ssize_t r = read(fd, q, left);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (r == 0) {
            return left == n ? 0 : -1; // EOF with nothing read vs a short read
        }
        q += r;
        left -= (size_t) r;
    }
    return 1;
}

// sends a request (header + payload) and parses the response; the storage keeps the payload bytes
bool invoke(int fd, uint32_t op, const std::vector<uint8_t> & req, uint32_t & status,
            std::vector<uint8_t> & storage) {
    std::vector<uint8_t> hdr;
    sgr::put_u32(hdr, sgr::kMagic);
    sgr::put_u32(hdr, op);
    sgr::put_u64(hdr, req.size());
    if (!write_all(fd, hdr.data(), hdr.size()) ||
        (!req.empty() && !write_all(fd, req.data(), req.size()))) {
        return false;
    }
    uint8_t rh[sgr::kHdrSize];
    if (read_all(fd, rh, sizeof(rh)) != 1) {
        return false;
    }
    sgr::cur hc{rh, sgr::kHdrSize};
    uint32_t magic = 0;
    uint64_t size = 0;
    if (!sgr::get_u32(hc, magic) || magic != sgr::kMagic || !sgr::get_u32(hc, status) ||
        !sgr::get_u64(hc, size) || size > sgr::kMaxPayload + 64 * 1024) {
        return false;
    }
    storage.resize((size_t) size);
    return size == 0 || read_all(fd, storage.data(), (size_t) size) == 1;
}

int connect_retry(const std::string & path, pid_t child) {
    for (int i = 0; i < 1000; ++i) {
        int wstatus = 0;
        if (waitpid(child, &wstatus, WNOHANG) == child) {
            return -1; // the server died before the socket came up
        }
        const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd >= 0) {
            sockaddr_un addr = {};
            addr.sun_family = AF_UNIX;
            std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
            if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
                return fd;
            }
            close(fd);
        }
        usleep(10000);
    }
    return -1;
}

// the test drives one connection; every read has a deadline so a wedged server fails, not hangs
void set_timeout(int fd) {
    timeval tv = {};
    tv.tv_sec  = 15;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

void run_tests(int fd) {
    // HELLO: the handshake must answer before anything else is tried
    {
        std::vector<uint8_t> req;
        sgr::put_u32(req, sgr::kVersion);
        uint32_t status = 0;
        std::vector<uint8_t> storage;
        CHECK(invoke(fd, sgr::SGR_OP_HELLO, req, status, storage));
        CHECK(status == SCRAPGPU_STATUS_OK);
    }

    // an oversized SHM_ENSURE is refused but the connection keeps working
    {
        std::vector<uint8_t> req;
        sgr::put_u64(req, sgr::kMaxShm + 1);
        uint32_t status = 0;
        std::vector<uint8_t> storage;
        CHECK(invoke(fd, sgr::SGR_OP_SHM_ENSURE, req, status, storage));
        CHECK(status != SCRAPGPU_STATUS_OK);
        std::vector<uint8_t> empty;
        status = 0;
        storage.clear();
        CHECK(invoke(fd, sgr::SGR_OP_GET_DEVICE_COUNT, empty, status, storage));
        CHECK(status == SCRAPGPU_STATUS_OK);
    }

    // a header announcing a payload above kMaxPayload must not be allocated: the server answers
    // with an error and closes the connection cleanly
    {
        std::vector<uint8_t> hdr;
        sgr::put_u32(hdr, sgr::kMagic);
        sgr::put_u32(hdr, sgr::SGR_OP_BUFFER_READ);
        sgr::put_u64(hdr, (uint64_t) sgr::kMaxPayload + 1);
        CHECK(write_all(fd, hdr.data(), hdr.size()));

        uint8_t rh[sgr::kHdrSize];
        CHECK(read_all(fd, rh, sizeof(rh)) == 1);
        sgr::cur hc{rh, sgr::kHdrSize};
        uint32_t magic = 0, status = 0;
        uint64_t size = 0;
        CHECK(sgr::get_u32(hc, magic) && magic == sgr::kMagic && sgr::get_u32(hc, status) &&
              sgr::get_u64(hc, size));
        CHECK(status == SCRAPGPU_STATUS_INVALID_ARGUMENT);
        CHECK(size <= sgr::kMaxPayload + 64 * 1024); // a sane error response, not a mirror of the request
        if (size > 0 && size <= sgr::kMaxPayload + 64 * 1024) {
            std::vector<uint8_t> err((size_t) size);
            CHECK(read_all(fd, err.data(), (size_t) size) == 1);
        }
        // the server then closes the connection
        uint8_t b = 0;
        CHECK(read_all(fd, &b, 1) == 0);
    }
}

} // namespace

int main() {
    char dir[] = "/tmp/scrapgpu-proto-XXXXXX";
    if (mkdtemp(dir) == nullptr) {
        std::fprintf(stderr, "mkdtemp: %s\n", strerror(errno));
        return 1;
    }
    const std::string sock = std::string(dir) + "/srv.sock";

    const pid_t child = fork();
    if (child < 0) {
        std::fprintf(stderr, "fork: %s\n", strerror(errno));
        rmdir(dir);
        return 1;
    }
    if (child == 0) {
        execl(SCRAPGPU_TEST_SERVER, "scrapgpu-device-server",
              "--backend", SCRAPGPU_TEST_SERVER_BACKEND, "--socket", sock.c_str(), (char *) nullptr);
        execlp("scrapgpu-device-server", "scrapgpu-device-server",
               "--backend", SCRAPGPU_TEST_SERVER_BACKEND, "--socket", sock.c_str(), (char *) nullptr);
        _exit(127);
    }

    int rc = 1;
    const int fd = connect_retry(sock, child);
    if (fd < 0) {
        std::fprintf(stderr, "the test server did not come up (SCRAPGPU_TEST_SERVER=%s)\n",
                     SCRAPGPU_TEST_SERVER);
        CHECK(fd >= 0);
    } else {
        set_timeout(fd);
        run_tests(fd);
        close(fd);
        rc = g_failures == 0 ? 0 : 1;
    }
    // the server exits when the client disconnects
    int wstatus = 0;
    for (int i = 0; i < 300 && waitpid(child, &wstatus, WNOHANG) != child; ++i) {
        usleep(10000);
    }
    if (waitpid(child, &wstatus, WNOHANG) != child) {
        kill(child, SIGKILL);
        waitpid(child, nullptr, 0);
    }
    unlink(sock.c_str());
    rmdir(dir);
    if (g_failures == 0) {
        std::printf("remote protocol tests OK\n");
    }
    return g_failures == 0 ? rc : 1;
}
