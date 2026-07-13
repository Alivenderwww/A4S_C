// ==========================================================================
// blocking_submit_shim.cpp — LD_PRELOAD wrapper for aecDeviceSubmit
//
// Compile (Linux only):
//   g++ -shared -fPIC -ldl -pthread -o blocking_submit_shim.so \
//       blocking_submit_shim.cpp
//
// Usage in test subprocess:
//   LD_PRELOAD=/abs/path/to/blocking_submit_shim.so python test_script.py
//
// Default: fully transparent — every call is forwarded to the real
// aecDeviceSubmit with zero overhead.
//
// When armed (aecTestArmSubmitBlock), the wrapper blocks the calling thread
// inside the *entry* of aecDeviceSubmit before forwarding, and does not
// return until aecTestReleaseSubmitBlock is called.  This gives the test
// thread a deterministic window to observe intermediate states (stream not
// destroyed, allocation not freed, registration not removed, etc.).
//
// Exported test-control functions (callable from Python via ctypes):
//   - aecTestArmSubmitBlock()        — arm the blocker (next submit blocks)
//   - aecTestWaitSubmitBlocked(ms)   — wait up to ms for a thread to block
//   - aecTestReleaseSubmitBlock()    — release the blocked thread
//   - aecTestResetSubmitBlock()      — disarm + reset (back to transparent)
// ==========================================================================

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cerrno>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// --------------------------------------------------------------------------
// Typedef for the real aecDeviceSubmit signature
// --------------------------------------------------------------------------
// We don't #include aec_device_abi.h so this file compiles standalone.
// The struct layouts don't matter — we forward the opaque pointers.
typedef struct aecDeviceCommand aecDeviceCommand;
typedef struct aecDeviceCompletion aecDeviceCompletion;

typedef int (*aecDeviceSubmit_fn)(const aecDeviceCommand *, aecDeviceCompletion *);

// --------------------------------------------------------------------------
// State (protected by mutex)
// --------------------------------------------------------------------------
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cond  = PTHREAD_COND_INITIALIZER;

// Guard: whether the blocker is currently armed.
static volatile int g_armed = 0;

// Count of blocked threads (should be 0 or 1 in deterministic tests).
static volatile int g_blocked = 0;

// Re-entry guard: inside wrapper we set this to avoid blocking when the
// real aecDeviceSubmit or internal device machinery calls back.
static __thread int g_inside = 0;

// --------------------------------------------------------------------------
// Wrapped entry point
// --------------------------------------------------------------------------
extern "C" int aecDeviceSubmit(const aecDeviceCommand *cmd,
                               aecDeviceCompletion *comp) {
    // Resolve the real function once (thread-safe: dlsym is idempotent and
    // the runtime linker guarantees single-initialisation).
    static aecDeviceSubmit_fn real_submit = nullptr;
    if (real_submit == nullptr) {
        real_submit = (aecDeviceSubmit_fn)dlsym(RTLD_NEXT, "aecDeviceSubmit");
        if (real_submit == nullptr) {
            fprintf(stderr,
                    "[blocking_submit_shim] FATAL: dlsym(RTLD_NEXT, "
                    "\"aecDeviceSubmit\") failed: %s\n",
                    dlerror());
            _exit(99);
        }
    }

    // Block if armed, not already inside the wrapper (avoids recursive
    // deadlock), and the caller is not the shim itself.
    if (g_armed && !g_inside) {
        pthread_mutex_lock(&g_mutex);
        g_blocked = 1;
        pthread_cond_broadcast(&g_cond);  // wake waiters
        while (g_armed) {
            pthread_cond_wait(&g_cond, &g_mutex);
        }
        g_blocked = 0;
        pthread_mutex_unlock(&g_mutex);
    }

    g_inside = 1;
    int rc = real_submit(cmd, comp);
    g_inside = 0;
    return rc;
}

// --------------------------------------------------------------------------
// Test-control functions (exported "C")
// --------------------------------------------------------------------------

extern "C" void aecTestArmSubmitBlock(void) {
    pthread_mutex_lock(&g_mutex);
    g_armed = 1;
    pthread_mutex_unlock(&g_mutex);
}

// Wait up to `timeout_ms` for a thread to be blocked in aecDeviceSubmit.
// Returns 1 if blocked before timeout, 0 on timeout.
extern "C" int aecTestWaitSubmitBlocked(uint64_t timeout_ms) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000L;
    }

    pthread_mutex_lock(&g_mutex);
    int rc = 0;
    while (!g_blocked) {
        int ret = pthread_cond_timedwait(&g_cond, &g_mutex, &ts);
        if (ret == ETIMEDOUT) {
            rc = 0;
            goto done;
        }
    }
    rc = 1;
done:
    pthread_mutex_unlock(&g_mutex);
    return rc;
}

extern "C" void aecTestReleaseSubmitBlock(void) {
    pthread_mutex_lock(&g_mutex);
    g_armed = 0;
    pthread_cond_broadcast(&g_cond);
    pthread_mutex_unlock(&g_mutex);
}

extern "C" void aecTestResetSubmitBlock(void) {
    pthread_mutex_lock(&g_mutex);
    g_armed = 0;
    g_blocked = 0;
    pthread_cond_broadcast(&g_cond);
    pthread_mutex_unlock(&g_mutex);
}
