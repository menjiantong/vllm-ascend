# [1110] vLLM worker 侧主机 NUMA / THP 控制（默认 no-op，行为不变）。
#
# 背景：engram DP 共享表是数百 GiB 的 /dev/shm 映射，页在首次写入（checkpoint load）时
# 按写入线程的 CPU 落位。默认 16 个 vLLM worker 可能都落在同一节点，把该节点抽干，
# 之后同节点的匿名页分配走 THP 直接压缩慢路径。本模块提供两个开关：
#   MY_ENV_DSV41_DISABLE_THP=1       进程级 prctl(PR_SET_THP_DISABLE)，靶向规避压缩慢路径。
#   MY_ENV_DSV41_ENGRAM_SPREAD=1     大表“分配 + 首次填充”整段按 rank 钉到对应 NUMA 节点，
#                                    16 rank 均摊到有内存节点（188/126 实测 first-touch 生效）。
#   MY_ENV_DSV41_NUMA_INTERLEAVE=1   若能用的容器允许 set_mempolicy(MPOL_INTERLEAVE) 则优先；
#                                    188/126 容器无 CAP_SYS_NICE，syscall 被 seccomp 拒绝(EPERM)，
#                                    自动回退 first-touch 并打日志。
# 注意：vllm-ascend 不依赖 verl，这里自带一份最小实现。

import contextlib
import ctypes
import os
import platform

PR_SET_THP_DISABLE = 41

MPOL_DEFAULT = 0
MPOL_INTERLEAVE = 3
_SYS_SET_MEMPOLICY = {"aarch64": 237, "x86_64": 238}
_AFFINITY_BYTES = 128  # 1024 CPU 足够本机型（320）


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _libc():
    return ctypes.CDLL("libc.so.6", use_errno=True)


def _read_text(path, default=""):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return default


def _parse_list(text):
    values = []
    for part in str(text or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def memory_nodes():
    """online 且有内存的节点（跳过 MemTotal=0 的无内存节点）。"""
    import re

    nodes = []
    for node in _parse_list(_read_text("/sys/devices/system/node/online", "0")):
        text = _read_text("/sys/devices/system/node/node%d/meminfo" % node)
        match = re.search(r"MemTotal:\s*(\d+)\s*kB", text)
        if match and int(match.group(1)) > 0:
            nodes.append(node)
    return nodes


def resolve_nodes():
    """有内存节点；若 MY_ENV_DSV41_NUMA_NODES 指定则取交集（如排除半容量 node3）。"""
    nodes = memory_nodes()
    spec = os.getenv("MY_ENV_DSV41_NUMA_NODES", "").strip()
    if spec:
        wanted = set(_parse_list(spec))
        filtered = [node for node in nodes if node in wanted]
        if filtered:
            return filtered
    return nodes


def cpu_list_for_node(node):
    return _parse_list(_read_text("/sys/devices/system/node/node%d/cpulist" % node, ""))


def get_affinity():
    buffer = ctypes.create_string_buffer(_AFFINITY_BYTES)
    _libc().sched_getaffinity(ctypes.c_int(0), ctypes.c_size_t(_AFFINITY_BYTES), buffer)
    allowed = []
    for byte_index, byte in enumerate(buffer.raw):
        for bit in range(8):
            if byte & (1 << bit):
                allowed.append(byte_index * 8 + bit)
    return allowed


def set_affinity(cpus):
    mask = (ctypes.c_ubyte * _AFFINITY_BYTES)()
    valid = 0
    for cpu in cpus:
        if 0 <= cpu < _AFFINITY_BYTES * 8:
            mask[cpu // 8] |= 1 << (cpu % 8)
            valid += 1
    if not valid:
        return False
    return _libc().sched_setaffinity(ctypes.c_int(0), ctypes.c_size_t(_AFFINITY_BYTES), mask) == 0


def disable_thp():
    return _libc().prctl(
        ctypes.c_int(PR_SET_THP_DISABLE),
        ctypes.c_ulong(1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    ) == 0


def _set_mempolicy(mode, nodes=None):
    machine = platform.machine()
    number = _SYS_SET_MEMPOLICY.get(machine)
    if number is None:
        raise OSError("set_mempolicy syscall number unknown on %s" % machine)
    libc = _libc()
    libc.syscall.restype = ctypes.c_long
    nodes = list(nodes or [])
    if mode == MPOL_DEFAULT or not nodes:
        ret = libc.syscall(ctypes.c_long(number), ctypes.c_int(mode),
                           ctypes.c_void_p(None), ctypes.c_ulong(0))
    else:
        maxnode = max(nodes) + 1
        mask = (ctypes.c_ubyte * ((maxnode + 7) // 8))()
        for node in nodes:
            mask[node // 8] |= 1 << (node % 8)
        ret = libc.syscall(ctypes.c_long(number), ctypes.c_int(mode),
                           ctypes.byref(mask), ctypes.c_ulong(maxnode))
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, "set_mempolicy: %s" % os.strerror(err))


def current_rank():
    """vLLM worker 身份：优先已初始化的 torch.distributed，其次环境变量。"""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001 - 身份只影响摊开位置，失败就退化
        pass
    for key in ("RANK", "VLLM_DP_RANK", "LOCAL_RANK"):
        value = os.getenv(key)
        if value is not None and str(value).strip().isdigit():
            return int(value)
    return os.getpid()


@contextlib.contextmanager
def host_alloc_spread_ctx(rank=None):
    """大表“分配 + 首次填充”的 NUMA 摊开上下文（默认 no-op）。

    优先 set_mempolicy(MPOL_INTERLEAVE)（若容器允许），否则按 rank%节点数 first-touch。
    """
    spread = _truthy(os.getenv("MY_ENV_DSV41_ENGRAM_SPREAD", "0"))
    interleave = _truthy(os.getenv("MY_ENV_DSV41_NUMA_INTERLEAVE", "0"))
    if not spread and not interleave:
        yield None
        return
    nodes = resolve_nodes()
    if not nodes:
        yield None
        return
    if interleave:
        try:
            _set_mempolicy(MPOL_INTERLEAVE, nodes)
        except OSError as exc:
            print(
                "[MY_DEBUG_LOG] vllm_alloc_interleave_failed pid=%s nodes=%s error=%s "
                "(fallback to first-touch spread=%s)"
                % (os.getpid(), nodes, exc, int(spread)),
                flush=True,
            )
        else:
            try:
                yield None
            finally:
                try:
                    _set_mempolicy(MPOL_DEFAULT, [])
                except OSError:
                    pass
            return
    if spread:
        index = current_rank() if rank is None else int(rank)
        target = nodes[index % len(nodes)]
        allowed = get_affinity()
        cpus = [cpu for cpu in cpu_list_for_node(target) if cpu in allowed] or allowed
        set_affinity(cpus)
        # copy_/zero_ 走 OMP 线程池时，页可能由旧亲和的工作线程 fault；临时单线程
        # 把首次触达压到主线程，保证落到目标节点（加载阶段一次性成本，可接受）。
        threads = None
        try:
            import torch

            threads = torch.get_num_threads()
            torch.set_num_threads(1)
        except Exception:  # noqa: BLE001 - torch 不可用则仅靠主线程亲和
            threads = None
        try:
            yield target
        finally:
            if threads is not None:
                try:
                    torch.set_num_threads(threads)
                except Exception:  # noqa: BLE001
                    pass
            set_affinity(allowed)
    else:
        yield None


def disable_thp_if_requested():
    """NPUWorker.__init__ 调用：在任何大块 pinned/shm 分配之前生效。"""
    requested = _truthy(os.getenv("MY_ENV_DSV41_DISABLE_THP", "0"))
    ok = None
    if requested:
        ok = disable_thp()
    print(
        "[MY_DEBUG_LOG] vllm_host_numa_env pid=%s rank=%s thp_off=%s thp_ok=%s "
        "spread=%s interleave=%s pin_process=%s nodes=%s"
        % (
            os.getpid(),
            current_rank(),
            int(requested),
            ok,
            int(_truthy(os.getenv("MY_ENV_DSV41_ENGRAM_SPREAD", "0"))),
            int(_truthy(os.getenv("MY_ENV_DSV41_NUMA_INTERLEAVE", "0"))),
            int(_truthy(os.getenv("MY_ENV_DSV41_PIN_PROCESS", "0"))),
            resolve_nodes(),
        ),
        flush=True,
    )
    return ok


def apply_process_pinning(rank=None):
    """进程级 first-touch（MY_ENV_DSV41_PIN_PROCESS=1）。

    把本 worker 进程的所有主机分配钉到 rank 对应 NUMA 节点：engram 共享表、
    checkpoint staging、camem 备份等都会按调用线程落页，从而整机摊开。
    """
    if not _truthy(os.getenv("MY_ENV_DSV41_PIN_PROCESS", "0")):
        return None
    nodes = resolve_nodes()
    if not nodes:
        return None
    index = current_rank() if rank is None else int(rank)
    target = nodes[index % len(nodes)]
    allowed = get_affinity()
    cpus = [cpu for cpu in cpu_list_for_node(target) if cpu in allowed] or allowed
    ok = set_affinity(cpus)
    print(
        "[MY_DEBUG_LOG] vllm_pin_process pid=%s index=%s node=%s ok=%s"
        % (os.getpid(), index, target, ok),
        flush=True,
    )
    return target if ok else None
