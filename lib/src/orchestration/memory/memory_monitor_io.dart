import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import 'memory_monitor.dart';

/// Native factory: real measurements on Apple platforms, fixed fallback
/// elsewhere.
SystemMemoryMonitor createSystemMemoryMonitor() =>
    Platform.isMacOS || Platform.isIOS
    ? DarwinMemoryMonitor()
    : FixedMemoryMonitor(fallbackMemorySnapshot);

typedef _SysctlByNameC =
    Int32 Function(
      Pointer<Utf8> name,
      Pointer<Void> oldp,
      Pointer<Size> oldlenp,
      Pointer<Void> newp,
      Size newlen,
    );
typedef _SysctlByNameDart =
    int Function(
      Pointer<Utf8> name,
      Pointer<Void> oldp,
      Pointer<Size> oldlenp,
      Pointer<Void> newp,
      int newlen,
    );

typedef _MachHostSelfC = Uint32 Function();
typedef _MachHostSelfDart = int Function();

typedef _HostStatistics64C =
    Int32 Function(
      Uint32 host,
      Int32 flavor,
      Pointer<Void> info,
      Pointer<Uint32> count,
    );
typedef _HostStatistics64Dart =
    int Function(
      int host,
      int flavor,
      Pointer<Void> info,
      Pointer<Uint32> count,
    );

typedef _TaskInfoC =
    Int32 Function(
      Uint32 task,
      Uint32 flavor,
      Pointer<Void> info,
      Pointer<Uint32> count,
    );
typedef _TaskInfoDart =
    int Function(
      int task,
      int flavor,
      Pointer<Void> info,
      Pointer<Uint32> count,
    );

typedef _AvailableMemoryC = Uint64 Function();
typedef _AvailableMemoryDart = int Function();

/// `HOST_VM_INFO64` in `<mach/host_info.h>`.
const int _hostVmInfo64 = 4;

/// `sizeof(vm_statistics64) / sizeof(integer_t)`.
const int _hostVmInfo64Count = 38;

/// `TASK_VM_INFO` in `<mach/task_info.h>`.
const int _taskVmInfo = 22;

/// Byte offset of `phys_footprint` in `task_vm_info_data_t`; the field is
/// part of revision 0, so every supported OS fills it.
const int _physFootprintOffset = 144;

/// `TASK_VM_INFO` count covering `phys_footprint` (in `integer_t` units).
const int _taskVmInfoCount = 38;

/// Measures system memory on macOS/iOS through libSystem.
///
/// Total memory comes from `sysctl hw.memsize`. Available memory prefers
/// `os_proc_available_memory` (the real jetsam headroom; iOS reports it,
/// macOS returns 0) and falls back to VM statistics
/// (free + inactive + purgeable pages). The app footprint uses
/// `task_info(TASK_VM_INFO).phys_footprint`, best-effort.
final class DarwinMemoryMonitor implements SystemMemoryMonitor {
  DarwinMemoryMonitor() : _lib = DynamicLibrary.process() {
    _sysctlByName = _lib.lookupFunction<_SysctlByNameC, _SysctlByNameDart>(
      'sysctlbyname',
    );
    _machHostSelf = _lib.lookupFunction<_MachHostSelfC, _MachHostSelfDart>(
      'mach_host_self',
    );
    _hostStatistics64 = _lib
        .lookupFunction<_HostStatistics64C, _HostStatistics64Dart>(
          'host_statistics64',
        );
    _taskInfo = _lib.lookupFunction<_TaskInfoC, _TaskInfoDart>('task_info');
    _availableMemory = _lib.providesSymbol('os_proc_available_memory')
        ? _lib.lookupFunction<_AvailableMemoryC, _AvailableMemoryDart>(
            'os_proc_available_memory',
          )
        : null;
    _machTaskSelf = _lib.providesSymbol('mach_task_self_')
        ? _lib.lookup<Uint32>('mach_task_self_').value
        : null;
  }

  final DynamicLibrary _lib;
  late final _SysctlByNameDart _sysctlByName;
  late final _MachHostSelfDart _machHostSelf;
  late final _HostStatistics64Dart _hostStatistics64;
  late final _TaskInfoDart _taskInfo;
  late final _AvailableMemoryDart? _availableMemory;
  late final int? _machTaskSelf;

  @override
  Future<MemorySnapshot> sample() async {
    final total = _sysctlInt('hw.memsize') ?? fallbackMemorySnapshot.totalBytes;
    final available = _sampleAvailable();
    return MemorySnapshot(
      totalBytes: total,
      availableBytes: available ?? fallbackMemorySnapshot.availableBytes,
      appFootprintBytes: _physFootprint(),
      isEstimated: available == null,
    );
  }

  int? _sampleAvailable() {
    // os_proc_available_memory is the honest number where it exists (iOS,
    // entitlement-aware jetsam headroom); it returns 0 on macOS.
    final direct = _availableMemory?.call();
    if (direct != null && direct > 0) return direct;
    return _vmAvailable();
  }

  /// Available memory from `host_statistics64`:
  /// (free + inactive + purgeable pages) x page size.
  int? _vmAvailable() {
    final pageSize = _sysctlInt('hw.pagesize');
    if (pageSize == null || pageSize <= 0) return null;
    final info = calloc<Uint8>(_hostVmInfo64Count * 4);
    final count = calloc<Uint32>()..value = _hostVmInfo64Count;
    try {
      final result = _hostStatistics64(
        _machHostSelf(),
        _hostVmInfo64,
        info.cast(),
        count,
      );
      if (result != 0) return null;
      final data = info.cast<Uint32>();
      final freePages = data[0];
      final inactivePages = data[2];
      // purgeable_count sits at byte offset 88 (see vm_statistics64).
      final purgeablePages = data[22];
      return (freePages + inactivePages + purgeablePages) * pageSize;
    } finally {
      calloc.free(info);
      calloc.free(count);
    }
  }

  /// This process's `phys_footprint`, or null when unavailable.
  int? _physFootprint() {
    final task = _machTaskSelf;
    if (task == null) return null;
    final info = calloc<Uint8>(_taskVmInfoCount * 4 + 8);
    final count = calloc<Uint32>()..value = _taskVmInfoCount;
    try {
      final result = _taskInfo(task, _taskVmInfo, info.cast(), count);
      if (result != 0) return null;
      return (info + _physFootprintOffset).cast<Uint64>().value;
    } finally {
      calloc.free(info);
      calloc.free(count);
    }
  }

  /// Reads an integer sysctl, tolerating 4- and 8-byte values.
  int? _sysctlInt(String name) {
    final namePtr = name.toNativeUtf8();
    final value = calloc<Uint64>();
    final length = calloc<Size>()..value = 8;
    try {
      final result = _sysctlByName(namePtr, value.cast(), length, nullptr, 0);
      if (result != 0) return null;
      return length.value == 4 ? value.cast<Uint32>().value : value.value;
    } finally {
      calloc.free(namePtr);
      calloc.free(value);
      calloc.free(length);
    }
  }
}
