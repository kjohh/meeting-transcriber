// System-audio capture helper for Meeting Transcriber.
//
// Uses a Core Audio process tap (macOS 14.4+) instead of ScreenCaptureKit.
// The permission macOS asks for is "System Audio Recording Only"
// (kTCCServiceAudioCapture / NSAudioCaptureUsageDescription) rather than
// "Screen Recording", so the user is never asked to let the app see the screen.
//
// Modes:
//   coreaudio_tap              capture: float32 PCM, 16 kHz mono, on stdout.
//                              stderr protocol: "READY" once audio flows,
//                              "ERROR:<msg>" on failure (process then exits).
//   coreaudio_tap --preflight  print "authorized" | "denied" | "unknown", exit 0.
//                              Never shows a prompt.
//   coreaudio_tap --request    show the system prompt (if not decided yet),
//                              print "authorized" | "denied", exit 0.
//
// TCC attributes the helper to its responsible process (the .app that spawned
// it), so the prompt shows the app's name and its Info.plist usage string.

import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation

// MARK: - Permission (TCC SPI, same approach as insidegui/AudioCap)

private let tccService = "kTCCServiceAudioCapture" as CFString

private func tccHandle() -> UnsafeMutableRawPointer? {
    dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW)
}

/// 0 = authorized, 1 = denied, anything else = not determined.
func permissionPreflight() -> String {
    typealias Preflight = @convention(c) (CFString, CFDictionary?) -> Int
    guard let h = tccHandle(), let sym = dlsym(h, "TCCAccessPreflight") else { return "unknown" }
    let fn = unsafeBitCast(sym, to: Preflight.self)
    switch fn(tccService, nil) {
    case 0: return "authorized"
    case 1: return "denied"
    default: return "unknown"
    }
}

func permissionRequest() -> String {
    typealias Request = @convention(c) (CFString, CFDictionary?, @escaping (Bool) -> Void) -> Void
    guard let h = tccHandle(), let sym = dlsym(h, "TCCAccessRequest") else { return "unknown" }
    let fn = unsafeBitCast(sym, to: Request.self)
    let sem = DispatchSemaphore(value: 0)
    var granted = false
    fn(tccService, nil) { ok in
        granted = ok
        sem.signal()
    }
    // The prompt waits on the user; give them plenty of time.
    _ = sem.wait(timeout: .now() + 120)
    return granted ? "authorized" : "denied"
}

// MARK: - Core Audio helpers

struct CAError: Error, CustomStringConvertible {
    let what: String
    let status: OSStatus
    var description: String { "\(what) failed (OSStatus \(status))" }
}

@inline(__always)
func check(_ status: OSStatus, _ what: String) throws {
    if status != noErr { throw CAError(what: what, status: status) }
}

func address(_ selector: AudioObjectPropertySelector,
             scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func defaultSystemOutputDevice() throws -> AudioDeviceID {
    var addr = address(kAudioHardwarePropertyDefaultSystemOutputDevice)
    var id = AudioDeviceID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    try check(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &id),
              "read default output device")
    return id
}

func deviceUID(_ id: AudioDeviceID) throws -> String {
    var addr = address(kAudioDevicePropertyDeviceUID)
    var uid: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    try check(AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &uid), "read device UID")
    guard let value = uid?.takeRetainedValue() else { throw CAError(what: "read device UID", status: -1) }
    return value as String
}

// MARK: - Capture

final class SystemAudioTap {
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?
    private var announcedReady = false
    private let ioQueue = DispatchQueue(label: "coreaudio_tap.io", qos: .userInitiated)
    // Property listeners must NOT share ioQueue: stop() waits for the IOProc,
    // which would deadlock if stop() itself were running on ioQueue.
    private let listenerQueue = DispatchQueue(label: "coreaudio_tap.listener")
    private let stopLock = NSLock()
    private var stopped = false

    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false
    )!

    func start() throws {
        // Tap every process's output. Our own app plays no audio, so nothing
        // needs excluding.
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        description.uuid = UUID()
        description.name = "Meeting Transcriber system audio"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        try check(AudioHardwareCreateProcessTap(description, &tapID), "create process tap")

        var fmtAddr = address(kAudioTapPropertyFormat)
        var asbd = AudioStreamBasicDescription()
        var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        try check(AudioObjectGetPropertyData(tapID, &fmtAddr, 0, nil, &asbdSize, &asbd), "read tap format")
        guard let fmt = AVAudioFormat(streamDescription: &asbd) else {
            throw CAError(what: "build tap format", status: -1)
        }
        sourceFormat = fmt
        converter = AVAudioConverter(from: fmt, to: targetFormat)

        // An aggregate device clocked by the current output device carries the
        // tap's stream to us through a normal IOProc.
        let outputID = try defaultSystemOutputDevice()
        let outputUID = try deviceUID(outputID)
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Meeting Transcriber Tap",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapDriftCompensationKey: true,
                kAudioSubTapUIDKey: description.uuid.uuidString,
            ]],
        ]
        try check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID),
                  "create aggregate device")

        try check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, ioQueue) {
            [weak self] _, inInputData, _, _, _ in
            self?.handle(inInputData)
        }, "create IOProc")
        try check(AudioDeviceStart(aggregateID, procID), "start aggregate device")

        watchDefaultOutput(startedOn: outputID)

        if ProcessInfo.processInfo.environment["MT_TAP_DEBUG"] != nil {
            var runAddr = address(kAudioDevicePropertyDeviceIsRunningSomewhere)
            var running: UInt32 = 0
            var size = UInt32(MemoryLayout<UInt32>.size)
            AudioObjectGetPropertyData(aggregateID, &runAddr, 0, nil, &size, &running)
            fputs("DEBUG tap=\(tapID) aggregate=\(aggregateID) output=\(outputID) running=\(running) "
                  + "format=\(fmt.sampleRate)Hz/\(fmt.channelCount)ch permission=\(permissionPreflight())\n", stderr)
        }
    }

    /// If the output device changes (headphones plugged in, Bluetooth drops),
    /// the aggregate is still clocked by the old device and can go silent or
    /// stop. Exit with an error so the parent's supervisor re-spawns us bound
    /// to the new device.
    ///
    /// Creating our private aggregate device itself fires this notification,
    /// so only act when the default device really is a different one.
    private func watchDefaultOutput(startedOn original: AudioDeviceID) {
        var addr = address(kAudioHardwarePropertyDefaultSystemOutputDevice)
        AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &addr, listenerQueue) {
            [weak self] _, _ in
            guard let now = try? defaultSystemOutputDevice(), now != original else { return }
            fputs("ERROR:output device changed (\(original) -> \(now))\n", stderr)
            self?.stop()
            exit(2)
        }
    }

    private func handle(_ input: UnsafePointer<AudioBufferList>) {
        guard let src = sourceFormat, let conv = converter else { return }
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: src, bufferListNoCopy: input, deallocator: nil),
              inBuf.frameLength > 0 else { return }

        let outCapacity = AVAudioFrameCount(
            Double(inBuf.frameLength) * targetFormat.sampleRate / src.sampleRate + 1
        )
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else { return }

        var consumed = false
        conv.convert(to: outBuf, error: nil) { _, status in
            if consumed { status.pointee = .noDataNow; return nil }
            consumed = true
            status.pointee = .haveData
            return inBuf
        }
        guard outBuf.frameLength > 0, let ch = outBuf.floatChannelData else { return }

        if !announcedReady {
            announcedReady = true
            fputs("READY\n", stderr)
        }
        let bytes = Int(outBuf.frameLength) * MemoryLayout<Float>.size
        var offset = 0
        while offset < bytes {
            let n = Darwin.write(STDOUT_FILENO, UnsafeRawPointer(ch[0]).advanced(by: offset), bytes - offset)
            if n <= 0 {
                if n < 0 && errno == EINTR { continue }
                // Reader is gone (EPIPE). Tear down off the IO queue.
                listenerQueue.async { [weak self] in
                    self?.stop()
                    exit(0)
                }
                return
            }
            offset += n
        }
    }

    func stop() {
        stopLock.lock()
        defer { stopLock.unlock() }
        if stopped { return }
        stopped = true
        if aggregateID != kAudioObjectUnknown {
            AudioDeviceStop(aggregateID, procID)
            if let p = procID { AudioDeviceDestroyIOProcID(aggregateID, p) }
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }
}

// MARK: - Entry

let args = CommandLine.arguments.dropFirst()
if args.contains("--preflight") {
    print(permissionPreflight())
    exit(0)
}
if args.contains("--request") {
    print(permissionRequest())
    exit(0)
}

let tap = SystemAudioTap()

// Tear the tap + aggregate device down on every exit path we control, so no
// private aggregate device outlives the process. Signals are routed through
// DispatchSource so cleanup runs on a normal queue, not in signal context.
var signalSources: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT, SIGPIPE, SIGHUP] {
    signal(sig, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler {
        tap.stop()
        exit(0)
    }
    src.resume()
    signalSources.append(src)
}

do {
    try tap.start()
} catch {
    fputs("ERROR:\(error)\n", stderr)
    tap.stop()
    exit(1)
}

// Parent went away (stdin closed / reparented to launchd) → stop.
DispatchQueue.global(qos: .utility).async {
    while getppid() != 1 { sleep(1) }
    tap.stop()
    exit(0)
}

RunLoop.main.run()
