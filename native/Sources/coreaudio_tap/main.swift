import Foundation
import ScreenCaptureKit
import AVFoundation

signal(SIGTERM) { _ in exit(0) }
signal(SIGPIPE) { _ in exit(0) }

@available(macOS 13.0, *)
final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {

    private var stream: SCStream?
    private var converter: AVAudioConverter?

    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16000,
        channels: 1,
        interleaved: false
    )!

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false
        )
        guard let display = content.displays.first else {
            throw NSError(domain: "coreaudio_tap", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No display found"])
        }

        let filter = SCContentFilter(
            display: display,
            excludingApplications: [],
            exceptingWindows: []
        )

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true

        stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream!.addStreamOutput(
            self, type: .audio,
            sampleHandlerQueue: .global(qos: .userInteractive)
        )
        try await stream!.startCapture()
        fputs("READY\n", stderr)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("ERROR:\(error.localizedDescription)\n", stderr)
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio else { return }
        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              var asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc)?.pointee
        else { return }

        guard let srcFormat = AVAudioFormat(streamDescription: &asbd) else { return }

        if converter == nil
            || converter!.inputFormat.sampleRate != srcFormat.sampleRate
            || converter!.inputFormat.channelCount != srcFormat.channelCount
        {
            converter = AVAudioConverter(from: srcFormat, to: targetFormat)
        }
        guard let conv = converter else { return }

        let frameCount = CMSampleBufferGetNumSamples(sampleBuffer)

        // Query required AudioBufferList size
        var ablByteSize = 0
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: &ablByteSize,
            bufferListOut: nil,
            bufferListSize: 0,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: 0,
            blockBufferOut: nil
        )
        guard ablByteSize > 0 else { return }

        let ablRaw = UnsafeMutableRawPointer.allocate(byteCount: ablByteSize, alignment: 8)
        defer { ablRaw.deallocate() }
        let ablPtr = ablRaw.bindMemory(to: AudioBufferList.self, capacity: 1)

        var blockBuffer: CMBlockBuffer?
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: ablPtr,
            bufferListSize: ablByteSize,
            blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer
        )
        guard status == noErr else { return }

        guard let inBuf = AVAudioPCMBuffer(
            pcmFormat: srcFormat,
            frameCapacity: AVAudioFrameCount(frameCount)
        ) else { return }
        inBuf.frameLength = AVAudioFrameCount(frameCount)

        // Copy audio data into AVAudioPCMBuffer
        let srcBufs = UnsafeMutableAudioBufferListPointer(ablPtr)
        let dstBufs = UnsafeMutableAudioBufferListPointer(inBuf.mutableAudioBufferList)
        for i in 0..<min(srcBufs.count, dstBufs.count) {
            guard let src = srcBufs[i].mData, let dst = dstBufs[i].mData else { continue }
            memcpy(dst, src, Int(srcBufs[i].mDataByteSize))
        }

        let outCapacity = AVAudioFrameCount(
            Double(frameCount) * targetFormat.sampleRate / srcFormat.sampleRate + 1
        )
        guard let outBuf = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: outCapacity
        ) else { return }

        var done = false
        conv.convert(to: outBuf, error: nil) { _, outStatus in
            if done { outStatus.pointee = .noDataNow; return nil }
            outStatus.pointee = .haveData
            done = true
            return inBuf
        }

        guard outBuf.frameLength > 0, let ch = outBuf.floatChannelData else { return }
        let data = Data(bytes: ch[0], count: Int(outBuf.frameLength) * MemoryLayout<Float>.size)
        FileHandle.standardOutput.write(data)
    }
}

guard #available(macOS 13.0, *) else {
    fputs("Requires macOS 13.0+\n", stderr)
    exit(1)
}

let capture = SystemAudioCapture()
Task {
    do {
        try await capture.start()
        // Keep alive until terminated
        await withCheckedContinuation { (_: CheckedContinuation<Void, Never>) in }
    } catch {
        fputs("ERROR:\(error.localizedDescription)\n", stderr)
        exit(1)
    }
}
RunLoop.main.run()
