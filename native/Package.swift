// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "coreaudio_tap",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "coreaudio_tap",
            path: "Sources/coreaudio_tap"
        )
    ]
)
