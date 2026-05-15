import Foundation

let exe = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
let projectDir = exe
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

func osascriptDialog(_ message: String) {
    let escaped = message.replacingOccurrences(of: "\"", with: "\\\"")
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
    p.arguments = ["-e", "display dialog \"\(escaped)\" with title \"Meeting Transcriber\" buttons {\"OK\"} default button \"OK\""]
    try? p.run(); p.waitUntilExit()
}

@discardableResult
func run(_ cmd: String, args: [String], cwd: URL? = nil, silent: Bool = false) -> Int32 {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: cmd)
    p.arguments = args
    if let cwd { p.currentDirectoryURL = cwd }
    if silent {
        p.standardOutput = FileHandle.nullDevice
        p.standardError  = FileHandle.nullDevice
    }
    try? p.run(); p.waitUntilExit()
    return p.terminationStatus
}

// ── Find Python ──
let pythonCandidates = [
    "/opt/homebrew/bin/python3.13",
    "/opt/homebrew/bin/python3.12",
    "/opt/homebrew/bin/python3",
]
guard let python = pythonCandidates.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
    osascriptDialog("Python not found.\n\nInstall it:\n  brew install python@3.13\n\nThen relaunch the app.")
    exit(1)
}

// ── Build native binary if missing ──
let binaryURL = projectDir.appendingPathComponent("native/.build/release/coreaudio_tap")
if !FileManager.default.fileExists(atPath: binaryURL.path) {
    osascriptDialog("First launch: building the audio engine.\n\nClick OK — the app will open in about 1 minute.")
    let status = run("/usr/bin/swift", args: ["build", "-c", "release"],
                     cwd: projectDir.appendingPathComponent("native"))
    if status != 0 {
        osascriptDialog("Build failed.\n\nOpen Terminal and run:\n  cd \(projectDir.appendingPathComponent("native").path)\n  swift build -c release")
        exit(1)
    }
}

// ── Install Python packages if missing ──
let check = run(python, args: ["-c", "import flask, groq, sounddevice, numpy, webview"], silent: true)
if check != 0 {
    run(python, args: ["-m", "pip", "install", "--break-system-packages",
                       "-r", projectDir.appendingPathComponent("requirements.txt").path],
        silent: true)
}

// ── Launch app ──
let process = Process()
process.executableURL = URL(fileURLWithPath: python)
process.arguments = [projectDir.appendingPathComponent("app.py").path]
process.currentDirectoryURL = projectDir
try! process.run()
process.waitUntilExit()
