import Foundation

let exe = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
let projectDir = exe
    .deletingLastPathComponent() // MacOS/
    .deletingLastPathComponent() // Contents/
    .deletingLastPathComponent() // .app/
    .deletingLastPathComponent() // project dir

let process = Process()
process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/python3.13")
process.arguments = [projectDir.appendingPathComponent("app.py").path]
process.currentDirectoryURL = projectDir
try! process.run()
process.waitUntilExit()
