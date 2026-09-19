import Foundation
import Vision
import CoreImage
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// Usage: segment <input.jpg> <output-mask.png>
// Writes an 8-bit grayscale soft-alpha mask (255 = subject, 0 = background).

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: segment <in> <out-mask.png>\n".data(using: .utf8)!)
    exit(1)
}
let inURL = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])

guard let src = CGImageSourceCreateWithURL(inURL as CFURL, nil),
      let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
    FileHandle.standardError.write("cannot read input image\n".data(using: .utf8)!)
    exit(2)
}

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
let request = VNGenerateForegroundInstanceMaskRequest()
do {
    try handler.perform([request])
} catch {
    FileHandle.standardError.write("vision failed: \(error)\n".data(using: .utf8)!)
    exit(3)
}

guard let observation = request.results?.first else {
    FileHandle.standardError.write("no foreground instance found\n".data(using: .utf8)!)
    exit(4)
}

let pixelBuffer: CVPixelBuffer
do {
    pixelBuffer = try observation.generateScaledMaskForImage(
        forInstances: observation.allInstances, from: handler)
} catch {
    FileHandle.standardError.write("mask generation failed: \(error)\n".data(using: .utf8)!)
    exit(5)
}

let ciContext = CIContext()
let maskImage = CIImage(cvPixelBuffer: pixelBuffer)
guard let maskCG = ciContext.createCGImage(maskImage, from: maskImage.extent) else {
    FileHandle.standardError.write("mask -> cgimage failed\n".data(using: .utf8)!)
    exit(6)
}

guard let dest = CGImageDestinationCreateWithURL(
        outURL as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    FileHandle.standardError.write("cannot create output\n".data(using: .utf8)!)
    exit(7)
}
CGImageDestinationAddImage(dest, maskCG, nil)
guard CGImageDestinationFinalize(dest) else {
    FileHandle.standardError.write("write failed\n".data(using: .utf8)!)
    exit(8)
}

print("ok \(maskCG.width)x\(maskCG.height)")
