import Foundation
import Vision
import CoreGraphics
import ImageIO

// Usage: facebox <input.jpg>
// Prints the face bounding box in pixel coords (origin top-left), plus the
// vertical position of the chin from the landmark points when available.

let args = CommandLine.arguments
guard args.count >= 2,
      let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[1]) as CFURL, nil),
      let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
    FileHandle.standardError.write("cannot read image\n".data(using: .utf8)!)
    exit(1)
}
let W = CGFloat(cg.width), H = CGFloat(cg.height)

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
let request = VNDetectFaceLandmarksRequest()
try handler.perform([request])

guard let faces = request.results, !faces.isEmpty else {
    print("no faces found")
    exit(2)
}
// largest face
let face = faces.max { a, b in
    a.boundingBox.width * a.boundingBox.height < b.boundingBox.width * b.boundingBox.height
}!

let bb = face.boundingBox   // normalized, origin bottom-left
let px = bb.minX * W
let py = (1 - bb.maxY) * H          // top edge in top-left coords
let pw = bb.width * W
let ph = bb.height * H
print(String(format: "face bbox px: x=%.0f y=%.0f w=%.0f h=%.0f", px, py, pw, ph))

if let lm = face.landmarks {
    func pts(_ region: VNFaceLandmarkRegion2D?) -> [CGPoint] {
        guard let r = region else { return [] }
        return r.normalizedPoints.map { p in
            CGPoint(x: (bb.minX + CGFloat(p.x) * bb.width) * W,
                    y: (1 - (bb.minY + CGFloat(p.y) * bb.height)) * H)
        }
    }
    let facePts = pts(lm.faceContour)
    if !facePts.isEmpty {
        let chinY = facePts.map { $0.y }.max()!
        let chinX = facePts.filter { $0.y > chinY - 6 }.map { $0.x }
        let cx = chinX.reduce(0, +) / CGFloat(chinX.count)
        print(String(format: "chin apex: x=%.0f y=%.0f", cx, chinY))
    }
    let eyePts = pts(lm.leftEye) + pts(lm.rightEye)
    if !eyePts.isEmpty {
        let ex = eyePts.map { $0.x }
        print(String(format: "eye line: x %.0f..%.0f  y avg %.0f",
                     ex.min()!, ex.max()!, eyePts.map { $0.y }.reduce(0, +) / CGFloat(eyePts.count)))
    }
    // eye centres (pupils) - use the centre of each eye region
    for (name, region) in [("leftEye", lm.leftEye), ("rightEye", lm.rightEye)] {
        let p = pts(region)
        if !p.isEmpty {
            let xs = p.map { $0.x }, ys = p.map { $0.y }
            let cx = xs.reduce(0, +) / CGFloat(p.count)
            let cy = ys.reduce(0, +) / CGFloat(p.count)
            print(String(format: "%@ centre: x=%.0f y=%.0f  xrange %.0f..%.0f  yrange %.0f..%.0f",
                         name, cx, cy, xs.min()!, xs.max()!, ys.min()!, ys.max()!))
        }
    }
}
print(String(format: "image px: %0.fx%0.f", W, H))
