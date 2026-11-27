// SAMPipeline.swift
// Runs the exported EfficientViT-SAM Core ML encoder + decoder on an image.
// Preprocessing matches export/export_coreml.py and team_sam/coreml_backend.py:
//   resize longest side to 512, pad bottom/right to 512x512 with RGB(124,116,104),
//   prompts in the 1024-long-side frame, padding point (0,0,-1) when there is no box.

import CoreML
import CoreVideo
import UIKit

enum SAMError: Error { case pixelBuffer, output }

struct Prompt {
    var points: [CGPoint] = []      // original-image pixel coordinates
    var labels: [Float] = []        // 1 = positive, 0 = negative
    var box: CGRect? = nil          // original-image pixel coordinates
}

final class SAMPipeline {
    static let encSize = 512
    static let frame: CGFloat = 1024
    static let padColor = UIColor(red: 124 / 255, green: 116 / 255, blue: 104 / 255, alpha: 1)

    let encoder: MLModel
    let decoder: MLModel
    private(set) var embedding: MLMultiArray?
    private(set) var originalSize: CGSize = .zero

    /// encoderURL / decoderURL: compiled .mlmodelc URLs (Xcode compiles .mlpackage files
    /// added to the target; use Bundle.main.url(forResource:withExtension: "mlmodelc")).
    init(encoderURL: URL, decoderURL: URL, computeUnits: MLComputeUnits) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        encoder = try MLModel(contentsOf: encoderURL, configuration: cfg)
        decoder = try MLModel(contentsOf: decoderURL, configuration: cfg)
    }

    // MARK: Encoder

    /// Expects an `.up`-oriented image (normalize orientation of camera photos first).
    func preprocess(_ image: UIImage) throws -> CVPixelBuffer {
        guard let cgIn = image.cgImage else { throw SAMError.pixelBuffer }
        originalSize = CGSize(width: cgIn.width, height: cgIn.height)  // pixels, not points
        let s = CGFloat(Self.encSize) / max(originalSize.width, originalSize.height)
        let nw = (originalSize.width * s).rounded(), nh = (originalSize.height * s).rounded()
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        CVPixelBufferCreate(nil, Self.encSize, Self.encSize, kCVPixelFormatType_32BGRA, attrs, &pb)
        guard let buf = pb else { throw SAMError.pixelBuffer }
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buf), width: Self.encSize, height: Self.encSize,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { throw SAMError.pixelBuffer }
        let cg = cgIn
        ctx.setFillColor(Self.padColor.cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: Self.encSize, height: Self.encSize))
        ctx.interpolationQuality = .medium
        // CGContext origin is bottom-left: draw at the top so padding is bottom/right.
        ctx.draw(cg, in: CGRect(x: 0, y: CGFloat(Self.encSize) - nh, width: nw, height: nh))
        return buf
    }

    /// Encodes a new image and caches its embedding (call once per photo).
    func encode(_ pixelBuffer: CVPixelBuffer) throws {
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)])
        let out = try encoder.prediction(from: input)
        guard let emb = out.featureValue(for: "image_embeddings")?.multiArrayValue else { throw SAMError.output }
        embedding = emb
    }

    // MARK: Decoder

    private func toFrame(_ p: CGPoint) -> (Float, Float) {
        let s = Self.frame / max(originalSize.width, originalSize.height)
        return (Float(p.x * s), Float(p.y * s))
    }

    /// Returns (best low-res 256x256 logits, predicted IoU). multimask: pick best of outputs 1...3.
    func decode(_ prompt: Prompt, multimask: Bool) throws -> (MLMultiArray, Int, Float) {
        guard let emb = embedding else { throw SAMError.output }
        var pts = prompt.points.map(toFrame)
        var labs = prompt.labels
        if let b = prompt.box {
            pts += [toFrame(CGPoint(x: b.minX, y: b.minY)), toFrame(CGPoint(x: b.maxX, y: b.maxY))]
            labs += [2, 3]
        } else {
            pts.append((0, 0)); labs.append(-1)
        }
        let n = pts.count
        let coords = try MLMultiArray(shape: [1, NSNumber(value: n), 2], dataType: .float32)
        let labels = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .float32)
        for i in 0..<n {
            coords[[0, i, 0] as [NSNumber]] = NSNumber(value: pts[i].0)
            coords[[0, i, 1] as [NSNumber]] = NSNumber(value: pts[i].1)
            labels[[0, i] as [NSNumber]] = NSNumber(value: labs[i])
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image_embeddings": MLFeatureValue(multiArray: emb),
            "point_coords": MLFeatureValue(multiArray: coords),
            "point_labels": MLFeatureValue(multiArray: labels),
        ])
        let out = try decoder.prediction(from: input)
        guard let masks = out.featureValue(for: "low_res_masks")?.multiArrayValue,
              let ious = out.featureValue(for: "iou_predictions")?.multiArrayValue else { throw SAMError.output }
        let candidates = multimask ? Array(1...3) : [0]
        let best = candidates.max { ious[[0, $0] as [NSNumber]].floatValue < ious[[0, $1] as [NSNumber]].floatValue }!
        return (masks, best, ious[[0, best] as [NSNumber]].floatValue)
    }

    // MARK: Mask -> displayable image

    /// Thresholds the selected 256x256 logits over the valid (unpadded) region and scales
    /// to the original image size. Nearest-neighbour + CoreGraphics scaling; close to,
    /// but not bit-identical with, the bilinear-logit upsampling used in the Python eval.
    func maskImage(_ masks: MLMultiArray, index: Int) -> UIImage {
        let side = 256
        let s = CGFloat(side) / max(originalSize.width, originalSize.height)
        let vw = max(1, Int((originalSize.width * s).rounded())), vh = max(1, Int((originalSize.height * s).rounded()))
        var bytes = [UInt8](repeating: 0, count: vw * vh)
        let ptr = masks.dataPointer
        let strides = masks.strides.map { $0.intValue }
        let isF16 = masks.dataType == .float16
        for y in 0..<vh {
            for x in 0..<vw {
                let off = index * strides[1] + y * strides[2] + x * strides[3]
                let v: Float = isF16
                    ? Float(ptr.assumingMemoryBound(to: Float16.self)[off])
                    : ptr.assumingMemoryBound(to: Float.self)[off]
                bytes[y * vw + x] = v > 0 ? 255 : 0
            }
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        let small = CGImage(width: vw, height: vh, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: vw,
                            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
        let fmt = UIGraphicsImageRendererFormat.default()
        fmt.scale = 1  // originalSize is in pixels
        let r = UIGraphicsImageRenderer(size: originalSize, format: fmt)
        return r.image { ctx in
            ctx.cgContext.interpolationQuality = .none
            UIImage(cgImage: small).draw(in: CGRect(origin: .zero, size: originalSize))  // handles UIKit flip
        }
    }
}
