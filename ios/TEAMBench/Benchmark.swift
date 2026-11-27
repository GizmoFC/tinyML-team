// Benchmark.swift
// On-device latency benchmark (proposal Sec. 3.1 / 3.3).
//
// For each compute-unit setting it measures, per image:
//   encoder      : encoder.prediction only
//   decoder      : one-click decoder.prediction only (embedding cached)
//   first_mask   : uncached image + available prompt -> displayable mask image
//                  (preprocess + encoder + decoder + mask rendering)
//   repeat_prompt: new click on the same image (decoder + mask rendering, embedding reused)
// Warm-up runs are discarded. Images are interleaved across trials to spread out
// run-order and thermal effects. Results: median and p95 (ms) plus a CSV of raw samples.

import CoreML
import Foundation
import UIKit

struct LatencyStats: Identifiable {
    let id = UUID()
    let computeUnits: String
    let metric: String
    let median: Double
    let p95: Double
    let n: Int
}

func computeUnitsName(_ cu: MLComputeUnits) -> String {
    switch cu {
    case .cpuOnly: return "cpuOnly"
    case .cpuAndGPU: return "cpuAndGPU"
    case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
    case .all: return "all"
    @unknown default: return "unknown"
    }
}

private func percentile(_ xs: [Double], _ p: Double) -> Double {
    let s = xs.sorted()
    guard !s.isEmpty else { return .nan }
    let r = p * Double(s.count - 1)
    let lo = Int(r.rounded(.down)), hi = Int(r.rounded(.up))
    return s[lo] + (s[hi] - s[lo]) * (r - Double(lo))
}

private func ms(_ block: () throws -> Void) rethrows -> Double {
    let t0 = DispatchTime.now().uptimeNanoseconds
    try block()
    return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}

final class Benchmark {
    struct Config {
        var encoderName = "L0_encoder_fp16"
        var decoderName = "L0_decoder_fp16"
        var computeUnits: [MLComputeUnits] = [.cpuOnly, .cpuAndGPU, .all]
        var warmup = 5
        var trials = 30   // timed passes over the image set
    }

    let config: Config
    let images: [UIImage]
    private(set) var csv = "compute_units,metric,trial,image,ms\n"

    init(config: Config = Config(), images: [UIImage]) {
        self.config = config
        self.images = images
    }

    private func url(_ name: String) throws -> URL {
        if let u = Bundle.main.url(forResource: name, withExtension: "mlmodelc") { return u }
        throw NSError(domain: "TEAMBench", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(name).mlmodelc not in bundle"])
    }

    /// Center click: a fixed, image-independent prompt so timing does not depend on labels.
    private func click(for img: UIImage, offset: CGFloat = 0) -> Prompt {
        let w = CGFloat(img.cgImage!.width), h = CGFloat(img.cgImage!.height)
        return Prompt(points: [CGPoint(x: w * (0.5 + offset), y: h * (0.5 + offset))], labels: [1])
    }

    func run(progress: (String) -> Void) throws -> [LatencyStats] {
        var results: [LatencyStats] = []
        for cu in config.computeUnits {
            let name = computeUnitsName(cu)
            progress("Loading models (\(name))")
            var loadMs = 0.0
            var pipe: SAMPipeline!
            loadMs = try ms { pipe = try SAMPipeline(encoderURL: url(config.encoderName),
                                                     decoderURL: url(config.decoderName), computeUnits: cu) }
            var samples: [String: [Double]] = ["encoder": [], "decoder": [], "first_mask": [], "repeat_prompt": []]

            for t in 0..<(config.warmup + config.trials) {
                let timed = t >= config.warmup
                progress("\(name): \(timed ? "trial \(t - config.warmup + 1)/\(config.trials)" : "warm-up \(t + 1)/\(config.warmup)")")
                // Rotate the starting image each trial (interleaving).
                for k in 0..<images.count {
                    let i = (k + t) % images.count
                    let img = images[i]
                    var encT = 0.0, decT = 0.0
                    let first = try ms {
                        let pb = try pipe.preprocess(img)
                        encT = try ms { try pipe.encode(pb) }
                        var out: (MLMultiArray, Int, Float)!
                        decT = try ms { out = try pipe.decode(click(for: img), multimask: true) }
                        _ = pipe.maskImage(out.0, index: out.1)
                    }
                    let rep = try ms {
                        let out = try pipe.decode(click(for: img, offset: 0.1), multimask: true)
                        _ = pipe.maskImage(out.0, index: out.1)
                    }
                    if timed {
                        let tr = t - config.warmup
                        for (m, v) in [("encoder", encT), ("decoder", decT), ("first_mask", first), ("repeat_prompt", rep)] {
                            samples[m]!.append(v)
                            csv += "\(name),\(m),\(tr),\(i),\(String(format: "%.3f", v))\n"
                        }
                    }
                }
            }
            csv += "\(name),model_load,0,-1,\(String(format: "%.3f", loadMs))\n"
            for m in ["encoder", "decoder", "first_mask", "repeat_prompt"] {
                let xs = samples[m]!
                results.append(LatencyStats(computeUnits: name, metric: m,
                                            median: percentile(xs, 0.5), p95: percentile(xs, 0.95), n: xs.count))
            }
        }
        return results
    }

    /// Writes raw samples + device info to Documents/ and returns the file URL.
    func saveCSV() throws -> URL {
        let dev = UIDevice.current
        var sys = utsname(); uname(&sys)
        let model = withUnsafePointer(to: &sys.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        let header = "# device=\(model) ios=\(dev.systemVersion) encoder=\(config.encoderName) decoder=\(config.decoderName) " +
            "warmup=\(config.warmup) trials=\(config.trials) images=\(images.count) thermal=\(ProcessInfo.processInfo.thermalState.rawValue)\n"
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let file = dir.appendingPathComponent("bench_\(config.encoderName)_\(stamp).csv")
        try (header + csv).write(to: file, atomically: true, encoding: .utf8)
        return file
    }
}
