// BenchView.swift
// Minimal SwiftUI screen: pick encoder variant, run the benchmark, show median/p95, share the CSV.
// Add bundled test images named bench_0.jpg ... bench_N.jpg to the app target.

import SwiftUI

struct BenchView: View {
    @State private var status = "Idle"
    @State private var stats: [LatencyStats] = []
    @State private var csvURL: URL?
    @State private var running = false
    @State private var encoderName = "L0_encoder_fp16"

    private func loadImages() -> [UIImage] {
        (0..<50).compactMap { i in
            Bundle.main.url(forResource: "bench_\(i)", withExtension: "jpg").flatMap { UIImage(contentsOfFile: $0.path) }
        }
    }

    var body: some View {
        NavigationStack {
            List {
                Section("Model") {
                    TextField("Encoder model name", text: $encoderName)
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                    Button(running ? "Running…" : "Run benchmark") { start() }.disabled(running)
                    Text(status).font(.footnote).foregroundStyle(.secondary)
                    if let u = csvURL { ShareLink("Share CSV", item: u) }
                }
                ForEach(Array(Set(stats.map(\.computeUnits))).sorted(), id: \.self) { cu in
                    Section(cu) {
                        ForEach(stats.filter { $0.computeUnits == cu }) { s in
                            HStack {
                                Text(s.metric)
                                Spacer()
                                Text(String(format: "p50 %.1f  p95 %.1f ms", s.median, s.p95)).monospacedDigit()
                            }
                        }
                    }
                }
            }
            .navigationTitle("TEAM bench")
        }
    }

    private func start() {
        running = true
        stats = []
        let images = loadImages()
        var cfg = Benchmark.Config()
        cfg.encoderName = encoderName
        DispatchQueue.global(qos: .userInitiated).async {
            let bench = Benchmark(config: cfg, images: images)
            do {
                guard !images.isEmpty else { throw NSError(domain: "TEAMBench", code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "No bench_*.jpg images in bundle"]) }
                let r = try bench.run { msg in DispatchQueue.main.async { status = msg } }
                let url = try bench.saveCSV()
                DispatchQueue.main.async { stats = r; csvURL = url; status = "Done: \(images.count) images"; running = false }
            } catch {
                DispatchQueue.main.async { status = "Error: \(error.localizedDescription)"; running = false }
            }
        }
    }
}
