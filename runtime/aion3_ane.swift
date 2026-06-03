// Aion-1.0-Instruct ANE runtime.
//
// Compile:
//   swiftc -O runtime/aion3_ane.swift -framework CoreML -framework Foundation -o runtime/aion3_ane_runtime
//
// Run:
//   ./runtime/aion3_ane_runtime --meta models/aion/ane/aion_runtime_meta.json --prompt-ids 1,2,3 --max-new 32
//
// Serve mode (JSONL):
//   ./runtime/aion3_ane_runtime --meta models/aion/ane/aion_runtime_meta.json --serve
//   stdin:  {"prompt_ids":[1,2,3],"max_new":32}
//   stdout: {"ok":true,"generated_ids":[...],"timing":{...}}

import CoreML
import Foundation

struct AionRuntimeMeta: Decodable {
    let modelFamily: String
    let dModel: Int
    let vocabSize: Int
    let maxSeqLen: Int
    let ropeDim: Int
    let ropeTheta: Double
    let eosTokenIds: [Int]
    let bosTokenId: Int?
    let embedBin: String
    let ropeCosBin: String?
    let ropeSinBin: String?
    let coremlCompiled: String
    let outputKind: String?

    enum CodingKeys: String, CodingKey {
        case modelFamily = "model_family"
        case dModel = "d_model"
        case vocabSize = "vocab_size"
        case maxSeqLen = "max_seq_len"
        case ropeDim = "rope_dim"
        case ropeTheta = "rope_theta"
        case eosTokenIds = "eos_token_ids"
        case bosTokenId = "bos_token_id"
        case embedBin = "embed_bin"
        case ropeCosBin = "rope_cos_bin"
        case ropeSinBin = "rope_sin_bin"
        case coremlCompiled = "coreml_compiled"
        case outputKind = "output_kind"
    }
}

struct AionServeRequest: Decodable {
    let promptIds: [Int]
    let maxNew: Int?
    let profile: Bool?
    let stopRepeatNgram: Int?

    enum CodingKeys: String, CodingKey {
        case promptIds = "prompt_ids"
        case maxNew = "max_new"
        case profile
        case stopRepeatNgram = "stop_repeat_ngram"
    }
}

struct AionServeTiming: Encodable {
    let prefillTokens: Int
    let prefillSeconds: Double
    let decodeTokens: Int
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double

    enum CodingKeys: String, CodingKey {
        case prefillTokens = "prefill_tokens"
        case prefillSeconds = "prefill_s"
        case decodeTokens = "decode_tokens"
        case decodeSeconds = "decode_s"
        case decodeTokensPerSecond = "decode_tok_s"
    }
}

struct AionServeResponse: Encodable {
    let ok: Bool
    let generatedIds: [Int]?
    let timing: AionServeTiming?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case generatedIds = "generated_ids"
        case timing
        case error
    }
}

final class FP16BinaryFile {
    let data: Data
    let count: Int

    init(path: String, expectedCount: Int) throws {
        let url = URL(fileURLWithPath: path)
        data = try Data(contentsOf: url, options: .mappedIfSafe)
        count = data.count / MemoryLayout<Float16>.size
        precondition(count == expectedCount, "\(path): got \(count) fp16 values, expected \(expectedCount)")
    }

    func writeRow(_ index: Int, dim: Int, into ptr: UnsafeMutablePointer<Float16>) {
        precondition(index >= 0 && index * dim + dim <= count, "token id \(index) out of embedding bounds")
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.assumingMemoryBound(to: Float16.self)
            memcpy(ptr, src + index * dim, dim * MemoryLayout<Float16>.size)
        }
    }
}

func resolvePath(_ relative: String, relativeTo metaPath: String) -> String {
    if relative.hasPrefix("/") { return relative }
    let base = (metaPath as NSString).deletingLastPathComponent
    return (base as NSString).appendingPathComponent(relative)
}

func buildFallbackRoPETable(maxSeqLen: Int, ropeHalf: Int, theta: Double, useSin: Bool) -> [Float16] {
    var table = [Float16](repeating: 0, count: maxSeqLen * ropeHalf)
    for pos in 0..<maxSeqLen {
        let base = pos * ropeHalf
        for j in 0..<ropeHalf {
            let inv = 1.0 / pow(theta, Double(j) / Double(ropeHalf))
            let angle = Double(pos) * inv
            table[base + j] = Float16(useSin ? sin(angle) : cos(angle))
        }
    }
    return table
}

func printStderr(_ message: String) {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
}

@available(macOS 15.0, *)
final class AionRuntime {
    let meta: AionRuntimeMeta
    let embed: FP16BinaryFile
    let ropeCos: FP16BinaryFile?
    let ropeSin: FP16BinaryFile?
    let model: MLModel

    let d: Int
    let vocab: Int
    let maxSeqLen: Int
    let ropeHalf: Int

    let xArr: MLMultiArray
    let cosArr: MLMultiArray
    let sinArr: MLMultiArray
    let attnMaskArr: MLMultiArray
    let kvWriteMaskArr: MLMultiArray

    let xPtr: UnsafeMutablePointer<Float16>
    let cosPtr: UnsafeMutablePointer<Float16>
    let sinPtr: UnsafeMutablePointer<Float16>
    let attnMaskPtr: UnsafeMutablePointer<Float16>
    let kvWriteMaskPtr: UnsafeMutablePointer<Float16>

    let fallbackRopeCosTable: [Float16]?
    let fallbackRopeSinTable: [Float16]?
    let baseAttnMask: [Float16]
    let baseKvWriteMask: [Float16]

    let layerProvider: MLDictionaryFeatureProvider

    init(meta: AionRuntimeMeta, embed: FP16BinaryFile, ropeCos: FP16BinaryFile?, ropeSin: FP16BinaryFile?, model: MLModel) throws {
        self.meta = meta
        self.embed = embed
        self.ropeCos = ropeCos
        self.ropeSin = ropeSin
        self.model = model
        d = meta.dModel
        vocab = meta.vocabSize
        maxSeqLen = meta.maxSeqLen
        ropeHalf = max(1, meta.ropeDim / 2)

        xArr = try MLMultiArray(shape: [1, NSNumber(value: d), 1, 1], dataType: .float16)
        cosArr = try MLMultiArray(shape: [1, NSNumber(value: ropeHalf)], dataType: .float16)
        sinArr = try MLMultiArray(shape: [1, NSNumber(value: ropeHalf)], dataType: .float16)
        attnMaskArr = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: maxSeqLen)], dataType: .float16)
        kvWriteMaskArr = try MLMultiArray(shape: [1, 1, NSNumber(value: maxSeqLen), 1], dataType: .float16)

        xPtr = xArr.dataPointer.assumingMemoryBound(to: Float16.self)
        cosPtr = cosArr.dataPointer.assumingMemoryBound(to: Float16.self)
        sinPtr = sinArr.dataPointer.assumingMemoryBound(to: Float16.self)
        attnMaskPtr = attnMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)
        kvWriteMaskPtr = kvWriteMaskArr.dataPointer.assumingMemoryBound(to: Float16.self)

        if ropeCos == nil || ropeSin == nil {
            fallbackRopeCosTable = buildFallbackRoPETable(maxSeqLen: maxSeqLen, ropeHalf: ropeHalf, theta: meta.ropeTheta, useSin: false)
            fallbackRopeSinTable = buildFallbackRoPETable(maxSeqLen: maxSeqLen, ropeHalf: ropeHalf, theta: meta.ropeTheta, useSin: true)
        } else {
            fallbackRopeCosTable = nil
            fallbackRopeSinTable = nil
        }

        baseAttnMask = [Float16](repeating: Float16(-65504), count: maxSeqLen)
        baseKvWriteMask = [Float16](repeating: 0, count: maxSeqLen)

        layerProvider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: xArr),
            "rope_cos": MLFeatureValue(multiArray: cosArr),
            "rope_sin": MLFeatureValue(multiArray: sinArr),
            "attn_mask": MLFeatureValue(multiArray: attnMaskArr),
            "kv_write_mask": MLFeatureValue(multiArray: kvWriteMaskArr),
        ])

        resetMasks()
    }

    func makeState() -> MLState {
        model.makeState()
    }

    func resetMasks() {
        _ = baseAttnMask.withUnsafeBufferPointer {
            memcpy(attnMaskPtr, $0.baseAddress!, maxSeqLen * MemoryLayout<Float16>.size)
        }
        _ = baseKvWriteMask.withUnsafeBufferPointer {
            memcpy(kvWriteMaskPtr, $0.baseAddress!, maxSeqLen * MemoryLayout<Float16>.size)
        }
    }

    func isEOS(_ token: Int) -> Bool {
        meta.eosTokenIds.contains(token)
    }

    func argmaxToken(from logits: MLMultiArray) -> Int {
        let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
        let stride = logits.strides.count > 1 ? Int(truncating: logits.strides[1]) : 1
        var bestToken = 0
        var bestScore = -Float.infinity
        for i in 0..<vocab {
            let value = Float(ptr[i * stride])
            if value > bestScore {
                bestScore = value
                bestToken = i
            }
        }
        return bestToken
    }

    func nextToken(from value: MLMultiArray) -> Int {
        switch value.dataType {
        case .float32:
            let ptr = value.dataPointer.assumingMemoryBound(to: Float32.self)
            return Int(ptr[0].rounded())
        case .float64:
            let ptr = value.dataPointer.assumingMemoryBound(to: Double.self)
            return Int(ptr[0].rounded())
        case .int32:
            let ptr = value.dataPointer.assumingMemoryBound(to: Int32.self)
            return Int(ptr[0])
        default:
            return Int(truncating: value[0])
        }
    }

    func hasRepeatedSuffix(_ tokens: [Int], ngram: Int) -> Bool {
        if ngram <= 0 || tokens.count < ngram * 2 { return false }
        let suffixStart = tokens.count - ngram
        for start in 0..<suffixStart {
            var matches = true
            for offset in 0..<ngram {
                if tokens[start + offset] != tokens[suffixStart + offset] {
                    matches = false
                    break
                }
            }
            if matches { return true }
        }
        return false
    }

    func forwardOne(tokenId: Int, pos: Int, cacheSeqLen: inout Int, state: MLState,
                    needLogits: Bool = true) throws -> Int {
        embed.writeRow(tokenId, dim: d, into: xPtr)
        precondition(pos >= 0 && pos < maxSeqLen)
        if let ropeCos = ropeCos, let ropeSin = ropeSin {
            ropeCos.writeRow(pos, dim: ropeHalf, into: cosPtr)
            ropeSin.writeRow(pos, dim: ropeHalf, into: sinPtr)
        } else {
            let base = pos * ropeHalf
            _ = fallbackRopeCosTable!.withUnsafeBufferPointer {
                memcpy(cosPtr, $0.baseAddress! + base, ropeHalf * MemoryLayout<Float16>.size)
            }
            _ = fallbackRopeSinTable!.withUnsafeBufferPointer {
                memcpy(sinPtr, $0.baseAddress! + base, ropeHalf * MemoryLayout<Float16>.size)
            }
        }

        attnMaskPtr[cacheSeqLen] = 0
        if cacheSeqLen > 0 { kvWriteMaskPtr[cacheSeqLen - 1] = 0 }
        kvWriteMaskPtr[cacheSeqLen] = 1

        let out = try model.prediction(from: layerProvider, using: state)
        cacheSeqLen += 1
        if !needLogits { return -1 }

        if meta.outputKind == "argmax" {
            guard let token = out.featureValue(for: "next_token")?.multiArrayValue else {
                throw NSError(domain: "AionRuntime", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "missing next_token output"])
            }
            return nextToken(from: token)
        }

        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            throw NSError(domain: "AionRuntime", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "missing logits output"])
        }
        return argmaxToken(from: logits)
    }

    func runGeneration(promptIds: [Int], maxNew: Int, state: MLState, stopRepeatNgram: Int = 0) throws -> (generated: [Int], timing: AionServeTiming) {
        precondition(!promptIds.isEmpty)
        precondition(promptIds.count + maxNew <= maxSeqLen)

        resetMasks()
        var cacheSeqLen = 0

        let prefillStart = CFAbsoluteTimeGetCurrent()
        var next = -1
        for idx in 0..<promptIds.count {
            let isLast = idx == promptIds.count - 1
            next = try forwardOne(tokenId: promptIds[idx], pos: cacheSeqLen, cacheSeqLen: &cacheSeqLen, state: state, needLogits: isLast)
        }
        let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart

        var generated = [next]
        let decodeStart = CFAbsoluteTimeGetCurrent()
        while generated.count < maxNew && !isEOS(generated.last!) {
            let tok = try forwardOne(tokenId: generated.last!, pos: cacheSeqLen, cacheSeqLen: &cacheSeqLen, state: state, needLogits: true)
            generated.append(tok)
            if hasRepeatedSuffix(generated, ngram: stopRepeatNgram) { break }
        }
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart
        let decodeTokens = max(0, generated.count - 1)

        let timing = AionServeTiming(
            prefillTokens: promptIds.count,
            prefillSeconds: prefillSeconds,
            decodeTokens: decodeTokens,
            decodeSeconds: decodeSeconds,
            decodeTokensPerSecond: decodeTokens > 0 && decodeSeconds > 0 ? Double(decodeTokens) / decodeSeconds : 0
        )
        return (generated, timing)
    }
}

func encodeJSON<T: Encodable>(_ value: T) throws -> String {
    let data = try JSONEncoder().encode(value)
    guard let s = String(data: data, encoding: .utf8) else {
        throw NSError(domain: "AionRuntime", code: 1, userInfo: [NSLocalizedDescriptionKey: "json encode failed"])
    }
    return s
}

@available(macOS 15.0, *)
func main() throws {
    var metaPath = "models/aion/ane/aion_runtime_meta.json"
    var promptIds = [1]
    var maxNew = 16
    var warmupCalls = 0
    var stopRepeatNgram = 0
    var serve = false

    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < args.count {
        switch args[i] {
        case "--meta":
            metaPath = args[i + 1]; i += 2
        case "--prompt-ids":
            promptIds = args[i + 1].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            i += 2
        case "--max-new":
            maxNew = Int(args[i + 1]) ?? 16; i += 2
        case "--warmup":
            warmupCalls = Int(args[i + 1]) ?? 0; i += 2
        case "--stop-repeat-ngram":
            stopRepeatNgram = Int(args[i + 1]) ?? 0; i += 2
        case "--serve":
            serve = true; i += 1
        default:
            i += 1
        }
    }

    let metaData = try Data(contentsOf: URL(fileURLWithPath: metaPath))
    let meta = try JSONDecoder().decode(AionRuntimeMeta.self, from: metaData)

    let embedPath = resolvePath(meta.embedBin, relativeTo: metaPath)
    let embed = try FP16BinaryFile(path: embedPath, expectedCount: meta.vocabSize * meta.dModel)

    let ropeExpectedCount = meta.maxSeqLen * max(1, meta.ropeDim / 2)
    let ropeCos = try meta.ropeCosBin.map { try FP16BinaryFile(path: resolvePath($0, relativeTo: metaPath), expectedCount: ropeExpectedCount) }
    let ropeSin = try meta.ropeSinBin.map { try FP16BinaryFile(path: resolvePath($0, relativeTo: metaPath), expectedCount: ropeExpectedCount) }

    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine
    let modelPath = resolvePath(meta.coremlCompiled, relativeTo: metaPath)
    let model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: cfg)

    let runtime = try AionRuntime(meta: meta, embed: embed, ropeCos: ropeCos, ropeSin: ropeSin, model: model)

    if warmupCalls > 0 {
        printStderr("Warming up \(warmupCalls) call(s)…")
        for _ in 0..<warmupCalls {
            let state = runtime.makeState()
            _ = try runtime.runGeneration(promptIds: [meta.bosTokenId ?? 1], maxNew: 1, state: state)
        }
        printStderr("Warmup done.")
    }

    if serve {
        print("READY {\"protocol\":\"aion-jsonl-v1\"}")
        fflush(stdout)
        while let line = readLine() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            do {
                let req = try JSONDecoder().decode(AionServeRequest.self, from: Data(trimmed.utf8))
                let state = runtime.makeState()
                let result = try runtime.runGeneration(
                    promptIds: req.promptIds,
                    maxNew: req.maxNew ?? 32,
                    state: state,
                    stopRepeatNgram: req.stopRepeatNgram ?? 0
                )
                let resp = AionServeResponse(ok: true, generatedIds: result.generated, timing: result.timing, error: nil)
                print(try encodeJSON(resp))
            } catch {
                let resp = AionServeResponse(ok: false, generatedIds: nil, timing: nil, error: error.localizedDescription)
                print((try? encodeJSON(resp)) ?? "{\"ok\":false,\"error\":\"encode failed\"}")
            }
            fflush(stdout)
        }
    } else {
        let state = runtime.makeState()
        let result = try runtime.runGeneration(promptIds: promptIds, maxNew: maxNew, state: state, stopRepeatNgram: stopRepeatNgram)
        let resp = AionServeResponse(ok: true, generatedIds: result.generated, timing: result.timing, error: nil)
        print(try encodeJSON(resp))
    }
}

if #available(macOS 15.0, *) {
    do {
        try main()
    } catch {
        printStderr("ERROR: \(error.localizedDescription)")
        exit(1)
    }
} else {
    printStderr("ERROR: macOS 15+ required for stateful CoreML")
    exit(1)
}
