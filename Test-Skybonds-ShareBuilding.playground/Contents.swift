import Swift
import Foundation


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: - Data Generation:
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

protocol ValuesGeneratable {
    
    associatedtype Value: Comparable
    func generateSingle() -> Value
}

extension ValuesGeneratable {
    
    func generate(count: Int = GeneratorDefaults.count) -> [Value] {
        return (0..<count).map { _ in generateSingle() }
    }
}

struct GeneratorDefaults {
    
    static let count: Int = 100_000
    static let range: ClosedRange<Double> = 1...10000000
}

struct DoubleGenerator: ValuesGeneratable {
    
    let range: ClosedRange<Double>
    
    init(range: ClosedRange<Double> = GeneratorDefaults.range) {
        self.range = range
    }
    
    let strangeNumberSequence: [Double] = [15.25, 1.0, 2.1, 1567.29, 1589.0, 20000123.2, 0.5123, 0.2, 2345252264.12389128922891289238923, 0.0000000123891238128933]
    
    func generateSingle() -> Double {
        return strangeNumberSequence.randomElement()!
    }
}

struct DoubleGrouppedGenerator: ValuesGeneratable {
    
    struct CustomGroup {
        let range: ClosedRange<Double>
        let probability: Double
    }
    
    private let groups: [CustomGroup]
    
    init(customGroups: [CustomGroup]) {
        self.groups = customGroups
    }
    
    func generateSingle() -> Double {
        
        let randomProbability = Double.random(in: 0...1)
        
        var spentProbability = 0.0
        var iterator = groups.makeIterator()
        var cusrorGroup: CustomGroup?
        
        while let group = iterator.next() {
            cusrorGroup = group
            
            if spentProbability <= randomProbability && (spentProbability + group.probability) > randomProbability {
                break
            }
            spentProbability += group.probability
        }
        
        guard let foundGroup = cusrorGroup else {
            return Double.random(in: GeneratorDefaults.range)
        }
        
        let newValue = Double.random(in: foundGroup.range)
        return newValue
    }
}

struct DecimalGenerator<G: ValuesGeneratable>: ValuesGeneratable where G.Value == Double {

    private let generator: G
    init (_ generator: G) {
        self.generator = generator
    }

    func generateSingle() -> Decimal {
        Decimal(generator.generateSingle())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: - About used FloatingPoint Numbers:
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Decimal -> base-10 Arithmetic, more suitable for currency operations, have big accuracy in general, and give less error results
// (usage of 100 as a numerator in algorithm) m * 10 ^ exp, m -> 38 digits, exp = [-2^7 + 1 .. 2^7] (8 bits)
// Decimal Swift Implementation problem -> this is a bridge to NSDecimal, thus some problems from here: (dynamic method dispatch, additional heavy convertations between Double/Float)
// Total: Spent more memory (Decimal - 16 bytes) + calculations are longer

// Double -> base-2 Arithmetic, have big enough accuracy. Optimized for speed
// m * 10 ^ exp, m = [-2^47 + 1 .. 2^47] (48 bits), exp = [-2^10 + 1 .. 2^10] (11 bits)
// Total: Spent less memory (Double - 8 bytes) + calculations are faster

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: - Possible Improvements:
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Algorithmic optimizations:
    // - Maybe, exist optimizations, which allow limit with one Array enumeration, but it's a pity, I didn't find anything

// Processing - optimizations:
    // 1. On systems with big CPU cores count - processes of summations and multiplication could be paralleled (+ Applied)
    // 2. With really big quantity of numbers with Floating point, maybe, GPU will be better choice for processing
    // 3. Maybe, exist possible optimizations to different architectures of CPU

// Optimizations, related with Floating Numbers capabilities/traits:
    // 1. Overflow predictions / dynamic conversion to more accurate types, by necessity
        // significand / exponent / normalization for this targets
    // 2. Special implementations of Floating Numbers to target (Unbelievable)
    // 3. Only 3 digits after point is required, thus for speed optimizations, separate especially long values can be truncated
        // use Float (clip digits after point before processing). It can append some error to result, but not critical, and give essential speed improvements
    // 4. Double-Float optimizations are senseless, because of widespread x64 systems

// Optimizations, related with Data nature:
    // 1. If the data have repeated share values, that we can use memory Optimizations. By example, remember all previous computation results, store and fetch them from Dictionary
    // 2. It the data have some minimal share quantum, the task can be reduced to Integer sum and significantly simplier operations


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: - Resources evaluations for current Solutions:
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// When used Double: 10M -> 80MB (till 160MB after result obtaining, because mapping with multiplication used another memory buffer)
// When used Decimal: in 2 times more memory, at least
// the Solutions have line asymptotic complexity O(2N) ~ O(N)

// ++++++++++++++++++++++ WARNING +++++++++++++++++++++++
// Debug QuickLook in Playground due to Preview-engine for huge arrays give some serious performance drawbacks
// inputValues.reduce(0, +) 1M values: 0.05 - 0.07 sec
// inputValues.reduce(0, { $0 + $1 }}) 1M: 10+ sec
// https://www.avanderlee.com/optimization/measure-performance-code/

// Accordingly, Playground - isn't best place for Performance measurements. I didn't find any way to disable Debug QuickLook

protocol SharesCalculatable {
    
    associatedtype Value: Comparable
    func calculate(_ inputValues: [Value]) throws -> [Value]
}

struct SharesCalculators {}

extension SharesCalculators {
    
    enum Error: LocalizedError {
        
        case nonPositiveInputValues
        case excessInputValues (details: ExcessDetails)
        
        case allowedTimeIsSmall (allowed: TimeInterval)
        
        case keepDigitsCountRange (value: Int, allowedRange: ClosedRange<Int>)
        
        enum ExcessDetails: CustomStringConvertible {
            case notAllowedExpectations (allowed: TimeInterval, expected: TimeInterval)
            
            var description: String {
                
                switch self {
                case let .notAllowedExpectations(allowedTime, expectedTime):
                    return "ExpectedTime = \(expectedTime), but AllowedTime = \(allowedTime)"
                }
            }
        }
        
        var localizedDescription: String {
            switch self {
                
            case .nonPositiveInputValues:
                return "InputValues have non-Positive Numbers"
                
            case .excessInputValues(let details):
                return "InputValues are excess, details = \(details)"
                
            case .allowedTimeIsSmall(let allowedTime):
                return "AllowedTime is very small \(allowedTime). Maybe, TimeConstraints it is not your choice"
                
            case let .keepDigitsCountRange(value, allowedRange):
                return "keepDigitsCount = \(value), But allowed digits Range for Truncating - \(allowedRange)"
            }
        }
    }
}

extension SharesCalculators {
    
    struct SafeReadable: SharesCalculatable {
        
        func calculate(_ inputValues: [Decimal]) throws -> [Decimal] {
            
            // Safety check - yet one Array enumeration, increase execution time to ~30%
            // Instead "guard" can be used Iterator + condition, that help to not extra fetch buffer data
            // As a one strategy for invalid values handling - ignoring them, but without context - FAIL FAST

            guard inputValues.contains(where: { $0 <= 0 }) == false else {
                throw Error.nonPositiveInputValues
            }
            
            // Just about ~2 times slower, than with Double type
            
            let inputSum = inputValues.reduce(0, +)
            let shareEquivalent = Decimal(100.0) / inputSum

            let shareValues = inputValues.map { $0 * shareEquivalent }
            return shareValues
        }
    }
    
    struct UnsafeShortest: SharesCalculatable {
        
        func calculate(_ inputValues: [Double]) throws -> [Double] {
            
            // Reduce in comparison with Iterator - give losing 5-6% with default optimizations
            // but as briefly and succinctly as possible
            
            let inputSum = inputValues.reduce(0, +)
            let shareEquivalent = 100.0 / inputSum
            
            let shareValues = inputValues.map { $0 * shareEquivalent }
            return shareValues
        }
    }
    
    struct UnsafeFastest: SharesCalculatable {
        
        func calculate(_ inputValues: [Double]) throws -> [Double] {
            
            // Simple iterator - bestest results by speed
            
            var inputSum: Double = 0.0
            var iterator = inputValues.makeIterator()

            while let value = iterator.next() {
                inputSum += value
            }
            
            let shareEquivalent = 100.0 / inputSum

            let shareValues = inputValues.map { $0 * shareEquivalent }
            return shareValues
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: - Optimizations with Task paralleling to CPU physical cores:
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// This implementation give growth in 4-5 times (!!!) (with 6 physical CPU cores, by example)

extension Array where Element: SignedNumeric {
    
    func chunksForConcurrency() -> [[Element]] {
        
        // !!! Slice array along VM Page alignment
        
        let size = Int(getpagesize()) / (MemoryLayout<Element>.size / 8)
        let chunks = stride(from: 0, to: self.count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, self.count)])
        }
        return chunks
    }
}

// Work with all SignedNumber: Int, Float, Double, Decimal. etc...

extension Array where Element: Collection, Element.Element: SignedNumeric, Element.Element: Comparable {
    
    typealias Value = Element.Element
    
    func concurrentSumWithSafetyCheck() throws -> Value {
        
        // For Resource lock on Thread - many techniques exist
        // GCD Semaphore - one of most simpliest technique, the truth is not the most effective
        
        let lock = DispatchSemaphore(value: 1)
        var sum: Element.Element = 0
        
        // Used POSIX read-write Lock
        // As most effective solver to Read-Write problem
        // (GCD barrier isn't useful with concurrentPerform method)
        
        var allValuesFlagLocker = pthread_rwlock_t()
        pthread_rwlock_init(&allValuesFlagLocker, nil)
        
        var allValuesArePositive = true
        
        // DispatchGroup and Interactive concurrent queue could be used here,
        // but DispatchQueue have convinient static concurrentPerform Method
        
        DispatchQueue.concurrentPerform(iterations: self.count, execute: {
            
            pthread_rwlock_rdlock(&allValuesFlagLocker)
            guard allValuesArePositive == true else {
                
                pthread_rwlock_unlock(&allValuesFlagLocker)
                return
            }
            pthread_rwlock_unlock(&allValuesFlagLocker)
            
            let chunk = self[$0]

            var chunkSum: Value = 0
            var iterator = chunk.makeIterator()

            // Check injected inward for speed effectiveness
            // so that not read Value few times
            
            while let value = iterator.next() {
                
                if value <= 0 {
                    
                    pthread_rwlock_wrlock(&allValuesFlagLocker)
                    allValuesArePositive = false
                    pthread_rwlock_unlock(&allValuesFlagLocker)
                    
                    return
                }
                chunkSum += value
            }

            lock.wait()
            sum += chunkSum
            lock.signal()
        })
        
        guard allValuesArePositive == true else {
            throw SharesCalculators.Error.nonPositiveInputValues
        }
        
        return sum
    }
    
    // .reduce concurrent equivalent (but with some other initialValue behaviour)
    // Also have .map concurrent equivalent below
    
    func concurrentReduce(
        initialValue: Value = 0,
        _ reducer: (Value, Value) -> Value) -> Value {
        
        let lock = DispatchSemaphore(value: 1)
        var result: Value = initialValue
        
        DispatchQueue.concurrentPerform(iterations: self.count, execute: {

            let chunk = self[$0]

            var chunkResult: Value = initialValue
            var iterator = chunk.makeIterator()

            while let value = iterator.next() {
                chunkResult = reducer(chunkResult, value)
            }

            lock.wait()
            result = reducer(result, chunkResult)
            lock.signal()
        })
        
        return result
    }
    
    func concurrentMapAndCollect(_ transform: (Value) -> Value) -> [Value] {
        
        let lock = DispatchSemaphore(value: 1)
        
        var results = [Int: [Value]]()
        DispatchQueue.concurrentPerform(iterations: self.count, execute: {

            let chunk = self[$0]
            let resultChunk = chunk.map(transform)
            
            lock.wait()
            results[$0] = resultChunk
            lock.signal()
        })
        
        let collectedValues = (0..<self.count)
            .reduce(into: [Value](), { $0 += results[$1]! })
        
        return collectedValues
    }
}

extension SharesCalculators {
    
    struct Parallelized {
        
        struct UnsafeFastest: SharesCalculatable {
            
            func calculate(_ inputValues: [Double]) throws -> [Double] {
                
                let chunkedValues = inputValues.chunksForConcurrency()
                  
                let inputSum = chunkedValues.concurrentReduce(+)
                let shareEquivalent = 100.0 / inputSum
                
                let shareValues = chunkedValues.concurrentMapAndCollect { $0 * shareEquivalent }
                return shareValues
            }
        }

        struct SafeAccurate: SharesCalculatable {
            
            func calculate(_ inputValues: [Decimal]) throws -> [Decimal] {
                
                let chunkedValues = inputValues.chunksForConcurrency()
                  
                let inputSum = try chunkedValues.concurrentSumWithSafetyCheck()
                let shareEquivalent = 100.0 / inputSum
                
                let shareValues = chunkedValues.concurrentMapAndCollect { $0 * shareEquivalent }
                return shareValues
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: - Time constrainted:
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Thereby, calculators can be decoreated with different Behaviours
    // It's other, old good approach, instead Mixins
    // By example: time and resource constraints. Perhaps, add some other steps with transformations

// Real Performance depend from many factors (build optimizations, hardware capabilities).
// Based on this, it hard to count, how many Values can be for 5 seconds time edges

// In this case, have 2 enough effective approaches for Time constraint:

    // - calculation Algorithm of "CPU effectivety parameter", and check to enough dedicated memory. Based on what it is possible to calculate approximately how many N values ​​will be executed in time x
    
    // - Perform special dry run on small sample, measure time, and extrapolate result to N values, calculate expected time (This approach was implemented, and as expected, most precise and effective)

extension SharesCalculators {
    
    struct TimeConstrainted {
        
        struct WithDryRun <V, C: SharesCalculatable>: SharesCalculatable where V == C.Value {
            
            private let calculator: C
            let allowedTime: TimeInterval
            
            init(_ calculator: C, allowedTime: TimeInterval) {
                self.calculator = calculator
                self.allowedTime = allowedTime
            }
            
            func calculate(_ inputValues: [V]) throws -> [V] {
                
                // Time constraint concept with small allowedTime values ->
                // it's a bit problem, it better to throw Error from here
                
                guard allowedTime > 1.0 else {
                    throw Error.allowedTimeIsSmall(allowed: allowedTime)
                }
                
                let validationResult = try validateExpectedTime(for: inputValues)
                
                if case let .invalidInput(allowedTime, expectedTime) = validationResult {
                    
                    throw Error.excessInputValues(
                        details: .notAllowedExpectations(
                            allowed: allowedTime,
                            expected: expectedTime
                        )
                    )
                }
                
                let resultValues = try calculator.calculate(inputValues)
                return resultValues
            }
            
            private enum ValidationResult {
                
                case validInput
                case invalidInput (allowed: TimeInterval, expected: TimeInterval)
            }
            
            private func validateExpectedTime(for inputValues: [V]) throws -> ValidationResult {
                
                // < 256k - Automatically considered as a valid values
                
                let noConstraintedRunCount = (1 << 18)
                guard inputValues.count > noConstraintedRunCount else {
                    return .validInput
                }
                
                // < 32k - Size for Dry rin sample
                
                let dryRunCount = (1 << 15)
                let dryRunValues = Array(inputValues[0..<dryRunCount])
                
                let dryRunTime = try evaluateTime {
                    _ = try calculator.calculate(dryRunValues)
                }
                let expectedTime = (Double(inputValues.count) / Double(dryRunCount)) * dryRunTime
                
                guard expectedTime > allowedTime else {
                    return .validInput
                }
                
                return .invalidInput(allowed: allowedTime, expected: expectedTime)
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: - Numbers Rounding (Truncating):
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// When we truncate values - we lose in accuracy. Final sum be not equal to 100, precisely due to the influence of the discarded remainder after the decimal. The best we can do - use IEEE 754 rounding mode "bankers rounding"

protocol SharesTruncatable {
    
    var keepDigitsCount: Int { get }
    
    associatedtype Value: Comparable
    func truncateSingle(_ inputValue: Value) -> Value
}

extension SharesTruncatable {
    
    func truncate(_ inputValues: [Value] ) throws -> [Value] {
        
        let allowedKeepDigitsRange = 0...6
        guard allowedKeepDigitsRange.contains(keepDigitsCount) else {
            
            throw SharesCalculators.Error.keepDigitsCountRange(
                value: keepDigitsCount,
                allowedRange: allowedKeepDigitsRange
            )
        }
        
        return inputValues.map(truncateSingle)
    }
}

// Convenience for Power operation

precedencegroup PowerPrecedence {
    higherThan: MultiplicationPrecedence
}
infix operator ^^ : PowerPrecedence

func ^^ (radix: Int, power: Int) -> Int {
    return Int(pow(Double(radix), Double(power)))
}


struct DoubleTruncation: SharesTruncatable {
    
    let keepDigitsCount: Int
    private let truncationParameter: Double
    
    init(keepDigitsCount: Int) {
        self.keepDigitsCount = keepDigitsCount
        self.truncationParameter = Double(10 ^^ keepDigitsCount)
    }
    
    func truncateSingle(_ inputValue: Double) -> Double {
        
        return (inputValue * truncationParameter).rounded(.toNearestOrEven) / truncationParameter
    }
}

struct DecimalTruncation: SharesTruncatable {
    
    let keepDigitsCount: Int
    
    func truncateSingle(_ inputValue: Decimal) -> Decimal {
        
        var baseValue: Decimal = inputValue
        var resultValue: Decimal = 0.0

        NSDecimalRound(&resultValue, &baseValue, keepDigitsCount, .bankers)
        return resultValue
    }
}

// Simple bankingRounding gives problems with a similar Truncating process. In real financial situations, this isn't practic solution, I suppose, to use such algorithm with randomness factor. But at least this experiment gave the result

// Experiments show, that randomization of rounding side choice in controversial situations, give best results - in 10+ times closer values to 100

struct ExperimentalDecimalTruncation: SharesTruncatable {
    
    let keepDigitsCount: Int
    
    init(keepDigitsCount: Int) {
        self.keepDigitsCount = keepDigitsCount
        
        self.truncationMultiplier_ = Decimal(10 ^^ keepDigitsCount)
        self.truncationTerm_ = Decimal(1) / truncationMultiplier_
        
        self.randomizationMultiplier = (10 ^^ randomizationDigitsCount)
        self.randomizationMultiplier_ = Decimal(randomizationMultiplier)
    }
    
    private let randomizationDigitsCount: Int = 4
    
    private let truncationMultiplier_: Decimal
    private let truncationTerm_: Decimal
    
    private let randomizationMultiplier: Int
    private let randomizationMultiplier_: Decimal
    
    func truncateSingle(_ inputValue: Decimal) -> Decimal {
        
        var baseValue: Decimal = inputValue
        var resultValue: Decimal = 0.0

        NSDecimalRound(&resultValue, &baseValue, keepDigitsCount, .down)
        
        let shiftedResult = baseValue * truncationMultiplier_ * randomizationMultiplier_
        let extraRemainder = Int(Double(truncating: NSDecimalNumber(decimal: shiftedResult))) % randomizationMultiplier
        let randomReminder = Int(arc4random() % UInt32(randomizationMultiplier))
        
        if extraRemainder > randomReminder {
            resultValue += truncationTerm_
        }
        if extraRemainder == randomReminder {
            if arc4random() % 2 == 0 {
                resultValue += truncationTerm_
            }
        }
        
        return resultValue
    }
}

struct ParallelTruncation <V, T: SharesTruncatable>: SharesTruncatable where V == T.Value, V: SignedNumeric {
    
    var keepDigitsCount: Int {
        return truncation.keepDigitsCount
    }
    private let truncation: T
    
    init(_ truncation: T) {
        self.truncation = truncation
    }
    
    func truncateSingle(_ inputValue: V) -> V {
        return truncation.truncateSingle(inputValue)
    }
    
    func truncate(_ inputValues: [V]) throws -> [V] {
        
        let chunkedValues = inputValues.chunksForConcurrency()
        let truncatedValues = chunkedValues.concurrentMapAndCollect(truncation.truncateSingle)
        
        return truncatedValues
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: - Supplementary Functions:
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

func evaluateTime(_ closure: () throws -> Void) rethrows -> TimeInterval {
    
    let start = Date()
    try closure()
    let end = Date()
    
    let diffTimeInSeconds = end.timeIntervalSince(start)
    return diffTimeInSeconds
}

@discardableResult func evaluateTimeAndPrint(
    taskName: String,
    _ closure: () throws -> Void) rethrows -> TimeInterval {
    
    let diffTimeInSeconds = try evaluateTime(closure)
    
    print("<Time> Estimated time for \(taskName) = \(diffTimeInSeconds) SEC")
    return diffTimeInSeconds
}

@discardableResult func evaluateAverageTimeAndPrint(
    iterationsCount: Int,
    _ timeMeasureAction: (Int) -> TimeInterval) -> TimeInterval {
    
    let averageTime =
        (0..<iterationsCount)
            .map(timeMeasureAction)
            .reduce(0, +)
    
    print("<Time> Average Time = \(averageTime) SEC")
    return averageTime
}

@discardableResult func validateResults(validationTask: String) -> ([Double]) throws -> [Double] {

    return { (inputValues: [Double]) throws -> [Double] in
        
        let resultSum = inputValues.reduce(0, +)
        print(String(format: "\t<Validation> \(validationTask), Sum = %.3f", resultSum))
        
        return inputValues
    }
}

@discardableResult func validateResults(validationTask: String) -> ([Decimal]) throws -> [Decimal] {
    
    return { (inputValues: [Decimal]) -> [Decimal] in
        
        var resultSum = inputValues.reduce(0, +)
        print(String(format: "\t<Validation> \(validationTask), Sum = %@", NSDecimalString(&resultSum, nil)))
        
        return inputValues
    }
}

func printResults(input: [Decimal], result: [Decimal]) {
    
    print("<Result> ----------------------------------------")
    
    for index in 0..<input.count {
        
        var inputValue = input[index]
        var resultValue = result[index]
        var readableInputValue: Decimal = 0
        
        NSDecimalRound(&readableInputValue, &inputValue, 3, .bankers)
        
        print(String(
            format: "<Result> share #\(index): \n\tInput = %@ \n\tOutput = %@",
            NSDecimalString(&readableInputValue, nil),
            NSDecimalString(&resultValue, nil)
        ))
    }
    print("<Result> ----------------------------------------")
}
func printResults(input: [Double], result: [Double]) {
    
    print("<Result> ----------------------------------------")
    
    for index in 0..<input.count {
        print(String(format: "<Result> share #\(index): Input = %.3f\t\t\t\tOutput = %.3f", input[index], result[index]))
    }
    print("<Result> ----------------------------------------")
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MARK: - Experiment:
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

precedencegroup ForwardPipe {
    
    associativity: left
    higherThan: LogicalConjunctionPrecedence
}
infix operator |> : ForwardPipe

func |> <T, U>(value: T, function: (T) throws -> U) rethrows -> U {
    return try function(value)
}

final class Experimentator {
    
    // Imitation of difference in revenue of different shares holders (sufficient differences)
    
    private let grouppedGenerator =
        DoubleGrouppedGenerator(customGroups: [
            DoubleGrouppedGenerator.CustomGroup(range: 1...100, probability: 0.2),
            DoubleGrouppedGenerator.CustomGroup(range: 100...1000, probability: 0.5),
            DoubleGrouppedGenerator.CustomGroup(range: 1000...100000, probability: 0.2),
            DoubleGrouppedGenerator.CustomGroup(range: 100000...10000000, probability: 0.05),
            DoubleGrouppedGenerator.CustomGroup(range: 10000000...100000000, probability: 0.04),
            DoubleGrouppedGenerator.CustomGroup(range: 100000000...1000000000, probability: 0.01)
        ])
    
    private let allowedTime: TimeInterval = 5.0
    private let keepDigits: Int = 3
    
    func performPerformanceMeasurements() {
        
        // WARNING: Do not run these measurements in Playground (QuickLook hangs)
        // NOTE: In reality, these methods use substantially more memory, than required, beacuse of executeCalculations___ extra internal copying
        
        #if canImport(PlaygroundSupport)
            preconditionFailure("Please, do not Run this code from XCode Playground")
        #endif
        
        print("\t[Fast Decimal] (1_000_000 Values)")
        executeCalculationsWithDecimal(
            iterations: 8,
            inputCount: 1_000_000,
            shouldPrintResult: false
        )
        print("\n")
        
        print("\t[Stress Decimal] (10_000_000 Values)")
        executeCalculationsWithDecimal(
            iterations: 2,
            inputCount: 10_000_000,
            shouldPrintResult: false
        )
        print("\n")
        
        print("\t[Fast Double] (1_000_000 Values)")
        executeCalculationsWithDouble(
            iterations: 8,
            inputCount: 1_000_000,
            testedCalculator: Experimentator.concurrentDoubleCalculator,
            shouldPrintResult: false
        )
        print("\n")
        
        print("\t[Stress Double] (10_000_000 Values)")
        executeCalculationsWithDouble(
            iterations: 2,
            inputCount: 10_000_000,
            testedCalculator: Experimentator.concurrentDoubleCalculator,
            shouldPrintResult: false
        )
        print("\n")
        
        print("\t[NonConcurrent Double] (10_000_000 Values)")
        executeCalculationsWithDouble(
            iterations: 2,
            inputCount: 10_000_000,
            testedCalculator: Experimentator.nonConcurrentDoubleCalculator,
            shouldPrintResult: false
        )
        print("\n")
    }
    
    func performCalculationExamples() {
        
        print("\t[Short Accurate Results] (50 Values)")
        executeCalculationsWithDecimal(
            iterations: 2,
            inputCount: 50,
            shouldPrintResult: true
        )
        print("\n")
    }
    
    func performInputOverflow() {
        
        print("\t[Input Overflow Double (nonConcurrent)] (50_000_000 Values)")
        executeCalculationsWithDouble(
            iterations: 1,
            inputCount: 50_000_000,
            testedCalculator: Experimentator.nonConcurrentDoubleCalculator,
            shouldPrintResult: false
        )
        print("\n")
    }
    
    private func executeCalculationsWithDecimal(
        iterations: Int,
        inputCount: Int,
        shouldPrintResult: Bool) {
        
        let generator = DecimalGenerator(grouppedGenerator)
        
        let calculator =
            SharesCalculators.TimeConstrainted.WithDryRun(
                SharesCalculators.Parallelized.SafeAccurate(),
                allowedTime: self.allowedTime
            )
        
        let fastTruncation =
            ParallelTruncation(
                DecimalTruncation(keepDigitsCount: self.keepDigits)
            )
        
        let experimentalTruncation =
            ParallelTruncation(
                ExperimentalDecimalTruncation(keepDigitsCount: self.keepDigits)
            )
        
        evaluateAverageTimeAndPrint(iterationsCount: iterations) { (_) -> TimeInterval in
            
            var calculationTime: TimeInterval = 0
            
            print("<Generation> will generate \(inputCount) Decimal values")
            let inputValues = generator.generate(count: inputCount)
            print("<Generation> did generate them")
            
            var values = inputValues
            
            do {
                calculationTime = try evaluateTimeAndPrint(taskName: "calculation \(type(of: calculator)) <\(inputCount)>") {
                    
                    values = try values
                        |> calculator.calculate
                }
                
                let resultValues = try values
                    |> validateResults(validationTask: "before_truncation")
                    |> fastTruncation.truncate
                    |> validateResults(validationTask: "after_truncation_simple")
                
                _ = try values
                    |> experimentalTruncation.truncate
                    |> validateResults(validationTask: "after_truncation_experimental")
                
                if shouldPrintResult == true {
                    printResults(input: inputValues, result: resultValues)
                }
                
            } catch {
                print("\t<Error> \(error)")
            }
            
            return calculationTime
        }
    }
    
    private static let concurrentDoubleCalculator = SharesCalculators.Parallelized.UnsafeFastest()
    private static let nonConcurrentDoubleCalculator = SharesCalculators.UnsafeFastest()
    
    private func executeCalculationsWithDouble <C: SharesCalculatable>(
        iterations: Int,
        inputCount: Int,
        testedCalculator: C,
        shouldPrintResult: Bool) where C.Value == Double {
        
        let generator = grouppedGenerator
        
        let calculator =
            SharesCalculators.TimeConstrainted.WithDryRun(
                testedCalculator,
                allowedTime: self.allowedTime
            )
        
        let truncation =
            ParallelTruncation(
                DoubleTruncation(keepDigitsCount: self.keepDigits)
            )
        
        evaluateAverageTimeAndPrint(iterationsCount: iterations) { (_) -> TimeInterval in
            
            var calculationTime: TimeInterval = 0
            
            print("<Generation> will generate \(inputCount) Double values")
            let inputValues = generator.generate(count: inputCount)
            print("<Generation> did generate them")
            
            var values = inputValues
            
            do {
                calculationTime = try evaluateTimeAndPrint(taskName: "calculation \(type(of: calculator)) <\(inputCount)>") {
                    
                    values = try values
                        |> calculator.calculate
                }
                
                values = try values
                    |> validateResults(validationTask: "before_truncation")
                    |> truncation.truncate
                    |> validateResults(validationTask: "after_truncation_simple")
                
                if shouldPrintResult == true {
                    printResults(input: inputValues, result: values)
                }
                
            } catch {
                print("\t<Error> \(error)")
            }
            
            return calculationTime
        }
    }
}
