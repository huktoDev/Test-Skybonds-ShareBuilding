//
//  main.swift
//  Test-Skybonds-ShareBuilding.console
//
//  Created by Alexandr Babenko on 20.12.2019.
//  Copyright © 2019 Alexandr Babenko. All rights reserved.
//

import Foundation

enum InputError: Swift.Error, CustomStringConvertible {
    
    case inputTextNil
    
    var description: String {
        switch self {
        case .inputTextNil:
            return "result with readline() == nil"
        }
    }
}

enum FileError: Swift.Error, CustomStringConvertible {
    
    case fileURLMalformed (String)
    case fileNotFoundByURL (URL)
    case dataNotObtained (URL, Swift.Error?)
    case dataDecodingError (Swift.Error)
    
    var description: String {
        switch self {
        case .fileURLMalformed (let inputText):
            return "file URL Malformed, inputText = \(inputText)"
        case .fileNotFoundByURL (let fileURL):
            return "file not found with URL = \(fileURL)"
        case .dataNotObtained (let fileURL, let error):
            return "data not Obtained from File with URL = \(fileURL), Error = \(String(describing: error))"
        case .dataDecodingError (let decodingError):
            return "Decodable error = \(decodingError)"
        }
    }
}

enum ParsingError: Swift.Error, CustomStringConvertible {
    
    case valuesRepresentationsNotEnough (Int)
    case foundNotDoublePresentations
    
    var description: String {
        switch self {
        case .valuesRepresentationsNotEnough (let count):
            return "Values representations aren't enough for Processing: \(count)"
        case .foundNotDoublePresentations:
            return "Not all Provided Values are Double's"
        }
    }
}

enum ValuesDataSource {
    
    case input
    case file
    
    func provider() -> ValuesDataProvider {
        
        switch self {
            case .input: return InputValuesProvider()
            case .file: return FileValuesProvider()
        }
    }
}

protocol ValuesDataProvider {
    
    func provideData() throws -> [Double]
}

final class InputValuesProvider: ValuesDataProvider {
    
    func provideData() throws -> [Double] {
        
        print("<Input> Enter here your Double Values with \',\' separator:")
        
        guard let userInput = readLine() else {
            throw InputError.inputTextNil
        }
        
        let valuesRepresentations =
            userInput
                .replacingOccurrences(of: " ", with: "")
                .components(separatedBy: ",")
                .filter { $0 != "" }
        
        guard valuesRepresentations.count > 1 else {
            throw ParsingError.valuesRepresentationsNotEnough(valuesRepresentations.count)
        }
        
        let inputValues =
            valuesRepresentations
                .compactMap {
                    Double($0)
                }
        
        guard valuesRepresentations.count == inputValues.count else {
            throw ParsingError.foundNotDoublePresentations
        }
        
        return inputValues
    }
}



struct FileValuesEntity: Decodable {
    
    enum CodingKeys: String, CodingKey {
        case values
    }
    let values: [Double]
}

final class FileValuesProvider: ValuesDataProvider {
    
    private let fileManager = FileManager.default
    
    func provideData() throws -> [Double] {
        
        print("<Input> File should have special Formatted data")
        print("<Input> JSON DataFormat: { \"values\": [v1, v2, v3] }")
        print("<Input> You can use \"testSharesData.json\" from Resources")
        print("<Input> Also don't forget about File access Permissions")
        print("<Input> Enter path to File in Absolute format:")
        
        guard let userInput = readLine() else {
            throw InputError.inputTextNil
        }

        guard let fileURL = URL(string: userInput) else {
            throw FileError.fileURLMalformed(userInput)
        }
        
        guard fileManager.fileExists(atPath: fileURL.absoluteString) else {
            throw FileError.fileNotFoundByURL(fileURL)
        }
        
        let fileData = try { () throws -> Data in
            
            var contentString: String!
            do {
                contentString = try String(contentsOfFile: fileURL.absoluteString)
            }
            catch {
                throw FileError.dataNotObtained(fileURL, error)
            }
            
            guard let contentData = contentString.data(using: .utf8) else {
                throw FileError.dataNotObtained(fileURL, nil)
            }
            return contentData
        }()
        
        let entity = try { () throws -> FileValuesEntity in
            do {
                return try JSONDecoder().decode(FileValuesEntity.self, from: fileData)
            }
            catch {
                throw FileError.dataDecodingError(error)
            }
        }()
        
        return entity.values
    }
}


func catchValuesDataSource() -> ValuesDataSource {
    
    repeat {
        let userSuggestion = """
        <Input> Will you to use .json values File, or console values Input?
        <Input> Possible Values: (\"FILE\": special file, \"INPUT\": from console)
        <Input> Select:
        """
        
        print(userSuggestion)

        let userInput = readLine()

        switch userInput {
        case "FILE":
            return .file
        case "INPUT":
            return .input
        default:
            print("<Error> Value Is Incorrect and couldn't be Recgonized... Please repeat")
            break
        }
    }
    while true
}

func catchValues(_ dataSource: ValuesDataSource) -> [Double] {
    
    repeat {
        do {
            let values = try dataSource.provider().provideData()
            return values
        }
        catch {
            print("<Error> \(error)")
            continue
        }
    }
    while true
}

func processValues(_ inputValues: [Double]) throws -> [Double] {
    
    let calculator =
        SharesCalculators.TimeConstrainted.WithDryRun(
            SharesCalculators.Parallelized.UnsafeFastest(),
            allowedTime: 5.0
        )

    let truncation =
        ParallelTruncation(
            DoubleTruncation(keepDigitsCount: 3)
        )
    
    return try inputValues
        |> calculator.calculate
        |> truncation.truncate
}



do {
    let valuesSource = catchValuesDataSource()
    let inputValues = catchValues(valuesSource)
    let outputValues = try processValues(inputValues)
    
    printResults(input: inputValues, result: outputValues)
}
catch {
    print("<Error> \(error)")
}
