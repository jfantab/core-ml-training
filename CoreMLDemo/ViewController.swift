
//
//  ViewController.swift
//  CoreMLDemo
//
//  Created by Sai Kambampati on 14/6/2017.
//  Copyright © 2017 AppCoda. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController, UINavigationControllerDelegate {
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    var model: LSTM!
    var permanentURL: URL!
    
    /// The location of the app's Application Support directory for the user.
    private static let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
                                                               in: .userDomainMask).first!
    
    // var updatedModelURL: URL = appDirectory.appendingPathComponent("LSTM.mlmodelc")
    var updatedModelURL = Bundle.main.url(forResource: "LSTM", withExtension: "mlmodelc")
    
    override func viewWillAppear(_ animated: Bool) {
        print("View will appear")

        // model = try LSTM()
        loadCompiledModel(url: updatedModelURL!)
        
        // Generate sample
        let array = try? MLMultiArray(shape: [1, 162, 1], dataType: .float32)
        
        let length = array!.count
        for i in 0..<length {
            // Generate a random float between 0 and 5
            array![i] = NSNumber(value: Float.random(in: 0..<5))
        }
        
        var recording = UserRecording()
        recording.array = array
        var dataLoader = loadData(data: recording, emotion: "happy")
        
        train(at: updatedModelURL!, with: dataLoader!) { context in
            print("Completion handler")
        }
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    func saveCompiledModel(compiledModelURL: URL) {
        let fileManager = FileManager.default
        let appSupportURL = fileManager.urls(for: .applicationSupportDirectory,
                                             in: .userDomainMask).first!
        
        permanentURL = appSupportURL.appendingPathComponent(compiledModelURL.absoluteString)

        // Copy the file to the permanent location, replacing it if necessary.
        do {
            _ = try fileManager.replaceItemAt(permanentURL, withItemAt: compiledModelURL)
        }
        catch {
            print(error)
            return
        }
    }
    
    func loadCompiledModel(url: URL) {
        print("Loading compiled model")
        guard FileManager.default.fileExists(atPath: url.path) else {
            // The updated model is not present at its designated path.
            return
        }
        
        guard let newModel = try? LSTM(contentsOf: updatedModelURL!) else {
            return
        }
        
        model = newModel
    }
    
    func loadData(data: UserRecording, emotion: String) -> MLArrayBatchProvider! {
        print("Loading data")
        var featureProviders = [MLFeatureProvider]()
        
        let inputName = "recording"
        let outputName = "emotion"
        
        let inputValue = data.featureValue
        let outputValue = MLFeatureValue(string: emotion)
        
        let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue,
                                                          outputName: outputValue]
        
        if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
            featureProviders.append(provider)
        }
        
        return MLArrayBatchProvider(array: featureProviders)
    }
    
    func train(at url: URL, with trainingData: MLBatchProvider) {
        print("Training")
        
        let progressHandler = { (context: MLUpdateContext) in
          switch context.event {
          case .trainingBegin:
            print("Training begin")

          case .miniBatchEnd:
            let batchIndex = context.metrics[.miniBatchIndex] as! Int
            let batchLoss = context.metrics[.lossValue] as! Double
            print("Mini batch \(batchIndex), loss: \(batchLoss)")

          case .epochEnd:
            let epochIndex = context.metrics[.epochIndex] as! Int
            let trainLoss = context.metrics[.lossValue] as! Double

            let predictor = Predictor(model: context.model)
            let (valLoss, valAcc) = predictor.evaluate(loader: self.validationLoader)

            // Tell the view controller about the results for this epoch:
            callback(.epochEnd(trainLoss: trainLoss, validationLoss: valLoss,
                               validationAccuracy: valAcc))
          default:
            print("Unknown event")
          }
        }

        
        guard let updateTask = try? MLUpdateTask(forModelAt: url, trainingData: trainingData, progressHandler: progressHandler)
        else {
            print("Couldn't create an MLUpdateTask.")
            return
        }
        
        print("Update task made")
        
        updateTask.resume()
    }
    
    func argmax(multiArray: MLMultiArray) -> Int {
        guard multiArray.count > 0 else {
            fatalError("MLMultiArray is empty")
        }
        
        var maxIndex = 0
        var maxValue: Double = multiArray[0].doubleValue
        
        for i in 1..<multiArray.count {
            let value = multiArray[i].doubleValue
            if value > maxValue {
                maxValue = value
                maxIndex = i
            }
        }
        
        return maxIndex
    }
    
    func predict(input: LSTMInput!) {
        print("Predicting")
        guard let prediction = try? model.prediction(input: input) else {
            return
        }
        
        print(prediction.Identity)
        print(argmax(multiArray: prediction.Identity))
    }
}

