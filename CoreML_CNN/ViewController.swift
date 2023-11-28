//
//  ViewController.swift
//  CoreML_CNN
//
//  Created by John Lu on 11/26/23.
//

import UIKit
import CoreML

class ViewController: UIViewController {
    
    var model: cnn!
    var modelURL = Bundle.main.url(forResource: "cnn", withExtension: "mlmodelc")
    
    override func viewDidLoad() {
        super.viewDidLoad()
        guard let newModel = try? cnn(contentsOf: modelURL!) else {
            return
        }
        
        model = newModel
        
        let trainingData = loadTrainingData()
        
        let progressHandler = { (context: MLUpdateContext) in
            switch context.event {
            case .trainingBegin:
                print("Training started..")
            case .miniBatchEnd:
                break
            case .epochEnd:
                let epochIndex = context.metrics[.epochIndex] as! Int
                let trainLoss = context.metrics[.lossValue] as! Double
                print("Epoch \(epochIndex + 1) end with loss \(trainLoss)")
                
            default:
                print("Unknown event")
            }
        }        
        
        let completionHandler = { (context: MLUpdateContext) in
            print("Training completed with state \(context.task.state.rawValue)")
            print("CoreML Error: \(context.task.error.debugDescription)")

            if context.task.state != .completed {
                print("Failed")
                return
            }

            let trainLoss = context.metrics[.lossValue] as! Double
            print("Final loss: \(trainLoss)")
            
//            let updatedModel = context.model
//            let updatedModelURL = URL(fileURLWithPath: retrainedCoreMLFilePath)
//            try! updatedModel.write(to: updatedModelURL)
            print("Model trained!")
        }

        let handlers = MLUpdateProgressHandlers(
                            forEvents: [.trainingBegin, .miniBatchEnd, .epochEnd],
                            progressHandler: progressHandler,
                            completionHandler: completionHandler)
        
        guard let updateTask = try? MLUpdateTask(forModelAt: modelURL!, trainingData: trainingData!, progressHandlers: handlers)
        else {
            print("Couldn't create an MLUpdateTask.")
            return
        }
        
        print("Update task made")
        
        updateTask.resume()
    }
    
    func generateSample() -> MLMultiArray! {
        let array = try? MLMultiArray(shape: [1, 28, 28], dataType: .double)
        
        for i in 0..<array!.shape[0].intValue {
            for j in 0..<array!.shape[1].intValue {
                for k in 0..<array!.shape[2].intValue {
                    array![[i as NSNumber, j as NSNumber, k as NSNumber]] = NSNumber(value: Double.random(in: 0..<1))
                }
            }
        }
        
        return array
    }
    
    func loadTrainingData() -> MLArrayBatchProvider! {
        print("Loading data")
        var featureProviders = [MLFeatureProvider]()
        
        let inputName = "input"
        let outputName = "prediction_true"
        
        for _ in 0..<5 {
            let inputValue = MLFeatureValue(multiArray: generateSample())
            
            let output: MLMultiArray = try! MLMultiArray(shape: [1], dataType: .int32)
            output[0] = Int.random(in: 0..<10) as NSNumber
            let outputValue = MLFeatureValue(multiArray: output)
            
            let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue,
                                                               outputName: outputValue]
            
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }
        
        return MLArrayBatchProvider(array: featureProviders)
    }
    
    func predict(array: MLMultiArray) {
        guard let result = try? model.prediction(input: array) else {
            return
        }

        print(result.prediction)
    }
}

