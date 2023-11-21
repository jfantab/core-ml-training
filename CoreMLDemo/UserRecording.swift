//
//  UserRecording.swift
//  CoreMLDemo
//
//  Created by John Lu on 11/19/23.
//  Copyright © 2023 AppCoda. All rights reserved.
//

import CoreML
import Foundation

struct UserRecording {
    var array: MLMultiArray!
    
    var featureValue: MLFeatureValue {
        let recordingFeatureValue = MLFeatureValue(multiArray: array!)
        
        return recordingFeatureValue
    }
}
