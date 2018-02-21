const Backprop = (() => {
  
  function shuffleArrays(array1, arrays) {
    let resultArray1 = [];
    let resultArrays = Array(arrays.length).fill([]);
    
    for (let i = 0; i < array1.length; i++) {
      let randomIndex = Math.floor(Math.random() * array1.length);
      
      resultArray1.push(array1[randomIndex]);
      resultArrays = resultArrays.map((innerArray, innerIndex) => innerArray.concat(arrays[innerIndex][randomIndex]));
      
      array1.splice(randomIndex, 1);
      arrays.forEach((innerArray) => innerArray.splice(randomIndex, 1));
    }
    resultArray1 = resultArray1.concat(array1);
    resultArrays = resultArrays.map((innerArray, innerIndex) => innerArray.concat(arrays[innerIndex]));
    
    return [resultArray1, resultArrays];
	}
  
  // this should not mutate its inputs
  // this automatically appends the bias to the output matrix
  function getNet(weights, lowerLayerOutputs, layerIndex, nodeIndex, bias) {
    const currentOutputs = lowerLayerOutputs.concat(bias);
    
    return weights[layerIndex].reduce( (sum, nodeWeights, currentNodeIndex) => {
      
      return sum + nodeWeights[nodeIndex] * currentOutputs[currentNodeIndex];
      
    }, 0);
  }
  
  // this is using the sigmoid activation function.
  function getOutput(nodeNet) {
    return 1 / (1 + Math.exp(-1 * nodeNet));
  }
  
  function getOutputDerivative(nodeOutput) {
    return nodeOutput * (1 - nodeOutput);
  }
  
  function getWeightChange(learningRate, nodeOutput, errorSignal, lastWeightChange, momentum) {
    
    const momentumAddition = momentum * lastWeightChange || 0;
    
    return learningRate * nodeOutput * errorSignal + momentumAddition;
  }
  
  function getOutputErrorSignal(nodeOutput, nodeOutputDerivative, nodeTarget) {
    return (nodeTarget - nodeOutput) * nodeOutputDerivative;
  }
  
  function getHiddenNodeErrorSignal(nodeOutputDerivative, weights, errorSignals, layerIndex, nodeIndex) {
    
    const aboveLayerIndex = layerIndex + 1;
    
    return nodeOutputDerivative * errorSignals[aboveLayerIndex].reduce( (sum, errorSignal, signalIndex) => {
      
      return sum + errorSignal * weights[aboveLayerIndex][nodeIndex][signalIndex];
      
    }, 0);
  }
  
  // initialValue baically meaningless since it will never actually get used, assuming backprop is working
  // NOTE: the last index of nodeCoutns indicates how many outputs there should be
  function initNodeValues(nodeCounts, includeBias, initValue = null) {
    return nodeCounts.map((count, index, array) => {
      if (includeBias && index < array.length - 1) {
        return Array(count + 1).fill(initValue);
      }
      else {
        return Array(count).fill(initValue);
      }
    });
  }
  
  function getAllNodeOutputs(nodeCounts, weights, inputs, bias, outputFunction = getOutput) {
    return initNodeValues(nodeCounts, false).reduce((outputs, layer, layerIndex) => {
      return outputs.concat([
        layer.map((node, nodeIndex) => {
          
          const net = getNet(weights, outputs[layerIndex - 1] || inputs, layerIndex, nodeIndex, bias);
          return outputFunction(net);
        })
      ]);
    }, []);
  }
  
  function getAllNodeErrorSignals(outputs, weights, targets, derivativeFunction = getOutputDerivative) {
    
    function mapErrorSignalLayer(layerIndex, node, nodeIndex) {
      
      const nodeOutput = outputs[layerIndex][nodeIndex];
      
      if (layerIndex === errorSignals.length - 1) {
        return getOutputErrorSignal(nodeOutput, derivativeFunction(nodeOutput), targets[nodeIndex]);
      }
      else {
        return getHiddenNodeErrorSignal(derivativeFunction(nodeOutput), weights, errorSignals, layerIndex, nodeIndex);
      }
    }
    
    let errorSignals = initNodeValues(outputs.reduce((sizes, layer) => sizes.concat(layer.length), []));

    // go backwards from the end of the array (the output layer)
    for (let layerIndex = errorSignals.length - 1; layerIndex >= 0; layerIndex--) {
      errorSignals[layerIndex] = errorSignals[layerIndex].map(mapErrorSignalLayer.bind(null, layerIndex));
    }
    
    return errorSignals;
  }
  
  function getAllWeightDeltas(learningRate, errorSignals, outputs, inputs, weights, lastWeightDeltas, bias, momentum) {
    
    return weights.map((layer, layerIndex, outputsArray) => {
      
      let belowLayerOutputs;
      
      if (layerIndex === 0) {
        belowLayerOutputs = inputs.concat(bias);
      }
      else {
        belowLayerOutputs = outputs[layerIndex-1].concat(bias);
      }
      
      return layer.map((fromNode, fromNodeIndex) => {
        return fromNode.map((toNode, toNodeIndex) => {
          return getWeightChange(learningRate, belowLayerOutputs[fromNodeIndex], errorSignals[layerIndex][toNodeIndex], (lastWeightDeltas || undefined) && lastWeightDeltas[layerIndex][fromNodeIndex][toNodeIndex], momentum);
        });
      });
    });
  }
  
  // non-mutating
  function getNewWeights(weights, weightDeltas) {
    return weights.map((layer, layerIndex) => {
      return layer.map((fromNodeWeights, fromNodeIndex) => {
        return fromNodeWeights.map((toNodeWeight, toNodeIndex) => {
          return toNodeWeight + weightDeltas[layerIndex][fromNodeIndex][toNodeIndex];
        });
      });
    });
  }
  
  function getSquaredError(outputs, targets) {
    return outputs.reduce((sum, output, outputIndex) => {
      return sum + Math.pow(output - targets[outputIndex], 2);
    }, 0);
  }
  
  function getClassificationAccuracy(outputs, targets) {
    const max = outputs.reduce((final, current, maxIndex) => {
      if (current > final.max) {
        return {
          max: current,
          index: maxIndex
        };
      }
      else {
        return final;
      }
    }, {max: -Infinity, index: -1});
    
    return targets[max.index] == 1;
  }
  
  function runSinglePattern(learningRate, weights, inputs, targets, bias, nodeCounts, momentum, lastWeightDeltas, outputFunction, derivativeFunction, classificationFunction) {
    
    const outputs = getAllNodeOutputs(nodeCounts, weights, inputs, bias, outputFunction);
    
    const errorSignals = getAllNodeErrorSignals(outputs, weights, targets, derivativeFunction);
    
    const weightDeltas = getAllWeightDeltas(learningRate, errorSignals, outputs, inputs, weights, lastWeightDeltas, bias, momentum);
    
    const newWeights = getNewWeights(weights, weightDeltas);
    
    const error = getSquaredError(outputs[outputs.length - 1], targets);
    
    const classifiedCorrectly = classificationFunction(outputs[outputs.length - 1], targets);
    
    return {
      outputs: outputs,
      errorSignals: errorSignals,
      weights: JSON.parse(JSON.stringify(weights)),
      weightDeltas: weightDeltas,
      newWeights: newWeights,
      squaredError: error,
      targets: targets,
      classifiedCorrectly: classifiedCorrectly
    };
  }
  
  function train(learningRate, inputs, targets, nodeCounts, initialWeights = null, validationInputs = null, validationTargets = null, momentum = 0, bias = 1, outputFunction = getOutput, outputDerivativeFunction = getOutputDerivative, declinePercent = 1, maxDeclineCount = 10, maxIterations = 9999, shuffle = true, classificationAccuracyFunction = getClassificationAccuracy) {    
    if (!Array.isArray(initialWeights) || initialWeights.length === 0) {
      initialWeights = randomWeights(nodeCounts, inputs[0].length);
    }
    
    let trainResults = {
      finalEpochIndex: 0,
      epochResults: [],
      inputs: inputs,
      initWeights: initialWeights,
      targets: targets,
      finalWeights: null,
      finalError: 0
    };
    let errors = [{
      msse: Infinity,
      epochIndex: -1,
      countSinceLastImprovement: 0
    }];
    let weights = JSON.parse(JSON.stringify(initialWeights));
    let deltas;
    let iteration = 0;
    let continueTraining = true;
    
    while (continueTraining && iteration < maxIterations) {
      let sumSquaredError = 0;
      let correctCount = 0;
      let lastWeights;
      
      const allPatternResults = inputs.reduce((results, pattern, patternIndex) => {       
        const patternTargets = targets.map((nodeTargets) => nodeTargets[patternIndex]);
        let patternResults = runSinglePattern(learningRate, weights, pattern, patternTargets, bias, nodeCounts, momentum, deltas, outputFunction, outputDerivativeFunction, classificationAccuracyFunction);
        
        weights = patternResults.newWeights;
        sumSquaredError += patternResults.squaredError;
        deltas = patternResults.weightDeltas;
        lastWeights = patternResults.weights;
        correctCount += patternResults.classifiedCorrectly;
        
        return results.concat({
          outputs: patternResults.outputs[patternResults.outputs.length - 1],
          targets: patternResults.targets
        });
      }, []);
      
      if (shuffle) {
        [inputs, targets] = shuffleArrays(inputs, targets);
      }
      
      let epochResult = {
        patternResults: allPatternResults,
        trainMeanSumSquaredError: sumSquaredError / inputs.length,
        outputs: allPatternResults[allPatternResults.length - 1].outputs,
        index: iteration,
        trainAccuracy: correctCount / inputs.length,
        validateMeanSumSquaredError: null,
        validateOutput: null,
        validateTargets: null,
        validateAccuracy: null
      };
      
      let errorTarget = epochResult.trainMeanSumSquaredError;
      
      if (validationInputs && validationTargets) {
        const validationOutput = run(nodeCounts, lastWeights, validationInputs, validationTargets, bias, outputFunction);
        
        errorTarget = validationOutput.meanSquaredError;
        epochResult.validateMeanSumSquaredError = validationOutput.meanSquaredError;
        epochResult.validateOutput = validationOutput.patternOutputs;
        epochResult.validateTargets = validationOutput.patternTargets;
        epochResult.validateAccuracy = validationOutput.accuracy;
      }
      
      const errorBetter = errorTarget - errors[0].msse <= -errors[0].msse * declinePercent / 100;
      
      if (errorBetter) {
        errors = [{
          msse: errorTarget,
          epochIndex: iteration,
          countSinceLastImprovement: iteration - errors[0].countSinceLastImprovement,
          weights: lastWeights
        }];
      }
      else {
        // error got worse
        errors.push({
          msse: errorTarget,
          epochIndex: iteration
        });
      }
      
      if (errors.length === maxDeclineCount) {
        continueTraining = false;
      }
      
      trainResults.epochResults.push(epochResult);
      
      iteration++;
    }
    
    trainResults.finalEpochIndex = errors[0].epochIndex;
    trainResults.finalError = errors[0].msse;
    trainResults.finalTrainError = trainResults.epochResults[errors[0].epochIndex].trainMeanSumSquaredError;
    trainResults.finalWeights = errors[0].weights;
    
    return trainResults;
  }
  
  function run(nodeCounts, weights, inputs, targets, bias = 1, outputFunction = getOutput, classificationAccuracyFunction = getClassificationAccuracy) {
    
    let sumSquaredError = 0;
    let accuracyCount = 0;
    let allPatternTargets = [];
    
    const allPatternOutputs = inputs.reduce((results, pattern, patternIndex) => {
      let patternOutputs = getAllNodeOutputs(nodeCounts, weights, pattern, bias, outputFunction);
      
      let patternTargets = targets.map((targets) => targets[patternIndex]);
      
      let outputNodeOutputs = patternOutputs[patternOutputs.length - 1];
      
      sumSquaredError += getSquaredError(outputNodeOutputs, patternTargets);
      accuracyCount += getClassificationAccuracy(outputNodeOutputs, patternTargets);
      
      allPatternTargets.push(patternTargets);
      return results.concat([outputNodeOutputs]);
    }, []);
    
    return {
      patternOutputs: allPatternOutputs,
      patternTargets: allPatternTargets,
      meanSquaredError: sumSquaredError / inputs.length,
      accuracy: accuracyCount / inputs.length
    };
  }
  
  function randomWeights(nodeCounts, featureCount) {
    
    function getRandomWeight() {
      return Math.random() - 0.5;
    }
    
    return [featureCount].concat(nodeCounts).slice(0, nodeCounts.length).map((layerNodeCount, layerIndex) => {
      return Array(layerNodeCount + 1).fill(null).map(() => {
        return Array(nodeCounts[layerIndex]).fill(null).map(() => getRandomWeight());
      });
    });
  }
  
  return {
    _test_getNet: getNet,
    _test_getHiddenNodeErrorSignal: getHiddenNodeErrorSignal,
    _test_getAllNodeOutputs: getAllNodeOutputs,
    _test_initLayerOutputs: initNodeValues,
    _test_getAllErrorSignals: getAllNodeErrorSignals,
    _test_getAllWeightDeltas: getAllWeightDeltas,
    _test_getNewWeights: getNewWeights,
    _test_runSinglePattern: runSinglePattern,
    _test_getSquaredError: getSquaredError,
    _test_shuffleArrays: shuffleArrays,
    
    generateRandomWeights: randomWeights,
    train: train,
    run: run
  };
})();


// set module exports if in node
if (typeof module !== 'undefined' && typeof module === 'object') {
  Object.assign(module.exports, Backprop);
}