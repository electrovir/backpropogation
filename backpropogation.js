const Backprop = (() => {
  
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
    // let errorSignals = initNodeValues(nodeCounts, false);
    
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
      return sum + Math.pow(output - outputIndex, 2);
    }, 0);
  }
  
  function runSinglePattern(learningRate, weights, inputs, targets, bias, nodeCounts, momentum, lastWeightDeltas, outputFunction, derivativeFunction) {
    
    const outputs = getAllNodeOutputs(nodeCounts, weights, inputs, bias, outputFunction);
    
    const errorSignals = getAllNodeErrorSignals(outputs, weights, targets, derivativeFunction);
    
    const weightDeltas = getAllWeightDeltas(learningRate, errorSignals, outputs, inputs, weights, lastWeightDeltas, bias, momentum);
    
    const newWeights = getNewWeights(weights, weightDeltas);
    
    const error = getSquaredError(outputs[outputs.length - 1], targets);
    
    return {
      outputs: outputs,
      errorSignals: errorSignals,
      weights: JSON.parse(JSON.stringify(weights)),
      weightDeltas: weightDeltas,
      newWeights: newWeights,
      squaredError: error
    };
  }
  
  function train(learningRate, initialWeights, inputs, targets, nodeCounts, validationInputs, validationTargets, momentum = 0, bias = 1, outputFunction = getOutput, outputDerivativeFunction = getOutputDerivative, declinePercent = 1, maxDeclineCount = 10) {
    
    let trainResults = {
      finalEpochIndex: 0,
      epochResults: [],
      inputs: inputs,
      initWeights: initialWeights,
      targets: targets,
      finalWeights: null,
      finalError: 0,
      finalOutputs: null
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
    
    while (continueTraining && iteration < 9999) {
      sumSquaredError = 0;
      
      const allPatternResults = inputs.reduce((results, pattern, patternIndex) => {        
        let patternResults = runSinglePattern(learningRate, weights, pattern, targets[patternIndex], bias, nodeCounts, momentum, deltas, outputFunction, outputDerivativeFunction);
        
        weights = patternResults.newWeights;
        sumSquaredError += patternResults.squaredError;
        deltas = patternResults.weightDeltas;
        
        return results.concat(patternResults);
      }, []);
      
      let epochResult = {
        patternResults: allPatternResults,
        trainMeanSumSquaredError: sumSquaredError / inputs.length,
        weights: allPatternResults[allPatternResults.length - 1].weights,
        outputs: allPatternResults[allPatternResults.length - 1].outputs
      };
      
      let errorChange = epochResult.trainMeanSumSquaredError - errors[0].msse;
      
      if (validationInputs && validationTargets) {
        const validationOutput = run(nodeCounts, epochResult.weights, validationInputs, validationTargets, bias, outputFunction);
        
        errorChange = validationOutput.meanSquaredError - errors[0].msse;
        epochResult.validateMeanSumSquaredError = validationOutput.meanSquaredError;
      }
      
      const errorBetter = errorChange <= -errors[0].msse * declinePercent / 100;
      
      if (errorBetter) {
        errors = [{
          msse: epochResult.trainMeanSumSquaredError,
          epochIndex: iteration,
          countSinceLastImprovement: iteration - errors[0].countSinceLastImprovement
        }];
      }
      else {
        // error got worse
        errors.push({
          msse: epochResult.trainMeanSumSquaredError,
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
    trainResults.finalWeights = trainResults.epochResults[trainResults.finalEpochIndex].weights;
    trainResults.finalOutputs = trainResults.epochResults[trainResults.finalEpochIndex].outputs;
    
    return trainResults;
  }
  
  function run(nodeCounts, weights, inputs, targets, bias = 1, outputFunction = getOutput) {
    
    let sumSquaredError = 0;
    
    const allPatternOutputs = inputs.reduce((results, pattern) => {        
      let patternOutputs = getAllNodeOutputs(nodeCounts, weights, pattern, bias, outputFunction);    
      sumSquaredError += getSquaredError(patternOutputs[patternOutputs.length - 1], targets);
      
      return results.concat(patternOutputs);
    }, []);
    
    return {
      patternOutputs: allPatternOutputs,
      meanSquaredError: sumSquaredError / inputs.length
    };
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
    
    train: train,
    run: run
  };
})();


// set module exports if in node
if (typeof module !== 'undefined' && typeof module === 'object') {
  Object.assign(module.exports, Perceptron);
}