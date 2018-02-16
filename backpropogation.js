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
  
  function getWeightChange(learningRate, nodeOutput, errorSignal) {
    return learningRate * nodeOutput * errorSignal;
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
  function initNodeValues(nodeCounts) {
    return nodeCounts.map(count => Array(count).fill(null));
  }
  
  // NOTE: THIS MUTATES THE lastOutputs VARIABLE
  function getAllNodeOutputs(lastOutputs, weights, inputs, bias, outputFunction = getOutput) {
    
    function mapOutputLayer(layerIndex, node, nodeIndex) {
        
        let net;
        
        if (layerIndex === 0) {
          net = getNet(weights, inputs, layerIndex, nodeIndex, bias);
        }
        else {
          net = getNet(weights, lastOutputs[layerIndex - 1], layerIndex, nodeIndex, bias);
        }
        
        return outputFunction(net);
      }
    
    for (let layerIndex = 0; layerIndex < lastOutputs.length; layerIndex++) {
      
      lastOutputs[layerIndex] = lastOutputs[layerIndex].map(mapOutputLayer.bind(null, layerIndex));
    }
    
    // THIS MUTATES
    return lastOutputs;
  }
  
  function getAllNodeErrorSignals(outputs, weights, targets, derivativeFunction = getOutputDerivative) {
    
    let errorSignals = initNodeValues(outputs.reduce((sizes, layer) => sizes.concat(layer.length), []));
    
    function mapErrorSignalLayer(layerIndex, node, nodeIndex) {
      
      const nodeOutput = outputs[layerIndex][nodeIndex];
      
      if (layerIndex === errorSignals.length - 1) {
        return getOutputErrorSignal(nodeOutput, derivativeFunction(nodeOutput), targets[nodeIndex]);
      }
      else {
        return getHiddenNodeErrorSignal(derivativeFunction(nodeOutput), weights, errorSignals, layerIndex, nodeIndex);
      }
    }
    
    // go backwards from the end of the array (the output layer)
    for (let layerIndex = errorSignals.length - 1; layerIndex >= 0; layerIndex--) {
      errorSignals[layerIndex] = errorSignals[layerIndex].map(mapErrorSignalLayer.bind(null, layerIndex));
    }
    
    return errorSignals;
  }
  
  return {
    _test_getNet: getNet,
    _test_getHiddenNodeErrorSignal: getHiddenNodeErrorSignal,
    _test_getAllNodeOutputs: getAllNodeOutputs,
    _test_initLayerOutputs: initNodeValues,
    _test_getAllErrorSignals: getAllNodeErrorSignals
  };
})();


// set module exports if in node
if (typeof module !== 'undefined' && typeof module === 'object') {
  Object.assign(module.exports, Perceptron);
}