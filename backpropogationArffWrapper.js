const ARFF = require('arff-toolkit');
const BACKPROP = require('./backpropogation.js');

function arffTrainValidateTest(inputFileName, trainSplit, validationSplit, layerSizes, callback, targetAttributes, patternAttributes, options = {}) {
  return ARFF.loadArff(inputFileName, (data) => {
    
    targetAttributes = ARFF.separateMultiClassArffData(data, targetAttributes);
    layerSizes[layerSizes.length - 1] = targetAttributes.length;
    
    
    const arffSplits = ARFF.splitArffTrainValidateTest(data, trainSplit, validationSplit);
    const arffInputs = {
      train: ARFF.arffToInputs(arffSplits.trainArffData, targetAttributes, patternAttributes, options),
      validate: ARFF.arffToInputs(arffSplits.validationArffData, targetAttributes, patternAttributes, options),
      test: ARFF.arffToInputs(arffSplits.testArffData, targetAttributes, patternAttributes, options)
    };
    
    const trainResults = BACKPROP.train(
      options.learningRate || 0.1,
      arffInputs.train.patterns,
      arffInputs.train.targetColumns,
      layerSizes,
      undefined,
      arffInputs.validate.patterns,
      arffInputs.validate.targetColumns,
      options.momentum || 0,
      options.bias || 1,
      options.activationFunction || undefined,
      options.derivativeFunction || undefined,
      options.declinePercent || 1,
      options.maxDeclineCount || 10,
      options.maxIterations || 999,
      options.shuffle || true,
      options.classificationAccuracyFunction || undefined
    );
    
    const testResults = BACKPROP.run(
      layerSizes,
      trainResults.finalWeights,
      arffInputs.test.patterns,
      arffInputs.test.targetColumns,
      options.bias || 1,
      options.activationFunction || undefined, 
      options.classificationAccuracyFunction || undefined
    );
    
    callback(trainResults, testResults);
  });
}

module.exports = {
  trainValidateTest: arffTrainValidateTest
};