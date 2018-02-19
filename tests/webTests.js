// these test parameters have three layers:
//    one input layer with 2 inputs + bias node
//    one hidden layer with 2 hidden nodes + bias node
//    output layer with a single output node
const TEST_VALUES = {
  targets: [
    0
  ],
  testWeights: [
    // between inputs layer and hidden layer
    [
      [
        1, 2
      ],
      [
        1, 1.4
      ],
      [
        1.00113, 0.5
      ]
    ],
    // between hidden layer and output layer
    [
      [
        1.0042
      ],
      [
        1.0042
      ],
      [
        1.00575
      ]
    ]
  ],
  inputs: [
    0,
    1
  ],
  testNodeOutputs: [
    [
      null, null
    ],
    [
      null
    ]
  ],
  outputNodeErrorSignalArray: [
    [],
    [-0.0522]
  ],
  learningRate: 0.1,
  size: [2, 1]
};

function runWebTests() {
  
  const results = [ 
    testHiddenNodeErrorSignal(), 
    testNet(),
    testInitLayerOutputs(),
    testGetAllNodeOutputs(),
    testAllErrorSignals(),
    testAllWeightDeltas(),
    testNewWeights(),
    testPattern(),
    testBackprop(),
    testRandomWeights()
  ];
  
  const passed = results.reduce( (total, current) => total && current.passed.reduce((a, b) => a && b, true), true);
  
  console.log('all passed:', passed, 'results:', results);
}

function testRandomWeights() {
  
  const weights = Backprop.generateRandomWeights(TEST_VALUES.size, TEST_VALUES.inputs.length);
  
  const results = [
    {
      data: weights,
      result: weights.length,
      target: 2
    },
    {
      data: weights[0],
      result: weights[0].length,
      target: 3
    },
    {
      data: weights[1],
      result: weights[1].length,
      target: 3
    },
    {
      data: weights[0][1],
      result: weights[0][1].length,
      target: 2
    },
    {
      data: weights[1][0],
      result: weights[1][0].length,
      target: 1
    }
  ];
  
  return returnResult(results, 'randomWeights');
}

function testBackprop() {
  const trainZeroMomentum = Backprop.train(TEST_VALUES.learningRate, [TEST_VALUES.inputs], [TEST_VALUES.targets], TEST_VALUES.size, TEST_VALUES.testWeights, undefined, undefined, 0, 1, undefined, undefined, 1, 10, 9999, false);
  const trainWithValidation = Backprop.train(TEST_VALUES.learningRate, [TEST_VALUES.inputs], [TEST_VALUES.targets], TEST_VALUES.size, TEST_VALUES.testWeights, [TEST_VALUES.inputs], [TEST_VALUES.targets], 0, 1, undefined, undefined, 1, 10, 9999, false);
  
  const results = [
    {
      data: trainZeroMomentum,
      result: trainZeroMomentum.finalEpochIndex,
      target: 1100,
      condition: trainZeroMomentum.finalOutputs[1][0] - TEST_VALUES.targets[0] < 0.1
    },
    {
      data: trainWithValidation,
      result: trainWithValidation.finalEpochIndex,
      target: 1100,
      condition: trainWithValidation.finalOutputs[1][0] - TEST_VALUES.targets[0] < 0.1
    }
  ];
  
  return returnResult(results, 'fullBackprop');
}

function testNewWeights() {
  
  const outputs = Backprop._test_getAllNodeOutputs(TEST_VALUES.size, TEST_VALUES.testWeights, TEST_VALUES.inputs, 1);
  const errorSignals = Backprop._test_getAllErrorSignals(outputs, TEST_VALUES.testWeights, TEST_VALUES.targets);
  const weightDeltas = Backprop._test_getAllWeightDeltas(TEST_VALUES.learningRate, errorSignals, outputs, TEST_VALUES.inputs, TEST_VALUES.testWeights, [[[0,0],[0,0],[0,0]],[[0],[0],[0]]], 1, 0);
  
  const results = [
    {
      result: Backprop._test_getNewWeights(TEST_VALUES.testWeights, weightDeltas),
      target: [[[1, 2], [0.9994471549545654, 1.3994035347424334], [1.0005771549545655, 0.4994035347424335]], [[0.9995769500471305], [0.9996348048116477], [1.000501994870844]]]
    }
  ];
  
  return returnResult(results, 'newWeights');
  
}

function testPattern() {
  const results = [
    {
      result: Backprop._test_runSinglePattern(TEST_VALUES.learningRate, TEST_VALUES.testWeights, TEST_VALUES.inputs, TEST_VALUES.targets, 1, TEST_VALUES.size, 0),
      target: {
        outputs: [[0.8809156696866742, 0.8698915256370021], [0.940694176634854]],
        errorSignals: [[-0.005528450454346137, -0.00596465257566539], [-0.052480051291559186]],
        weights: TEST_VALUES.testWeights,
        weightDeltas: [[[0, 0], [-0.0005528450454346137, -0.0005964652575665391], [-0.0005528450454346137, -0.0005964652575665391]], [[-0.004623049952869488], [-0.004565195188352254], [-0.005248005129155919]]],
        squaredError: 0.8849055339547259,
        newWeights: [[[1, 2], [0.9994471549545654, 1.3994035347424334], [1.0005771549545655, 0.4994035347424335]], [[0.9995769500471305], [0.9996348048116477], [1.000501994870844]]]
      }
    }
  ];
  
  return returnResult(results, 'singlePattern');
}

function testAllWeightDeltas() {
  
  const outputs = Backprop._test_getAllNodeOutputs(TEST_VALUES.size, TEST_VALUES.testWeights, TEST_VALUES.inputs, 1);
  const errorSignals = Backprop._test_getAllErrorSignals(outputs, TEST_VALUES.testWeights, TEST_VALUES.targets);
  
  const results = [
    {
      result: Backprop._test_getAllWeightDeltas(TEST_VALUES.learningRate, errorSignals, outputs, TEST_VALUES.inputs, TEST_VALUES.testWeights, [[[0,0],[0,0],[0,0]],[[0],[0],[0]]], 1, 0),
      target: [[[0, 0], [-0.0005528450454346137, -0.0005964652575665391], [-0.0005528450454346137, -0.0005964652575665391]], [[-0.004623049952869488], [-0.004565195188352254], [-0.005248005129155919]]]
    }
  ];
  
  return returnResult(results, 'allWeightDeltas');
}

function testAllErrorSignals() {
  
  const outputs = Backprop._test_getAllNodeOutputs(TEST_VALUES.size, TEST_VALUES.testWeights, TEST_VALUES.inputs, 1);
  
  let results = [
    {
      result: Backprop._test_getAllErrorSignals(outputs, TEST_VALUES.testWeights, TEST_VALUES.targets),
      target: [[-0.005528450454346137, -0.00596465257566539], [-0.052480051291559186]]
    }
  ];
  
  return returnResult(results, 'allErrorSignals');
}

function testHiddenNodeErrorSignal() {
  let results = [
    {
      result: Backprop._test_getHiddenNodeErrorSignal(0.104839, TEST_VALUES.testWeights, TEST_VALUES.outputNodeErrorSignalArray, 0, 0),
      target: -0.005495580702360001
    }
  ];
  
  return returnResult(results, 'hiddenNodeErrorSignal');
}

function testNet() {
  let results = [
    {
      result: Backprop._test_getNet(TEST_VALUES.testWeights, TEST_VALUES.inputs, 0, 0, 1),
      target: 2.00113
    },
    {
      result: Backprop._test_getNet(TEST_VALUES.testWeights, TEST_VALUES.inputs, 0, 1, 1),
      target: 1.9
    }
  ];
  
  return returnResult(results, 'net');  
}

function testInitLayerOutputs() {
  results = [
    {
      result: Backprop._test_initLayerOutputs([2, 1]),
      target: [[null, null],[null]]
    },
    {
      result: Backprop._test_initLayerOutputs([2, 1], true),
      target: [[null, null, null],[null]]
    },
    {
      result: Backprop._test_initLayerOutputs([3, 3, 2], true),
      target: [[null, null, null, null], [null, null, null, null], [null, null]]
    },
  ];
    
  return returnResult(results, 'initLayers');
}

function testGetAllNodeOutputs() {
  results = [
    {
      result: Backprop._test_getAllNodeOutputs(TEST_VALUES.size, TEST_VALUES.testWeights, TEST_VALUES.inputs, 1),
      target: [[0.8809156696866742, 0.8698915256370021], [0.940694176634854]]
    }
  ];
  
  return returnResult(results, 'allNodeOutputs');
}

function returnResult(results, testName) {
  return {
    passed: compareResults(results),
    results: results,
    test: testName
  };
}

function compareResults(results) {
  return results.map((testObject) => {
    return equal(testObject.result, testObject.target)  && ((testObject.condition !== undefined && testObject.condition) || testObject.condition === undefined);
  });
}



// ripped this straight out of fast-deep-equal: https://github.com/epoberezkin/fast-deep-equal
// made it browser friendly and reformated some weird lines
function equal(a, b) {
  if (a === b) return true;

  let arrA = Array.isArray(a);
  let arrB = Array.isArray(b);
  let i;

  if (arrA && arrB) {
    if (a.length != b.length) return false;
    for (i = 0; i < a.length; i++)
      if (!equal(a[i], b[i])) return false;
    return true;
  }

  if (arrA != arrB) return false;

  if (a && b && typeof a === 'object' && typeof b === 'object') {
    let keys = Object.keys(a);
    if (keys.length !== Object.keys(b).length) return false;

    let dateA = a instanceof Date;
    let dateB = b instanceof Date;
    if (dateA && dateB) return a.getTime() == b.getTime();
    if (dateA != dateB) return false;

    let regexpA = a instanceof RegExp;
    let regexpB = b instanceof RegExp;
    if (regexpA && regexpB) return a.toString() == b.toString();
    if (regexpA != regexpB) return false;

    for (i = 0; i < keys.length; i++)
      if (!Object.prototype.hasOwnProperty.call(b, keys[i])) return false;

    for (i = 0; i < keys.length; i++)
      if(!equal(a[keys[i]], b[keys[i]])) return false;

    return true;
  }

  return false;
}