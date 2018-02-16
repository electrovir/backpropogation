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
  ]
};

function runWebTests() {
  
  const results = [ 
    testHiddenNodeErrorSignal(), 
    testNet(),
    testInitLayerOutputs(),
    testGetAllNodeOutputs(),
    testAllErrorSignals()
  ];
  
  const passed = results.reduce( (total, current) => total && current.passed.reduce((a, b) => a && b, true), true);
  
  console.log('all passed:', passed, 'results:', results);
}

function testAllErrorSignals() {
  
  const outputs = Backprop._test_getAllNodeOutputs(Backprop._test_initLayerOutputs([2, 1]), TEST_VALUES.testWeights, TEST_VALUES.inputs, 1);
  
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
    }
  ];
    
  return returnResult(results, 'initLayers');
}

function testGetAllNodeOutputs() {
  results = [
    {
      result: Backprop._test_getAllNodeOutputs(Backprop._test_initLayerOutputs([2, 1]), TEST_VALUES.testWeights, TEST_VALUES.inputs, 1),
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
    return equal(testObject.result, testObject.target);
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