let x_vals = [];
let y_vals = [];
let x_vals_tensor;
let y_vals_tensor;

let model;

var table;

/*let education = table.get("EDUCATION");
let souuth = table.get("SOUTH");
let sex = table.get("SEX");
let experience = table.get("EXPERIENCE");
let union = table.get("UNION");
let wage = table.get("WAGE");
let age = table.get("AGE");
let race = table.get("RACE");
let occupation = table.get("OCCUPATION");
let sector = table.get("SECTOR");
let marriage = table.get("MARR");

function preload() {
  table = loadTable('wages2.csv', 'csv', 'header');
}*/

function setup() {
  createCanvas(400, 400);
  setupTfModel();
}


//neural network builder
function setupTfModel() {
  model = tf.sequntial();
  //first layer
  let hidden1 = tf.layers.dense( {
  inputShape: [10], 
  units: 64, 
  activation: 'linear'
  });
  //second layer
  let hidden2 = tf.layers.dense( {
  units: 128, 
  activation: 'linear'
  });
  //output
  let output = tf.layers.dense( {
  units: 1, 
  activation: 'linear'
  });
  //putting the network togther
  model.add(hidden1);
  model.add(hidden2);
  model.add(output);

  //training
  const optimizer = tf.train.adam(0.2);
  model.compile( {
  optimizer: optimizer, 
  loss: 'meanSquaredError'
  })
}


function draw() {
  background(0);
  //drawing the 
  drawSelectedPoints();
  predictCurveAndDraw();

  if (x_vals.length > 0) {
    model.fit(x_vals_tensor, y_vals_tensor, {
    shuffle: false, 
    epochs: 1
    });
  }
  console.log(tf.memory().numTensors);
}

function predictCurveAndDraw() {
  tf.tidy(() => {

    let curveX = [];
    let curveY = [];
    for (let x = -1; x <= 1; x += 0.05) {
      curveX.push([x]);
    }

    //get predictions
    ys = model.predict(tf.tensor2d(curveX));
    curveY = ys.dataSync();
    drawPredictionCurve(curveX, CurveY);
  });
}

function drawPredictionCurve(curveX, curveY) {
  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);
    vertex(x, y);
  }
  endShape();
}

function drawSelectedPoints(){
  stroke(255);
  strokeWeight(8);
  for(let i = 0; i < x_vals.length; i++){
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }
}

function mousePressed(){
  if(x_vals_tensor) x_vals_tensor.dispose();
  if(y_vals_tensor) y_vals_tensor.dispose();
  let x = map(mouseX, 0, width, -1, 1);
  let y = map(mouseY, 0, height, 1, -1);
  x_vals.push([x]);
  y_vals.push([y]);
  x_vals_tensor = tf.tensor2d(x_vals);
  y_vals_tensor = tf.tensor2d(y_vals);
}
