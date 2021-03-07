import './App.css';
import React, { useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Button, makeStyles, Paper, Grid, Typography } from '@material-ui/core';
import CameraAltIcon from '@material-ui/icons/CameraAlt';

var model;
var array = ["Acanthosis nigricans", "Acne", "Atopic dermatitis", "Cafe au lait macule", "Candidiasis", "Cellulitis", "Contact dermatitis", "Cutaneous larva migrans", "Eczema herpeticum", "Erythema multiforme", "Erythema nodosum", "Folliculitis", "Gianotti Crosti syndrome", "Hand, foot, and mouth disease", "Herpes simplex", "Herpes zoster", "Impetigo", "Insect bites, bed bug bites, or flea bites", "Molluscum contagiosum", "Morbilliform drug reaction", "Pityriasis alba", "Pityriasis rosea", "Purpura", "Scabies", "Seborrheic dermatitis", "Staphylococcal scalded skin syndrome", "Stevens Johnson syndrome / Toxic epidermal necrolysis", "Tinea corporis", "Tinea versicolor", "Urticaria", "Varicella", "Viral exanthem"]

const useStyles = makeStyles((theme) => ({
  root: {
    '& > *': {
      margin: theme.spacing(1),
    },
  },
  input: {
    display: 'none',
  },
  results: {
    display: 'flex',
    flexWrap: 'wrap',
    justifyContent: 'center',
    '& > *': {
      marginLeft: window.innerWidth>576 ? 40 : 0,
      width: 400,
      height: 400,
    },
  },
}));


function App() {

  const [predicting, setPredicting] = React.useState(false)
  const [instructions, setInstructions] = React.useState(true)
  const [results, setResults] = React.useState()
  const classes = useStyles();

  useEffect(() => {
    tf.setBackend('webgl').then(() => {
      loadModel()
    })
  }, [])

  const takeImage = async (fileInput) => {
    setInstructions(false)
    setPredicting(true)
    var file = fileInput.target.files[0]
    var imageType = /image.*/;

    if (file.type.match(imageType)) {
      var canvas;
      var img = new Image();
      img.src = window.URL.createObjectURL(file);
      img.onload = async () => {
        const tfImg = tf.browser.fromPixels(img)
        const smallImg = tf.image.resizeBilinear(tfImg, [224, 224]) // 600, 450
        const displayImg = tf.image.resizeBilinear(tfImg, [300, 300])
        var imgBuffer = []
        var displayBuffer = []
        for (var i = 0; i < smallImg.dataSync().length; i++) {
          imgBuffer.push(smallImg.dataSync()[i] / 255)
        }
        for (var i = 0; i < displayImg.dataSync().length; i++) {
          displayBuffer.push(displayImg.dataSync()[i] / 255)
        }
        const tf3d = tf.tensor3d(Array.from(displayBuffer), [300, 300, 3])
        const tf4d = tf.tensor4d(Array.from(imgBuffer), [1, 224, 224, 3])
        const tf4d1 = tf.cast(tf4d, 'float32')
        canvas = document.getElementById("canvas");
        canvas.width = tf3d.shape.width
        canvas.height = tf3d.shape.height
        await tf.browser.toPixels(tf3d, canvas);
        canvas.toDataURL()
        let predictions = await model.execute({ "Image": tf4d1 }, "e7a45c55-0e03-47cd-9ce3-f829fed48eeb/dense_2/BiasAdd")
        var result = predictions.dataSync()
        var top3s = [
          [array[predictions.dataSync().indexOf(Math.max(...result))], Math.max(...result).toFixed(2)],
        ]
        result[result.indexOf(Math.max(...result))] = -Infinity
        top3s.push([array[predictions.dataSync().indexOf(Math.max(...result))], Math.max(...result).toFixed(2)])
        result[result.indexOf(Math.max(...result))] = -Infinity
        top3s.push([array[predictions.dataSync().indexOf(Math.max(...result))], Math.max(...result).toFixed(2)])
        setResults(top3s)
        setPredicting(false)
      }

    } else {
      setPredicting(false)
    }
}


const loadModel = async () => {
  if (('indexedDB' in window)) {
  try{
    model = await tf.loadGraphModel('indexeddb://my-model');
    console.log("loaded from indexeddb")
}
catch(e){
  console.log(e)
  model = await tf.loadGraphModel('model/model.json');
  await model.save('indexeddb://my-model');
}
  }
  else{
  model = await tf.loadGraphModel('model/model.json');
  }
  }


  return (

    <div className="App">
      <div className="main">
        <Grid container style={{ marginTop: "20%" }} justify="center">
          <Grid>
            <div className="upload-image">
              <input
                accept="image/*"
                className={classes.input}
                id="contained-button-file"
                type="file"
                onChange={takeImage}
              />
              <label htmlFor="contained-button-file">
                <Button variant="contained" color="primary" component="span">
                  Upload &nbsp;
          <CameraAltIcon />
                </Button>
              </label>
              <canvas className="mt-3 round" id="canvas" />
            </div>
          </Grid>
          <Grid style={{maxWidth:"100%"}}>
            <div className={classes.results}>
              <Paper>
                {instructions && <div className="mt-3">
                  <Typography variant="h4" className="mt-2">Instructions</Typography>
                  <div className="mt-3 px-3 text-left">
                    <Typography variant="h6">1. Press upload button.</Typography>
                    <Typography variant="h6">2. Select the picture.</Typography>
                    <Typography variant="h6">3. Get the results</Typography>
                  </div>
                </div>}
                {predicting && <Typography style={{ marginTop: "43%" }} variant="h5">Predicting...</Typography>}
                {!predicting && !instructions && <div>
                  <Typography variant="h4" className="mt-2">Results</Typography>
                  {results.map((v, i) => {
                    return (
                      <div className='mt-3' key={i}>
                        <p className="px-3 text-left"><b>{v[0]}</b></p>
                        <p className="px-3 text-left">{v[1]}</p>
                        <div className="progress" style={{ width: "90%", marginLeft: "5%" }}>
                          <div className="progress-bar" style={{ width: (v[1] * 100) / results[0][1] + "%" }}></div><br />
                        </div>
                      </div>
                    )
                  })
                  }
                </div>
                }
              </Paper>
            </div>
          </Grid>
          </Grid>
        <p style={{marginTop:"20px",fontSize:"19px"}}><b>NOTE:</b> This app is for educational purposes only and should not be used to diagnose or treat any medical condition.</p>

      </div>
    </div>
  );
}

export default App;
