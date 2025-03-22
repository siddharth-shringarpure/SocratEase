const https = require('https');
const fs = require('fs');
const path = require('path');

const models = [
  {
    name: 'tiny_face_detector_model-weights_manifest.json',
    url: 'https://github.com/justadudewhohacks/face-api.js/raw/master/weights/tiny_face_detector_model-weights_manifest.json'
  },
  {
    name: 'tiny_face_detector_model-shard1',
    url: 'https://github.com/justadudewhohacks/face-api.js/raw/master/weights/tiny_face_detector_model-shard1'
  },
  {
    name: 'face_expression_model-weights_manifest.json',
    url: 'https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_expression_model-weights_manifest.json'
  },
  {
    name: 'face_expression_model-shard1',
    url: 'https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_expression_model-shard1'
  }
];

const modelsDir = path.join(process.cwd(), 'public', 'models');

// Create models directory if it doesn't exist
if (!fs.existsSync(modelsDir)) {
  fs.mkdirSync(modelsDir, { recursive: true });
}

const downloadFile = (url, dest) => {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, response => {
      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', err => {
      fs.unlink(dest, () => reject(err));
    });
  });
};

const downloadModels = async () => {
  console.log('Downloading face-api.js models...');
  for (const model of models) {
    const dest = path.join(modelsDir, model.name);
    console.log(`Downloading ${model.name}...`);
    await downloadFile(model.url, dest);
  }
  console.log('All models downloaded successfully!');
};

downloadModels().catch(console.error); 