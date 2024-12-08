const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');
const admin = require('firebase-admin');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 8080;

// Firebase setup
const serviceAccount = require('./serviceAccountKey.json');
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    storageBucket: 'gs://asclepius-backend.firebasestorage.app',
});
const db = admin.firestore();

// Multer setup for file uploads
const upload = multer({
    limits: { fileSize: 1000000 }, // Max file size 1MB
    fileFilter(req, file, cb) {
        if (!file.mimetype.startsWith('image/')) {
            return cb(new Error('File must be an image'));
        }
        cb(null, true);
    }
});

// Model setup
const BUCKET_URL = 'https://storage.googleapis.com/bucket-applied/submissions-model/model.json';
let model;

async function loadModel() {
    try {
        model = await tf.loadLayersModel(BUCKET_URL);
        console.log('Model loaded successfully!');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Load model when the server starts
loadModel();

// API Endpoint
app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        if (!model) {
            return res.status(500).send({
                status: 'fail',
                message: 'Model is not loaded yet. Please try again later.',
            });
        }

        const file = req.file;
        if (!file) {
            return res.status(400).send({
                status: 'fail',
                message: 'File not uploaded',
            });
        }

        // Prepare the image for prediction
        const buffer = file.buffer;
        const imageTensor = tf.node.decodeImage(buffer, 3)
            .resizeBilinear([224, 224])
            .expandDims(0)
            .div(tf.scalar(255));

        // Predict using the loaded model
        const prediction = model.predict(imageTensor);
        const predictionArray = await prediction.array();
        const result = predictionArray[0][0] > 0.5 ? 'Cancer' : 'Non-cancer';
        const suggestion = result === 'Cancer'
            ? 'Segera periksa ke dokter!'
            : 'Penyakit kanker tidak terdeteksi.';
        const id = uuidv4();
        const createdAt = new Date().toISOString();

        // Save prediction to Firestore
        await db.collection('predictions').doc(id).set({
            id,
            result,
            suggestion,
            createdAt,
        });

        // Response
        res.status(200).send({
            status: 'success',
            message: 'Prediction successful',
            data: {
                id,
                result,
                suggestion,
                createdAt,
            },
        });
    } catch (error) {
        console.error('Error during prediction:', error);
        res.status(500).send({
            status: 'fail',
            message: 'An error occurred during prediction',
        });
    }
});

// Firestore test endpoint
app.get('/test-firestore', async (req, res) => {
    try {
        const docRef = db.collection('test').doc('testDoc');
        await docRef.set({
            message: 'Hello, Firestore!',
            timestamp: new Date().toISOString(),
        });
        res.status(200).send({ status: 'success', message: 'Firestore connected!' });
    } catch (error) {
        console.error('Error testing Firestore:', error);
        res.status(500).send({ status: 'fail', message: error.message });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    if (err.message === 'File too large') {
        return res.status(413).send({
            status: 'fail',
            message: 'Payload content length greater than maximum allowed: 1MB',
        });
    }
    res.status(err.status || 500).send({
        status: 'fail',
        message: err.message || 'Internal Server Error',
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
