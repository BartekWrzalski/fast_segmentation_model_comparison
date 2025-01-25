package com.example.seg_java;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;
import android.app.Activity;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import ai.onnxruntime.*;

public class ModelTester extends Activity {

    private OrtEnvironment env;
    private OrtSession session;

    private static final String IMAGES_PATH = "images";
    private static final String MASKS_PATH = "masks";

    @SuppressLint("SetTextI18n")
    public double[] runTests(Context context, String modelName) throws IOException {
        List<String> imageFiles = getImageNamesFromAssets(context);

        try {
            env = OrtEnvironment.getEnvironment();
            InputStream modelStream = context.getAssets().open("models/" + modelName);
            byte[] modelBytes = new byte[modelStream.available()];
            modelStream.read(modelBytes);
            session = env.createSession(modelBytes, new OrtSession.SessionOptions());
            Log.i("ModelTester", "Testing model: " + modelName);
        } catch (IOException | OrtException e) {
            Log.e("ModelTester", "Failed loading " + modelName);
            e.printStackTrace();
            return null;
        }

        long totalInferenceTime = 0;
        double totalMiou = 0;
        int processedImages = 0;

        for (String imageFile : imageFiles) {
            String maskFile = imageFile.replace(".jpg", ".png");

            // Preprocess image
            Bitmap bitmap = null;
            try {
                bitmap = BitmapFactory.decodeStream(context.getAssets().open(IMAGES_PATH + "/" + imageFile));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 512, 512, true);
            float[] inputTensor = preprocessImage(resizedBitmap);

            // Run inference
            OnnxTensor input;
            OrtSession.Result result;
            long inferenceTime;
            float[][][][] outputTensor;

            try {
                input = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensor), new long[]{1, 3, 512, 512});

                long startTime = System.nanoTime();
                result = session.run(Collections.singletonMap("l_x_", input));
                inferenceTime = System.nanoTime() - startTime;

                // Postprocess output
                outputTensor = (float[][][][]) result.get(0).getValue();
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
            Bitmap outputMask = postprocessOutput(outputTensor);

            // Load ground truth mask
            Bitmap groundTruthMask = null;
            try {
                groundTruthMask = BitmapFactory.decodeStream(context.getAssets().open(MASKS_PATH + "/" + maskFile));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            Bitmap resizedMask = Bitmap.createScaledBitmap(groundTruthMask, 512, 512, false);

            // Calculate IoU
            double iou = calculateMiou(outputMask, resizedMask);
            totalMiou += iou;

            // Track inference time
            totalInferenceTime += inferenceTime;
            processedImages++;

            Log.i("ModelTester", "Model: " + modelName + "Image: " + imageFile + " - IoU: " + iou + ", Time: " + inferenceTime / 1e6 + " ms");
        }

        // Calculate averages
        double avgInferenceTime = (double) totalInferenceTime / processedImages / 1e6;
        double avgMiou = totalMiou / processedImages;

        double[] metrics = new double[]{avgMiou, avgInferenceTime};

        Log.i("ModelTester", "Model: " + modelName + " - Avg Time: " + avgInferenceTime + " ms, Avg MIoU: " + avgMiou);
        try {
            session.close();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return metrics;
    }

    private List<String> getImageNamesFromAssets(Context context) {
        List<String> imageNames = new ArrayList<>();
        try {
            String[] files = context.getAssets().list("images");
            List<String> filesList = Arrays.asList(files);
            Object[] results = filesList.stream().filter(s -> s.endsWith("jpg")).toArray();
            String[] stringResults = Arrays.copyOf(results, results.length, String[].class);
            if (files != null) {
                Collections.addAll(imageNames, stringResults);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return imageNames;
    }

    private float[] preprocessImage(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float[] input = new float[3 * width * height];

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            input[i] = r / 255.0f;                     // Normalizing to [0, 1]
            input[i + width * height] = g / 255.0f;
            input[i + 2 * width * height] = b / 255.0f;
        }
        return input;
    }

    private Bitmap postprocessOutput(float[][][][] outputTensor) {
        int numClasses = outputTensor[0].length;
        int height = outputTensor[0][0].length;
        int width = outputTensor[0][0][0].length;
        Bitmap outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int predictedClass = 0;
                float maxProbability = -Float.MAX_VALUE;

                for (int c = 0; c < numClasses; c++) {
                    if (outputTensor[0][c][i][j] > maxProbability) {
                        maxProbability = outputTensor[0][c][i][j];
                        predictedClass = c;
                    }
                }

                outputBitmap.setPixel(j, i, getColorForClass(predictedClass));
            }
        }

        return outputBitmap;
    }

    private int getColorForClass(int classId) {
        int[] colors = {
                0xFF000000, // Klasa 0:'background'
                0xFF800000, // Klasa 1:'aeroplane'
                0xFF008000, // Klasa 2:'bicycle'
                0xFF808000, // Klasa 3:'bird'
                0xFF000080, // Klasa 4:'boat'
                0xFF800080, // Klasa 5:'bottle'
                0xFF008080, // Klasa 6:'bus'
                0xFF808080, // Klasa 7:'car'
                0xFF400000, // Klasa 8:'cat'
                0xFFC00000, // Klasa 9:'chair'
                0xFF408000, // Klasa 10:'cow'
                0xFFC08080, // Klasa 11:'diningtable'
                0xFF400080, // Klasa 12:'dog'
                0xFFC00080, // Klasa 13:'horse'
                0xFF408080, // Klasa 14:'motorbike'
                0xFFC08080, // Klasa 15:'person'
                0xFF004000, // Klasa 16:'pottedplant'
                0xFF804000, // Klasa 17:'sheep'
                0xFF00C000, // Klasa 18:'sofa'
                0xFF80C000, // Klasa 19:'train'
                0xFF004080  // Klasa 20:'tvmonitor'
        };
        return colors[classId % colors.length];
    }

    private double calculateMiou(Bitmap outputMask, Bitmap groundTruth) {
        int width = outputMask.getWidth();
        int height = outputMask.getHeight();

        HashMap<Integer, Integer> intersection = new HashMap<>();
        HashMap<Integer, Integer> union = new HashMap<>();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int outputPixel = outputMask.getPixel(x, y) & 0xFFFFFF;  // Ignore alpha
                int groundTruthPixel = groundTruth.getPixel(x, y) & 0xFFFFFF;

                if (outputPixel == groundTruthPixel) {
                    intersection.putIfAbsent(outputPixel, 0);
                    union.putIfAbsent(outputPixel, 0);

                    intersection.put(outputPixel, intersection.get(outputPixel) + 1);
                    union.put(outputPixel, union.get(outputPixel) + 1);
                }
                else if (groundTruthPixel != 14737600) { // white line
                    union.putIfAbsent(outputPixel, 0);
                    union.putIfAbsent(groundTruthPixel, 0);

                    union.put(outputPixel, union.get(outputPixel) + 1);
                    union.put(groundTruthPixel, union.get(groundTruthPixel) + 1);
                }
            }
        }
        double miou = 0.0;
        int classes = 0;

        for (Integer value: union.keySet()) {
            if (intersection.containsKey(value)) {
                miou += (double) intersection.get(value) / union.get(value);
            }
            classes += 1;
        }

        miou = miou / classes;
        return miou;
    }
}
