package com.example.seg_java;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;

import androidx.annotation.Nullable;

import ai.onnxruntime.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;


public class MainActivity extends Activity {

    private TextView statusText;
    private TextView imageNameText;
    private TextView miouText;
    private TextView timeText;
    private ImageView inputImageView, outputImageView, maskView;
    private OrtEnvironment env;
    private OrtSession session;
    private static final String MODEL_PATH = "models/";

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

//      // case 1 - select model and image
        setContentView(R.layout.activity_main);
        try {
            runHandTest();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // case 2 - select model and run all images
//        setContentView(R.layout.tester);
//        try {
//            testModel();
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
    }

    @SuppressLint("SetTextI18n")
    private void testModel() throws IOException {
        ListView listView = findViewById(R.id.modelListView);
        timeText = findViewById(R.id.timeView);
        miouText = findViewById(R.id.miouView);

        String[] modelFiles = getAssets().list(MODEL_PATH);
        List<String> modelNames = Arrays.asList(modelFiles);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, modelNames);
        listView.setAdapter(adapter);

        listView.setOnItemClickListener((parent, view, position, id) -> {
            String modelName = modelNames.get(position);
            ModelTester tester = new ModelTester();
            try {
                double[] metrics = tester.runTests(this, modelName);

                miouText.setText("mIoU: " + metrics[0]);
                timeText.setText("Time: " + metrics[1]);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @SuppressLint("SetTextI18n")
    private void runHandTest() throws IOException {
        // Wczytaj model
        statusText = findViewById(R.id.statusText);
        imageNameText = findViewById(R.id.imageName);
        miouText = findViewById(R.id.miou);
        timeText = findViewById(R.id.time);
        inputImageView = findViewById(R.id.inputImageView);
        outputImageView = findViewById(R.id.outputImageView);
        maskView = findViewById(R.id.maskView);
        ListView imageView = findViewById(R.id.imageListView);
        ListView modelView = findViewById(R.id.modelListView);

        // Wyświetl informację o ładowaniu modelu
        statusText.setText("Choose model...");

        // Wczytaj listę modeli z assets/models
        String[] modelFiles = getAssets().list(MODEL_PATH);
        List<String> modelNames = Arrays.asList(modelFiles);
        ArrayAdapter<String> model_adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, modelNames);
        modelView.setAdapter(model_adapter);

        modelView.setOnItemClickListener((parent, view, position, id) -> {
            String modelName = modelNames.get(position);

            try {
                env = OrtEnvironment.getEnvironment();
                InputStream modelStream = getAssets().open(MODEL_PATH + modelName);
                byte[] modelBytes = new byte[modelStream.available()];
                modelStream.read(modelBytes);
                session = env.createSession(modelBytes, new OrtSession.SessionOptions());

                statusText.setText("Model " + modelName + " loaded.");
            } catch (IOException | OrtException e) {
                statusText.setText("Error loading model: " + e.getMessage());
            }
        });

        // Wczytaj listę zdjęć z assets/images
        List<String> imageNames = getImageNamesFromAssets();
        ArrayAdapter<String> image_adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, imageNames);
        imageView.setAdapter(image_adapter);

        // Obsługa wyboru obrazu z listy
        imageView.setOnItemClickListener((parent, view, position, id) -> {
            String imageName = imageNames.get(position);
            String maskName = imageName.replace(".jpg", ".png");
            imageNameText.setText("Image " + imageName);

            // Wczytaj obraz
            Bitmap inputBitmap = loadBitmapFromAssets("images/" + imageName);
            Bitmap maskBitMap = loadBitmapFromAssets("masks/" + maskName);
            if (inputBitmap != null) {
                // Wyświetl obraz wejściowy po preprocessingu
                Bitmap preprocessedBitmap = preprocessImage(inputBitmap);
                inputImageView.setImageBitmap(preprocessedBitmap);

                Bitmap outputBitmap = runInference(preprocessedBitmap);
                outputImageView.setImageBitmap(outputBitmap);

                Bitmap processedMask = preprocessMask(maskBitMap);
                maskView.setImageBitmap(processedMask);
                calculateMiou(outputBitmap, processedMask);
            } else {
                statusText.setText("Error loading image: " + imageName);
            }
        });
    }

    private List<String> getImageNamesFromAssets() {
        List<String> imageNames = new ArrayList<>();
        try {
            String[] files = getAssets().list("images");
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

    private Bitmap loadBitmapFromAssets(String filePath) {
        try (InputStream is = getAssets().open(filePath)) {
            return BitmapFactory.decodeStream(is);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private Bitmap preprocessImage(Bitmap inputBitmap) {
        // Przeskaluj obraz do rozmiaru wymaganego przez model (512x512)
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(inputBitmap, 512, 512, true);
        return resizedBitmap;
    }

    @SuppressLint({"ResourceType", "SetTextI18n"})
    private Bitmap preprocessMask(Bitmap mask) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mask, 512, 512, false);
        return resizedBitmap;
    }

    private String getRGB(int color) {
        int A = (color >> 24) & 0xff; // or color >>> 24
        int R = (color >> 16) & 0xff;
        int G = (color >>  8) & 0xff;
        int B = (color      ) & 0xff;
        return (
                "(" + A + ", " + R + ", " + G + ", " + B + ") "
        );
    }

    private int[][] getMask(float[][][][] outputTensor) {
        int numClasses = outputTensor[0].length;
        int height = outputTensor[0][0].length;
        int width = outputTensor[0][0][0].length;

        // Wynikowy obraz (po argmax na wymiarze num_classes)
        int[][] mask = new int[height][width];

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float maxVal = -Float.MAX_VALUE;
                int maxClass = -1;

                for (int c = 0; c < numClasses; c++) {
                    if (outputTensor[0][c][h][w] > maxVal) {
                        maxVal = outputTensor[0][c][h][w];
                        maxClass = c;
                    }
                }
                mask[h][w] = maxClass; // Klasa z największą wartością
            }
        }

        return mask;
    }

    @SuppressLint("SetTextI18n")
    private Bitmap runInference(Bitmap inputBitmap) {
        try {
            // Konwersja obrazu do tensora
            float[] inputTensor = bitmapToTensor(inputBitmap);
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensor), new long[]{1, 3, 512, 512});

            // Wykonanie inferencji
            long startTime = System.nanoTime();
            OrtSession.Result result = session.run(Collections.singletonMap("l_x_", tensor));
            long inferenceTime = System.nanoTime() - startTime;
            timeText.setText("Time: " + (int) (inferenceTime / 1e6) + " ms");

            float[][][][] outputTensor = (float[][][][]) result.get(0).getValue();
            int[][] mask = getMask(outputTensor);

            // Konwersja tensora na obraz wyjściowy
            return segmentationMask(mask);
        } catch (OrtException e) {
            e.printStackTrace();
            return null;
        }
    }

    @SuppressLint({"SetTextI18n", "DefaultLocale"})
    private void calculateMiou(Bitmap outputMask, Bitmap groundTruth) {
        int width = outputMask.getWidth();
        int height = outputMask.getHeight();

        HashMap<Integer, Integer> intersection = new HashMap<Integer, Integer>();
        HashMap<Integer, Integer> union = new HashMap<Integer, Integer>();

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
        miouText.setText("mIoU: " + String.format("%.3g%n", miou));
    }

    private Bitmap segmentationMask(int[][] mask) {
        int height = mask.length;
        int width = mask[0].length;
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int classId = mask[h][w];
                int color = getColorForClass(classId); // Funkcja, która przypisuje kolor do klasy
                bitmap.setPixel(w, h, color);
            }
        }
        return bitmap;
    }

    private int getColorForClass(int classId) {
        // Przykładowe kolory dla klas (zależne od liczby klas w modelu)
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
        return colors[classId % colors.length]; // Obsługuje więcej klas niż kolorów
    }

    private float[] bitmapToTensor(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float[] tensor = new float[3 * width * height]; // Kanały RGB
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            tensor[i] = ((pixel >> 16) & 0xFF) / 255.0f; // Red
            tensor[i + width * height] = ((pixel >> 8) & 0xFF) / 255.0f; // Green
            tensor[i + 2 * width * height] = (pixel & 0xFF) / 255.0f; // Blue
        }

        return tensor;
    }
}