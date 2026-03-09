package com.example.handgesture

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * ============================================================
 * EJEMPLO DE USO: Pipeline completo en una Activity
 * ============================================================
 *
 * Pipeline:
 *   Camera → MediaPipe Hands → 21 landmarks
 *   → HandFeatureExtractor (73 features)
 *   → GestureClassifier (TFLite)
 *   → Predicción del gesto
 *
 * ARCHIVOS NECESARIOS EN assets/:
 *   - hand_gesture_model.tflite  (modelo entrenado)
 *   - label_map.json             (mapeo de clases)
 *   - hand_landmarker.task       (modelo MediaPipe)
 *
 * DEPENDENCIAS en build.gradle:
 *   implementation 'com.google.mediapipe:tasks-vision:0.10.14'
 *   implementation 'org.tensorflow:tensorflow-lite:2.16.1'
 *   implementation 'androidx.camera:camera-core:1.3.4'
 *   implementation 'androidx.camera:camera-camera2:1.3.4'
 *   implementation 'androidx.camera:camera-lifecycle:1.3.4'
 *   implementation 'androidx.camera:camera-view:1.3.4'
 *
 * NOTA: Este es un ejemplo conceptual. Adapta según tu layout y
 *       la versión exacta de las dependencias que uses.
 * ============================================================
 */
class ExampleUsage : AppCompatActivity() {

    private lateinit var featureExtractor: HandFeatureExtractor
    private lateinit var gestureClassifier: GestureClassifier
    private var handLandmarker: HandLandmarker? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // setContentView(R.layout.activity_main)

        // 1. Inicializar extractor de features y clasificador
        featureExtractor = HandFeatureExtractor()
        gestureClassifier = GestureClassifier(this).apply {
            confidenceThreshold = 0.6f  // Solo mostrar si confianza > 60%
        }

        // 2. Configurar MediaPipe HandLandmarker
        setupHandLandmarker()

        // 3. Iniciar cámara
        // startCamera()
    }

    /**
     * Configura MediaPipe HandLandmarker en modo LIVE_STREAM
     * para procesar frames de la cámara en tiempo real.
     */
    private fun setupHandLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, _ ->
                // Este callback se ejecuta cada vez que MediaPipe detecta una mano
                processHandResult(result)
            }
            .setErrorListener { error ->
                Log.e(TAG, "MediaPipe error: ${error.message}")
            }
            .build()

        handLandmarker = HandLandmarker.createFromOptions(this, options)
    }

    /**
     * ============================================================
     * NÚCLEO DEL PIPELINE
     * ============================================================
     * Esta función recibe los landmarks de MediaPipe,
     * extrae las features y ejecuta la clasificación.
     */
    private fun processHandResult(result: HandLandmarkerResult) {
        // Paso 1: Extraer 73 features (63 coords normalizadas + 10 ángulos)
        val features = featureExtractor.extract(result) ?: return

        // Paso 2: Clasificar con TFLite
        val gesture = gestureClassifier.classify(features) ?: return

        // Paso 3: Usar el resultado
        Log.d(TAG, "Gesto detectado: ${gesture.label} (${(gesture.confidence * 100).toInt()}%)")

        // Actualizar UI en el hilo principal
        runOnUiThread {
            // Ejemplo: actualizar un TextView con el gesto detectado
            // tvGesture.text = gesture.label
            // tvConfidence.text = "${(gesture.confidence * 100).toInt()}%"

            // Si quieres ver las top 3 predicciones:
            val top3 = gestureClassifier.classifyTopN(features, 3)
            for ((label, confidence) in top3) {
                Log.d(TAG, "  $label: ${(confidence * 100).toInt()}%")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        gestureClassifier.close()
        handLandmarker?.close()
    }

    companion object {
        private const val TAG = "HandGesture"
    }
}
