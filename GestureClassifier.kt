package com.example.handgesture

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Clasificador de gestos usando TensorFlow Lite.
 *
 * Uso:
 *   val classifier = GestureClassifier(context)
 *   val result = classifier.classify(featuresArray)  // FloatArray de 73
 *   Log.d("Gesture", "Gesto: ${result.label}, Confianza: ${result.confidence}")
 *   classifier.close()
 *
 * Archivos necesarios en assets/:
 *   - hand_gesture_model.tflite
 *   - label_map.json
 */
class GestureClassifier(context: Context) {

    private val interpreter: Interpreter
    private val labelMap: Map<Int, String>
    private val numClasses: Int

    // Umbral mínimo de confianza para considerar una predicción válida
    var confidenceThreshold: Float = 0.5f

    data class GestureResult(
        val label: String,        // Letra predicha (ej: "A", "B", ...)
        val confidence: Float,    // Confianza (0.0 - 1.0)
        val allProbabilities: Map<String, Float>  // Probabilidades de todas las clases
    )

    init {
        // Cargar modelo TFLite desde assets
        val model = loadModelFile(context, "hand_gesture_model.tflite")
        val options = Interpreter.Options().apply {
            setNumThreads(4)  // Usar 4 hilos para mejor rendimiento
        }
        interpreter = Interpreter(model, options)

        // Cargar mapeo de labels desde assets
        labelMap = loadLabelMap(context, "label_map.json")
        numClasses = labelMap.size
    }

    /**
     * Clasifica un vector de 73 features y devuelve el resultado.
     *
     * @param features FloatArray de 73 elementos (63 coords + 10 ángulos)
     * @return GestureResult con label, confianza y todas las probabilidades,
     *         o null si la confianza está por debajo del umbral
     */
    fun classify(features: FloatArray): GestureResult? {
        require(features.size == HandFeatureExtractor.FEATURE_SIZE) {
            "Se esperan ${HandFeatureExtractor.FEATURE_SIZE} features, se recibieron ${features.size}"
        }

        // Preparar input: [1, 73]
        val inputBuffer = ByteBuffer.allocateDirect(4 * features.size).apply {
            order(ByteOrder.nativeOrder())
            features.forEach { putFloat(it) }
            rewind()
        }

        // Preparar output: [1, numClasses]
        val outputBuffer = ByteBuffer.allocateDirect(4 * numClasses).apply {
            order(ByteOrder.nativeOrder())
        }

        // Ejecutar inferencia
        interpreter.run(inputBuffer, outputBuffer)

        // Leer resultados
        outputBuffer.rewind()
        val probabilities = FloatArray(numClasses) { outputBuffer.float }

        // Encontrar la clase con mayor probabilidad
        var maxIdx = 0
        var maxProb = probabilities[0]
        for (i in 1 until numClasses) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIdx = i
            }
        }

        // Verificar umbral de confianza
        if (maxProb < confidenceThreshold) {
            return null
        }

        // Construir mapa de todas las probabilidades
        val allProbs = mutableMapOf<String, Float>()
        for (i in 0 until numClasses) {
            labelMap[i]?.let { label ->
                allProbs[label] = probabilities[i]
            }
        }

        return GestureResult(
            label = labelMap[maxIdx] ?: "Unknown",
            confidence = maxProb,
            allProbabilities = allProbs
        )
    }

    /**
     * Clasifica y devuelve las top-N predicciones ordenadas por confianza.
     */
    fun classifyTopN(features: FloatArray, n: Int = 3): List<Pair<String, Float>> {
        require(features.size == HandFeatureExtractor.FEATURE_SIZE)

        val inputBuffer = ByteBuffer.allocateDirect(4 * features.size).apply {
            order(ByteOrder.nativeOrder())
            features.forEach { putFloat(it) }
            rewind()
        }

        val outputBuffer = ByteBuffer.allocateDirect(4 * numClasses).apply {
            order(ByteOrder.nativeOrder())
        }

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val probabilities = FloatArray(numClasses) { outputBuffer.float }

        return probabilities
            .mapIndexed { idx, prob -> (labelMap[idx] ?: "?") to prob }
            .sortedByDescending { it.second }
            .take(n)
    }

    /**
     * Liberar recursos del intérprete TFLite.
     * Llamar cuando ya no se necesite el clasificador.
     */
    fun close() {
        interpreter.close()
    }

    // --- Métodos privados ---

    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabelMap(context: Context, filename: String): Map<Int, String> {
        val jsonString = context.assets.open(filename).bufferedReader().use { it.readText() }
        val jsonObject = JSONObject(jsonString)
        val map = mutableMapOf<Int, String>()
        jsonObject.keys().forEach { key ->
            map[key.toInt()] = jsonObject.getString(key)
        }
        return map
    }
}
