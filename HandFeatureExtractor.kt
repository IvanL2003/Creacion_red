package com.example.handgesture

import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Extrae el vector de 73 features a partir de los 21 landmarks de MediaPipe Hands.
 *
 * Estructura del vector:
 *   [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20, angle0, angle1, ..., angle9]
 *    └─────────── 63 valores ───────────────┘ └────── 10 valores ───────┘
 *
 * Normalización aplicada (igual que en el entrenamiento):
 *   1. Centrar respecto a la muñeca (landmark 0)
 *   2. Escalar dividiendo por la máxima distancia al origen
 */
class HandFeatureExtractor {

    companion object {
        const val NUM_LANDMARKS = 21
        const val NUM_COORDS = 63    // 21 * 3
        const val NUM_ANGLES = 10
        const val FEATURE_SIZE = 73  // 63 + 10

        // Índices de los dedos según MediaPipe Hands
        private val FINGER_INDICES = arrayOf(
            intArrayOf(1, 2, 3, 4),     // Pulgar
            intArrayOf(5, 6, 7, 8),     // Índice
            intArrayOf(9, 10, 11, 12),  // Medio
            intArrayOf(13, 14, 15, 16), // Anular
            intArrayOf(17, 18, 19, 20)  // Meñique
        )
    }

    /**
     * Extrae el vector de 73 features desde un HandLandmarkerResult de MediaPipe.
     *
     * @param result Resultado de MediaPipe HandLandmarker
     * @param handIndex Índice de la mano (0 para la primera detectada)
     * @return FloatArray de 73 elementos, o null si no hay mano detectada
     */
    fun extract(result: HandLandmarkerResult, handIndex: Int = 0): FloatArray? {
        if (result.landmarks().isEmpty() || handIndex >= result.landmarks().size) {
            return null
        }

        val handLandmarks = result.landmarks()[handIndex]

        // Extraer los 21 landmarks como array de [x, y, z]
        val landmarks = Array(NUM_LANDMARKS) { i ->
            floatArrayOf(
                handLandmarks[i].x(),
                handLandmarks[i].y(),
                handLandmarks[i].z()
            )
        }

        return extractFromLandmarks(landmarks)
    }

    /**
     * Extrae el vector de 73 features desde un array de landmarks raw.
     * Útil si obtienes los landmarks de otra fuente que no sea HandLandmarkerResult.
     *
     * @param landmarks Array de 21 elementos, cada uno con [x, y, z]
     * @return FloatArray de 73 elementos
     */
    fun extractFromLandmarks(landmarks: Array<FloatArray>): FloatArray {
        require(landmarks.size == NUM_LANDMARKS) {
            "Se esperan $NUM_LANDMARKS landmarks, se recibieron ${landmarks.size}"
        }

        // --- PASO 1: Calcular ángulos ANTES de normalizar ---
        // (Los ángulos son invariantes a traslación y escala,
        //  pero los calculamos con las coordenadas originales
        //  tal como lo hace cargar_csv.py)
        val angles = computeAngles(landmarks)

        // --- PASO 2: Normalizar landmarks ---
        val normalized = normalizeLandmarks(landmarks)

        // --- PASO 3: Construir el vector de 73 features ---
        val features = FloatArray(FEATURE_SIZE)

        // Copiar 63 coordenadas normalizadas
        for (i in 0 until NUM_LANDMARKS) {
            features[i * 3 + 0] = normalized[i][0]
            features[i * 3 + 1] = normalized[i][1]
            features[i * 3 + 2] = normalized[i][2]
        }

        // Copiar 10 ángulos
        for (i in 0 until NUM_ANGLES) {
            features[NUM_COORDS + i] = angles[i]
        }

        return features
    }

    /**
     * Normaliza los landmarks:
     * 1. Centra respecto a la muñeca (landmark 0)
     * 2. Escala dividiendo por la distancia máxima
     */
    private fun normalizeLandmarks(landmarks: Array<FloatArray>): Array<FloatArray> {
        val wrist = landmarks[0].copyOf()

        // Centrar respecto a la muñeca
        val centered = Array(NUM_LANDMARKS) { i ->
            floatArrayOf(
                landmarks[i][0] - wrist[0],
                landmarks[i][1] - wrist[1],
                landmarks[i][2] - wrist[2]
            )
        }

        // Encontrar la distancia máxima al origen (muñeca)
        var maxDist = 0f
        for (point in centered) {
            val dist = sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])
            if (dist > maxDist) maxDist = dist
        }

        // Escalar
        if (maxDist > 0f) {
            for (point in centered) {
                point[0] /= maxDist
                point[1] /= maxDist
                point[2] /= maxDist
            }
        }

        return centered
    }

    /**
     * Calcula los 10 ángulos articulares de los 5 dedos.
     * Por cada dedo: 2 ángulos (en las articulaciones intermedias).
     */
    private fun computeAngles(landmarks: Array<FloatArray>): FloatArray {
        val angles = FloatArray(NUM_ANGLES)
        var idx = 0

        for (finger in FINGER_INDICES) {
            // Ángulo en la articulación proximal: finger[0]-finger[1]-finger[2]
            angles[idx++] = angleBetween(
                landmarks[finger[0]],
                landmarks[finger[1]],
                landmarks[finger[2]]
            )
            // Ángulo en la articulación distal: finger[1]-finger[2]-finger[3]
            angles[idx++] = angleBetween(
                landmarks[finger[1]],
                landmarks[finger[2]],
                landmarks[finger[3]]
            )
        }

        return angles
    }

    /**
     * Calcula el ángulo (en grados) entre tres puntos 3D: A-B-C
     * El ángulo se mide en el vértice B.
     */
    private fun angleBetween(a: FloatArray, b: FloatArray, c: FloatArray): Float {
        // Vectores BA y BC
        val ba = floatArrayOf(a[0] - b[0], a[1] - b[1], a[2] - b[2])
        val bc = floatArrayOf(c[0] - b[0], c[1] - b[1], c[2] - b[2])

        // Producto punto
        val dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]

        // Magnitudes
        val magBA = sqrt(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2])
        val magBC = sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2])

        if (magBA == 0f || magBC == 0f) return 0f

        // Clamp para evitar errores numéricos con acos
        val cosAngle = (dot / (magBA * magBC)).coerceIn(-1f, 1f)

        // Convertir a grados
        return Math.toDegrees(acos(cosAngle).toDouble()).toFloat()
    }
}
