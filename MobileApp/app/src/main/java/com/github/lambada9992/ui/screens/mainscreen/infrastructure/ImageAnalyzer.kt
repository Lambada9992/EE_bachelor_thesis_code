package com.github.lambada9992.ui.screens.mainscreen.infrastructure

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.github.lambada9992.inference.models.InferenceModel
import com.github.lambada9992.utils.rotate
import com.github.lambada9992.utils.toBitmap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ImageAnalyzer(private val model: InferenceModel?) : ImageAnalysis.Analyzer {
    private var isProcessing = false

    override fun analyze(image: ImageProxy) {
        if (isProcessing) {
            image.close()
            return
        }

        isProcessing = true

        val imageBitmap = image.use { it.toBitmap().rotate(image.imageInfo.rotationDegrees.toFloat()) }
        image.close()

        CoroutineScope(Dispatchers.Default).launch {
            runInference(imageBitmap)
        }
    }

    private suspend fun runInference(image: Bitmap){
        withContext(Dispatchers.Default){
            model?.runInference(image)
            withContext(Dispatchers.Main){
                isProcessing = false
            }
        }
    }
}

