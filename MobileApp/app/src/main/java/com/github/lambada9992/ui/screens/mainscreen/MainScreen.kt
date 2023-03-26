package com.github.lambada9992.ui.screens.mainscreen

import android.Manifest
import android.content.Context
import android.util.Log
import android.util.Size
import androidx.activity.viewModels
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.github.lambada9992.MainViewModel
import com.github.lambada9992.inference.models.tensorflow.TensorflowInferenceModel
import com.github.lambada9992.ui.screens.mainscreen.camerapreview.CameraPreview
import com.github.lambada9992.ui.screens.mainscreen.inferencevisualization.InferenceVisualization
import com.github.lambada9992.ui.screens.mainscreen.infrastructure.ImageAnalyzer
import com.github.lambada9992.utils.getCameraProvider
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import java.util.concurrent.Executors

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun MainScreen(){
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val cameraPermission = rememberPermissionState(permission = Manifest.permission.CAMERA)
    LaunchedEffect(Unit){ cameraPermission.launchPermissionRequest() }

    val model = TensorflowInferenceModel()
    model.initialize(context)

    val imageAnalyzer = ImageAnalyzer(model)

    val previewView: PreviewView = remember { PreviewView(context).apply { implementationMode = PreviewView.ImplementationMode.PERFORMANCE } }
    val cameraSelector: CameraSelector by remember { mutableStateOf(CameraSelector.DEFAULT_BACK_CAMERA) }
    LaunchedEffect(previewView, cameraPermission.status.isGranted, cameraSelector) {
        if (cameraPermission.status.isGranted) initializeCamera(previewView, cameraSelector, context, lifecycleOwner, imageAnalyzer)
    }

    Box {
        CameraPreview(
            previewView,
            cameraPermission.status.isGranted,
            onPermissionRequest =  { cameraPermission.launchPermissionRequest() }
        )
        InferenceVisualization()
    }
}

suspend fun initializeCamera(
    previewView: PreviewView,
    cameraSelector: CameraSelector,
    context: Context,
    lifecycleOwner: LifecycleOwner,
    imageAnalyzer: ImageAnalysis.Analyzer? = null
){
    val preview = Preview.Builder().build().apply { setSurfaceProvider(previewView.surfaceProvider) }

    val cameraProvider = context.getCameraProvider().apply { unbindAll() }

    val imageAnalysis = imageAnalyzer?.let { analyzer ->
        val resolution = context.resources.displayMetrics.let { Size(it.widthPixels, it.heightPixels) }
        ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setTargetResolution(resolution)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .apply { setAnalyzer(Executors.newSingleThreadExecutor(), analyzer) }
    }

    cameraProvider.bindToLifecycle(
        lifecycleOwner,
        cameraSelector,
        *listOfNotNull(preview, imageAnalysis).toTypedArray()
    )
}
