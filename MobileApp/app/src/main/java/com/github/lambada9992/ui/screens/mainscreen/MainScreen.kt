package com.github.lambada9992.ui.screens.mainscreen

import android.Manifest
import android.content.Context
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.LifecycleOwner
import com.github.lambada9992.MainViewModel
import com.github.lambada9992.ui.screens.mainscreen.components.CameraPreview
import com.github.lambada9992.ui.screens.mainscreen.components.InferenceModelPicker
import com.github.lambada9992.ui.screens.mainscreen.components.InferenceStats
import com.github.lambada9992.ui.screens.mainscreen.components.InferenceVisualization
import com.github.lambada9992.ui.screens.mainscreen.infrastructure.ImageAnalyzer
import com.github.lambada9992.utils.getCameraProvider
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import java.util.concurrent.Executors
import kotlin.time.Duration.Companion.minutes
import kotlinx.coroutines.delay

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun MainScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val cameraPermission = rememberPermissionState(permission = Manifest.permission.CAMERA)
    val externalStoragePermission = rememberPermissionState(permission = Manifest.permission.READ_EXTERNAL_STORAGE)

    LaunchedEffect(Unit) {
        cameraPermission.launchPermissionRequest()
        externalStoragePermission.launchPermissionRequest()
    }

    val viewModel: MainViewModel = hiltViewModel()

    val imageAnalyzer = ImageAnalyzer(viewModel.inferenceService, viewModel.statisticsService)

    val previewView: PreviewView = remember { PreviewView(context).apply { implementationMode = PreviewView.ImplementationMode.PERFORMANCE } }
    val cameraSelector: CameraSelector by remember { mutableStateOf(CameraSelector.DEFAULT_BACK_CAMERA) }
    LaunchedEffect(previewView, cameraPermission.status.isGranted, cameraSelector) {
        if (cameraPermission.status.isGranted) initializeCamera(previewView, cameraSelector, context, lifecycleOwner, imageAnalyzer)
    }

    var autoEnabled by remember { mutableStateOf(false) }

    LaunchedEffect(autoEnabled) {
        while (autoEnabled) {
            if (viewModel.inferenceService.selectedInferenceModel == null) {
                autoEnabled = viewModel.inferenceService.switchToNextModel(context, viewModel.statisticsService)
            }
            delay(1.minutes)
            viewModel.statisticsService.clear()
            delay(2.minutes)
            viewModel.inferenceService.selectedInferenceModel?.let {
                viewModel.statisticsService.saveStatisticsToFile(
                    context,
                    it.name,
                    it.name.prepareFileName() + ".csv",
                )
            }
            autoEnabled = viewModel.inferenceService.switchToNextModel(context, viewModel.statisticsService)
        }
    }

    Box {
        CameraPreview(previewView, cameraPermission.status.isGranted, onPermissionRequest = { cameraPermission.launchPermissionRequest() })
        InferenceVisualization(viewModel.inferenceService)
        Column(
            modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.Top, horizontalAlignment = Alignment.CenterHorizontally
        ) {
            InferenceModelPicker(viewModel.inferenceService)
            Button(onClick = {
                autoEnabled = !autoEnabled
            }) {
                Text(text = if (autoEnabled) "true" else "false")
            }

            InferenceStats(viewModel.inferenceService, viewModel.statisticsService)
        }
    }
}

fun String.prepareFileName(): String {
    return this.replace("s:", "").replace("q:", "").replace("PT:", "PT_").replace("TF:", "TF_").replace(" ", "_")
}

suspend fun initializeCamera(
    previewView: PreviewView,
    cameraSelector: CameraSelector,
    context: Context,
    lifecycleOwner: LifecycleOwner,
    imageAnalyzer: ImageAnalysis.Analyzer? = null
) {
    val preview = Preview.Builder().build().apply { setSurfaceProvider(previewView.surfaceProvider) }

    val cameraProvider = context.getCameraProvider().apply { unbindAll() }

    val imageAnalysis = imageAnalyzer?.let { analyzer ->
        val resolution = context.resources.displayMetrics.let { Size(it.widthPixels, it.heightPixels) }
        ImageAnalysis.Builder().setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).setTargetResolution(resolution)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build().apply { setAnalyzer(Executors.newSingleThreadExecutor(), analyzer) }
    }

    cameraProvider.bindToLifecycle(
        lifecycleOwner, cameraSelector, *listOfNotNull(preview, imageAnalysis).toTypedArray()
    )
}
