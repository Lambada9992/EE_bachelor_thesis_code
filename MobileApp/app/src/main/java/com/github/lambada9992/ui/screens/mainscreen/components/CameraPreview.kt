package com.github.lambada9992.ui.screens.mainscreen.components

import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun CameraPreview(
    previewView: PreviewView,
    cameraPermissionGranted: Boolean,
    onPermissionRequest: () -> Unit
) {
    if (cameraPermissionGranted) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
    } else {
        Button(onClick = { onPermissionRequest() }) {
            Text(text = "Camera permission")
        }
    }
}
