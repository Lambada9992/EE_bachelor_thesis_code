package com.github.lambada9992.ui.screens.mainscreen.inferencevisualization

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp

@Composable
fun InferenceVisualization() {
    Canvas(modifier =  Modifier.fillMaxSize()) {
        drawRect(
            color = Color.Blue,
            size = Size(600f, 250f),
            topLeft = Offset(100f, 100f),
            style = Stroke(8.dp.toPx())
        )
    }

}
