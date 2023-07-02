package com.github.lambada9992.ui.screens.mainscreen.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.github.lambada9992.inference.InferenceService

@Composable
fun InferenceVisualization(inferenceService: InferenceService) {
    val classificationText by inferenceService.classificationResult.observeAsState("")

    Box(modifier = Modifier.fillMaxSize()) {
//        Canvas(modifier =  Modifier.fillMaxSize()) {
//            drawRect(
//                color = Color.Blue,
//                size = Size(600f, 250f),
//                topLeft = Offset(100f, 700f),
//                style = Stroke(8.dp.toPx())
//            )
//        }
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Bottom,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = classificationText
            )
        }
    }
}
