package com.github.lambada9992.ui.screens.mainscreen.components

import androidx.compose.material.ExperimentalMaterialApi
import androidx.compose.material.ExposedDropdownMenuBox
import androidx.compose.material.Text
import androidx.compose.material.TextField
import androidx.compose.material.ExposedDropdownMenuDefaults
import androidx.compose.material.DropdownMenuItem
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import com.github.lambada9992.inference.InferenceService

@OptIn(ExperimentalMaterialApi::class)
@Composable
fun InferenceModelPicker(
    inferenceService: InferenceService
){
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = {expanded = !expanded}
    ) {
        TextField(
            value = inferenceService.selectedInferenceModel?.name ?: "NOT SELECTED",
            onValueChange = {},
            readOnly = true,
            label = { Text(text = "Selected model") },
            trailingIcon = {
                ExposedDropdownMenuDefaults.TrailingIcon(
                    expanded = expanded
                )
            },
        )

        ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            (listOf("NONE") + inferenceService.getModelsNames).forEach {
                DropdownMenuItem(onClick = {
                    inferenceService.selectModel(it)
                    expanded = false
                }) {
                    Text(text = it)
                }
            }
        }
    }
}
