package com.github.lambada9992.ui

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.github.lambada9992.ui.screens.mainscreen.MainScreen

sealed class Screen(val route: String) {
    object Home: Screen(route = "home_screen")
}

@Composable
fun Navigation(
    navController: NavHostController
) {
    NavHost(
    navController = navController,
    startDestination = Screen.Home.route
    ) {
        composable(route = Screen.Home.route) { MainScreen() }
    }
}
