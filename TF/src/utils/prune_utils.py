import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

def prune_model(
        model: tf.keras.Model,
        train_ds,
        val_ds,
        train_ds_size,
        batch_size=64,
        epochs=3,
        initial_sparsity = 0.0,
        final_sparsity = 0.5,
        optimizer = tf.keras.optimizers.SGD(),
):
    end_step = np.ceil(train_ds_size/batch_size).astype(np.int32) * epochs
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparsity,
                                                                 final_sparsity=final_sparsity,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    model_for_pruning.fit(train_ds,
                          validation_data=val_ds,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=callbacks,
                          verbose=1)
    pruned_model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    return pruned_model_for_export
