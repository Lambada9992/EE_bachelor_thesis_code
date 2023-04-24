import openpyxl
def save_model_stats_to_file(
        file_name,
        model_name: str,
        pruned: bool,
        quantized: str,
        base_model_accuracy: float,
        base_model_size,
        base_pruned_model_accuracy: float,
        base_pruned_model_size,
        accuracy: float,
        size
):
    try:
        # If the file exists, load it
        wb = openpyxl.load_workbook(file_name)
        sheet = wb['Sheet1']
    except FileNotFoundError:
        # If the file doesn't exist, create a new workbook and sheet
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = 'Sheet1'
        sheet.append(['Model', 'Pruned', 'Quantized', 'Base model accuracy', 'Base Model Size', 'Pruned model accuracy', "Pruned model size", 'Accuracy', 'Size'])

    sheet.append([model_name, pruned, quantized, base_model_accuracy, base_model_size, base_pruned_model_accuracy, base_pruned_model_size,  accuracy, size])
    wb.save(file_name)
