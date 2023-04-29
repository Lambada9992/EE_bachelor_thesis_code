import openpyxl
from .qunatize_utils import QuantizeMode

def save_model_stats_to_file(
        file_name,
        model_name: str,
        pruned: float,
        quantized: QuantizeMode,
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
        sheet.append(['Model', 'Pruned', 'Quantized', 'Accuracy', 'Size'])

    quantized_string = next(enum_name for enum_name, value in vars(QuantizeMode).items() if value == quantized)
    sheet.append([model_name, pruned, quantized_string, accuracy, size])
    wb.save(file_name)
