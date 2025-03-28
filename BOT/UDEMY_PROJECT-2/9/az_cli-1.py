import openpyxl

def extract_hyperlinks(input_file, output_file):
    # Load the workbook and the active sheet
    wb = openpyxl.load_workbook(input_file)
    sheet = wb.active

    # Loop through all the cells in Column A to extract the full hyperlink and copy it to Column C
    for row in range(2, sheet.max_row + 1):  # Assuming the first row contains headers
        cell = sheet[f'A{row}']
        if cell.hyperlink:
            full_url = cell.hyperlink.target  # Extract the full hyperlink, including the fragment part after #
            sheet[f'C{row}'] = full_url  # Write the URL to Column C
        else:
            sheet[f'C{row}'] = "No URL"  # If no hyperlink is present, indicate as "No URL"

    # Save the workbook with the extracted URLs
    wb.save(output_file)
    print(f"Hyperlinks extracted and saved to {output_file}")

if __name__ == "__main__":
    input_file = input("Enter the path to the input Excel file: ")
    output_file = input("Enter the path to save the output Excel file: ")
    
    extract_hyperlinks(input_file, output_file)
