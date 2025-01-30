import pandas as pd

def convert_csv_to_excel(csv_file, excel_file):
    # Membaca file CSV
    df = pd.read_csv(csv_file, names=["review", "actual", "predicted"], header=None)
    
    # Menyimpan ke file Excel
    df.to_excel(excel_file, index=False)
    print(f"File berhasil dikonversi ke {excel_file}")

# Contoh penggunaan
convert_csv_to_excel("naive_bayes_results.csv", "naive_bayes_results.xlsx")
