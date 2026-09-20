from client import data_collector as dc
from datetime import date

if __name__ == "__main__":

    input_data = dc.collect_np('AMD', date.fromisoformat('2026-08-15'), date.fromisoformat('2026-09-15'), 26)
    print(len(input_data))

