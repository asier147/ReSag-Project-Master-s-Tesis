import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'common')))
from correlation_matrices import load_and_prepare_data, calculate_and_combine_correlation, plot_correlation_matrix

def main():
    comarcas = {
        'III': {
            'convencional': {'start': '2022-07-01', 'end': '2023-08-31'},
            'conservacion': {'start': '2022-11-01', 'end': '2023-08-31'}
        },
        'V': {
            'convencional': {'start': '2022-07-01', 'end': '2023-08-31'},
            'conservacion': {'start': '2022-11-01', 'end': '2023-08-31'}
        }
    }
    for comarca, date_filters in comarcas.items():
        directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'Comarca_{comarca}'))
        fich_trabajo_convencional, fich_trabajo_conservacion = load_and_prepare_data(directory_path, comarca, date_filters)
        combined_matrix, index_names, upper_triangle_indices, lower_triangle_indices = calculate_and_combine_correlation(fich_trabajo_convencional, fich_trabajo_conservacion)
        plot_correlation_matrix(combined_matrix, index_names, upper_triangle_indices, lower_triangle_indices)

if __name__ == '__main__':
    main()
