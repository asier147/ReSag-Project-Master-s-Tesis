# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:19:35 2024

@author: Asier

Correlación índices
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_prepare_data(directory_path, comarca_name, date_filters):
    """Loads and prepares data for a given comarca."""
    convencional_indices_path = os.path.join(directory_path, f"convencional_{comarca_name}_indices_filtrado.csv")
    conservacion_indices_path = os.path.join(directory_path, f"conservacion_{comarca_name}_indices_filtrado.csv")
    convencional_bandas_path = os.path.join(directory_path, f"convencional_{comarca_name}_bandas_filtrado.csv")
    conservacion_bandas_path = os.path.join(directory_path, f"conservacion_{comarca_name}_bandas_filtrado.csv")
    radar_path = os.path.join(directory_path, f"S1_bandas_filtrado_{comarca_name}.csv")

    convencional_indices = pd.read_csv(convencional_indices_path)
    conservacion_indices = pd.read_csv(conservacion_indices_path)
    del convencional_indices['SINDRI_median']
    del conservacion_indices['SINDRI_median']

    convencional_bandas = pd.read_csv(convencional_bandas_path)
    desired_order = [col for col in convencional_bandas.columns if col not in ['B8A_median','B11_median', 'B12_median']] + ['B8A_median','B11_median', 'B12_median']
    convencional_bandas=convencional_bandas[desired_order]

    conservacion_bandas = pd.read_csv(conservacion_bandas_path)
    desired_order = [col for col in conservacion_bandas.columns if col not in ['B8A_median','B11_median', 'B12_median']] + ['B8A_median','B11_median', 'B12_median']
    conservacion_bandas=conservacion_bandas[desired_order]

    fich_trabajo_radar=pd.read_csv(radar_path)
    radar_convencional=fich_trabajo_radar[fich_trabajo_radar['Manejo']=='Convencional']
    radar_conservacion=fich_trabajo_radar[fich_trabajo_radar['Manejo']=='Conservacion']

    columnas=['date','Manejo','REFSIGPAC','Cultivo']
    indices = sorted([x for x in convencional_indices.columns.values if 'median' in x])
    convencional_indices=convencional_indices[columnas+indices]
    conservacion_indices=conservacion_indices[columnas+indices]

    fich_trabajo_convencional = pd.merge(convencional_indices, convencional_bandas,on=['date','Manejo','REFSIGPAC','Cultivo'], how='outer')
    fich_trabajo_convencional = pd.merge(fich_trabajo_convencional, radar_convencional,on=['date','Manejo','REFSIGPAC'], how='outer')
    fich_trabajo_convencional = fich_trabajo_convencional[(fich_trabajo_convencional['date'] >= date_filters['convencional']['start']) & (fich_trabajo_convencional['date'] <= date_filters['convencional']['end'])]

    fich_trabajo_conservacion = pd.merge(conservacion_indices, conservacion_bandas,on=['date','Manejo','REFSIGPAC','Cultivo'], how='outer')
    fich_trabajo_conservacion = fich_trabajo_conservacion[(fich_trabajo_conservacion['date'] >= date_filters['conservacion']['start']) & (fich_trabajo_conservacion['date'] <= date_filters['conservacion']['end'])]
    fich_trabajo_conservacion = pd.merge(fich_trabajo_conservacion, radar_conservacion,on=['date','Manejo','REFSIGPAC'], how='outer')

    return fich_trabajo_convencional, fich_trabajo_conservacion

def calculate_and_combine_correlation(fich_trabajo_convencional, fich_trabajo_conservacion):
    """Calculates and combines correlation matrices."""
    columnas_convencional = ['date'] + [col for col in fich_trabajo_convencional.columns if 'median' in col]
    df_convencional = fich_trabajo_convencional[columnas_convencional].set_index('date')
    correlation_matrix_convencional = df_convencional.corr().values

    columnas_conservacion = ['date'] + [col for col in fich_trabajo_conservacion.columns if 'median' in col]
    df_conservacion = fich_trabajo_conservacion[columnas_conservacion].set_index('date')
    correlation_matrix_conservacion = df_conservacion.corr().values

    assert correlation_matrix_convencional.shape == correlation_matrix_conservacion.shape
    index_names = df_convencional.columns.tolist()

    combined_matrix = np.zeros_like(correlation_matrix_convencional)
    upper_triangle_indices = np.triu_indices_from(combined_matrix, k=0)
    combined_matrix[upper_triangle_indices] = correlation_matrix_conservacion[upper_triangle_indices]
    lower_triangle_indices = np.tril_indices_from(combined_matrix, k=-1)
    combined_matrix[lower_triangle_indices] = correlation_matrix_convencional[lower_triangle_indices]

    return combined_matrix, index_names, upper_triangle_indices, lower_triangle_indices

def plot_correlation_matrix(combined_matrix, index_names, upper_triangle_indices, lower_triangle_indices):
    """Plots the combined correlation matrix."""
    specific_mapping = {
        "median_B2": "VH",
        "median_B3": "VV",
        "median_B4": "VH/VV"
    }

    def process_name(name):
        if '_median' in name:
            return name.split('_median')[0]
        return specific_mapping.get(name, name)

    modified_index_names = [process_name(name) for name in index_names]

    plt.figure(figsize=(10, 8))
    plt.imshow(combined_matrix, cmap='coolwarm', interpolation='nearest')
    plt.xticks(range(len(modified_index_names)), modified_index_names, rotation=45, ha='right')
    plt.yticks(range(len(modified_index_names)), modified_index_names)

    for i in range(len(index_names)):
        for j in range(len(index_names)):
            if (i, j) in zip(*upper_triangle_indices) or (i, j) in zip(*lower_triangle_indices):
                plt.text(j, i, f'{combined_matrix[i, j]:.2f}', ha='center', va='center', color='white')

    plt.colorbar(label='Correlation')
    plt.title('Combined Correlation Matrix\nUpper: Conservation, Lower: Conventional')
    plt.tight_layout()
    plt.show()
