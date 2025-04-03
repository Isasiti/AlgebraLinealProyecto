import numpy as np
import pandas as pd

# Definir los activos del portafolio
activos = ["Acciones", "Bonos", "Fondos Indexados", "Criptomonedas"]

# Asignación del portafolio (porcentajes que suman 100%)
pesos = np.array([0.50, 0.30, 0.15, 0.05])  # 50% acciones, 30% bonos, etc.

# Rentabilidad anual esperada de cada activo (basada en datos históricos estimados)
rendimientos_esperados = np.array([0.08, 0.04, 0.06, 0.20])  # 8%, 4%, 6%, 20%

# Calcular la rentabilidad esperada del portafolio
rentabilidad_portafolio = np.dot(pesos, rendimientos_esperados)

# Crear un DataFrame para visualizar mejor
df = pd.DataFrame({
    "Activo": activos,
    "Asignación (%)": pesos * 100,
    "Rentabilidad Esperada (%)": rendimientos_esperados * 100
})

print(df)
print(f"\nRentabilidad esperada del portafolio: {rentabilidad_portafolio * 100:.2f}% anual")
