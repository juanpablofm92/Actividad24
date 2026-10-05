"""
Actividad 24: Ejemplo Numérico de Índices de Validación Internos de Agrupamiento
Maestría en Inteligencia Artificial - TecNM / ITSU
Materia: Aprendizaje Automático
Autor: Juan Pablo Figueroa Moran (M26040059)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples


def main():
    print("=" * 70)
    print(" ACTIVIDAD 24: ÍNDICES DE VALIDACIÓN INTERNA DE CLUSTERING")
    print(" Juan Pablo Figueroa Moran | TecNM / ITSU - MIA")
    print("=" * 70)

    # 1. Definición del conjunto de datos controlado N=6, k=2
    # C1: (1,1), (1,2), (2,1)  |  C2: (5,5), (5,6), (6,5)
    X = np.array([
        [1.0, 1.0],  # p1
        [1.0, 2.0],  # p2
        [2.0, 1.0],  # p3
        [5.0, 5.0],  # p4
        [5.0, 6.0],  # p5
        [6.0, 5.0]   # p6
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    n_samples = len(X)
    k = len(np.unique(labels))

    print(f"\n[+] Puntos del conjunto (N={n_samples}, k={k}):")
    for i, (pt, lbl) in enumerate(zip(X, labels), 1):
        print(f"  p{i}: {pt.tolist()} -> Cluster C{lbl+1}")

    # 2. CÁLCULO MANUAL PASO A PASO
    print("\n" + "-" * 70)
    print(" 1. CÁLCULO PASO A PASO: COEFICIENTE SILHOUETTE")
    print("-" * 70)

    a = np.zeros(n_samples)
    b = np.zeros(n_samples)
    s = np.zeros(n_samples)

    for i in range(n_samples):
        same_cluster_pts = X[(labels == labels[i]) & (np.arange(n_samples) != i)]
        other_cluster_pts = X[labels != labels[i]]

        # a_i: cohesión intra-cluster
        dist_same = np.linalg.norm(same_cluster_pts - X[i], axis=1)
        a[i] = np.mean(dist_same)

        # b_i: separación inter-cluster
        dist_other = np.linalg.norm(other_cluster_pts - X[i], axis=1)
        b[i] = np.mean(dist_other)

        # s_i: Coeficiente Silhouette individual
        s[i] = (b[i] - a[i]) / max(a[i], b[i])

        print(f"  Punto p{i+1} {X[i].tolist()}: a_{i+1}={a[i]:.4f}, b_{i+1}={b[i]:.4f} => s_{i+1} = {s[i]:.4f}")

    s_mean = np.mean(s)
    print(f"\n  -> Coeficiente Silhouette Global Promedio (Manual): s_prom = {s_mean:.4f}")

    print("\n" + "-" * 70)
    print(" 2. CÁLCULO PASO A PASO: ÍNDICE DAVIES-BOULDIN (DB)")
    print("-" * 70)

    # Centroides
    c1 = np.mean(X[labels == 0], axis=0)
    c2 = np.mean(X[labels == 1], axis=0)
    print(f"  Centroide C1: {c1.tolist()}")
    print(f"  Centroide C2: {c2.tolist()}")

    # Dispersión intra-cluster (distancia media al centroide)
    S1 = np.mean(np.linalg.norm(X[labels == 0] - c1, axis=1))
    S2 = np.mean(np.linalg.norm(X[labels == 1] - c2, axis=1))
    print(f"  Dispersión S1 (C1): {S1:.4f}")
    print(f"  Dispersión S2 (C2): {S2:.4f}")

    # Distancia entre centroides
    M12 = np.linalg.norm(c1 - c2)
    print(f"  Distancia entre centroides M_1,2: {M12:.4f}")

    # Similitud R12 y DB
    R12 = (S1 + S2) / M12
    db_manual = R12  # Para k=2, DB = (R12 + R21) / 2 = R12
    print(f"  Similitud R_1,2 = (S1 + S2) / M_1,2 = {R12:.4f}")
    print(f"  -> Índice Davies-Bouldin (Manual): DB = {db_manual:.4f}")

    print("\n" + "-" * 70)
    print(" 3. CÁLCULO PASO A PASO: ÍNDICE CALINSKI-HARABASZ (CH)")
    print("-" * 70)

    c_global = np.mean(X, axis=0)
    print(f"  Centroide Global: {c_global.tolist()}")

    # Traza matriz de dispersión entre clusters (SSB / Between)
    n1 = np.sum(labels == 0)
    n2 = np.sum(labels == 1)
    SSB = n1 * np.sum((c1 - c_global) ** 2) + n2 * np.sum((c2 - c_global) ** 2)

    # Traza matriz de dispersión dentro de clusters (SSW / Within)
    SSW = np.sum((X[labels == 0] - c1) ** 2) + np.sum((X[labels == 1] - c2) ** 2)

    ch_manual = (SSB / (k - 1)) / (SSW / (n_samples - k))
    print(f"  SSB (Dispersión Inter-Cluster / Between): {SSB:.4f}")
    print(f"  SSW (Dispersión Intra-Cluster / Within):  {SSW:.4f}")
    print(f"  -> Índice Calinski-Harabasz (Manual): CH = {ch_manual:.4f}")

    # 3. VERIFICACIÓN CON SCIKIT-LEARN
    print("\n" + "=" * 70)
    print(" VERIFICACIÓN CON BIBLIOTECA OFICIAL (scikit-learn)")
    print("=" * 70)
    sk_silhouette = silhouette_score(X, labels)
    sk_db = davies_bouldin_score(X, labels)
    sk_ch = calinski_harabasz_score(X, labels)

    print(f"  [scikit-learn] Silhouette Score:      {sk_silhouette:.4f}  (Coincidencia: {np.isclose(s_mean, sk_silhouette)})")
    print(f"  [scikit-learn] Davies-Bouldin Score:  {sk_db:.4f}  (Coincidencia: {np.isclose(db_manual, sk_db)})")
    print(f"  [scikit-learn] Calinski-Harabasz Score: {sk_ch:.4f} (Coincidencia: {np.isclose(ch_manual, sk_ch)})")

    # 4. GENERACIÓN DE GRÁFICO
    os.makedirs("docs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico de dispersión con centroides
    axes[0].scatter(X[labels == 0, 0], X[labels == 0, 1], color='#2563EB', s=120, label='Cluster C1', edgecolors='black')
    axes[0].scatter(X[labels == 1, 0], X[labels == 1, 1], color='#DC2626', s=120, label='Cluster C2', edgecolors='black')
    axes[0].scatter([c1[0], c2[0]], [c1[1], c2[1]], color='#F59E0B', marker='X', s=200, label='Centroides', edgecolors='black')
    for i, pt in enumerate(X):
        axes[0].annotate(f"p{i+1}({int(pt[0])},{int(pt[1])})", (pt[0]+0.15, pt[1]+0.1), fontsize=9)
    axes[0].set_title("Partición de Datos en 2D (N=6, k=2)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Eje X1")
    axes[0].set_ylabel("Eje X2")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Gráfico de barras de Silhouette individual
    colors = ['#2563EB' if l == 0 else '#DC2626' for l in labels]
    axes[1].bar(range(1, n_samples + 1), s, color=colors, edgecolor='black', alpha=0.85)
    axes[1].axhline(s_mean, color='green', linestyle='--', linewidth=2, label=f'Promedio Global ({s_mean:.3f})')
    axes[1].set_xticks(range(1, n_samples + 1))
    axes[1].set_xticklabels([f"p{i}" for i in range(1, n_samples + 1)])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Coeficiente Silhouette por Muestra ($s_i$)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Muestra")
    axes[1].set_ylabel("Valor Silhouette $s_i$")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join("docs", "cluster_validation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[+] Gráfico exportado exitosamente en: {plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
