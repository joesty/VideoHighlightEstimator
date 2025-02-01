import numpy as np

def compute_average_precision(recall, precision):
    """
    Calcula a AP (Average Precision) dado os vetores de recall e precisão.
    """
    # Adiciona 0 no início e 1 no final para recall e precisão
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Garante que a precisão seja monotonicamente decrescente
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Calcula a área sob a curva P-R
    indices = np.where(recall[1:] != recall[:-1])[0]  # Pontos onde o recall muda
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def compute_mAP(y_true, y_scores, num_classes):
    """
    Calcula a mAP (Mean Average Precision) para múltiplas classes.
    
    Args:
        y_true: Lista de listas, onde cada lista contém os rótulos reais para cada classe.
        y_scores: Lista de listas, onde cada lista contém os scores preditos para cada classe.
        num_classes: Número total de classes.
    
    Returns:
        mAP: Média das APs (Average Precision) para todas as classes.
    """
    aps = []
    for c in range(num_classes):
        # Ordena predições por score (confiança)
        scores = np.array(y_scores[c])
        labels = np.array(y_true[c])
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]

        # Calcula precisão e recall
        tp = np.cumsum(sorted_labels)
        fp = np.cumsum(1 - sorted_labels)
        recall = tp / (np.sum(labels) + 1e-10)  # Evita divisão por zero
        precision = tp / (tp + fp + 1e-10)

        # Calcula AP para a classe
        ap = compute_average_precision(recall, precision)
        aps.append(ap)

    # Calcula a média das APs
    mAP = np.mean(aps)
    return mAP