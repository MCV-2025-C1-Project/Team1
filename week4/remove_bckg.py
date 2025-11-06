# Import necessary libraries
import imageio.v2 as imageio
import numpy as np
import os
import matplotlib.pyplot as plt
from bckg_rmv import *
from descriptors import preprocess_image
from image_split import split_images

def extract_final_mask(res):
    """
    Intenta extraer la máscara final desde 'res' (tupla/lista o dict),
    sin asumir un índice concreto.
    """
    import numpy as np

    # Caso diccionario: busca claves típicas
    if isinstance(res, dict):
        for key in ('fm', 'final_mask', 'mask', 'pred_mask'):
            if key in res:
                return res[key]

    # Caso lista/tupla: prueba posiciones típicas y, si no, busca el último array 2D
    if isinstance(res, (list, tuple)):
        candidates = [-1, 7, 3, 1]   # fm suele ser el último; si no, algunos índices comunes
        for k in candidates:
            if -len(res) <= k < len(res):
                x = res[k]
                if isinstance(x, np.ndarray) and x.ndim == 2:
                    return x

        # Último recurso: elige el último ndarray 2D
        arrays2d = [x for x in res if isinstance(x, np.ndarray) and x.ndim == 2]
        if arrays2d:
            return arrays2d[-1]

    raise ValueError("No se pudo extraer la máscara final de 'res'.")


# Path to the images
image_folder = "qsd1_w4"  # Update this path as necessary
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
mask_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
image_files.sort()
mask_files.sort()


# Iterar sobre imágenes, comparar con GT, calcular métricas y mostrar resultados
import matplotlib.patches as mpatches

# Toggle plotting of images. Set to False to speed-up and only print metrics.
SHOW_PLOTS = False
os.makedirs("outputs", exist_ok=True)

precisions = []
recalls = []
f1s = []

it = 0

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    im = imageio.imread(image_path)  # RGB

    _, splitted = split_images(preprocess_image(im))


    # split_images devuelve imágenes en RGB → no hace falta reconvertir
    if isinstance(splitted, (list, tuple)):
        parts = list(splitted)
    else:
        parts = [splitted]

    part_labels = ["izquierda", "derecha"][:len(parts)]
    results = []

    # 3️⃣ Procesar cada subimagen individualmente
    for i, part in enumerate(parts):
        result = remove_background_morphological_gradient(part)
        results.append(result)

        if SHOW_PLOTS:
            original_image, pred_mask, foreground, grad_norm, gn, sc, sc2, fm = result

            plt.figure(figsize=(16, 8))
            plt.suptitle(f"{image_file} - Cuadro {part_labels[i].capitalize()}")

            plt.subplot(2, 4, 1)
            plt.title('Original')
            plt.imshow(original_image)
            plt.axis('off')

            plt.subplot(2, 4, 2)
            plt.title('Gradient')
            plt.imshow(grad_norm, cmap='viridis')
            plt.axis('off')

            plt.subplot(2, 4, 3)
            plt.title('Intermediate gradient')
            plt.imshow(gn, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 4, 4)
            plt.title('Intermediate Binary Mask')
            plt.imshow(pred_mask, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 4, 5)
            plt.title('Candidates')
            plt.imshow(sc2)
            plt.axis('off')

            plt.subplot(2, 4, 6)
            plt.title('Candidates Filtered')
            plt.imshow(sc)
            plt.axis('off')

            plt.subplot(2, 4, 7)
            plt.title('Binary Mask (fm)')
            plt.imshow(fm, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 4, 8)
            plt.title('Foreground')
            plt.imshow(foreground)
            plt.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join("outputs", f"{os.path.splitext(image_file)[0]}_{part_labels[i]}.png")
            plt.savefig(save_path)
            plt.show()

    # 4️⃣ Combinar máscaras si hubo división
    if len(results) == 1:
        fm = extract_final_mask(results[0])
        pred_bool = (fm > 0) if fm.dtype != bool else fm
    else:
        masks = []
        for r in results:
            fm = extract_final_mask(r)
            fm_bool = (fm > 0) if fm.dtype != bool else fm
            masks.append(fm_bool)
        pred_bool = np.hstack(masks)

    base = os.path.splitext(image_file)[0]

    # A 0/255 uint8
    mask_uint8 = (pred_bool.astype(np.uint8)) * 255

    # Carpeta de salida
    os.makedirs("outputs_mask", exist_ok=True)

    # Guarda binaria (blanco/negro)
    save_mask_path = os.path.join("outputs_mask", f"{base}_mask.png")
    imageio.imwrite(save_mask_path, mask_uint8)
    print(f"✅ Saved binary mask: {save_mask_path}")
    # ⚙️ Reconstruir foreground combinado aplicando la máscara sobre la imagen original
    h_pred, w_pred = pred_bool.shape
    im_resized = im[:h_pred, :w_pred, :]  # imagen original recortada al tamaño válido
    combined_foreground = im_resized * pred_bool[..., np.newaxis]

    # 5️⃣ Intentar leer GT
    base = os.path.splitext(image_file)[0]
    gt_path = os.path.join(image_folder, base + '.png')

    if os.path.exists(gt_path):
        # 🟡 Solo se hace esta parte si existe GT
        gt_raw = imageio.imread(gt_path)
        gt_bool = gt_raw > 127

        # 6️⃣ Alinear tamaños
        h = min(pred_bool.shape[0], gt_bool.shape[0])
        w = min(pred_bool.shape[1], gt_bool.shape[1])
        pred_bool = pred_bool[:h, :w]
        gt_bool = gt_bool[:h, :w]
        im_resized = im_resized[:h, :w, :]

        # 7️⃣ Calcular métricas conjuntas
        TP = np.logical_and(pred_bool, gt_bool).sum()
        FP = np.logical_and(pred_bool, np.logical_not(gt_bool)).sum()
        FN = np.logical_and(np.logical_not(pred_bool), gt_bool).sum()
        TN = np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).sum()

        P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

        precisions.append(P)
        recalls.append(R)
        f1s.append(F1)

        print(f"{image_file} -> Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")

        # 8️⃣ Mostrar máscara combinada + GT + diferencias + foreground combinado
        if SHOW_PLOTS:
            diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            diff_rgb[np.logical_and(pred_bool, gt_bool)] = [0, 255, 0]     # TP
            diff_rgb[np.logical_and(pred_bool, np.logical_not(gt_bool))] = [255, 0, 0]  # FP
            diff_rgb[np.logical_and(np.logical_not(pred_bool), gt_bool)] = [0, 0, 255]  # FN
            diff_rgb[np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool))] = [255, 255, 255]  # TN

            plt.figure(figsize=(14, 5))
            plt.suptitle(f"{image_file} - Evaluación conjunta  (P={P:.3f}, R={R:.3f}, F1={F1:.3f})")

            plt.subplot(1, 4, 1)
            plt.title('Foreground combinado')
            plt.imshow(combined_foreground)
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.title('Máscara Predicha (Combinada)')
            plt.imshow(pred_bool, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.title('GT Mask')
            plt.imshow(gt_bool, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.title('Difference Mask')
            plt.imshow(diff_rgb)
            plt.axis('off')

            patches = [
                mpatches.Patch(color=(0, 1, 0), label='TP'),
                mpatches.Patch(color=(1, 0, 0), label='FP'),
                mpatches.Patch(color=(0, 0, 1), label='FN'),
                mpatches.Patch(color=(1, 1, 1), label='TN')
            ]
            plt.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            save_path = os.path.join("outputs", f"{base}_eval.png")
            plt.savefig(save_path)
            plt.show()

    else:
        # 🟡 Si no hay GT, solo mostrar resultados sin métricas ni diff
        print(f"{image_file} -> ⚠️ No se encontró GT. Solo se muestra procesamiento visual.")
        if SHOW_PLOTS:
            plt.figure(figsize=(10, 4))
            plt.suptitle(f"{image_file} - Sin GT disponible")

            plt.subplot(1, 2, 1)
            plt.title('Foreground combinado')
            plt.imshow(combined_foreground)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Máscara Predicha (Combinada)')
            plt.imshow(pred_bool, cmap='gray')
            plt.axis('off')

            # Crear carpeta de salida de máscaras si no existe
            os.makedirs("outputs_mask", exist_ok=True)
            # Guardar la máscara predicha combinada como PNG
            mask_save_path = os.path.join("outputs_mask", f"{base}_mask.png")
            imageio.imwrite(mask_save_path, (pred_bool.astype(np.uint8) * 255))

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join("outputs", f"{base}_pred_only.png")
            plt.savefig(save_path)
            plt.show()


# 🟡 Solo calcular globales si hubo alguna GT evaluada
if len(precisions) > 0:
    global_precision = np.mean(precisions)
    global_recall = np.mean(recalls)
    global_f1 = np.mean(f1s)

    print('\n=== Global pixel-wise metrics ===')
    print(f'Precision: {global_precision:.4f} | Recall: {global_recall:.4f} | F1: {global_f1:.4f}')
else:
    print('\n⚠️ No se encontró ninguna GT. No se calcularon métricas globales.')